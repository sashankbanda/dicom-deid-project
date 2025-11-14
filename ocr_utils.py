#!/usr/bin/env python3
"""
ocr_utils.py

Precise OCR-driven PHI masking for DICOM burned-in text.

Flow:
1. Extract original pixel array and an 8-bit preview image.
2. Run Tesseract word-level OCR (boxes + confidence).
3. For each OCR word box:
   a. crop the *preview* region (same coords as original preview)
   b. build a tight per-pixel text mask using local adaptive threshold + morphology
   c. extract the word string, normalize, compare to metadata PHI and keywords
   d. if matched -> mark those exact stroke pixels (with a tiny padding) for redaction
4. Inpaint+soft-blend only on the union of those precise pixel regions.
5. Write PixelData back, preserve PhotometricInterpretation, BitsAllocated/Stored, SamplesPerPixel where feasible.
"""

import re
from typing import Tuple, Dict, List

import numpy as np
from PIL import Image
import pytesseract
import cv2
import pydicom


# ---------- small tunables ----------
OCR_CONF_THRESHOLD = 40       # keep words with conf >= this (tune per your OCR)
LOCAL_PADDING_PIXELS = 3      # tiny padding around detected text strokes (not box)
MIN_COMPONENT_AREA = 8        # minimum connected component area (pixels) to keep in local mask
KEYWORDS = ['ID', 'NAME', 'DOB', 'DATE', 'TIME', 'ACCESSION', 'AGE', 'HOSPITAL',
            'PATIENT', 'PHYSICIAN', 'INSTITUTION', 'PORTABLE', 'MRN']


# ---------- utilities ----------
def normalize_text(text: str) -> str:
    if not text:
        return ''
    s = str(text)
    s = s.replace('\u2019', "'")
    s = re.sub(r'[^\x00-\x7F]+', '', s)
    s = re.sub(r'[^A-Za-z0-9\s]', '', s)
    return s.upper().strip()


# ---------- preview & conversion ----------
def get_image_from_dicom(ds: pydicom.dataset.Dataset) -> Tuple[np.ndarray, Image.Image]:
    """
    Return (original_array, ocr_preview_pil_image).
    Original array is the raw ds.pixel_array.
    ocr_preview is an 8-bit PIL image used for OCR and local mask extraction.
    """
    try:
        orig = ds.pixel_array
    except Exception as e:
        raise RuntimeError(f"Could not get pixel_array: {e}")

    # build 8-bit preview image preserving contrast approx.
    if orig.ndim == 3:
        # channel-last: RGB or RGBA or other
        if orig.shape[2] == 4:
            vis = cv2.cvtColor(orig, cv2.COLOR_RGBA2RGB)
        elif orig.shape[2] == 3:
            vis = orig
        else:
            vis = orig[..., 0]
        if vis.dtype != np.uint8:
            vis8 = ((vis.astype(np.float32) - float(vis.min())) / max(1.0, float(vis.max() - vis.min())) * 255.0).astype(np.uint8)
        else:
            vis8 = vis.copy()
        pil = Image.fromarray(vis8)
    else:
        if orig.dtype != np.uint8:
            vis8 = ((orig.astype(np.float32) - float(orig.min())) / max(1.0, float(orig.max() - orig.min())) * 255.0).astype(np.uint8)
        else:
            vis8 = orig.copy()
        pil = Image.fromarray(vis8)

    return orig, pil


# ---------- local text stroke mask (very tight) ----------
def build_precise_text_mask(crop_gray: np.ndarray) -> np.ndarray:
    """
    Given a grayscale crop (uint8) containing one word (or small phrase),
    return a mask (same shape) where strokes are marked 255 and background 0.

    Steps:
    - small blur to remove noise
    - adaptive threshold (local) to separate strokes
    - morphological open to remove specks
    - keep components with area >= MIN_COMPONENT_AREA
    - return mask
    """
    if crop_gray.dtype != np.uint8:
        crop = ((crop_gray.astype(np.float32) - float(crop_gray.min())) / max(1.0, float(crop_gray.max() - crop_gray.min())) * 255.0).astype(np.uint8)
    else:
        crop = crop_gray.copy()

    # Slight blur
    b = cv2.GaussianBlur(crop, (3, 3), 0)

    # Adaptive threshold - use mean C with a small negative constant to favor faint strokes
    try:
        th = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 6)
    except Exception:
        _, th = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological open to remove tiny noise, keep letters
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep only reasonably sized connected components (to avoid speckle)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    mask = np.zeros_like(opened)
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= MIN_COMPONENT_AREA:
            mask[labels == i] = 255

    # final small dilate to recover thin strokes that may have been broken
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
    return mask


# ---------- inpaint + soft blend (applies only inside mask) ----------
def inpaint_and_blend_region(orig_arr: np.ndarray, pixel_mask: np.ndarray) -> np.ndarray:
    """
    Inpaint only the pixels where pixel_mask == 255.
    pixel_mask is 2D, same size as orig_arr's image plane.
    Returns array with same dtype/shape as orig_arr.
    """
    if pixel_mask.dtype != np.uint8:
        pixel_mask = pixel_mask.astype(np.uint8)

    # Convert original to 3-channel uint8 for inpainting (so color/grayscale handled uniformly)
    if orig_arr.ndim == 2:
        vis8 = ((orig_arr.astype(np.float32) - float(orig_arr.min())) / max(1.0, float(orig_arr.max() - orig_arr.min())) * 255.0).astype(np.uint8)
        vis_bgr = cv2.cvtColor(vis8, cv2.COLOR_GRAY2BGR)
        is_gray = True
    else:
        vis = orig_arr.copy()
        if vis.dtype != np.uint8:
            vis8 = ((vis.astype(np.float32) - float(vis.min())) / max(1.0, float(vis.max() - vis.min())) * 255.0).astype(np.uint8)
        else:
            vis8 = vis
        if vis8.shape[2] == 4:
            vis_bgr = cv2.cvtColor(vis8, cv2.COLOR_RGBA2BGR)
        elif vis8.shape[2] == 3:
            vis_bgr = vis8
        else:
            vis_bgr = cv2.cvtColor(vis8[..., 0], cv2.COLOR_GRAY2BGR)
        is_gray = False

    # If mask size differs, resize mask nearest
    if pixel_mask.shape != vis_bgr.shape[:2]:
        pixel_mask = cv2.resize(pixel_mask, (vis_bgr.shape[1], vis_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Prepare inpaint mask: must be 8-bit single channel
    inpaint_mask = (pixel_mask > 0).astype(np.uint8) * 255

    # If mask is empty, return original
    if inpaint_mask.sum() == 0:
        return orig_arr

    # Inpaint
    try:
        inpainted = cv2.inpaint(vis_bgr, inpaint_mask, 3, cv2.INPAINT_TELEA)
    except Exception as e:
        raise RuntimeError(f"Inpainting failed: {e}")

    # Soft blend on mask boundary: blur mask and alpha-blend
    blurred = cv2.GaussianBlur(inpaint_mask, (9, 9), 0)
    alpha = (blurred.astype(np.float32) / 255.0)[:, :, None]
    blended = (alpha * inpainted.astype(np.float32) + (1.0 - alpha) * vis_bgr.astype(np.float32)).astype(np.uint8)

    # Convert back to original dtype & channels
    if is_gray:
        blended_gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
        if orig_arr.dtype == np.uint8:
            return blended_gray
        else:
            out = (blended_gray.astype(np.float32) / 255.0 * (orig_arr.max() - orig_arr.min()) + orig_arr.min())
            return out.astype(orig_arr.dtype)
    else:
        if orig_arr.dtype == np.uint8 and blended.shape == orig_arr.shape:
            return blended
        else:
            out = (blended.astype(np.float32) / 255.0 * (orig_arr.max() - orig_arr.min()) + orig_arr.min())
            out_arr = out.astype(orig_arr.dtype)
            if orig_arr.ndim == 3 and orig_arr.shape[2] == 4:
                alpha_chan = np.zeros((out_arr.shape[0], out_arr.shape[1], 1), dtype=out_arr.dtype)
                out_arr = np.concatenate([out_arr, alpha_chan], axis=2)
            return out_arr


# ---------- writeback ----------
def _write_back_pixeldata(ds: pydicom.dataset.Dataset, arr: np.ndarray):
    """
    Safely write arr back into ds.PixelData and adjust basic attributes.
    Only supports 2D grayscale and 3-channel RGB arrays.
    """
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, arr.min(), arr.max())
        arr = ((arr - arr.min()) / max(1e-8, (arr.max() - arr.min())) * 255.0).astype(np.uint8)

    arr = np.ascontiguousarray(arr)

    if arr.ndim == 2:
        ds.Rows, ds.Columns = int(arr.shape[0]), int(arr.shape[1])
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
        if arr.dtype == np.uint8:
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
        elif arr.dtype == np.uint16:
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
        ds.PixelData = arr.tobytes()
        return

    if arr.ndim == 3:
        rows, cols, ch = arr.shape
        ds.Rows, ds.Columns = int(rows), int(cols)
        if ch == 3:
            ds.SamplesPerPixel = 3
            ds.PhotometricInterpretation = getattr(ds, 'PhotometricInterpretation', 'RGB')
            ds.PlanarConfiguration = 0
            if arr.dtype == np.uint8:
                ds.BitsAllocated = 8
                ds.BitsStored = 8
                ds.HighBit = 7
            elif arr.dtype == np.uint16:
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
            ds.PixelData = arr.tobytes()
            return
        elif ch == 4:
            rgb = arr[..., :3]
            ds.SamplesPerPixel = 3
            ds.PhotometricInterpretation = getattr(ds, 'PhotometricInterpretation', 'RGB')
            ds.PlanarConfiguration = 0
            ds.PixelData = rgb.tobytes()
            return
        else:
            gray = arr[..., 0]
            _write_back_pixeldata(ds, gray)
            return

    raise ValueError("Unsupported array shape for PixelData writeback.")


# ---------- main entry ----------
def run_ocr_and_mask(ds: pydicom.dataset.Dataset, phis_to_mask: Dict[str, str], output_path: str):
    """
    Core function called from deidentify_runner.
    Matches OCR against phis_to_mask (dict of metadata PHI) and keywords.
    Redacts only the precise stroke pixels for matched words.
    """
    try:
        original_arr, preview_pil = get_image_from_dicom(ds)
    except Exception as e:
        print(f"  Could not extract image: {e}")
        try:
            ds.save_as(output_path)
        except Exception:
            pass
        return

    # Run Tesseract word-level detection
    tconf = '--oem 1 --psm 6'  # word/line layout
    try:
        ocr = pytesseract.image_to_data(preview_pil, config=tconf, output_type=pytesseract.Output.DICT)
    except Exception as e:
        print(f"  Tesseract failed: {e}")
        try:
            ds.save_as(output_path)
        except Exception:
            pass
        return

    # Prepare PHI list and keywords
    metadata_values = [normalize_text(v) for v in phis_to_mask.values() if v and len(str(v)) > 2]
    keywords = KEYWORDS

    # Prepare an empty per-pixel mask on the preview resolution
    preview_gray = np.array(preview_pil.convert('L'))
    h, w = preview_gray.shape
    final_pixel_mask = np.zeros((h, w), dtype=np.uint8)

    n_words = len(ocr.get('text', []))
    for i in range(n_words):
        text = ocr['text'][i].strip()
        if not text:
            continue
        conf_str = ocr['conf'][i]
        try:
            conf = float(conf_str) if conf_str not in (None, '') else -1.0
        except Exception:
            conf = -1.0
        # Skip very low confidence words
        if conf < OCR_CONF_THRESHOLD:
            continue

        # Read box
        x = int(ocr['left'][i])
        y = int(ocr['top'][i])
        ww = int(ocr['width'][i])
        hh = int(ocr['height'][i])
        if ww <= 0 or hh <= 0:
            continue

        # Crop corresponding area from preview_gray
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w, x + ww)
        y1 = min(h, y + hh)
        crop = preview_gray[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        # Build precise stroke mask for this crop
        local_mask = build_precise_text_mask(crop)

        # If nothing detected inside box (OCR saw it but our stroke mask didn't), try a tiny loosened threshold:
        if local_mask.sum() == 0:
            # gentle fallback: small threshold to pick thin strokes
            try:
                _, th2 = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
                # keep only components big enough
                nl, labs, stats, _ = cv2.connectedComponentsWithStats(th2, connectivity=8)
                mask2 = np.zeros_like(th2)
                for j in range(1, nl):
                    if int(stats[j, cv2.CC_STAT_AREA]) >= max(2, MIN_COMPONENT_AREA // 2):
                        mask2[labs == j] = 255
                local_mask = mask2
            except Exception:
                pass

        # Normalize OCR text and check match against metadata and keywords
        nt = normalize_text(text)
        matched = False
        # exact / substring match against metadata values
        for mv in metadata_values:
            if not mv:
                continue
            if mv in nt or nt in mv:
                matched = True
                break
        # keyword match
        if not matched:
            for kw in keywords:
                if kw in nt:
                    matched = True
                    break
        # date or long numeric heuristics
        if not matched:
            if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text) or re.search(r'\d{6,}', text):
                matched = True

        # If matched, copy local_mask into final_pixel_mask (with tiny padding)
        if matched and local_mask.sum() > 0:
            # pad local stroke mask by LOCAL_PADDING_PIXELS inside the crop coords
            pad = LOCAL_PADDING_PIXELS
            # dilate mask within crop to get small padding
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
            padded_local = cv2.dilate(local_mask, k, iterations=1)

            # place into final mask at correct coordinates
            final_pixel_mask[y0:y1, x0:x1] = cv2.bitwise_or(final_pixel_mask[y0:y1, x0:x1], padded_local)

        # If matched but local mask empty (rare), fall back to a minimal rectangular mask of box with tiny pad
        elif matched and local_mask.sum() == 0:
            pad = max(1, min(LOCAL_PADDING_PIXELS, hh // 6))
            rx0 = max(0, x0 - pad)
            ry0 = max(0, y0 - pad)
            rx1 = min(w, x1 + pad)
            ry1 = min(h, y1 + pad)
            final_pixel_mask[ry0:ry1, rx0:rx1] = 255

    # If final_pixel_mask is empty -> nothing matched; save unchanged
    if final_pixel_mask.sum() == 0:
        print("  No matched PHI words found (precise). Saving metadata-cleaned DICOM unchanged.")
        try:
            ds.save_as(output_path)
        except Exception as e:
            print(f"  Failed to save DICOM: {e}")
        return

    # At this stage, final_pixel_mask marks the exact strokes to redact on the preview resolution.
    # We need to map this to the original pixel-array coordinates. The preview was created by scaling 16-bit/float -> 8-bit
    # but kept same spatial resolution, so coordinates are consistent with original ds.pixel_array shape if same HxW.
    # If original array has different size, we must scale mask accordingly.

    orig = original_arr
    # Determine mapping: preview shape vs orig shape
    if orig.ndim == 3:
        orig_h, orig_w = orig.shape[0], orig.shape[1]
    else:
        orig_h, orig_w = orig.shape[0], orig.shape[1]

    mask_to_write = final_pixel_mask
    if mask_to_write.shape != (orig_h, orig_w):
        # resize mask to original resolution nearest
        mask_to_write = cv2.resize(mask_to_write, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Inpaint and blend only on mask_to_write (precise pixels)
    try:
        out_arr = inpaint_and_blend_region(orig, mask_to_write)
    except Exception as e:
        print(f"  Inpaint failed: {e}. Applying conservative opaque fallback.")
        # fallback: cover the exact mask pixels with black
        fallback = orig.copy()
        if fallback.ndim == 2:
            fallback[mask_to_write > 0] = 0
        else:
            fallback[mask_to_write > 0, :] = 0
        try:
            _write_back_pixeldata(ds, fallback)
            ds.save_as(output_path)
            print(f"  Saved fallback masked file to {output_path}")
        except Exception as ee:
            print(f"  Failed to save fallback: {ee}")
        return

    # Write final image back to DICOM
    try:
        _write_back_pixeldata(ds, out_arr)
        ds.save_as(output_path)
        print(f"  Saved de-identified file to {output_path}")
    except Exception as e:
        print(f"  Failed to write PixelData back: {e}")
        try:
            ds.save_as(output_path)
        except Exception as ee:
            print(f"  Failed to save file: {ee}")
