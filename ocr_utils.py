#!/usr/bin/env python3
"""
ocr_utils.py - Improved OCR + faint-text detection + inpainting blending for DICOM de-id.

Features
- Tesseract OCR token detection (with confidence handling)
- Keyword and metadata-driven matches (explicit matching against metadata PHI)
- Faint / washed-out text detector using top-hat + adaptive threshold
- Merge OCR and faint-text masks, morphological smoothing
- Inpaint using OpenCV (TELEA) and soft-edge blending to avoid sharp seams
- Safe PixelData writeback preserving PhotometricInterpretation, SamplesPerPixel,
  BitsAllocated/BitsStored/HighBit where possible
"""

import re
from typing import Tuple, Dict, List

import numpy as np
from PIL import Image
import pytesseract
import cv2
import pydicom


# -----------------------
# Utility / normalization
# -----------------------
def normalize_text(text: str) -> str:
    if not text:
        return ''
    s = str(text)
    s = s.replace('\u2019', "'")
    s = re.sub(r'[^\x00-\x7F]+', '', s)
    s = re.sub(r'[^A-Za-z0-9\s]', '', s)
    return s.upper().strip()


# -----------------------
# Preprocessing for OCR
# -----------------------
def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    """
    Convert PIL image to an 8-bit image tuned for Tesseract:
    - Grayscale, percentile contrast stretch
    - Conditional inversion (if background dark)
    - Otsu threshold fallback to adaptive threshold
    """
    gray = pil_img.convert('L')
    arr = np.array(gray).astype(np.uint8)

    # Contrast stretch 2nd-98th percentile
    p2, p98 = np.percentile(arr, (2, 98))
    if p98 > p2:
        arr = np.clip((arr - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)

    # If overall background is dark, invert so text becomes dark on light
    if arr.mean() < 127:
        arr = 255 - arr

    # Try Otsu; fallback to adaptive
    try:
        _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        arr = th
    except Exception:
        arr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

    return Image.fromarray(arr)


# -----------------------
# DICOM -> preview helper
# -----------------------
def get_image_from_dicom(ds: pydicom.dataset.Dataset) -> Tuple[np.ndarray, Image.Image]:
    """
    Return (original_array, ocr_preview_pil_image)
    - original_array is the raw numpy array from ds.pixel_array (unchanged)
    - ocr_preview is an 8-bit PIL image used for OCR and faint-text detection
    """
    try:
        orig = ds.pixel_array  # may raise
    except Exception as e:
        raise RuntimeError(f"Could not read pixel array: {e}")

    # Build an 8-bit preview image for OCR (keep aspect & approximate contrast)
    if orig.ndim == 3:
        # If channel-last RGB/RGBA, convert to RGB PIL
        if orig.shape[2] == 4:
            vis = cv2.cvtColor(orig, cv2.COLOR_RGBA2RGB)
        elif orig.shape[2] == 3:
            vis = orig
        else:
            # unexpected channels, take first channel
            vis = orig[..., 0]
        if vis.dtype != np.uint8:
            vis8 = ((vis.astype(np.float32) - float(vis.min())) / max(1.0, float(vis.max() - vis.min())) * 255.0).astype(np.uint8)
        else:
            vis8 = vis.copy()
        pil = Image.fromarray(vis8)
    else:
        # grayscale/single-channel: scale to 8-bit for preview
        if orig.dtype != np.uint8:
            vis8 = ((orig.astype(np.float32) - float(orig.min())) / max(1.0, float(orig.max() - orig.min())) * 255.0).astype(np.uint8)
        else:
            vis8 = orig.copy()
        pil = Image.fromarray(vis8)

    ocr_img = preprocess_image_for_ocr(pil)
    return orig, ocr_img


# -----------------------
# Faint-text detector
# -----------------------
def detect_faint_text(preview_img: Image.Image) -> np.ndarray:
    """
    Detect faint/washed-out bright text using morphological top-hat and adaptive thresholding.
    Returns a single-channel uint8 mask (255 where text likely is).
    """
    gray = np.array(preview_img.convert('L')).astype(np.uint8)

    # Small Gaussian to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Use a relatively large structuring element for top-hat to reveal bright strokes on uneven background
    h = max(9, int(max(3, min(gray.shape) * 0.02)))  # adaptive kernel size
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (h, h))
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, k)

    # Adaptive threshold on top-hat image; invert so bright strokes become white
    try:
        th = cv2.adaptiveThreshold(tophat, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 31, -8)
    except Exception:
        _, th = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean with morphological close to fill letters
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Remove very small noise
    nb_components, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    mask = np.zeros_like(th)
    for i in range(1, nb_components):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 30:  # small area filter (tunable)
            mask[labels == i] = 255

    return mask


# -----------------------
# OCR mask builder
# -----------------------
def _build_mask_from_ocr(ocr_data: dict, image_shape: Tuple[int, int], conf_threshold: int = 30) -> np.ndarray:
    """
    Build a mask from OCR tokens. image_shape is (h, w) of the OCR preview image.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    heights = [int(h) for h in ocr_data.get('height', []) if h and int(h) > 0]
    median_h = int(np.median(heights)) if heights else 10

    n = len(ocr_data.get('text', []))
    for i in range(n):
        text = ocr_data['text'][i]
        conf_str = ocr_data['conf'][i]
        try:
            conf = float(conf_str) if conf_str not in (None, '') else -1.0
        except Exception:
            conf = -1.0
        if not text.strip() or conf < conf_threshold:
            continue

        x = int(ocr_data['left'][i])
        y = int(ocr_data['top'][i])
        w = int(ocr_data['width'][i])
        h = int(ocr_data['height'][i])
        buf = max(3, int(h * 0.4))
        x0 = max(0, x - buf)
        y0 = max(0, y - buf)
        x1 = min(image_shape[1], x + w + buf)
        y1 = min(image_shape[0], y + h + buf)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)

    # Connect nearby boxes
    ksize = max(3, (median_h // 2) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


# -----------------------
# PHI matching heuristic
# -----------------------
def _matches_phi(text: str, metadata_values: List[str], keywords: List[str]) -> bool:
    nt = normalize_text(text)
    if not nt:
        return False
    # metadata exact / substring match
    for m in metadata_values:
        if not m:
            continue
        if m in nt or nt in m:
            return True
    # keyword
    for k in keywords:
        if k in nt:
            return True
    # date or long numeric
    if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text) or re.search(r'\d{6,}', text):
        return True
    return False


# -----------------------
# Inpaint + soft blend
# -----------------------
def _inpaint_and_blend(orig_arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Inpaint orig_arr on mask (mask: 0/255, same HxW).
    Returns array with same dtype/shape where possible.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Create 3-channel uint8 image for inpainting
    if orig_arr.ndim == 2:
        vis8 = ((orig_arr.astype(np.float32) - float(orig_arr.min())) / max(1.0, float(orig_arr.max() - orig_arr.min())) * 255.0).astype(np.uint8)
        vis_bgr = cv2.cvtColor(vis8, cv2.COLOR_GRAY2BGR)
        is_gray = True
    else:
        vis = orig_arr.copy()
        if vis.dtype != np.uint8:
            vis8 = ((vis.astype(np.float32) - float(vis.min())) / max(1.0, float(vis.max() - vis.min())) * 255.0).astype(np.uint8)
        else:
            vis8 = vis.copy()
        if vis8.shape[2] == 4:
            vis_bgr = cv2.cvtColor(vis8, cv2.COLOR_RGBA2BGR)
        elif vis8.shape[2] == 3:
            vis_bgr = vis8
        else:
            vis_bgr = cv2.cvtColor(vis8[..., 0], cv2.COLOR_GRAY2BGR)
        is_gray = False

    # Resize mask to match vis_bgr in case shapes differ
    if mask.shape != vis_bgr.shape[:2]:
        mask_resized = cv2.resize(mask, (vis_bgr.shape[1], vis_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask

    # Inpaint
    inpaint_radius = 3
    try:
        inpainted = cv2.inpaint(vis_bgr, mask_resized, inpaint_radius, cv2.INPAINT_TELEA)
    except Exception as e:
        raise RuntimeError(f"inpaint failed: {e}")

    # Soft-edge blend: Gaussian blur mask and linear alpha
    blurred = cv2.GaussianBlur(mask_resized, (9, 9), 0)
    alpha = (blurred.astype(np.float32) / 255.0)[:, :, None]
    blended = (alpha * inpainted.astype(np.float32) + (1.0 - alpha) * vis_bgr.astype(np.float32)).astype(np.uint8)

    # Convert back to original dtype/range
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
            # If original had 4 channels, attempt to re-attach an alpha channel (zeros)
            if orig_arr.ndim == 3 and orig_arr.shape[2] == 4:
                alpha_chan = np.zeros((out_arr.shape[0], out_arr.shape[1], 1), dtype=out_arr.dtype)
                out_arr = np.concatenate([out_arr, alpha_chan], axis=2)
            return out_arr


# -----------------------
# PixelData writeback
# -----------------------
def _write_back_pixeldata(ds: pydicom.dataset.Dataset, arr: np.ndarray):
    """
    Update ds.PixelData and related metadata fields to reflect arr.
    Handles 2D grayscale and 3-channel RGB arrays.
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
            # drop alpha, store as RGB
            rgb = arr[..., :3]
            ds.SamplesPerPixel = 3
            ds.PhotometricInterpretation = getattr(ds, 'PhotometricInterpretation', 'RGB')
            ds.PlanarConfiguration = 0
            ds.PixelData = rgb.tobytes()
            return
        else:
            # unexpected channel count: reduce to first channel
            gray = arr[..., 0]
            _write_back_pixeldata(ds, gray)
            return

    raise ValueError("Unsupported array shape for PixelData writeback.")


# -----------------------
# Main entry
# -----------------------
def run_ocr_and_mask(ds: pydicom.dataset.Dataset, phis_to_mask: Dict[str, str], output_path: str):
    """
    Extracts pixel array and OCR preview, detects PHI regions using:
      - direct OCR token matches against metadata and keywords
      - geometry-based OCR mask fallback
      - faint-text detector (top-hat)
      - keyword-driven global expansion
    Then inpaints + blends and writes PixelData back safely.
    """
    try:
        original_arr, ocr_preview = get_image_from_dicom(ds)
    except Exception as e:
        print(f"  Could not prepare preview image: {e}")
        try:
            ds.save_as(output_path)
        except Exception as ee:
            print(f"  Failed saving original: {ee}")
        return

    # Run OCR
    tconf = '--oem 1 --psm 6'
    try:
        ocr_data = pytesseract.image_to_data(ocr_preview, config=tconf, output_type=pytesseract.Output.DICT)
    except Exception as e:
        print(f"  Tesseract failed: {e}. Saving cleaned metadata DICOM.")
        try:
            ds.save_as(output_path)
        except Exception as ee:
            print(f"  Failed to save DICOM: {ee}")
        return

    # Prepare metadata and keyword lists
    metadata_values = [normalize_text(v) for v in phis_to_mask.values() if v and len(str(v)) > 2]
    keywords = ['ID', 'NAME', 'DOB', 'DATE', 'TIME', 'ACCESSION', 'AGE',
                'HOSPITAL', 'PATIENT', 'PHYSICIAN', 'INSTITUTION', 'PORTABLE', 'MRN']

    # Build OCR mask from tokens that match PHI heuristics
    preview_gray = np.array(ocr_preview.convert('L'))
    image_shape = preview_gray.shape  # (h, w)
    used_mask = np.zeros(image_shape, dtype=np.uint8)

    n = len(ocr_data.get('text', []))
    for i in range(n):
        text = ocr_data['text'][i]
        conf_str = ocr_data['conf'][i]
        try:
            conf = float(conf_str) if conf_str not in (None, '') else -1.0
        except Exception:
            conf = -1.0
        if not text.strip() or conf < 30:
            continue
        if _matches_phi(text, metadata_values, keywords):
            x = int(ocr_data['left'][i])
            y = int(ocr_data['top'][i])
            w = int(ocr_data['width'][i])
            h = int(ocr_data['height'][i])
            buf = max(3, int(h * 0.45))
            x0 = max(0, x - buf)
            y0 = max(0, y - buf)
            x1 = min(image_shape[1], x + w + buf)
            y1 = min(image_shape[0], y + h + buf)
            cv2.rectangle(used_mask, (x0, y0), (x1, y1), 255, -1)

    # If no direct matches found, fall back to OCR geometry mask (wider)
    if used_mask.sum() == 0:
        used_mask = _build_mask_from_ocr(ocr_data, image_shape, conf_threshold=30)

    # Faint-text mask (detect very light text that OCR may miss)
    faint_mask = detect_faint_text(ocr_preview)

    # Merge masks
    merged = cv2.bitwise_or(used_mask, faint_mask)

    # Keyword-driven expansion:
    # if any OCR token contains partial keyword (like HOSP or HOSPT), expand mask along same horizontal band
    for i in range(n):
        txt = ocr_data['text'][i]
        if not txt.strip():
            continue
        nt = normalize_text(txt)
        for kw in keywords:
            if kw in nt or (len(kw) > 4 and kw[:4] in nt) or (len(nt) >= 4 and nt in kw):
                # expand along line
                y = int(ocr_data['top'][i])
                h = int(ocr_data['height'][i]) if ocr_data['height'][i] else 12
                band = max(12, int(h * 3))
                y0 = max(0, y - band)
                y1 = min(image_shape[0], y + band)
                # fill across entire width to catch broken fragments (but keep morphological ops to avoid huge masks)
                merged[y0:y1, :] = cv2.bitwise_or(merged[y0:y1, :], 255)

    # Morphological smoothing
    heights = [int(h) for h in ocr_data.get('height', []) if h and int(h) > 0]
    median_h = int(np.median(heights)) if heights else 10
    ksize = max(3, (median_h // 2) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    merged = cv2.dilate(merged, kernel, iterations=1)
    merged = cv2.medianBlur(merged, 5)

    # If still empty, nothing to inpaint
    if merged.sum() == 0:
        print("  No likely PHI regions found (including faint text). Saving metadata-cleaned DICOM unchanged.")
        try:
            ds.save_as(output_path)
        except Exception as e:
            print(f"  Failed to save DICOM: {e}")
        return

    # Inpaint and blend back into original array
    try:
        inpainted = _inpaint_and_blend(original_arr, merged)
    except Exception as e:
        print(f"  Inpainting failed: {e}. Falling back to opaque rectangles.")
        # Fallback: apply opaque zero rectangles on original array for safety
        fallback = original_arr.copy()
        ys, xs = np.where(merged > 0)
        if ys.size:
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            if fallback.ndim == 2:
                fallback[y0:y1 + 1, x0:x1 + 1] = 0
            else:
                fallback[y0:y1 + 1, x0:x1 + 1, :] = 0
        try:
            _write_back_pixeldata(ds, fallback)
            ds.save_as(output_path)
            print(f"  Saved fallback masked file to {output_path}")
        except Exception as ee:
            print(f"  Failed saving fallback: {ee}")
        return

    # Write back and save final DICOM
    try:
        _write_back_pixeldata(ds, inpainted)
        ds.save_as(output_path)
        print(f"  Saved de-identified file to {output_path}")
    except Exception as e:
        print(f"  Failed to write PixelData back: {e}")
        try:
            ds.save_as(output_path)
        except Exception as ee:
            print(f"  Failed to save file: {ee}")
