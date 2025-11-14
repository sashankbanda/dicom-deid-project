#!/usr/bin/env python3
"""
ocr_utils.py

Improved OCR-driven PHI masking for DICOM burned-in text.

Replaces original build_precise_text_mask with a multi-method approach,
adds char-level fallback, expands/merges masks before inpainting,
and provides an optional debug saving mode controlled by SAVE_DEBUG.
"""

import re
import uuid
import os
from typing import Tuple, Dict, List

import numpy as np
from PIL import Image
import pytesseract
import cv2
import pydicom

# ---------- tuning flags ----------
OCR_CONF_THRESHOLD = 30       # lowered to capture faint words; tune on dataset
LOCAL_PADDING_PIXELS = 3      # tiny padding around detected text strokes (not box)
MIN_COMPONENT_AREA = 8        # minimum connected component area (pixels) to keep in local mask
SAVE_DEBUG = True           # default off; set True to persist debug images to cwd
DEBUG_DIR = 'ocr_debug'       # directory to save debug artifacts when SAVE_DEBUG=True

KEYWORDS = [
    "PATIENT", "PATIENTID", "PATIENT ID",
    "ID", "MRN", "MEDICAL RECORD",
    "NAME", "PATIENT NAME", "INSTITUTION", "HOSPITAL",
    "FACILITY", "CENTER", "CLINIC",
    "PHYSICIAN", "DOCTOR", "REFERRING", "PERFORMING", "OPERATOR",
    "STUDY", "STUDY DATE", "EXAM", "ACQUISITION",
    "DATE", "TIME", "AGE", "DOB",

    # DICOM fields (kept in normalized form too)
    "PATIENTNAME", "PATIENTID", "PATIENTBIRTHDATE", "PATIENTBIRTHTIME",
    "PATIENTSEX", "PATIENTAGE", "OTHERPATIENTIDS", "OTHERPATIENTNAMES",
    "PATIENTADDRESS", "PATIENTTELEPHONENUMBERS", "INSTITUTIONNAME",
    "INSTITUTIONADDRESS", "REFERRINGPHYSICIANNAME", "REFERRINGPHYSICIANTELEPHONENUMBERS",
    "REQUESTINGPHYSICIAN", "INSTITUTIONALDEPARTMENTNAME", "PHYSICIANSOFRECORD",
    "PERFORMINGPHYSICIANNAME", "NAMEOFPHYSICIANSREADINGSTUDY", "OPERATORSNAME",
    "STUDYDATE", "STUDYTIME", "ACCESSIONNUMBER", "STUDYINSTANCEUID",
    "SERIESINSTANCEUID", "DEVICESERIALNUMBER", "DEVICESERIESNUMBER", "STATIONNAME",
    "STUDYID", "SERIESNUMBER", "SCHEDULEDPERFORMINGPHYSICIANNAME", "VERIFYINGOBSERVERNAME",
    "VERIFYINGOBSERVERIDENTIFICATIONCODESEQUENCE",
]


# ---------- utilities ----------
def _ensure_debug_dir():
    if SAVE_DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)


def _save_debug(preview_pil: Image.Image, final_pixel_mask: np.ndarray, prefix: str):
    if not SAVE_DEBUG:
        return
    _ensure_debug_dir()
    uid = uuid.uuid4().hex[:8]
    preview_path = os.path.join(DEBUG_DIR, f'{prefix}_{uid}_preview.png')
    mask_path = os.path.join(DEBUG_DIR, f'{prefix}_{uid}_mask_preview.png')
    overlay_path = os.path.join(DEBUG_DIR, f'{prefix}_{uid}_overlay.png')
    try:
        preview_pil.save(preview_path)
    except Exception:
        pass
    try:
        cv2.imwrite(mask_path, final_pixel_mask)
    except Exception:
        pass
    try:
        pg = np.array(preview_pil.convert('RGB'))
        mask3 = np.zeros_like(pg)
        mask3[final_pixel_mask > 0] = (0, 0, 255)
        overlay = cv2.addWeighted(pg, 1.0, mask3, 0.5, 0)
        Image.fromarray(overlay).save(overlay_path)
    except Exception:
        pass


def normalize_text(text: str) -> str:
    if not text:
        return ''
    s = str(text)
    s = s.replace('\u2019', "'")
    # drop non-ascii
    s = re.sub(r'[^\x00-\x7F]+', '', s)
    # remove punctuation but keep digits/letters/space
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
            vis8 = ((vis.astype(np.float32) - float(vis.min())) /
                    max(1.0, float(vis.max() - vis.min())) * 255.0).astype(np.uint8)
        else:
            vis8 = vis.copy()
        pil = Image.fromarray(vis8)
    else:
        if orig.dtype != np.uint8:
            vis8 = ((orig.astype(np.float32) - float(orig.min())) /
                    max(1.0, float(orig.max() - orig.min())) * 255.0).astype(np.uint8)
        else:
            vis8 = orig.copy()
        pil = Image.fromarray(vis8)

    return orig, pil


# ---------- robust local text stroke mask (multi-method) ----------
def build_precise_text_mask(crop_gray: np.ndarray) -> np.ndarray:
    """
    More robust local mask:
    - use adaptive threshold (existing)
    - use Otsu threshold
    - use top-hat (morphological) to reveal light text on dark background
    - use morphological gradient / edge dilate to capture anti-aliased strokes
    - combine results, filter by area, and lightly dilate to reconnect broken strokes
    """
    if crop_gray.dtype != np.uint8:
        crop = ((crop_gray.astype(np.float32) - float(crop_gray.min())) /
                max(1.0, float(crop_gray.max() - crop_gray.min())) * 255.0).astype(np.uint8)
    else:
        crop = crop_gray.copy()

    h, w = crop.shape
    # small blur
    b = cv2.GaussianBlur(crop, (3, 3), 0)

    masks = []

    # adaptive threshold
    try:
        th_adapt = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY_INV, 15, 6)
        masks.append(th_adapt)
    except Exception:
        pass

    # Otsu threshold
    try:
        _, th_otsu = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        masks.append(th_otsu)
    except Exception:
        pass

    # top-hat to reveal light text on darker backgrounds
    try:
        kx = 9 if w > 30 else 5
        ky = 3 if h > 12 else 2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, kx), max(2, ky)))
        tophat = cv2.morphologyEx(b, cv2.MORPH_TOPHAT, kernel)
        _, th_top = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(th_top)
    except Exception:
        pass

    # morphological gradient to capture edges/anti-aliased strokes
    try:
        grad = cv2.morphologyEx(b, cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        _, th_grad = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_grad = cv2.dilate(th_grad, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
        masks.append(th_grad)
    except Exception:
        pass

    # combine masks
    if len(masks) == 0:
        combined = np.zeros_like(crop)
    else:
        combined = masks[0].copy()
        for m in masks[1:]:
            combined = cv2.bitwise_or(combined, m)

    # clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    # keep only connected components above area threshold
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    mask = np.zeros_like(combined)
    min_area = max(2, MIN_COMPONENT_AREA // 2)
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            mask[labels == i] = 255

    # small dilate to reconnect thin strokes
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
    return mask


# ---------- inpaint + soft blend (applies only inside mask) ----------
def inpaint_and_blend_region(orig_arr: np.ndarray, pixel_mask: np.ndarray) -> np.ndarray:
    """
    Inpaint only the pixels where pixel_mask == 255.
    Returns array with same dtype/shape as orig_arr.
    """
    if pixel_mask.dtype != np.uint8:
        pixel_mask = pixel_mask.astype(np.uint8)

    # Convert original to 3-channel uint8 for inpainting
    if orig_arr.ndim == 2:
        vis8 = ((orig_arr.astype(np.float32) - float(orig_arr.min())) /
                max(1.0, float(orig_arr.max() - orig_arr.min())) * 255.0).astype(np.uint8)
        vis_bgr = cv2.cvtColor(vis8, cv2.COLOR_GRAY2BGR)
        is_gray = True
    else:
        vis = orig_arr.copy()
        if vis.dtype != np.uint8:
            vis8 = ((vis.astype(np.float32) - float(vis.min())) /
                    max(1.0, float(vis.max() - vis.min())) * 255.0).astype(np.uint8)
        else:
            vis8 = vis
        if vis8.shape[2] == 4:
            vis_bgr = cv2.cvtColor(vis8, cv2.COLOR_RGBA2BGR)
        elif vis8.shape[2] == 3:
            vis_bgr = vis8
        else:
            vis_bgr = cv2.cvtColor(vis8[..., 0], cv2.COLOR_GRAY2BGR)
        is_gray = False

    # Resize mask if needed
    if pixel_mask.shape != vis_bgr.shape[:2]:
        pixel_mask = cv2.resize(pixel_mask, (vis_bgr.shape[1], vis_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    inpaint_mask = (pixel_mask > 0).astype(np.uint8) * 255
    if inpaint_mask.sum() == 0:
        return orig_arr

    try:
        inpainted = cv2.inpaint(vis_bgr, inpaint_mask, 3, cv2.INPAINT_TELEA)
    except Exception as e:
        raise RuntimeError(f"Inpainting failed: {e}")

    # soft blend on mask boundary
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

    # prepare Tesseract configs
    tconf = '--oem 1 --psm 6'  # word/line layout

    # Run Tesseract word-level detection
    try:
        ocr = pytesseract.image_to_data(preview_pil, config=tconf, output_type=pytesseract.Output.DICT)
    except Exception as e:
        print(f"  Tesseract failed (word-level): {e}")
        try:
            ds.save_as(output_path)
        except Exception:
            pass
        return

    # Optionally get char boxes (helps when word boxes are fragmented)
    try:
        char_boxes_raw = pytesseract.image_to_boxes(preview_pil, config=tconf)
    except Exception:
        char_boxes_raw = ''

    # Prepare PHI list and keywords
    metadata_values = [normalize_text(v) for v in phis_to_mask.values() if v and len(str(v)) > 1]
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

        # accept slightly lower confidences but keep numeric fallback for IDs/dates
        if conf < OCR_CONF_THRESHOLD:
            if not re.search(r'\d{4,}', text):
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

        # If nothing detected inside box (OCR saw it but our stroke mask didn't), try char-box fallback:
        if local_mask.sum() == 0 and char_boxes_raw:
            # build a local mask by mapping char boxes that overlap this word box
            try:
                grid = np.zeros_like(crop)
                for line in char_boxes_raw.strip().splitlines():
                    parts = line.split(' ')
                    if len(parts) < 5:
                        continue
                    ch, bx0, by0, bx1, by1 = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                    # pytesseract image_to_boxes uses origin at bottom-left
                    ch_x0 = bx0
                    ch_y0 = h - by1
                    ch_x1 = bx1
                    ch_y1 = h - by0
                    # check overlap with current word bbox
                    if ch_x1 < x0 or ch_x0 > x1 or ch_y1 < y0 or ch_y0 > y1:
                        continue
                    # intersect coordinates in crop-local space
                    lx0 = max(0, ch_x0 - x0); ly0 = max(0, ch_y0 - y0)
                    lx1 = min(ww, ch_x1 - x0); ly1 = min(hh, ch_y1 - y0)
                    if lx1 > lx0 and ly1 > ly0:
                        grid[ly0:ly1, lx0:lx1] = 255
                # filter small comps
                nl, labs, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
                mask2 = np.zeros_like(grid)
                for j in range(1, nl):
                    if int(stats[j, cv2.CC_STAT_AREA]) >= max(2, MIN_COMPONENT_AREA // 2):
                        mask2[labs == j] = 255
                local_mask = mask2
            except Exception:
                pass

        # gentle fallback using Otsu if still empty (already present in old code)
        if local_mask.sum() == 0:
            try:
                _, th2 = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
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
            pad = LOCAL_PADDING_PIXELS
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
            padded_local = cv2.dilate(local_mask, k, iterations=1)
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

    # Optionally save debug previews
    try:
        _save_debug(preview_pil, final_pixel_mask, prefix='ocr_preview')
    except Exception:
        pass

    # Map mask to original resolution
    orig = original_arr
    if orig.ndim == 3:
        orig_h, orig_w = orig.shape[0], orig.shape[1]
    else:
        orig_h, orig_w = orig.shape[0], orig.shape[1]

    mask_to_write = final_pixel_mask
    if mask_to_write.shape != (orig_h, orig_w):
        mask_to_write = cv2.resize(mask_to_write, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Expand and merge mask components before inpainting
    if mask_to_write.sum() > 0:
        kernel_size = max(3, LOCAL_PADDING_PIXELS * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_to_write = cv2.morphologyEx(mask_to_write, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_to_write = cv2.dilate(mask_to_write, kernel, iterations=1)

    # Inpaint and blend only on mask_to_write (precise pixels)
    try:
        out_arr = inpaint_and_blend_region(orig, mask_to_write)
        # write back
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
        return
    except Exception as e:
        print(f"  Inpaint failed: {e}. Applying conservative fallback.")

    # Fallback: apply blurred + opaque region fill per connected component
    try:
        fallback = orig.copy()
        nl, labs, stats, _ = cv2.connectedComponentsWithStats((mask_to_write > 0).astype(np.uint8), connectivity=8)
        for j in range(1, nl):
            area = int(stats[j, cv2.CC_STAT_AREA])
            if area < 5:
                continue
            x = int(stats[j, cv2.CC_STAT_LEFT])
            y = int(stats[j, cv2.CC_STAT_TOP])
            wbox = int(stats[j, cv2.CC_STAT_WIDTH])
            hbox = int(stats[j, cv2.CC_STAT_HEIGHT])
            rx0 = max(0, x - 2); ry0 = max(0, y - 2)
            rx1 = min(orig_w, x + wbox + 2); ry1 = min(orig_h, y + hbox + 2)
            roi = fallback[ry0:ry1, rx0:rx1]
            if roi.size == 0:
                continue
            try:
                tmp = roi.copy()
                if tmp.dtype != np.uint8:
                    tmp8 = ((tmp.astype(np.float32) - tmp.min()) / max(1e-8, tmp.max() - tmp.min()) * 255.0).astype(np.uint8)
                    tmp8 = cv2.GaussianBlur(tmp8, (7, 7), 0)
                    tmp = (tmp8.astype(np.float32) / 255.0 * (roi.max() - roi.min()) + roi.min()).astype(roi.dtype)
                else:
                    tmp = cv2.GaussianBlur(tmp, (7, 7), 0)
                # fill with blurred content then an opaque rectangle to be safe
                fallback[ry0:ry1, rx0:rx1] = tmp
                if fallback.ndim == 2:
                    fallback[ry0:ry1, rx0:rx1] = 0
                else:
                    fallback[ry0:ry1, rx0:rx1, :] = 0
            except Exception:
                # final conservative fallback: black rectangle
                try:
                    if fallback.ndim == 2:
                        fallback[ry0:ry1, rx0:rx1] = 0
                    else:
                        fallback[ry0:ry1, rx0:rx1, :] = 0
                except Exception:
                    pass
        # write fallback
        try:
            _write_back_pixeldata(ds, fallback)
            ds.save_as(output_path)
            print(f"  Saved fallback masked file to {output_path}")
        except Exception as ee:
            print(f"  Failed to save fallback: {ee}")
    except Exception as ee:
        print(f"  Fallback failed: {ee}. Saving metadata-cleaned DICOM without pixel changes.")
        try:
            ds.save_as(output_path)
        except Exception:
            pass
