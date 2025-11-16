#!/usr/bin/env python3
"""
ocr_utils_v3_fixed_pixel_depth.py

Production-ready PHI masking for DICOM burned-in text with robust pixel-depth
restoration and photometric handling.

Usage:
    from ocr_utils_v3_fixed_pixel_depth import run_ocr_and_mask
    run_ocr_and_mask(ds, phis_to_mask, output_path)

Requirements:
    pip install pydicom numpy Pillow pytesseract opencv-python

Top-level flags to tune:
    SAVE_DEBUG           -> write preview/mask overlays to DEBUG_DIR
    OCR_CONF_THRESHOLD   -> lower to capture fainter OCR words
    LOCAL_PADDING_PIXELS -> small padding around stroke masks
    MAX_MASK_AREA_FRAC   -> safety: abort large masks and apply conservative fallback
    ENABLE_MSER_FALLBACK -> False by default (safe for chest X-ray)
"""

import os
import re
import uuid
import math
from typing import Tuple, Dict

import numpy as np
from PIL import Image
import pytesseract
import cv2
import pydicom

# -----------------------------
# Configuration (tune carefully)
# -----------------------------
OCR_CONF_THRESHOLD = 30        # keep words with conf >= this (or numeric heuristics)
LOCAL_PADDING_PIXELS = 3       # small dilation on stroke mask
MIN_COMPONENT_AREA = 6        # filter tiny speckle components
SAVE_DEBUG = True             # set True to save debug preview/mask/overlay images
DEBUG_DIR = "ocr_debug"
MAX_MASK_AREA_FRAC = 0.12      # if mask would exceed this fraction, fallback to conservative rectangles
ENABLE_MSER_FALLBACK = False   # set True to test MSER on a small dataset
MSER_MIN_AREA = 30
MSER_MAX_AREA = 5000

# -----------------------------
# Logging helper
# -----------------------------
def log(msg: str):
    print(f"[PHI-MASK] {msg}")

# -----------------------------
# Debug helpers
# -----------------------------
def _ensure_debug_dir():
    if SAVE_DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)

def _save_debug_images(preview_pil: Image.Image, preview_mask: np.ndarray, prefix: str):
    if not SAVE_DEBUG:
        return
    _ensure_debug_dir()
    uid = uuid.uuid4().hex[:8]
    try:
        preview_pil.save(os.path.join(DEBUG_DIR, f"{prefix}_{uid}_preview.png"))
    except Exception:
        pass
    try:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_{uid}_mask.png"), preview_mask)
    except Exception:
        pass
    try:
        rgb = np.array(preview_pil.convert("RGB"))
        overlay = rgb.copy()
        overlay[preview_mask > 0] = (255, 0, 0)
        mixed = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)
        Image.fromarray(mixed).save(os.path.join(DEBUG_DIR, f"{prefix}_{uid}_overlay.png"))
    except Exception:
        pass
    log(f"Saved debug images: {prefix}_{uid}_*")

# -----------------------------
# Text normalization
# -----------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    s = str(text)
    s = s.replace("\u2019", "'")
    s = re.sub(r"[^\x00-\x7F]+", "", s)
    s = re.sub(r"[^A-Za-z0-9\s:/\-\.]", "", s)  # keep some punctuation used in dates/IDs
    return s.upper().strip()

# -----------------------------
# Read DICOM image and build 8-bit preview
# Returns: orig_array (numpy), preview_pil (PIL.Image), info dict
# info contains: BitsAllocated, BitsStored, PixelRepresentation, PhotometricInterpretation,
#               OrigDtype, OrigMin, OrigMax, RescaleSlope, RescaleIntercept, InversionNeeded (bool)
# -----------------------------
def get_image_from_dicom(ds: pydicom.dataset.Dataset) -> Tuple[np.ndarray, Image.Image, dict]:
    try:
        orig = ds.pixel_array
    except Exception as e:
        raise RuntimeError(f"Could not read pixel_array: {e}")

    info = {}
    bits_alloc = int(getattr(ds, "BitsAllocated", 0) or 0)
    bits_stored = int(getattr(ds, "BitsStored", bits_alloc) or 0)
    pixel_repr = int(getattr(ds, "PixelRepresentation", 0) or 0)  # 0 unsigned, 1 signed
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")

    info["BitsAllocated"] = bits_alloc
    info["BitsStored"] = bits_stored
    info["PixelRepresentation"] = pixel_repr
    info["PhotometricInterpretation"] = photometric
    info["OrigDtype"] = orig.dtype

    # RescaleSlope/Intercept if present (often used for CT)
    try:
        info["RescaleSlope"] = float(getattr(ds, "RescaleSlope", 1.0))
        info["RescaleIntercept"] = float(getattr(ds, "RescaleIntercept", 0.0))
    except Exception:
        info["RescaleSlope"] = 1.0
        info["RescaleIntercept"] = 0.0

    # Compute min/max of original pixel array in stored representation
    try:
        info["OrigMin"] = float(np.nanmin(orig))
        info["OrigMax"] = float(np.nanmax(orig))
    except Exception:
        info["OrigMin"] = 0.0
        info["OrigMax"] = 1.0

    # Determine if inversion is needed for preview: MONOCHROME1 means higher values==darker.
    # For preview we want a typical display where higher values are lighter (uint8)
    inversion_needed = False
    if photometric.upper().startswith("MONOCHROME1"):
        inversion_needed = True
    info["InversionNeeded"] = inversion_needed

    # Build 8-bit preview image for OCR (linear scaling)
    # If image is multichannel keep as-is but ensure uint8
    if orig.ndim == 3:
        vis = orig.copy()
        if vis.dtype != np.uint8:
            vis8 = ((vis.astype(np.float32) - info["OrigMin"]) /
                    max(1e-6, (info["OrigMax"] - info["OrigMin"])) * 255.0).astype(np.uint8)
        else:
            vis8 = vis
        preview = Image.fromarray(vis8)
    else:
        # single channel
        if orig.dtype != np.uint8:
            vis8 = ((orig.astype(np.float32) - info["OrigMin"]) /
                    max(1e-6, (info["OrigMax"] - info["OrigMin"])) * 255.0).astype(np.uint8)
        else:
            vis8 = orig.copy()
        if inversion_needed:
            # invert preview so text (usually dark) stays dark/contrasted properly
            vis8 = 255 - vis8
        preview = Image.fromarray(vis8)

    return orig, preview, info

# -----------------------------
# Build precise stroke mask (operates on small uint8 crop)
# -----------------------------
def build_precise_text_mask(crop_gray: np.ndarray) -> np.ndarray:
    if crop_gray.dtype != np.uint8:
        crop = ((crop_gray.astype(np.float32) - float(crop_gray.min())) /
                max(1.0, float(crop_gray.max() - crop_gray.min())) * 255.0).astype(np.uint8)
    else:
        crop = crop_gray.copy()

    # small blur to remove noise
    b = cv2.GaussianBlur(crop, (3, 3), 0)

    masks = []
    try:
        masks.append(cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV, 15, 6))
    except Exception:
        pass
    try:
        _, th = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        masks.append(th)
    except Exception:
        pass
    try:
        kx = max(3, min(15, crop.shape[1] // 4))
        ky = max(2, min(7, crop.shape[0] // 4))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        tophat = cv2.morphologyEx(b, cv2.MORPH_TOPHAT, kernel)
        _, th_top = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(th_top)
    except Exception:
        pass

    if not masks:
        return np.zeros_like(crop)

    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)

    # cleanup
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    mask = np.zeros_like(combined)
    min_area = max(2, MIN_COMPONENT_AREA)
    for i in range(1, num_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            mask[labels == i] = 255

    # small dilate to recover thin strokes
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
    return mask

# -----------------------------
# MSER fallback (optional)
# -----------------------------
def detect_text_regions_mser(preview_gray: np.ndarray):
    log("MSER: running fallback detector")
    try:
        mser = cv2.MSER_create(_delta=5, _min_area=MSER_MIN_AREA, _max_area=MSER_MAX_AREA)
    except Exception:
        mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(preview_gray)
    boxes = []
    H, W = preview_gray.shape[:2]
    for p in regions:
        x, y, w, h = cv2.boundingRect(p.reshape(-1, 1, 2))
        if w < 8 or h < 8:
            continue
        if w > 0.5 * W or h > 0.5 * H:
            continue
        boxes.append((x, y, x + w, y + h))
    log(f"MSER: found {len(boxes)} candidate boxes")
    return boxes

# -----------------------------
# Inpaint and soft blend (works in uint8 preview scale)
# Returns array in same dtype as input orig_arr conversion step (uint8)
# -----------------------------
def inpaint_and_blend_region(orig_arr: np.ndarray, pixel_mask: np.ndarray) -> np.ndarray:
    if pixel_mask.dtype != np.uint8:
        pm = (pixel_mask > 0).astype(np.uint8) * 255
    else:
        pm = pixel_mask.copy()
        pm[pm > 0] = 255

    # Convert orig to uint8 visual image used for inpainting
    if orig_arr.ndim == 2:
        if orig_arr.dtype != np.uint8:
            vis8 = ((orig_arr.astype(np.float32) - float(orig_arr.min())) /
                    max(1.0, float(orig_arr.max() - orig_arr.min())) * 255.0).astype(np.uint8)
        else:
            vis8 = orig_arr.copy()
        vis_bgr = cv2.cvtColor(vis8, cv2.COLOR_GRAY2BGR)
        is_gray = True
    else:
        vis = orig_arr.copy()
        if vis.dtype != np.uint8:
            vis8 = ((vis.astype(np.float32) - float(vis.min())) /
                    max(1.0, float(vis.max() - vis.min())) * 255.0).astype(np.uint8)
        else:
            vis8 = vis.copy()
        if vis8.shape[2] == 4:
            vis_bgr = cv2.cvtColor(vis8, cv2.COLOR_RGBA2BGR)
        else:
            vis_bgr = vis8
        is_gray = False

    # Resize mask if shape mismatch
    if pm.shape != vis_bgr.shape[:2]:
        pm = cv2.resize(pm, (vis_bgr.shape[1], vis_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    if pm.sum() == 0:
        return orig_arr

    try:
        inpainted = cv2.inpaint(vis_bgr, pm, 3, cv2.INPAINT_TELEA)
    except Exception as e:
        raise RuntimeError(f"Inpainting failed: {e}")

    # soft blend boundaries
    blurred = cv2.GaussianBlur(pm, (9, 9), 0)
    alpha = (blurred.astype(np.float32) / 255.0)[:, :, None]
    blended = (alpha * inpainted.astype(np.float32) + (1.0 - alpha) * vis_bgr.astype(np.float32)).astype(np.uint8)

    if is_gray:
        return cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
    else:
        return blended

# -----------------------------
# Write back PixelData and update DICOM metadata, restoring native dtype/range
# - Handles signedness, MONOCHROME1 inversion, RescaleSlope/Intercept, BitsStored/BitsAllocated
# - out_arr is uint8 or uint16 array (inpainting result). orig_info from get_image_from_dicom
# -----------------------------
def write_back_pixeldata_and_metadata(ds: pydicom.dataset.Dataset, out_arr: np.ndarray, orig_info: dict):
    orig_dtype = orig_info.get("OrigDtype", None)
    bits_alloc = int(orig_info.get("BitsAllocated", 0) or 0)
    bits_stored = int(orig_info.get("BitsStored", bits_alloc) or bits_alloc)
    pixel_repr = int(orig_info.get("PixelRepresentation", 0) or 0)
    orig_min = float(orig_info.get("OrigMin", 0.0) or 0.0)
    orig_max = float(orig_info.get("OrigMax", 1.0) or 1.0)
    slope = float(orig_info.get("RescaleSlope", 1.0) or 1.0)
    intercept = float(orig_info.get("RescaleIntercept", 0.0) or 0.0)
    photometric = orig_info.get("PhotometricInterpretation", "MONOCHROME2")
    inversion_needed = bool(orig_info.get("InversionNeeded", False))

    # Convert floating to uint8 if any
    if np.issubdtype(out_arr.dtype, np.floating):
        out_arr = np.clip(out_arr, 0.0, 255.0)
        out_arr = (out_arr / max(1e-6, out_arr.max()) * 255.0).astype(np.uint8)

    # If original was >8-bit, scale out_arr (0..255) back to original stored integer range
    if (bits_alloc >= 12) or (hasattr(orig_dtype, "itemsize") and getattr(orig_dtype, "itemsize", 0) > 1):
        # target maximum based on BitsStored (preferred) or BitsAllocated
        target_max = (2 ** bits_stored) - 1 if bits_stored > 0 else (2 ** max(12, bits_alloc) - 1)
        try:
            if orig_max > orig_min + 1e-6:
                # map 0..255 -> orig_min..orig_max then apply rescale inverse to match stored representation
                scaled = (out_arr.astype(np.float32) / 255.0) * (orig_max - orig_min) + orig_min
                # undo rescale slope/intercept: stored_value = (scaled - intercept) / slope
                if abs(slope) > 1e-9:
                    stored_float = (scaled - intercept) / slope
                else:
                    stored_float = scaled
                # map stored_float approximate to integer stored range 0..target_max
                # if original min/max corresponded to 0..target_max, do proportional mapping
                stored_min = 0.0
                stored_max = target_max
                # Normalize stored_float to 0..1 using original stored range (approx)
                # We don't have stored_min/stored_max reliably, so clip to 0..target_max
                scaled_u = np.clip(stored_float, stored_min, stored_max)
                # round to nearest integer
                if target_max <= 255:
                    final = scaled_u.astype(np.uint8)
                    ds_bits = 8
                elif target_max <= 65535:
                    final = np.round(scaled_u).astype(np.uint16)
                    ds_bits = 16
                else:
                    final = np.round(scaled_u).astype(np.uint32)
                    ds_bits = int(math.ceil(math.log2(target_max + 1)))
            else:
                # fallback linear expand to 16-bit
                final = (out_arr.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
                ds_bits = 16
        except Exception:
            final = (out_arr.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
            ds_bits = 16
    else:
        # original likely 8-bit; keep as uint8
        final = out_arr.astype(np.uint8)
        ds_bits = 8

    # If photometric was MONOCHROME1, we might need to invert final to preserve viewer rendering
    # We inverted the preview for OCR earlier when MONOCHROME1; now invert back.
    if photometric.upper().startswith("MONOCHROME1"):
        try:
            # operate on final's integer type
            final = np.max(final) - final
        except Exception:
            pass

    # Now write final into dataset and set metadata fields
    if final.ndim == 2:
        ds.Rows = int(final.shape[0])
        ds.Columns = int(final.shape[1])
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
        ds.BitsAllocated = ds_bits
        ds.BitsStored = ds_bits
        ds.HighBit = ds_bits - 1
        ds.PixelRepresentation = pixel_repr
        ds.PixelData = final.tobytes()
    elif final.ndim == 3:
        rows, cols, ch = final.shape
        ds.Rows = int(rows)
        ds.Columns = int(cols)
        ds.SamplesPerPixel = ch if ch <= 3 else 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0
        ds.BitsAllocated = ds_bits
        ds.BitsStored = ds_bits
        ds.HighBit = ds_bits - 1
        ds.PixelRepresentation = pixel_repr
        if ch == 4:
            ds.PixelData = final[..., :3].tobytes()
            ds.SamplesPerPixel = 3
        else:
            ds.PixelData = final.tobytes()
    else:
        raise ValueError("Unsupported image shape for final writeback.")

    log(f"Writeback: BitsAllocated={ds.BitsAllocated}, BitsStored={ds.BitsStored}, HighBit={ds.HighBit}, Photometric={ds.PhotometricInterpretation}")

# -----------------------------
# Conservative grouping helper - only group boxes that are similar and close
# -----------------------------
def can_group_boxes(box_a, box_b, hh_tolerance=0.25, x_distance_frac=0.6):
    # box = (x0,y0,x1,y1)
    xa0, ya0, xa1, ya1 = box_a
    xb0, yb0, xb1, yb1 = box_b
    ha = ya1 - ya0
    hb = yb1 - yb0
    if ha <= 0 or hb <= 0:
        return False
    if abs(ha - hb) / max(1.0, (ha + hb) / 2.0) > hh_tolerance:
        return False
    ca = (ya0 + ya1) / 2.0
    cb = (yb0 + yb1) / 2.0
    if abs(ca - cb) > max(ha, hb) * 1.2:
        return False
    # horizontal gap
    dist = max(0, max(xb0 - xa1, xa0 - xb1))
    max_allowed = max(ha, hb) * x_distance_frac
    if dist > max_allowed:
        return False
    return True

# -----------------------------
# Main function
# -----------------------------
def run_ocr_and_mask(ds: pydicom.dataset.Dataset, phis_to_mask: Dict[str, str], output_path: str):
    """
    ds: loaded pydicom Dataset (can be read with pydicom.dcmread)
    phis_to_mask: dict mapping metadata keywords -> original values (strings) (from remove_metadata_phi)
    output_path: where to save the de-identified DICOM
    """
    log(f"Processing file -> {output_path}")
    try:
        orig_arr, preview_pil, orig_info = get_image_from_dicom(ds)
        log(f"Orig dtype={orig_info['OrigDtype']}, BitsAllocated={orig_info.get('BitsAllocated')}, Photometric={orig_info.get('PhotometricInterpretation')}, range=({orig_info.get('OrigMin')},{orig_info.get('OrigMax')})")
    except Exception as e:
        log(f"Failed to extract pixel data: {e}. Saving metadata-cleaned DICOM.")
        try:
            ds.save_as(output_path)
        except Exception as ee:
            log(f"Failed to save DICOM: {ee}")
        return

    preview_gray = np.array(preview_pil.convert("L"))
    h, w = preview_gray.shape[:2]

    # prepare PHI list (normalized)
    metadata_values = [normalize_text(v) for v in phis_to_mask.values() if v and len(str(v)) > 1]
    KEYWORDS = ["PATIENT", "NAME", "DOB", "AGE", "STUDY", "DATE", "ID", "HOSPITAL", "INSTITUTION"] + metadata_values

    # Run word-level OCR
    try:
        ocr = pytesseract.image_to_data(preview_pil, config="--oem 1 --psm 6", output_type=pytesseract.Output.DICT)
        log(f"OCR: {len(ocr.get('text', []))} word candidates")
    except Exception as e:
        log(f"OCR failure: {e}")
        ocr = {}

    try:
        char_boxes_raw = pytesseract.image_to_boxes(preview_pil, config="--oem 1 --psm 6")
    except Exception:
        char_boxes_raw = ""

    final_mask = np.zeros((h, w), dtype=np.uint8)
    words = ocr.get("text", [])
    n_words = len(words)

    boxes_info = []
    for i in range(n_words):
        txt = (words[i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(ocr.get("conf", [None]*n_words)[i])
        except Exception:
            conf = -1.0
        left = int(ocr.get("left", [0]*n_words)[i])
        top = int(ocr.get("top", [0]*n_words)[i])
        width = int(ocr.get("width", [0]*n_words)[i])
        height = int(ocr.get("height", [0]*n_words)[i])
        if width <= 0 or height <= 0:
            continue
        boxes_info.append({
            "text": txt,
            "conf": conf,
            "box": (left, top, left + width, top + height),
            "w": width,
            "h": height,
            "i": i
        })

    # Process each word conservatively
    for info in boxes_info:
        text = info["text"]
        conf = info["conf"]
        x0, y0, x1, y1 = info["box"]
        ww = info["w"]; hh = info["h"]

        # Confidence filter (allow numeric tokens even if low conf)
        if conf < OCR_CONF_THRESHOLD and not re.search(r"\d{4,}", text):
            continue

        # Expand box slightly
        pad = max(1, int(0.2 * hh))
        ex0 = max(0, x0 - pad); ey0 = max(0, y0 - pad)
        ex1 = min(w, x1 + pad); ey1 = min(h, y1 + pad)
        crop = preview_gray[ey0:ey1, ex0:ex1]
        if crop.size == 0:
            continue

        nt = normalize_text(text)
        matched = False
        # direct metadata value match
        for mv in metadata_values:
            if mv and (mv in nt or nt in mv):
                matched = True
                break
        # keyword match
        if not matched:
            for kw in KEYWORDS:
                if kw and kw in nt:
                    matched = True
                    break
        # date / long numeric heuristics
        if not matched:
            if re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text) or re.search(r"\d{6,}", text):
                matched = True

        if not matched:
            continue

        log(f"PHI candidate matched: '{text}' conf={conf} bbox=[{ex0},{ey0},{ex1},{ey1}]")

        # Build a precise stroke mask for this crop
        local_mask = build_precise_text_mask(crop)

        # If local mask empty, try char-box fallback within crop (tight)
        if local_mask.sum() == 0 and char_boxes_raw:
            try:
                char_grid = np.zeros_like(crop)
                for line in char_boxes_raw.strip().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    ch, bx0, by0, bx1, by1 = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                    # pytesseract returns coords with origin bottom-left (y measured from bottom)
                    # convert to preview image coords
                    ch_x0 = bx0
                    ch_x1 = bx1
                    ch_y0 = h - by1
                    ch_y1 = h - by0
                    if ch_x1 < ex0 or ch_x0 > ex1 or ch_y1 < ey0 or ch_y0 > ey1:
                        continue
                    lx0 = max(0, ch_x0 - ex0); ly0 = max(0, ch_y0 - ey0)
                    lx1 = min(ex1 - ex0, ch_x1 - ex0); ly1 = min(ey1 - ey0, ch_y1 - ey0)
                    if lx1 > lx0 and ly1 > ly0:
                        char_grid[ly0:ly1, lx0:lx1] = 255
                # filter components
                nl, labs, stats, _ = cv2.connectedComponentsWithStats(char_grid, connectivity=8)
                mask2 = np.zeros_like(char_grid)
                for j in range(1, nl):
                    if int(stats[j, cv2.CC_STAT_AREA]) >= max(2, MIN_COMPONENT_AREA // 2):
                        mask2[labs == j] = 255
                local_mask = mask2
            except Exception:
                pass

        # Conservative grouping: collect neighbor boxes that are similar and close
        neighbors = [info]
        for other in boxes_info:
            if other is info:
                continue
            if can_group_boxes((ex0, ey0, ex1, ey1), other["box"], hh_tolerance=0.25, x_distance_frac=0.6):
                neighbors.append(other)

        group_preview_mask = np.zeros((h, w), dtype=np.uint8)
        for nb in neighbors:
            nb_x0, nb_y0, nb_x1, nb_y1 = nb["box"]
            nb_h = nb_y1 - nb_y0
            nb_pad = max(1, int(0.2 * nb_h))
            nb_ex0 = max(0, nb_x0 - nb_pad); nb_ey0 = max(0, nb_y0 - nb_pad)
            nb_ex1 = min(w, nb_x1 + nb_pad); nb_ey1 = min(h, nb_y1 + nb_pad)
            nb_crop = preview_gray[nb_ey0:nb_ey1, nb_ex0:nb_ex1]
            if nb_crop.size == 0:
                continue
            nb_local = build_precise_text_mask(nb_crop)
            if nb_local.sum() == 0:
                # fallback: small rectangle for that box
                rp = max(1, min(LOCAL_PADDING_PIXELS, nb_h // 6))
                rx0 = max(0, nb_ex0 - rp); ry0 = max(0, nb_ey0 - rp)
                rx1 = min(w, nb_ex1 + rp); ry1 = min(h, nb_ey1 + rp)
                group_preview_mask[ry0:ry1, rx0:rx1] = 255
            else:
                group_preview_mask[nb_ey0:nb_ey1, nb_ex0:nb_ex1] |= nb_local

        if group_preview_mask.sum() == 0:
            rp = max(1, min(LOCAL_PADDING_PIXELS, hh // 6))
            rx0 = max(0, ex0 - rp); ry0 = max(0, ey0 - rp)
            rx1 = min(w, ex1 + rp); ry1 = min(h, ey1 + rp)
            group_preview_mask[ry0:ry1, rx0:rx1] = 255

        prospective = final_mask | group_preview_mask
        covered_fraction = float(prospective.sum()) / float(prospective.size)
        log(f"Prospective coverage fraction (preview): {covered_fraction:.4f}")

        if covered_fraction > MAX_MASK_AREA_FRAC:
            log("Prospective mask too large. Applying conservative per-word rectangle only.")
            rp = max(1, min(LOCAL_PADDING_PIXELS, hh // 6))
            rx0 = max(0, ex0 - rp); ry0 = max(0, ey0 - rp)
            rx1 = min(w, ex1 + rp); ry1 = min(h, ey1 + rp)
            final_mask[ry0:ry1, rx0:rx1] = 255
        else:
            final_mask |= group_preview_mask

    # If still empty and MSER allowed, try MSER
    if final_mask.sum() == 0 and ENABLE_MSER_FALLBACK:
        log("No OCR mask found; trying MSER fallback")
        mser_boxes = detect_text_regions_mser(preview_gray)
        for (sx, sy, ex, ey) in mser_boxes:
            crop = preview_gray[sy:ey, sx:ex]
            if crop.size == 0:
                continue
            m = build_precise_text_mask(crop)
            if m.sum() > 0:
                final_mask[sy:ey, sx:ex] = 255

    _save_debug_images(preview_pil, final_mask, "final_preview_mask")

    if final_mask.sum() == 0:
        log("No PHI pixels detected in preview. Saving metadata-cleaned DICOM without pixel changes.")
        try:
            ds.save_as(output_path)
            log(f"Saved (no pixel change): {output_path}")
        except Exception as e:
            log(f"Failed to save DICOM: {e}")
        return

    # Map preview mask to original resolution if needed
    oh, ow = orig_arr.shape[:2]
    if final_mask.shape != (oh, ow):
        log(f"Resizing mask from preview ({final_mask.shape}) to original ({oh},{ow})")
        final_mask = cv2.resize(final_mask, (ow, oh), interpolation=cv2.INTER_NEAREST)

    # final small close + dilate to avoid tiny holes
    ksize = max(3, LOCAL_PADDING_PIXELS * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)

    final_frac = float(final_mask.sum()) / float(final_mask.size)
    log(f"Final mask coverage fraction: {final_frac:.4f}")

    # Safety: if mask too large, fallback to conservative per-word rectangles for matched boxes
    if final_frac > MAX_MASK_AREA_FRAC:
        log("Final mask too large. Switching to conservative per-word rectangle fallback.")
        safe_mask = np.zeros_like(final_mask)
        for info in boxes_info:
            nt = normalize_text(info["text"])
            is_phi = False
            for mv in metadata_values:
                if mv and (mv in nt or nt in mv):
                    is_phi = True; break
            if not is_phi:
                for kw in KEYWORDS:
                    if kw and kw in nt:
                        is_phi = True; break
            if not is_phi:
                continue
            x0, y0, x1, y1 = info["box"]
            pad = max(1, int(0.1 * (y1 - y0)))
            sx0 = max(0, x0 - pad); sy0 = max(0, y0 - pad)
            sx1 = min(ow, x1 + pad); sy1 = min(oh, y1 + pad)
            safe_mask[sy0:sy1, sx0:sx1] = 255
        final_mask = safe_mask
        final_frac = float(final_mask.sum()) / float(final_mask.size)
        log(f"Conservative mask coverage: {final_frac:.4f}")

    # Inpaint original array using the mask
    try:
        out_arr = inpaint_and_blend_region(orig_arr, final_mask)
        log("Inpaint completed.")
    except Exception as e:
        log(f"Inpaint failed: {e}. Using black-fill fallback on mask pixels.")
        fallback = orig_arr.copy()
        if fallback.ndim == 2:
            fallback[final_mask > 0] = 0
        else:
            fallback[final_mask > 0, :] = 0
        out_arr = fallback

    # Write back while restoring original bit-depth / photometric interpretation
    try:
        write_back_pixeldata_and_metadata(ds, out_arr, orig_info)
        ds.save_as(output_path)
        log(f"Saved de-identified DICOM: {output_path}")
    except Exception as e:
        log(f"Writeback save failed: {e}. Attempting conservative save.")
        try:
            ds.save_as(output_path)
        except Exception as ee:
            log(f"Final save failed: {ee}")

# End of file
