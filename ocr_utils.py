#!/usr/bin/env python3
"""
ocr_utils.py

OCR-based burned-in PHI masking for DICOM images.
- Uses pytesseract for OCR
- Uses OpenCV for mask morphology and inpainting
- Tries to preserve DICOM image attributes
"""

import re
import numpy as np
from PIL import Image
import pytesseract
import cv2
import pydicom
from typing import Tuple, Dict


def normalize_text(text: str) -> str:
    """Conservative normalization for OCR tokens and metadata values."""
    if not text:
        return ''
    text = str(text)
    text = text.replace('\u2019', "'")
    # remove non-ascii
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.upper().strip()


def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    """
    Convert PIL image to a uint8 PIL image optimized for Tesseract:
    - grayscale, contrast stretch, conditional inversion, Otsu or adaptive threshold
    """
    img_gray = pil_img.convert('L')
    arr = np.array(img_gray).astype(np.uint8)

    # Contrast stretching (2nd to 98th percentile)
    p2, p98 = np.percentile(arr, (2, 98))
    if p98 > p2:
        arr = np.clip((arr - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)

    # If background is dark, invert
    if arr.mean() < 127:
        arr = 255 - arr

    # Apply Otsu threshold, fallback to adaptive if Otsu fails
    try:
        _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        arr = th
    except Exception:
        arr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

    return Image.fromarray(arr)


def get_image_from_dicom(ds: pydicom.dataset.Dataset) -> Tuple[np.ndarray, Image.Image]:
    """
    Returns:
      - original_pixel_array: the original pixel array read from DICOM (numpy)
      - ocr_preview_img: a PIL image (8-bit) suitable for tesseract preprocessing
    The original array is not modified here.
    """
    try:
        orig = ds.pixel_array  # may raise if PixelData malformed
    except Exception as e:
        raise RuntimeError(f"Could not get pixel_array: {e}")

    # For OCR preview, convert to an 8-bit PIL image.
    if orig.ndim == 3:
        # If 3 or 4 channels, attempt to convert to RGB PIL image
        if orig.shape[2] == 4:
            vis = cv2.cvtColor(orig, cv2.COLOR_RGBA2RGB)
        elif orig.shape[2] == 3:
            vis = orig
        else:
            # weird channel count: collapse to first channel
            vis = orig[..., 0]
        # Ensure uint8 for preview
        if vis.dtype != np.uint8:
            vis = ((vis - vis.min()) / (max(1, vis.max() - vis.min())) * 255).astype(np.uint8)
        pil = Image.fromarray(vis)
    else:
        # single-channel
        if orig.dtype != np.uint8:
            vis = ((orig - orig.min()) / (max(1, orig.max() - orig.min())) * 255).astype(np.uint8)
        else:
            vis = orig.copy()
        pil = Image.fromarray(vis)

    # Preprocess for OCR
    ocr_img = preprocess_image_for_ocr(pil)
    return orig, ocr_img


def _matches_phi(text: str, metadata_phis: Dict[str, str], keywords: Tuple[str, ...]) -> bool:
    nt = normalize_text(text)
    if not nt:
        return False
    # direct metadata match
    for v in metadata_phis:
        if v and v in nt:
            return True
    # keyword match
    for k in keywords:
        if k in nt:
            return True
    # date pattern or long digit sequence
    if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text) or re.search(r'\d{6,}', text):
        return True
    return False


def _build_mask_from_ocr(ocr_data: dict, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Build an initial mask from OCR text boxes. Returns a single-channel mask (uint8) same shape as image.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    heights = [int(h) for h in ocr_data.get('height', []) if h and int(h) > 0]
    median_h = int(np.median(heights)) if heights else 10

    n = len(ocr_data.get('text', []))
    for i in range(n):
        try:
            txt = ocr_data['text'][i]
            conf_str = ocr_data['conf'][i]
            conf = float(conf_str) if conf_str not in (None, '') else -1
        except Exception:
            continue
        if not txt.strip() or conf < 30:
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

    # Morphological ops to connect nearby boxes
    ksize = max(3, (median_h // 2) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _inpaint_and_blend(orig_arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Inpaint masked regions and blend softly to avoid visible seams.
    Returns array with same shape and dtype as orig_arr where possible.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Prepare a 3-channel uint8 image for OpenCV inpaint
    if orig_arr.ndim == 2:
        # Grayscale: convert to 3-channel BGR for inpainting
        vis8 = ((orig_arr - orig_arr.min()) / (max(1, orig_arr.max() - orig_arr.min())) * 255).astype(np.uint8)
        vis_bgr = cv2.cvtColor(vis8, cv2.COLOR_GRAY2BGR)
        is_gray = True
    else:
        # Multi-channel: ensure 3-channel BGR uint8
        vis = orig_arr.copy()
        if vis.dtype != np.uint8:
            vis8 = ((vis - vis.min()) / (max(1, vis.max() - vis.min())) * 255).astype(np.uint8)
        else:
            vis8 = vis
        if vis8.shape[2] == 4:
            vis_bgr = cv2.cvtColor(vis8, cv2.COLOR_RGBA2BGR)
        elif vis8.shape[2] == 3:
            vis_bgr = vis8
        else:
            # fallback: take first channel expanded
            vis_bgr = cv2.cvtColor(vis8[..., 0], cv2.COLOR_GRAY2BGR)
        is_gray = False

    # Ensure mask shape matches vis_bgr shape
    mask_resized = mask
    if mask.shape != vis_bgr.shape[:2]:
        mask_resized = cv2.resize(mask, (vis_bgr.shape[1], vis_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Inpaint using TELEA
    try:
        inpainted = cv2.inpaint(vis_bgr, mask_resized, 3, cv2.INPAINT_TELEA)
    except Exception as e:
        raise RuntimeError(f"Inpainting failed: {e}")

    # Soft blend edges: blur mask and alpha blend
    blurred = cv2.GaussianBlur(mask_resized, (9, 9), 0)
    alpha = (blurred.astype(np.float32) / 255.0)[:, :, None]

    blended = (alpha * inpainted.astype(np.float32) + (1.0 - alpha) * vis_bgr.astype(np.float32)).astype(np.uint8)

    # Convert back to original dtype and channels
    if is_gray:
        blended_gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
        if orig_arr.dtype == np.uint8:
            return blended_gray
        else:
            # scale back to original numeric range
            out = (blended_gray.astype(np.float32) / 255.0 * (orig_arr.max() - orig_arr.min()) + orig_arr.min())
            return out.astype(orig_arr.dtype)
    else:
        if orig_arr.dtype == np.uint8 and blended.shape == orig_arr.shape:
            return blended
        else:
            # scale back to original dtype/range
            out = (blended.astype(np.float32) / 255.0 * (orig_arr.max() - orig_arr.min()) + orig_arr.min())
            # If original had 4 channels, try to restore alpha as zeros
            if orig_arr.ndim == 3 and orig_arr.shape[2] == 4:
                alpha_chan = np.zeros((out.shape[0], out.shape[1], 1), dtype=out.dtype)
                out = np.concatenate([out, alpha_chan], axis=2)
            return out.astype(orig_arr.dtype)


def run_ocr_and_mask(ds: pydicom.dataset.Dataset, phis_to_mask: Dict[str, str], output_path: str):
    """
    Main entry point:
      - extracts pixel array and tesseract preview image
      - runs OCR
      - builds a precise mask using metadata matches and keywords
      - inpaints masked regions and writes PixelData back while preserving DICOM attributes
    """
    try:
        original_arr, ocr_img = get_image_from_dicom(ds)
    except Exception as e:
        print(f"  Could not prepare image for OCR: {e}")
        try:
            ds.save_as(output_path)
        except Exception as ee:
            print(f"  Failed to save original: {ee}")
        return

    # OCR configuration: LSTM engine, page segmentation suited for line/word detection
    tconf = '--oem 1 --psm 6'
    try:
        ocr_data = pytesseract.image_to_data(ocr_img, config=tconf, output_type=pytesseract.Output.DICT)
    except Exception as e:
        print(f"  Tesseract failed: {e}. Saving cleaned metadata DICOM.")
        try:
            ds.save_as(output_path)
        except Exception as ee:
            print(f"  Failed to save DICOM: {ee}")
        return

    # Build metadata search list and keyword list
    metadata_values = [normalize_text(v) for v in phis_to_mask.values() if v and len(str(v)) > 2]
    keywords = ('ID', 'NAME', 'DOB', 'DATE', 'TIME', 'ACCESSION', 'AGE',
                'HOSPITAL', 'PATIENT', 'PHYSICIAN', 'INSTITUTION', 'PORTABLE', 'MRN')

    # Determine image shape for mask (OCR preview size)
    ocr_preview_gray = np.array(ocr_img.convert('L'))
    image_shape = ocr_preview_gray.shape  # (h, w)

    # 1) Try exact/content matches first to avoid overmasking
    used_mask = np.zeros(image_shape, dtype=np.uint8)
    n = len(ocr_data.get('text', []))
    for i in range(n):
        txt = ocr_data['text'][i]
        conf_str = ocr_data['conf'][i]
        try:
            conf = float(conf_str) if conf_str not in (None, '') else -1
        except Exception:
            conf = -1
        if not txt.strip() or conf < 30:
            continue

        if _matches_phi(txt, metadata_values, keywords):
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

    # 2) If no used_mask found, fall back to a broader mask using geometry
    if used_mask.sum() == 0:
        used_mask = _build_mask_from_ocr(ocr_data, image_shape)

    # Final morphological smoothing of the mask
    heights = [int(h) for h in ocr_data.get('height', []) if h and int(h) > 0]
    median_h = int(np.median(heights)) if heights else 10
    ksize = max(3, (median_h // 2) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    used_mask = cv2.dilate(used_mask, kernel, iterations=1)
    used_mask = cv2.medianBlur(used_mask, 5)

    # If mask empty, skip
    if used_mask.sum() == 0:
        print("  No likely PHI regions detected. Saving cleaned metadata image unchanged.")
        try:
            ds.save_as(output_path)
        except Exception as e:
            print(f"  Failed saving file: {e}")
        return

    # Inpaint and blend
    try:
        inpainted = _inpaint_and_blend(original_arr, used_mask)
    except Exception as e:
        print(f"  Inpainting failed: {e}. Falling back to opaque rectangles.")
        # Fallback: apply opaque zero rectangles on original array
        masked = original_arr.copy()
        ys, xs = np.where(used_mask > 0)
        if ys.size:
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            if masked.ndim == 2:
                masked[y0:y1 + 1, x0:x1 + 1] = 0
            else:
                masked[y0:y1 + 1, x0:x1 + 1, :] = 0
        # Write fallback masked PixelData and save
        try:
            _write_back_pixeldata(ds, masked)
            ds.save_as(output_path)
            print(f"  Saved fallback masked file to {output_path}")
        except Exception as ee:
            print(f"  Failed to save fallback: {ee}")
        return

    # Write final inpainted array back to ds and save
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


def _write_back_pixeldata(ds: pydicom.dataset.Dataset, arr: np.ndarray):
    """
    Write numpy array arr as PixelData into ds while trying to preserve
    SamplesPerPixel, PhotometricInterpretation, BitsAllocated/Stored/HighBit, and shape.
    """
    # If arr is float, convert to suitable integer
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, arr.min(), arr.max())
        arr = (arr - arr.min()) / max(1e-8, (arr.max() - arr.min()))
        arr = (arr * 255).astype(np.uint8)

    # Ensure contiguous
    arr = np.ascontiguousarray(arr)

    # Update dataset attributes
    if arr.ndim == 2:
        rows, cols = arr.shape
        ds.Rows = int(rows)
        ds.Columns = int(cols)
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
        # BitsAllocated/Stored/HighBit based on dtype
        if arr.dtype == np.uint8:
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
        elif arr.dtype == np.uint16:
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
        ds.PixelData = arr.tobytes()
    elif arr.ndim == 3:
        rows, cols, ch = arr.shape
        ds.Rows = int(rows)
        ds.Columns = int(cols)
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
        elif ch == 4:
            # store as 3-channel RGB, dropping alpha if present
            rgb = arr[..., :3]
            ds.SamplesPerPixel = 3
            ds.PhotometricInterpretation = getattr(ds, 'PhotometricInterpretation', 'RGB')
            if rgb.dtype == np.uint8:
                ds.BitsAllocated = 8
                ds.BitsStored = 8
                ds.HighBit = 7
            ds.PixelData = rgb.tobytes()
        else:
            # unexpected channel count; flatten first channel
            gray = arr[..., 0]
            _write_back_pixeldata(ds, gray)
    else:
        raise ValueError("Unsupported array shape when writing back pixel data.")
