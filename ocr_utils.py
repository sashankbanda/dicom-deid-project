import pydicom
import numpy as np
from PIL import Image
import pytesseract
import re
from skimage.filters import threshold_otsu
import os

def normalize_text(text: str) -> str:
    """Removes special characters, standardizes text, and handles common OCR errors."""
    if not text:
        return ""
    text = text.replace('l', 'I').replace('1', 'I')
    text = text.replace('O', '0')
    return re.sub(r'[^\w\s]', '', text).upper().strip()

def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Applies simple inversion and optional thresholding to create black text on a white background 
    (Tesseract's preferred format).
    """
    img_array = np.array(img.convert('L')) 
    
    # 1. Simple Inversion: Makes light text black and bright background white.
    inverted_array = 255 - img_array
    
    # 2. Optional Thresholding
    try:
        thresh = threshold_otsu(inverted_array)
        final_array = (inverted_array > thresh) * 255
        final_array = final_array.astype(np.uint8)
        
    except ValueError:
        final_array = inverted_array.astype(np.uint8)
        
    return Image.fromarray(final_array)

def get_image_from_dicom(ds: pydicom.Dataset) -> tuple:
    """
    Converts DICOM pixel data to a 8-bit NumPy array for processing and returns the array 
    and the original DICOM array copy.
    """
    
    original_pixel_array = ds.pixel_array
    pixel_array = original_pixel_array.copy()
    
    # Normalize 16-bit data to 8-bit for OCR
    if pixel_array.dtype in [np.int16, np.uint16]:
        # Scale to 0-255 range
        normalized_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
        img_data = normalized_array.astype(np.uint8)
    else:
        img_data = pixel_array.astype(np.uint8)

    # Handle multi-frame or color images (take the first frame for simplicity)
    if img_data.ndim > 2:
        img_data = img_data[0] if img_data.ndim == 3 else img_data
        
    img = Image.fromarray(img_data)
    
    # Apply pre-processing before OCR, but return the original array for masking
    return original_pixel_array, preprocess_image_for_ocr(img)


def run_ocr_and_mask(ds: pydicom.Dataset, phis_to_mask: dict, output_path: str):
    """
    Runs OCR, identifies PHI, masks it, and saves the modified DICOM file.
    """
    masked_count = 0
    
    try:
        # 1. Get the original pixel array and the enhanced image for OCR
        original_pixel_array, ocr_img = get_image_from_dicom(ds)
        
        # Preserve original Photometric Interpretation
        original_photometric_interpretation = ds.get('PhotometricInterpretation', 'MONOCHROME2')
        
        # 2. Run Tesseract
        ocr_data = pytesseract.image_to_data(ocr_img, config='--psm 6', output_type=pytesseract.Output.DICT)
        
    except Exception as e:
        print(f"  Could not run OCR or process pixel data: {e}")
        ds.save_as(output_path)
        return

    # --- Debugging Log (Identical to previous, useful for checking detection) ---
    detected_words = [t for t, conf in zip(ocr_data['text'], ocr_data['conf']) if t.strip() and float(conf) >= 50]
    if not detected_words:
        print("  ❌ Tesseract detected zero high-confidence words (conf >= 50).")
    else:
        print(f"  ✅ Tesseract detected words (conf >= 50): {detected_words}")
    # --------------------
    
    masked_array = original_pixel_array.copy()

    # 3. PHI Match Lists (Logic remains correct)
    metadata_phis_normalized = [normalize_text(v) for v in phis_to_mask.values() if v and len(v) > 3]
    phi_keywords = ['ID', 'NAME', 'DOB', 'DATE', 'TIME', 'ACCESSION', 'AGE', 'HOSPITAL', 'PATIENT', 'PHYSICIAN', 'INSTITUTION', 'PORTABLE']
    
    # 4. Loop through OCR results and apply masking
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        conf = float(ocr_data['conf'][i])
        
        if not text or conf < 50: 
            continue

        normalized_text = normalize_text(text)
        is_phi = False
        
        # Identification Logic (A, B, C checks)
        if any(phi_val in normalized_text for phi_val in metadata_phis_normalized) or \
           any(keyword in normalized_text for keyword in phi_keywords) or \
           re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text) or \
           re.search(r'\d{4}\s\w{3}\s\d{1,2}', text) or \
           re.search(r'\d{6,}', text):
            is_phi = True
            
        # --- Masking Logic ---
        if is_phi:
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            
            # Draw a black rectangle (0 intensity) on the original array
            # Buffer is increased to mask the whole field value
            buffer = 15 
            
            y_start = max(0, y - buffer)
            y_end = min(masked_array.shape[0], y + h + buffer)
            x_start = max(0, x - buffer)
            x_end = min(masked_array.shape[1], x + w + buffer)
            
            # Set the pixel values to 0 (black) in the detected region
            # We use 0 (the lowest value) for a safe, visible redaction.
            masked_array[y_start:y_end, x_start:x_end] = 0
            
            masked_count += 1
            
    if masked_count > 0:
        print(f"  Masked {masked_count} text blocks in the image.")
        
        # Replace the original pixel data with the masked data
        ds.PixelData = masked_array.tobytes()
        
        # CRITICAL FIX: Retain original Photometric Interpretation for correct contrast display.
        # This resolves the darkening issue on the final output image.
        ds.PhotometricInterpretation = original_photometric_interpretation 
        ds.BitsStored = masked_array.dtype.itemsize * 8
        
    # Save the final DICOM file (metadata was already cleaned)
    ds.save_as(output_path)
    print(f"  Saved de-identified file to {output_path}")