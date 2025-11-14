import pydicom
import os
import shutil
from ocr_utils import run_ocr_and_mask

# # --- Configuration ---
# INPUT_DIR = './input_dcm'
# OUTPUT_DIR = './output_dcm'

# --- Configuration ---
# Use r-strings for Windows paths to handle backslashes correctly
INPUT_DIR = r"D:\0000 study spacd\06.1 SEM8\00 internship\01 go zeal\02 projects\02 dicom-deid-project\input_dcm"
OUTPUT_DIR = r"D:\0000 study spacd\06.1 SEM8\00 internship\01 go zeal\02 projects\02 dicom-deid-project\output_dcm"

# HIPAA Safe Harbor Identifiers List (for metadata removal)
PHI_TAGS = [
    (0x0010, 0x0010), # PatientName
    (0x0010, 0x0020), # PatientID
    (0x0010, 0x0030), # PatientBirthDate
    (0x0010, 0x0032), # PatientBirthTime
    (0x0010, 0x0040), # PatientSex (Often retained for research, but removing per your list)
    # ... Add all other tags from your list (e.g., InstitutionName, StudyDate, etc.)
    # We will use a smaller, illustrative list here for brevity:
    (0x0008, 0x0020), # StudyDate
    (0x0008, 0x0030), # StudyTime
    (0x0008, 0x0050), # AccessionNumber
    (0x0008, 0x0080), # InstitutionName
    # ... many more ...
]

# DICOM SOP Class UID for Secondary Capture (High-risk images)
SECONDARY_CAPTURE_UID = '1.2.840.10008.5.1.4.1.1.7' 

# --- Main Functions ---

def remove_metadata_phi(ds: pydicom.Dataset) -> dict:
    """Removes PHI from metadata and returns the original PHI values."""
    original_phis = {}
    
    for tag in PHI_TAGS:
        if tag in ds:
            # Store the original value for OCR comparison later
            tag_name = ds[tag].keyword
            original_phis[tag_name] = str(ds[tag].value) 
            
            # Remove the tag or replace with a blank string
            if tag_name.endswith('UID') or tag_name.endswith('Number'):
                 # Remap UIDs/Numbers (simple unique hash or deletion is common)
                 ds[tag].value = '0' # Placeholder value
            else:
                 # Delete the tag entirely
                 del ds[tag]
                 
    # Critical step: Re-UID the study/series/instance to ensure uniqueness
    pydicom.uid.generate_uid() # Initialize UID generator if needed
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    
    return original_phis

def should_run_ocr(ds: pydicom.Dataset) -> bool:
    """Implements the targeted filtering logic."""
    
    # 1. Check Modality (e.g., Ultrasound is high-risk)
    if ds.get('Modality', '').upper() == 'US':
        print("  - Running OCR: Modality is Ultrasound (US).")
        return True
        
    # 2. Check for Secondary Capture Flag
    if ds.get('SOPClassUID') == SECONDARY_CAPTURE_UID:
        print("  - Running OCR: Image is a Secondary Capture.")
        return True
        
    # 3. Check for Burned-In Annotation Flag
    if ds.get('BurnedInAnnotation', '').upper() == 'YES':
        print("  - Running OCR: Burned-In Annotation flag is YES.")
        return True

    return True

def process_dicom_files(input_dir: str, output_dir: str):
    """Iterates through files and applies de-identification."""
    
    # Setup directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.dcm')]
    total_files = len(file_list)
    ocr_run_count = 0
    
    print(f"Processing {total_files} DICOM files...")
    
    for i, filename in enumerate(file_list):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"\n[{i+1}/{total_files}] Processing {filename}:")

        try:
            ds = pydicom.dcmread(input_path)
        except Exception as e:
            print(f"  Error reading DICOM file: {e}. Skipping.")
            continue
            
        # 1. Metadata De-identification
        original_phis = remove_metadata_phi(ds)
        
        # 2. Targeted OCR Filter Check
        if hasattr(ds, 'PixelData') and should_run_ocr(ds):
            ocr_run_count += 1
            # 3. Identify and Mask PHI Text
            run_ocr_and_mask(ds, original_phis, output_path)
        else:
            # If no OCR needed, just save the metadata-cleaned file
            print("  - Skipping OCR: Low risk of burned-in PHI.")
            ds.save_as(output_path)

    print(f"\n--- Summary ---")
    print(f"Total Files Processed: {total_files}")
    print(f"OCR Engine Run Count: {ocr_run_count} ({ocr_run_count/total_files*100:.2f}%)")

if __name__ == '__main__':
    # Create a dummy input folder for testing
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Created dummy input directory: {INPUT_DIR}")
        print("Please place your test DICOM files (.dcm) inside this folder.")
    else:
        process_dicom_files(INPUT_DIR, OUTPUT_DIR)
