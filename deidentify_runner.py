#!/usr/bin/env python3
"""
deidentify_runner.py
Iterate DICOM files, remove metadata PHI, and run OCR-based burned-in PHI masking when needed.
"""

import os
import pydicom
import pydicom.uid
from ocr_utils import run_ocr_and_mask

# --- Configuration: adjust paths before running ---
INPUT_DIR = r"D:\0000 study spacd\06.1 SEM8\00 internship\01 go zeal\02 projects\02 dicom-deid-project\input_dcm"
OUTPUT_DIR = r"D:\0000 study spacd\06.1 SEM8\00 internship\01 go zeal\02 projects\02 dicom-deid-project\output_dcm"

# --- PHI Tags list (tag tuples) ---
PHI_TAGS = [
    (0x0010, 0x0010), (0x0010, 0x0020), (0x0010, 0x0030), (0x0010, 0x0032),
    (0x0010, 0x0040), (0x0010, 0x1010), (0x0010, 0x1000), (0x0010, 0x1001),
    (0x0010, 0x1040), (0x0010, 0x2150),
    (0x0008, 0x0080), (0x0008, 0x0081), (0x0008, 0x1040), (0x0008, 0x1010),
    (0x0008, 0x0090), (0x0008, 0x0094), (0x0008, 0x0096), (0x0008, 0x9000),
    (0x0008, 0x1050), (0x0008, 0x1084), (0x0008, 0x1070),
    (0x0040, 0x0006), (0x0040, 0xA075), (0x0040, 0xA078),
    (0x0008, 0x0020), (0x0008, 0x0030), (0x0008, 0x0050), (0x0020, 0x0010),
    (0x0020, 0x0011),
    (0x0018, 0x1000), (0x0018, 0x1002),
    (0x0020, 0x000D), (0x0020, 0x000E),
]

# SOP Class UID for Secondary Capture
SECONDARY_CAPTURE_UID = '1.2.840.10008.5.1.4.1.1.7'


def remove_metadata_phi(ds: pydicom.dataset.Dataset) -> dict:
    """
    Remove or replace PHI-containing metadata fields. Returns a dict of original values.
    UIDs are regenerated; basic IDs replaced with '0'; other text tags deleted.
    """
    original = {}
    for tag in PHI_TAGS:
        try:
            if tag in ds:
                elem = ds[tag]
                tag_name = getattr(elem, 'keyword', f'{tag[0]:04X},{tag[1]:04X}')
                original[tag_name] = str(elem.value)
                # Replace UIDs
                if 'UID' in tag_name.upper() or elem.VR == 'UI':
                    ds[tag].value = pydicom.uid.generate_uid()
                # Numeric identifiers: replace with zero-like sentinel
                elif isinstance(elem.value, (int,)) or 'ID' in tag_name.upper() or 'NUMBER' in tag_name.upper():
                    ds[tag].value = '0'
                else:
                    # remove text-style PHI
                    try:
                        del ds[tag]
                    except Exception:
                        # Some elements may not be deletable; set to empty string
                        ds[tag].value = ''
        except Exception:
            # Ignore problematic tags but continue
            continue

    # Always regenerate critical UIDs
    try:
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
    except Exception:
        pass
    try:
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    except Exception:
        pass

    return original


def should_run_ocr(ds: pydicom.dataset.Dataset) -> bool:
    """
    Decide whether to run OCR-based burned-in PHI detection.
    Use modality, SOPClassUID, burned-in annotation flag, and textual hints.
    """
    modality = str(ds.get('Modality', '')).upper()
    study_desc = str(ds.get('StudyDescription', '')).upper()
    series_desc = str(ds.get('SeriesDescription', '')).upper()

    if modality in ('US', 'XA', 'CR', 'DX'):
        return True

    if str(ds.get('SOPClassUID', '')) == SECONDARY_CAPTURE_UID:
        return True

    if 'PORTABLE' in study_desc or 'PORTABLE' in series_desc or 'SCANNED' in study_desc or 'SCANNED' in series_desc or 'SC' in series_desc:
        return True

    burn = ds.get('BurnedInAnnotation', None)
    if burn is not None and str(burn).upper() == 'YES':
        return True

    # PixelData presence alone isn't sufficient; we prefer to avoid OCR where unlikely
    return False


def process_dicom_files(input_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.dcm')]
    total = len(files)
    if total == 0:
        print("No DICOM files found in input directory.")
        return

    ocr_count = 0
    print(f"Processing {total} DICOM files...")

    for idx, fn in enumerate(files, start=1):
        in_path = os.path.join(input_dir, fn)
        out_path = os.path.join(output_dir, fn)
        print(f"\n[{idx}/{total}] {fn}")

        try:
            ds = pydicom.dcmread(in_path, force=True)
        except Exception as e:
            print(f"  Error reading DICOM: {e} — skipping")
            continue

        # 1. Remove metadata PHI, but keep a small record for matching burned-in text
        original_phis = remove_metadata_phi(ds)

        # 2. If PixelData exists and detection heuristics say OCR is needed, run OCR masking
        if hasattr(ds, 'PixelData') and should_run_ocr(ds):
            print(f"  - Running OCR/masking (Modality: {ds.get('Modality', 'N/A')})")
            try:
                run_ocr_and_mask(ds, original_phis, out_path)
                ocr_count += 1
            except Exception as e:
                print(f"  - OCR/masking failed: {e}. Saving cleaned metadata DICOM.")
                try:
                    ds.save_as(out_path)
                except Exception as ee:
                    print(f"  - Failed to save DICOM: {ee}")
        else:
            print("  - Skipping OCR (low burn-in risk). Saving cleaned metadata file.")
            try:
                ds.save_as(out_path)
            except Exception as e:
                print(f"  - Failed to save DICOM: {e}")

    print("\n--- Summary ---")
    print(f"Total files: {total}")
    print(f"OCR mask applied: {ocr_count} ({ocr_count/total*100:.1f}%)")


if __name__ == '__main__':
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory not found: {INPUT_DIR}")
    else:
        process_dicom_files(INPUT_DIR, OUTPUT_DIR)
