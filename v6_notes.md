detects original bit-depth and signedness per DICOM,

handles MONOCHROME1 vs MONOCHROME2 inversion correctly,

preserves/respects RescaleSlope/RescaleIntercept where present,

builds precise stroke masks from OCR boxes (tight masks, small padding),

avoids aggressive row/group masking by default,

offers an optional MSER fallback (disabled by default),

inpaints only masked pixels and restores the final array to the original dynamic range,

writes PixelData and updates BitsAllocated/BitsStored/HighBit/PixelRepresentation correctly,

logs all important steps to the terminal,

can save debug preview + mask overlays when SAVE_DEBUG = True.