Notes and next steps

Set SAVE_DEBUG = True and run on a small subset (10–20 files). Inspect files in ocr_debug/ (*_preview.png, *_mask_preview.png, *_overlay.png) to see whether missed strokes are due to OCR or mask creation.

If you still see residual pixels, try increasing LOCAL_PADDING_PIXELS to 4 or 5 and rerun.

If inpaint artifacts remain visible, consider testing a learned inpainting model later (heavy but great results).