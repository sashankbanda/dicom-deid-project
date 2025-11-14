This code does not make assumptions about where text appears. It relies on Tesseract to find words first.

For each Tesseract word we build a local stroke mask so we redact only the exact letter pixels (plus a tiny padding), not the entire rectangle.

If the stroke mask fails, minimal rectangular fallback is used only for that small box. That keeps risk low.

The faint-text detector approach you rejected is not used here. This respects your requirement: find the text first, then redact precisely.

Tune OCR_CONF_THRESHOLD and LOCAL_PADDING_PIXELS a bit on your dataset — they are intentionally conservative.
