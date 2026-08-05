import sys
import os
import io
import numpy as np
from PIL import Image

sys.path.insert(0, '.')

from app import run_localization, build_annotated_image, annotated_to_png_bytes

# Load sample image files
ref_path = os.path.join('dataset', 'validation', 'reference', '00174.png')
search_path = os.path.join('dataset', 'validation', 'search', '00174.png')

with open(ref_path, 'rb') as f:
    ref_bytes = f.read()

with open(search_path, 'rb') as f:
    search_bytes = f.read()

print("Running run_localization...")
res = run_localization(ref_bytes, search_bytes)

print("Result keys:", list(res.keys()))
print("Predicted X:", res["predicted_x"])
print("Predicted Y:", res["predicted_y"])
print("Confidence:", res["confidence"])
print("Candidate Rank:", res["candidate_rank"])
print("Runtime:", res["runtime_sec"])
print("Status:", res["status"])
print("Error Message:", res["error_message"])

assert res["error_message"] is None, f"Error: {res['error_message']}"
assert res["predicted_x"] is not None
assert res["predicted_y"] is not None

print("Building annotated image...")
vis = build_annotated_image(
    search_img=res["search_arr"],
    pred_x=res["predicted_x"],
    pred_y=res["predicted_y"],
    ref_img=res["ref_arr"],
    gt_x=513.87,
    gt_y=783.43
)

png_b = annotated_to_png_bytes(vis)
print("PNG bytes size:", len(png_b))
assert len(png_b) > 0, "PNG encoding failed"

print("All tests passed cleanly!")
