"""
app.py — DriftSense-X Streamlit Web Application

Demo-ready localization interface using the validated Global Landmark pipeline.
Method: Global Landmark (431.93 px mean error on held-out benchmark).
Hybrid Ranker (534.74 px) was evaluated and rejected.

Run:
    streamlit run app.py
"""

import io
import math
import os
import sys
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Ensure project root is on path so all localization sub-modules resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="DriftSense-X",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0d1117; }

    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #7b61ff 50%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #8b949e;
        margin-bottom: 0.5rem;
        font-weight: 400;
    }
    .hero-desc {
        font-size: 0.92rem;
        color: #6e7681;
        max-width: 680px;
        line-height: 1.6;
    }

    .metric-card {
        background: linear-gradient(145deg, #161b22, #1c2128);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s ease;
    }
    .metric-card:hover { border-color: #58a6ff; }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b949e;
        margin-bottom: 0.3rem;
        font-weight: 500;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e6edf3;
        line-height: 1.1;
    }
    .metric-unit {
        font-size: 0.85rem;
        color: #8b949e;
        margin-left: 0.25rem;
        font-weight: 400;
    }

    .status-success {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #1a3a2a, #1f4535);
        border: 1px solid #2ea043;
        color: #3fb950;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .status-failed {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #3a1a1a, #451f1f);
        border: 1px solid #da3633;
        color: #f85149;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .status-running {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #1a2a3a, #1f3545);
        border: 1px solid #388bfd;
        color: #58a6ff;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .section-header {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8b949e;
        font-weight: 600;
        margin-bottom: 0.5rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #21262d;
    }

    .upload-zone {
        background: linear-gradient(145deg, #161b22, #1c2128);
        border: 1px dashed #30363d;
        border-radius: 12px;
        padding: 1rem;
    }

    .benchmark-badge {
        display: inline-block;
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.3rem 0.75rem;
        font-size: 0.75rem;
        color: #8b949e;
        margin-top: 0.5rem;
    }
    .benchmark-badge b { color: #58a6ff; }

    .how-step {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }
    .how-num {
        min-width: 26px;
        height: 26px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .how-text {
        font-size: 0.88rem;
        color: #c9d1d9;
        line-height: 1.5;
    }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(31, 111, 235, 0.35);
        width: 100%;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(31, 111, 235, 0.5);
    }

    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #2ea043, #3fb950);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.25rem;
        font-weight: 600;
        font-size: 0.88rem;
        transition: all 0.2s ease;
        width: 100%;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(46, 160, 67, 0.4);
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #30363d, transparent);
        margin: 1.5rem 0;
    }

    .coord-display {
        font-family: 'Courier New', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
        letter-spacing: -0.5px;
    }

    .stExpander {
        border: 1px solid #21262d !important;
        border-radius: 10px !important;
        background: #161b22 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helper: load image from uploaded file ─────────────────────────────────────
def load_image_from_upload(uploaded_file) -> np.ndarray | None:
    """Convert Streamlit UploadedFile to a grayscale numpy array."""
    try:
        pil_img = Image.open(uploaded_file).convert("L")   # grayscale
        return np.array(pil_img, dtype=np.uint8)
    except Exception:
        return None


def numpy_to_pil_rgb(img_bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB for Streamlit display."""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def numpy_gray_to_pil(img_gray: np.ndarray) -> Image.Image:
    """Convert grayscale numpy array to PIL for display."""
    return Image.fromarray(img_gray)


# ── Annotated image builder (pure OpenCV — no localization imports needed) ───
def build_annotated_image(
    search_img: np.ndarray,
    pred_x: float,
    pred_y: float,
    ref_img: np.ndarray,
    scale: float = 0.10,
    gt_x: float = None,
    gt_y: float = None,
) -> np.ndarray:
    """Draw crosshair, bounding box, and coordinate label on search image."""
    vis = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
    H, W = vis.shape[:2]

    box_w = max(20, int(round(ref_img.shape[1] * scale)))
    box_h = max(20, int(round(ref_img.shape[0] * scale)))

    cx = int(round(pred_x))
    cy = int(round(pred_y))

    # --- Bounding box (cyan glow effect) ---
    tl = (max(0, cx - box_w // 2), max(0, cy - box_h // 2))
    br = (min(W - 1, cx + box_w // 2), min(H - 1, cy + box_h // 2))
    # Shadow / glow
    cv2.rectangle(vis, (tl[0]-1, tl[1]-1), (br[0]+1, br[1]+1), (0, 120, 180), 3)
    # Main box
    cv2.rectangle(vis, tl, br, (0, 220, 255), 2)

    # --- Crosshair ---
    arm = 22
    cross_color = (0, 255, 160)
    cv2.line(vis, (max(0, cx - arm), cy), (min(W-1, cx + arm), cy), cross_color, 2, cv2.LINE_AA)
    cv2.line(vis, (cx, max(0, cy - arm)), (cx, min(H-1, cy + arm)), cross_color, 2, cv2.LINE_AA)
    # Center dot
    cv2.circle(vis, (cx, cy), 4, (255, 255, 80), -1, cv2.LINE_AA)
    cv2.circle(vis, (cx, cy), 4, (0, 0, 0), 1, cv2.LINE_AA)

    # --- Coordinate label ---
    label = f"Target: ({pred_x:.1f}, {pred_y:.1f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.55
    thickness = 1
    (tw, th), bl = cv2.getTextSize(label, font, fscale, thickness)
    tx = max(5, min(W - tw - 8, cx - tw // 2))
    ty_label = max(th + 8, tl[1] - 8)

    # Background pill
    cv2.rectangle(vis, (tx - 4, ty_label - th - 4), (tx + tw + 4, ty_label + 2), (0, 0, 0), -1)
    cv2.rectangle(vis, (tx - 4, ty_label - th - 4), (tx + tw + 4, ty_label + 2), (0, 180, 220), 1)
    cv2.putText(vis, label, (tx, ty_label), font, fscale, (0, 220, 255), thickness, cv2.LINE_AA)

    # --- Ground-truth marker (red) ---
    if gt_x is not None and gt_y is not None:
        gx, gy = int(round(gt_x)), int(round(gt_y))
        cv2.drawMarker(vis, (gx, gy), (60, 60, 255), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
        gt_label = f"GT: ({gt_x:.1f}, {gt_y:.1f})"
        (gtw, gth), _ = cv2.getTextSize(gt_label, font, fscale, thickness)
        gtx_pos = max(5, min(W - gtw - 8, gx + 14))
        gty_pos = max(gth + 8, gy - 8)
        cv2.rectangle(vis, (gtx_pos - 4, gty_pos - gth - 4), (gtx_pos + gtw + 4, gty_pos + 2), (0, 0, 0), -1)
        cv2.rectangle(vis, (gtx_pos - 4, gty_pos - gth - 4), (gtx_pos + gtw + 4, gty_pos + 2), (60, 60, 255), 1)
        cv2.putText(vis, gt_label, (gtx_pos, gty_pos), font, fscale, (100, 100, 255), thickness, cv2.LINE_AA)

        # Error line (orange)
        err = math.hypot(pred_x - gt_x, pred_y - gt_y)
        if err < 400:
            cv2.line(vis, (cx, cy), (gx, gy), (0, 140, 255), 1, cv2.LINE_AA)
            mid_x, mid_y = (cx + gx) // 2, (cy + gy) // 2
            err_label = f"{err:.1f}px"
            cv2.putText(vis, err_label, (mid_x + 3, mid_y - 3), font, 0.42, (0, 165, 255), 1, cv2.LINE_AA)

    return vis


def annotated_to_png_bytes(vis: np.ndarray) -> bytes:
    """Encode annotated BGR image to PNG bytes for download."""
    success, buf = cv2.imencode(".png", vis)
    if success:
        return buf.tobytes()
    return b""


# ── Cached localization call (keyed on image content hashes) ─────────────────
@st.cache_data(show_spinner=False)
def run_localization(ref_bytes: bytes, search_bytes: bytes) -> dict:
    """
    Run Global Landmark localization on uploaded images.
    Results are cached: same image pair won't re-run on re-renders.
    """
    import numpy as np
    from PIL import Image
    import io
    import time

    # Decode images from bytes
    ref_arr = np.array(Image.open(io.BytesIO(ref_bytes)).convert("L"), dtype=np.uint8)
    search_arr = np.array(Image.open(io.BytesIO(search_bytes)).convert("L"), dtype=np.uint8)

    t0 = time.perf_counter()
    try:
        from localization.final_localizer_hybrid import locate_target
        import tempfile, os

        # locate_target requires file paths — save to temp files
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as rf:
            ref_path = rf.name
            Image.fromarray(ref_arr).save(ref_path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as sf:
            search_path = sf.name
            Image.fromarray(search_arr).save(search_path)

        try:
            result = locate_target(ref_path, search_path)
        finally:
            os.unlink(ref_path)
            os.unlink(search_path)

        result["ref_arr"] = ref_arr
        result["search_arr"] = search_arr
        result["runtime_sec"] = float(result.get("runtime_sec", time.perf_counter() - t0))
        result["error_message"] = None
        return result

    except Exception as e:
        return {
            "predicted_x": None,
            "predicted_y": None,
            "pixel_error": None,
            "confidence": None,
            "candidate_rank": None,
            "runtime_sec": time.perf_counter() - t0,
            "status": "FAILED",
            "ref_arr": ref_arr,
            "search_arr": search_arr,
            "error_message": str(e),
        }


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-title">DriftSense-X</div>
    <div class="hero-sub">AI-Powered Navigation-Error Recovery for Wafer Inspection</div>
    <div class="hero-desc">
        DriftSense-X locates a target site in a search image using global landmark-based
        visual localization and reports the predicted coordinates with confidence, candidate
        rank, and runtime.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="benchmark-badge">
    Production method: <b>Global Landmark</b> &nbsp;|&nbsp;
    Validation mean error: <b>431.93 px</b> (40-image held-out set) &nbsp;|&nbsp;
    Oracle upper bound: <b>21.53 px</b>
    </div>""",
    unsafe_allow_html=True,
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Two-column upload layout ──────────────────────────────────────────────────
col_ref, col_search = st.columns(2, gap="large")

with col_ref:
    st.markdown('<div class="section-header">Reference Image</div>', unsafe_allow_html=True)
    ref_upload = st.file_uploader(
        "Upload Reference Image",
        type=["png", "jpg", "jpeg"],
        key="ref_uploader",
        label_visibility="collapsed",
        help="The reference image contains the target pattern to locate (typically 1000×1000 px).",
    )
    if ref_upload is not None:
        ref_img_np = load_image_from_upload(ref_upload)
        if ref_img_np is not None:
            st.image(
                numpy_gray_to_pil(ref_img_np),
                caption=f"Reference — {ref_img_np.shape[1]}×{ref_img_np.shape[0]} px",
                width="stretch",
            )
        else:
            st.error("Could not decode reference image. Please upload a valid PNG/JPG.")
    else:
        st.markdown(
            '<div style="text-align:center;padding:2.5rem 0;color:#484f58;font-size:0.9rem;">'
            '📷 No reference image uploaded</div>',
            unsafe_allow_html=True,
        )

with col_search:
    st.markdown('<div class="section-header">Search Image</div>', unsafe_allow_html=True)
    search_upload = st.file_uploader(
        "Upload Search Image",
        type=["png", "jpg", "jpeg"],
        key="search_uploader",
        label_visibility="collapsed",
        help="The search image is scanned to find the target location (typically 1000×1000 px).",
    )
    if search_upload is not None:
        search_img_np = load_image_from_upload(search_upload)
        if search_img_np is not None:
            st.image(
                numpy_gray_to_pil(search_img_np),
                caption=f"Search — {search_img_np.shape[1]}×{search_img_np.shape[0]} px",
                width="stretch",
            )
        else:
            st.error("Could not decode search image. Please upload a valid PNG/JPG.")
    else:
        st.markdown(
            '<div style="text-align:center;padding:2.5rem 0;color:#484f58;font-size:0.9rem;">'
            '🔍 No search image uploaded</div>',
            unsafe_allow_html=True,
        )

# ── Optional ground-truth inputs ──────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
with st.expander("Optional: Provide Ground-Truth Coordinates for Error Calculation", expanded=False):
    gt_col1, gt_col2 = st.columns(2, gap="medium")
    with gt_col1:
        gt_x_input = st.number_input(
            "Ground Truth X (px)", value=None, placeholder="e.g. 513.87",
            format="%.2f", key="gt_x"
        )
    with gt_col2:
        gt_y_input = st.number_input(
            "Ground Truth Y (px)", value=None, placeholder="e.g. 783.43",
            format="%.2f", key="gt_y"
        )

# ── Locate Target button ──────────────────────────────────────────────────────
st.markdown("")
btn_col, _ = st.columns([1, 2])
with btn_col:
    locate_clicked = st.button("🎯  LOCATE TARGET", key="locate_btn")

# ═════════════════════════════════════════════════════════════════════════════
#  LOCALIZATION RESULT
# ═════════════════════════════════════════════════════════════════════════════

if locate_clicked:
    # ── Validate inputs ───────────────────────────────────────────────────────
    if ref_upload is None:
        st.error("Please upload a Reference Image before locating.")
        st.stop()
    if search_upload is None:
        st.error("Please upload a Search Image before locating.")
        st.stop()

    ref_img_np = load_image_from_upload(ref_upload)
    search_img_np = load_image_from_upload(search_upload)

    if ref_img_np is None:
        st.error("Reference image is invalid or cannot be decoded.")
        st.stop()
    if search_img_np is None:
        st.error("Search image is invalid or cannot be decoded.")
        st.stop()

    # ── Read raw bytes for cache key ──────────────────────────────────────────
    ref_upload.seek(0)
    ref_bytes = ref_upload.read()
    search_upload.seek(0)
    search_bytes = search_upload.read()

    # ── Run localization with progress indicator ───────────────────────────────
    with st.spinner("Running Global Landmark localization… this may take ~1–2 seconds."):
        result = run_localization(ref_bytes, search_bytes)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Error handling ────────────────────────────────────────────────────────
    if result.get("error_message"):
        st.error(f"Localization error: {result['error_message']}")
        st.info("Make sure all project dependencies are installed (`pip install -r requirements.txt`) "
                "and run from the D:\\DriftSense-X directory.")
        st.stop()

    pred_x = result.get("predicted_x")
    pred_y = result.get("predicted_y")
    confidence = result.get("confidence")
    cand_rank = result.get("candidate_rank")
    runtime_sec = result.get("runtime_sec", 0.0)
    status = result.get("status", "UNKNOWN")

    if pred_x is None or pred_y is None:
        st.error("Localization returned no valid coordinates. Status: " + str(status))
        st.stop()

    # ── Pixel error (optional GT) ─────────────────────────────────────────────
    gt_x_val = gt_x_input if gt_x_input is not None else None
    gt_y_val = gt_y_input if gt_y_input is not None else None
    pixel_error = None
    if gt_x_val is not None and gt_y_val is not None:
        pixel_error = math.hypot(pred_x - gt_x_val, pred_y - gt_y_val)

    # ── Status badge ──────────────────────────────────────────────────────────
    status_cls = "status-success" if status == "SUCCESS" else "status-failed"
    status_icon = "✓" if status == "SUCCESS" else "✕"
    st.markdown(
        f'<div style="margin-bottom:1rem;">'
        f'<span class="{status_cls}">{status_icon} TARGET {status}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Metric cards row ──────────────────────────────────────────────────────
    mc1, mc2, mc3, mc4, mc5 = st.columns(5, gap="small")

    with mc1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Predicted X</div>'
            f'<div class="metric-value">{pred_x:.2f}<span class="metric-unit">px</span></div></div>',
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Predicted Y</div>'
            f'<div class="metric-value">{pred_y:.2f}<span class="metric-unit">px</span></div></div>',
            unsafe_allow_html=True,
        )
    with mc3:
        conf_pct = f"{confidence * 100:.1f}%" if confidence is not None else "N/A"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Confidence</div>'
            f'<div class="metric-value">{conf_pct}</div></div>',
            unsafe_allow_html=True,
        )
    with mc4:
        rank_str = f"#{cand_rank}" if cand_rank is not None and cand_rank > 0 else "N/A"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Candidate Rank</div>'
            f'<div class="metric-value">{rank_str}</div></div>',
            unsafe_allow_html=True,
        )
    with mc5:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Runtime</div>'
            f'<div class="metric-value">{runtime_sec:.3f}<span class="metric-unit">s</span></div></div>',
            unsafe_allow_html=True,
        )

    # ── Pixel error display ───────────────────────────────────────────────────
    if pixel_error is not None:
        err_color = "#3fb950" if pixel_error <= 50 else ("#f0883e" if pixel_error <= 200 else "#f85149")
        st.markdown(
            f'<div class="metric-card" style="max-width:260px;">'
            f'<div class="metric-label">Pixel Error (vs. Ground Truth)</div>'
            f'<div class="metric-value" style="color:{err_color};">{pixel_error:.2f}'
            f'<span class="metric-unit">px</span></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "Pixel error: Available only when ground-truth coordinates are provided. "
            "Use the 'Optional: Provide Ground-Truth Coordinates' section above.",
            icon="ℹ️",
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Annotated image ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Localization Result</div>', unsafe_allow_html=True)

    vis_bgr = build_annotated_image(
        search_img=result["search_arr"],
        pred_x=pred_x,
        pred_y=pred_y,
        ref_img=result["ref_arr"],
        gt_x=gt_x_val,
        gt_y=gt_y_val,
    )

    vis_pil = numpy_to_pil_rgb(vis_bgr)
    st.image(vis_pil, caption=f"Predicted target at ({pred_x:.1f}, {pred_y:.1f}) px", width="stretch")

    # ── Download button ────────────────────────────────────────────────────────
    png_bytes = annotated_to_png_bytes(vis_bgr)
    if png_bytes:
        dl_col, _ = st.columns([1, 3])
        with dl_col:
            st.download_button(
                label="⬇  Download Annotated Result",
                data=png_bytes,
                file_name="driftsense_result.png",
                mime="image/png",
                key="download_btn",
            )

# ═════════════════════════════════════════════════════════════════════════════
#  HOW IT WORKS PANEL
# ═════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

with st.expander("How DriftSense-X Works", expanded=False):
    steps = [
        ("Reference pattern", "The reference image defines the target site pattern (1000×1000 px SEM crop). It is downscaled to 100×100 and used as a macro template."),
        ("Candidate generation", "Multi-scale template matching (7 scales, 0.085–0.115×) across grayscale, Sobel gradient, LoG, and Gaussian blur maps extracts up to 500 candidate locations from the search image."),
        ("Global landmark heatmap", "A macro-level probability heatmap is computed by matching low-frequency Gaussian and Canny edge density maps at full image scale. This encodes global intensity context to discriminate between periodically aliased candidates."),
        ("Candidate scoring", "Each candidate receives a combined score: 60% global landmark heatmap value + 40% local template correlation peak score. Candidates are ranked by this combined score."),
        ("Fine localization", "Sub-pixel refinement is applied around the winning candidate center using 100×100 template matching, recovering the final (X, Y) to sub-pixel precision."),
        ("Output", "The predicted center coordinates, alignment confidence, candidate rank, and runtime are returned. The search image is annotated with a bounding box, crosshair, and coordinate label."),
        ("Honest benchmark", "Validated mean error: 431.93 px on a 40-image held-out set. Oracle upper bound (best possible with Top-500 pool): 21.53 px. The gap is caused by periodic aliasing — visually identical lattice alias candidates that only global structural features can disambiguate."),
    ]

    for i, (title, text) in enumerate(steps, 1):
        st.markdown(
            f'<div class="how-step">'
            f'<div class="how-num">{i}</div>'
            f'<div class="how-text"><b>{title}</b> — {text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
        padding:0.9rem 1.1rem;margin-top:0.75rem;font-size:0.8rem;color:#8b949e;line-height:1.6;">
        <b style="color:#c9d1d9;">Benchmark note:</b>
        Global Landmark validation mean error: <b style="color:#58a6ff;">431.93 px</b> on the
        40-image held-out set (images 00161–00200). Oracle upper bound: <b style="color:#58a6ff;">21.53 px</b>.
        The Hybrid Ranker (56-D features, triplet margin loss) was evaluated but
        achieved <b style="color:#f85149;">534.74 px</b> mean error and was <b>rejected</b> in favour of Global Landmark.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-top:3rem;padding:1rem 0;border-top:1px solid #21262d;
    text-align:center;color:#484f58;font-size:0.75rem;">
    DriftSense-X &nbsp;·&nbsp; Global Landmark Localization &nbsp;·&nbsp;
    Production method validated 2026-08 &nbsp;·&nbsp;
    <span style="color:#30363d;">Hybrid Ranker preserved for reference only</span>
    </div>
    """,
    unsafe_allow_html=True,
)
