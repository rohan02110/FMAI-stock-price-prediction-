"""
=============================================================================
Vision AI Broken Window & Structural Element Re-verification Dashboard
File: vision_damage_dashboard.py
Built with Streamlit, PIL, and Gemini Multimodal Vision API
=============================================================================
"""

import os
import sys
import json
import time

# Ensure script directory and portable site-packages are in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

PORTABLE_PACKAGES = r"C:\Users\Hp\PythonPortable\Lib\site-packages"
if os.path.exists(PORTABLE_PACKAGES) and PORTABLE_PACKAGES not in sys.path:
    sys.path.insert(0, PORTABLE_PACKAGES)

import pandas as pd  # pyrefly: ignore [missing-import] # type: ignore
from PIL import Image, ImageDraw, ImageFont  # pyrefly: ignore [missing-import] # type: ignore
import streamlit as st  # pyrefly: ignore [missing-import] # type: ignore

# Import the core verifier engine
from broken_element_verifier import (  # pyrefly: ignore [missing-import] # type: ignore
    CandidateElement,
    PatchProcessor,
    GeminiVisionVerifier,
    InspectionPipeline,
    create_demo_building_image
)

# Set page configuration
st.set_page_config(
    page_title="Broken Element Vision AI Verifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: white;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .metric-title {
        font-size: 13px;
        font-weight: 700;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-number {
        font-size: 28px;
        font-weight: 800;
        margin-top: 4px;
    }
    .metric-green { color: #28a745; }
    .metric-red { color: #dc3545; }
    .metric-blue { color: #007bff; }
    .metric-purple { color: #6f42c1; }
    .badge-fp {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
    }
    .badge-broken {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("🛠️ Configuration")

# API Key Management
default_env_key = os.environ.get("GEMINI_API_KEY", "").strip()
api_key_input = st.sidebar.text_input(
    "🔑 Google Gemini API Key:",
    value=default_env_key,
    type="password",
    help="Enter your Gemini API Key. If empty, the system will check environment variable GEMINI_API_KEY."
)

active_api_key = api_key_input.strip() if api_key_input.strip() else default_env_key

if active_api_key:
    st.sidebar.success("✅ API Key Active")
else:
    st.sidebar.warning("⚠️ Enter your Gemini API Key above to run live Vision AI verification.")

model_choice = st.sidebar.selectbox(
    "🧠 Gemini Vision Model:",
    ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Pixelation & Inspection Settings")

pixel_size = st.sidebar.slider(
    "Patch Pixelation Block Size (px):",
    min_value=2,
    max_value=24,
    value=8,
    step=2,
    help="Higher values create coarser blocks to analyze macro structural continuity versus micro specular glares."
)

grid_cols = st.sidebar.slider("Grid Columns (Auto-Scan):", 1, 5, 3)
grid_rows = st.sidebar.slider("Grid Rows (Auto-Scan):", 1, 5, 3)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How the Re-verification Engine Works:**
1. **Candidate Extraction**: Isolates candidate regions flagged by the initial model.
2. **Patch Pixelation**: Applies forensic multi-scale block downsampling.
3. **Gemini Vision AI Evaluation**: Dissects reflections, tree shadows, and window mullions vs genuine fracture cracks.
4. **False Positive Elimination**: Reclassifies intact reflective windows as `UNBROKEN`.
""")


# -----------------------------------------------------------------------------
# MAIN APP BODY
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🔍 Broken Element & Window Vision AI Verifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Two-stage forensic pipeline eliminating false-positive broken flags using patch pixelation and Gemini Vision AI</div>', unsafe_allow_html=True)

# Image Source Selection
img_source = st.radio(
    "Select Image Input Source:",
    ["🏢 Demo Building Facade (Intact Reflective Windows vs Real Damage)", "📁 Upload Custom Inspection Image"],
    horizontal=True
)

if img_source.startswith("📁"):
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG, WEBP):", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        main_image = Image.open(uploaded_file).convert("RGB")
    else:
        st.info("👆 Please upload an image above to begin inspection.")
        st.stop()
        main_image = None
else:
    main_image = create_demo_building_image()

if main_image is None:
    st.stop()

# Initialize pipeline
pipeline = InspectionPipeline(api_key=active_api_key, model_name=model_choice)

# Session state initialization for results
if "verified_candidates" not in st.session_state or st.session_state.get("current_source") != img_source:
    st.session_state["candidates"] = pipeline.generate_candidate_grid(main_image, grid_cols=grid_cols, grid_rows=grid_rows)
    st.session_state["verified_candidates"] = None
    st.session_state["current_source"] = img_source

candidates = st.session_state["candidates"]

# Action Button
col_btn, col_info = st.columns([1, 2])
with col_btn:
    run_verify = st.button("🚀 Run AI Re-check & Filter False Positives", type="primary", use_container_width=True)

with col_info:
    if not active_api_key:
        st.warning("⚠️ Live AI re-verification requires an API Key. Please enter it in the sidebar.")

if run_verify:
    if not active_api_key:
        st.error("❌ Cannot execute Vision AI verification without an API Key. Please provide a Gemini API Key in the sidebar.")
    else:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(ratio, msg):
            progress_bar.progress(ratio)
            status_text.text(msg)

        # Run verification
        verified = pipeline.run_reverification(
            image=main_image,
            candidates=candidates,
            pixel_size=pixel_size,
            progress_callback=update_progress
        )
        st.session_state["verified_candidates"] = verified
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        st.success("🎉 Forensic Vision AI re-verification finished successfully!")

# Compute Summary Statistics
verified_list = st.session_state["verified_candidates"]
total_elements = len(candidates)
initial_flagged_broken = total_elements  # All candidate elements initially flagged as broken

if verified_list:
    truly_broken_count = sum(1 for c in verified_list if c.is_verified_broken is True)
    false_positives_cleared = sum(1 for c in verified_list if c.is_verified_broken is False and c.status == "FALSE_POSITIVE_UNBROKEN")
    accuracy_gain = (false_positives_cleared / total_elements) * 100 if total_elements > 0 else 0
else:
    truly_broken_count = "?"
    false_positives_cleared = "?"
    accuracy_gain = 0

# KPI Stat Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Detected Elements</div>
        <div class="metric-number metric-blue">{total_elements}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Initial Raw Flagged 'Broken'</div>
        <div class="metric-number metric-purple">{initial_flagged_broken}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">False Positives Eliminated</div>
        <div class="metric-number metric-green">{false_positives_cleared}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Verified Genuine Broken</div>
        <div class="metric-number metric-red">{truly_broken_count}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VISUAL COMPARISON: RAW PREDICTION VS RE-VERIFIED
# -----------------------------------------------------------------------------
tab_visual, tab_patches, tab_audit = st.tabs([
    "🖼️ Visual Comparison (Raw vs Re-verified)",
    "🔬 Pixelation & Patch Inspector",
    "📋 Forensic Audit Table & Export"
])

with tab_visual:
    st.subheader("Side-by-Side Visual Verification")
    col_raw, col_post = st.columns(2)

    with col_raw:
        st.markdown("**1️⃣ Initial Detection Model (High False-Positive Rate)**")
        raw_annotated = pipeline.render_annotated_image(main_image, candidates, show_raw=True)
        st.image(raw_annotated, caption="Raw Model Output: Intact windows with reflections falsely flagged as broken", use_container_width=True)

    with col_post:
        st.markdown("**2️⃣ Post-Gemini Vision & Pixelation Re-check (Accurate)**")
        if verified_list:
            verified_annotated = pipeline.render_annotated_image(main_image, verified_list, show_raw=False)
            st.image(verified_annotated, caption="Re-verified Output: 🟢 Green = Intact Window Cleared | 🔴 Red = Confirmed Broken", use_container_width=True)
        else:
            st.info("Click '🚀 Run AI Re-check' to execute Vision AI verification and generate cleared annotations.")
            st.image(main_image, caption="Awaiting verification...", use_container_width=True)

with tab_patches:
    st.subheader("Patch-Level Forensic Inspection (Original Crop | Pixelated Blocks | Edge Discontinuities)")
    
    current_elements = verified_list if verified_list else candidates
    
    selected_elem_idx = st.selectbox(
        "Select an Element to Inspect at Forensic Level:",
        range(len(current_elements)),
        format_func=lambda i: f"Element #{current_elements[i].element_id} - BBox {current_elements[i].bbox} ({'🔴 BROKEN' if getattr(current_elements[i], 'is_verified_broken', None) is True else '🟢 INTACT' if getattr(current_elements[i], 'is_verified_broken', None) is False else '⚪ PENDING'})"
    )
    
    active_cand = current_elements[selected_elem_idx]
    patch = PatchProcessor.crop_element(main_image, active_cand.bbox)
    composite = PatchProcessor.create_inspection_composite(patch, pixel_size=pixel_size)
    
    st.image(composite, caption=f"Element #{active_cand.element_id}: [Left] Original Crop | [Center] {pixel_size}px Block Pixelation | [Right] Edge Discontinuity Filter", use_container_width=True)
    
    if verified_list:
        st.markdown(f"""
        **AI Forensic Analysis Report:**
        - **Status:** `{'🔴 VERIFIED BROKEN' if active_cand.is_verified_broken else '🟢 FALSE POSITIVE CLEARED (INTACT)'}`
        - **Confidence:** `{active_cand.verified_confidence:.1%}`
        - **Damage Classification:** `{active_cand.damage_type}`
        - **Visual Evidence & Findings:** *{active_cand.visual_evidence}*
        - **Recommended Action:** `{active_cand.recommended_action}`
        """)

with tab_audit:
    st.subheader("Audit Log & Exportable Verification Data")
    
    if verified_list:
        records = [c.to_dict() for c in verified_list]
        df_records = pd.DataFrame(records)
        
        st.dataframe(df_records, use_container_width=True)
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_data = df_records.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Audit Report (CSV)", csv_data, "damage_verification_report.csv", "text/csv")
        with col_dl2:
            json_data = json.dumps(records, indent=2)
            st.download_button("📥 Download Audit Report (JSON)", json_data, "damage_verification_report.json", "application/json")
    else:
        st.info("Run verification to generate the complete forensic audit log.")
