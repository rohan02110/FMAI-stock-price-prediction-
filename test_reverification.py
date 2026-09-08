"""
=============================================================================
Broken Element Re-verification Test Script (CLI & Automated Validation)
File: test_reverification.py
=============================================================================
"""

import os
import sys

# Ensure script directory and portable site-packages are in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

PORTABLE_PACKAGES = r"C:\Users\Hp\PythonPortable\Lib\site-packages"
if os.path.exists(PORTABLE_PACKAGES) and PORTABLE_PACKAGES not in sys.path:
    sys.path.insert(0, PORTABLE_PACKAGES)

from PIL import Image  # pyrefly: ignore [missing-import] # type: ignore

from broken_element_verifier import (  # pyrefly: ignore [missing-import] # type: ignore
    CandidateElement,
    PatchProcessor,
    GeminiVisionVerifier,
    InspectionPipeline,
    create_demo_building_image
)


def main():
    print("=" * 80)
    print("TESTING BROKEN WINDOW & ELEMENT RE-VERIFICATION PIPELINE")
    print("=" * 80)

    # 1. Generate Synthetic Building Test Image
    print("[1/4] Generating synthetic building image with reflective intact windows & 1 broken window...")
    img = create_demo_building_image()
    os.makedirs(os.path.join("outputs", "vision_tests"), exist_ok=True)
    test_img_path = os.path.join("outputs", "vision_tests", "building_facade_test.png")
    img.save(test_img_path)
    print(f"  -> Saved test scene to: {test_img_path}")

    # 2. Check API Key
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    print(f"[2/4] Checking Gemini API Key: {'Found (' + api_key[:4] + '...' + api_key[-4:] + ')' if api_key else 'Not set in environment'}")

    pipeline = InspectionPipeline(api_key=api_key)

    # 3. Generate candidate bounding boxes (simulating initial model that flagged all as broken)
    print("[3/4] Generating 3x3 candidate window grid (simulating raw model flags)...")
    candidates = pipeline.generate_candidate_grid(img, grid_cols=3, grid_rows=3)
    print(f"  -> Extracted {len(candidates)} candidate elements.")

    # Render raw initial detection image
    raw_img = pipeline.render_annotated_image(img, candidates, show_raw=True)
    raw_path = os.path.join("outputs", "vision_tests", "raw_detections_all_flagged.png")
    raw_img.save(raw_path)
    print(f"  -> Saved raw flagged visualization to: {raw_path}")

    # 4. Test Patch Pixelation and Composite Generation
    print("[4/4] Testing patch extraction and multi-resolution pixelation...")
    sample_patch = PatchProcessor.crop_element(img, candidates[0].bbox)
    composite = PatchProcessor.create_inspection_composite(sample_patch, pixel_size=8)
    composite_path = os.path.join("outputs", "vision_tests", "sample_patch_pixelated_composite.png")
    composite.save(composite_path)
    print(f"  -> Saved forensic inspection composite to: {composite_path}")

    if api_key:
        print("\n[AI RE-VERIFICATION] Running Gemini Vision API re-verification across candidates...")
        verified = pipeline.run_reverification(img, candidates, pixel_size=8)
        
        post_img = pipeline.render_annotated_image(img, verified, show_raw=False)
        post_path = os.path.join("outputs", "vision_tests", "reverified_results.png")
        post_img.save(post_path)
        print(f"  -> Saved post-verification annotated image to: {post_path}")

        print("\n" + "=" * 80)
        print("VERIFICATION RESULTS SUMMARY:")
        print("=" * 80)
        for c in verified:
            status_icon = "🔴 BROKEN" if c.is_verified_broken else "🟢 INTACT (FALSE POSITIVE CLEARED)"
            print(f"Element #{c.element_id} ({c.bbox}): {status_icon} | Conf: {c.verified_confidence:.1%} | Evidence: {c.visual_evidence}")
    else:
        print("\n[NOTE] To run live AI verification, set GEMINI_API_KEY or launch the Streamlit dashboard via:")
        print("       streamlit run vision_damage_dashboard.py")

    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
