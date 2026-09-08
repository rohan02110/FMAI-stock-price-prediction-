"""
=============================================================================
Broken Window & Structural Element Verifier
Module: broken_element_verifier.py
Purpose: Re-verifies candidate broken windows/elements using patch pixelation
         and Gemini Vision AI to filter out false positives (e.g. reflections,
         lighting glares, frame shadows, undamaged glass).
=============================================================================
"""

import os
import sys
import json
import base64
import io
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Tuple

# Ensure portable site-packages is in sys.path
PORTABLE_PACKAGES = r"C:\Users\Hp\PythonPortable\Lib\site-packages"
if os.path.exists(PORTABLE_PACKAGES) and PORTABLE_PACKAGES not in sys.path:
    sys.path.insert(0, PORTABLE_PACKAGES)

from PIL import Image, ImageDraw, ImageFont, ImageFilter  # pyrefly: ignore [missing-import] # type: ignore


class CandidateElement:
    """Represents a detected window or structural element candidate."""
    def __init__(
        self,
        element_id: int,
        bbox: Tuple[int, int, int, int],  # (xmin, ymin, xmax, ymax)
        raw_label: str = "broken_window",
        raw_confidence: float = 0.85,
        element_type: str = "window"
    ):
        self.element_id = element_id
        self.bbox = bbox
        self.raw_label = raw_label
        self.raw_confidence = raw_confidence
        self.element_type = element_type
        
        # Post-verification results
        self.is_verified_broken: Optional[bool] = None
        self.verified_confidence: float = 0.0
        self.damage_type: str = "Unknown"
        self.visual_evidence: str = ""
        self.recommended_action: str = ""
        self.status: str = "PENDING_VERIFICATION"  # PENDING, VERIFIED_BROKEN, FALSE_POSITIVE_UNBROKEN, ERROR

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_id": self.element_id,
            "bbox": list(self.bbox),
            "element_type": self.element_type,
            "initial_prediction": self.raw_label,
            "initial_confidence": round(self.raw_confidence, 3),
            "is_verified_broken": self.is_verified_broken,
            "verification_status": self.status,
            "verified_confidence": round(self.verified_confidence, 3),
            "damage_type": self.damage_type,
            "visual_evidence": self.visual_evidence,
            "recommended_action": self.recommended_action
        }


class PatchProcessor:
    """Handles image cropping, zooming, and forensic pixelation analysis."""

    @staticmethod
    def crop_element(image: Image.Image, bbox: Tuple[int, int, int, int], padding_pct: float = 0.05) -> Image.Image:
        """Crops candidate element with optional context padding."""
        width, height = image.size
        xmin, ymin, xmax, ymax = bbox
        
        # Add slight padding for context
        pad_w = int((xmax - xmin) * padding_pct)
        pad_h = int((ymax - ymin) * padding_pct)
        
        crop_xmin = max(0, xmin - pad_w)
        crop_ymin = max(0, ymin - pad_h)
        crop_xmax = min(width, xmax + pad_w)
        crop_ymax = min(height, ymax + pad_h)
        
        return image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

    @staticmethod
    def pixelate_patch(patch: Image.Image, pixel_size: int = 8) -> Image.Image:
        """
        Applies pixelation to downsample and cluster high-frequency texture blocks.
        Useful for inspecting structural discontinuity vs smooth surface gradients.
        """
        w, h = patch.size
        if w < pixel_size or h < pixel_size:
            return patch
        
        # Downscale then upscale with NEAREST interpolation to achieve crisp pixelation blocks
        small = patch.resize((max(1, w // pixel_size), max(1, h // pixel_size)), resample=Image.Resampling.NEAREST)
        pixelated = small.resize((w, h), resample=Image.Resampling.NEAREST)
        return pixelated

    @staticmethod
    def create_inspection_composite(patch: Image.Image, pixel_size: int = 8) -> Image.Image:
        """
        Creates a side-by-side composite containing:
        1. High-Res Original Crop
        2. Forensic Pixelated Block Patch
        3. Edge Discontinuity Filter
        """
        w, h = patch.size
        pixelated = PatchProcessor.pixelate_patch(patch, pixel_size)
        
        # Generate edge discontinuity map
        gray = patch.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES).convert('RGB')
        
        composite = Image.new('RGB', (w * 3 + 20, h), color=(30, 30, 30))
        composite.paste(patch, (0, 0))
        composite.paste(pixelated, (w + 10, 0))
        composite.paste(edges, (w * 2 + 20, 0))
        
        return composite


class GeminiVisionVerifier:
    """Communicates directly with Google Gemini Multimodal Vision API."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        self.model_name = model_name

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip()

    def is_api_key_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 5)

    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def verify_candidate_patch(
        self,
        patch: Image.Image,
        pixelated_patch: Optional[Image.Image] = None,
        element_type: str = "window"
    ) -> Dict[str, Any]:
        """
        Calls Gemini Vision API with high-precision chain-of-thought prompt
        to verify if the window/element is genuinely broken or a false positive.
        """
        if not self.is_api_key_configured():
            return {
                "error": "API Key not configured",
                "is_broken": False,
                "confidence": 0.5,
                "damage_type": "Unverified (API Key Required)",
                "visual_evidence": "Please provide a valid Gemini API Key to enable AI re-verification.",
                "recommended_action": "Set GEMINI_API_KEY"
            }

        # Build multimodal prompt
        prompt_text = f"""
You are an expert Structural Forensic Engineer and Computer Vision Quality Inspector.
Examine this cropped image patch of a building/structure {element_type}.

BACKGROUND PROBLEM:
The initial automated detection model incorrectly flagged this {element_type} as 'BROKEN'.
However, many unbroken windows and smooth elements are mistakenly flagged as broken due to:
1. Exterior reflections (trees, clouds, sun glare, opposite buildings, power lines).
2. Window mullions, grids, blinds, tinted films, or curtains.
3. Shadows cast across glass or frames.
4. Dust, slight smudges, or normal surface textures that do NOT constitute structural fractures.

YOUR TASK:
Perform a strict forensic inspection to determine if this {element_type} is GENUINELY BROKEN / DAMAGED or UNBROKEN (FALSE POSITIVE).

CRITICAL CRITERIA FOR 'BROKEN':
- Genuine cracks (spiderweb fractures, radiating impact breaks, jagged sharp glass fissures).
- Missing glass panes or large gaping holes.
- Shattered tempered glass pebble patterns.
- Severely deformed or split structural frames.

CRITICAL CRITERIA FOR 'UNBROKEN' (False Positive):
- Continuous intact glass pane, even if there are reflections of tree branches or window frames.
- Intact frame with glare/reflections.
- Blinds/shades visible behind glass.

OUTPUT REQUIREMENT:
Respond ONLY with a valid JSON object matching this exact schema:
{{
    "is_broken": true or false,
    "confidence": float between 0.00 and 1.00,
    "damage_type": "None / Intact" OR specific type like "Spiderweb Glass Fracture", "Missing Pane", "Corner Impact Crack", "Frame Distortion",
    "visual_evidence": "Brief explanation of specific visual signs confirming broken status or explaining why it is a false positive reflection/shadow",
    "recommended_action": "e.g. 'No action needed (Glass intact)', 'Immediate glass replacement', 'Seal hairline crack'"
}}
"""

        endpoint_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

        # Prepare payload
        parts: List[Dict[str, Any]] = [
            {"text": prompt_text},
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": self._image_to_base64(patch)
                }
            }
        ]

        if pixelated_patch is not None:
            parts.append({"text": "Here is the pixelated/block analysis patch for inspecting structural edge continuity:"})
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": self._image_to_base64(pixelated_patch)
                }
            })

        payload = {
            "contents": [
                {
                    "parts": parts
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                endpoint_url,
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                result_json = json.loads(response.read().decode("utf-8"))

            candidates = result_json.get("candidates", [])
            if not candidates:
                return {
                    "is_broken": False,
                    "confidence": 0.5,
                    "damage_type": "API No Response",
                    "visual_evidence": "No response returned from Vision API",
                    "recommended_action": "Retry"
                }

            text_response = candidates[0]["content"]["parts"][0]["text"]
            # Clean possible markdown wrapping
            text_response = text_response.strip()
            if text_response.startswith("```json"):
                text_response = text_response[7:]
            if text_response.startswith("```"):
                text_response = text_response[3:]
            if text_response.endswith("```"):
                text_response = text_response[:-3]
            text_response = text_response.strip()

            parsed = json.loads(text_response)
            return {
                "is_broken": bool(parsed.get("is_broken", False)),
                "confidence": float(parsed.get("confidence", 0.9)),
                "damage_type": str(parsed.get("damage_type", "None")),
                "visual_evidence": str(parsed.get("visual_evidence", "")),
                "recommended_action": str(parsed.get("recommended_action", ""))
            }

        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8") if hasattr(e, "read") else str(e)
            return {
                "error": f"HTTP Error {e.code}: {err_msg}",
                "is_broken": False,
                "confidence": 0.0,
                "damage_type": "API Error",
                "visual_evidence": f"Failed to call Gemini API: {e.code}",
                "recommended_action": "Check API key & connectivity"
            }
        except Exception as e:
            return {
                "error": str(e),
                "is_broken": False,
                "confidence": 0.0,
                "damage_type": "Verification Exception",
                "visual_evidence": f"Processing error: {str(e)}",
                "recommended_action": "Check input format"
            }


class InspectionPipeline:
    """Coordinates detection, patch pixelation, re-verification, and visualization."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        self.verifier = GeminiVisionVerifier(api_key=api_key, model_name=model_name)
        self.patch_processor = PatchProcessor()

    def generate_candidate_grid(self, image: Image.Image, grid_cols: int = 3, grid_rows: int = 3) -> List[CandidateElement]:
        """
        Generates simulated/default grid of candidate elements across the image
        if an external object detector (YOLO/Faster-RCNN) is not connected.
        """
        w, h = image.size
        candidates = []
        elem_id = 1
        
        margin_x = int(w * 0.08)
        margin_y = int(h * 0.08)
        cell_w = (w - 2 * margin_x) // grid_cols
        cell_h = (h - 2 * margin_y) // grid_rows
        
        for r in range(grid_rows):
            for c in range(grid_cols):
                xmin = margin_x + c * cell_w + int(cell_w * 0.1)
                ymin = margin_y + r * cell_h + int(cell_h * 0.1)
                xmax = margin_x + (c + 1) * cell_w - int(cell_w * 0.1)
                ymax = margin_y + (r + 1) * cell_h - int(cell_h * 0.1)
                
                candidates.append(
                    CandidateElement(
                        element_id=elem_id,
                        bbox=(xmin, ymin, xmax, ymax),
                        raw_label="broken_window_candidate",
                        raw_confidence=0.75 + (elem_id % 3) * 0.08,
                        element_type="window"
                    )
                )
                elem_id += 1
                
        return candidates

    def run_reverification(
        self,
        image: Image.Image,
        candidates: List[CandidateElement],
        pixel_size: int = 8,
        progress_callback=None
    ) -> List[CandidateElement]:
        """
        Iterates over each candidate element, pixelates/analyzes patch, and calls Gemini Vision.
        """
        total = len(candidates)
        for idx, cand in enumerate(candidates):
            if progress_callback:
                progress_callback(idx / total, f"Re-verifying candidate #{cand.element_id} ({idx+1}/{total})...")

            patch = self.patch_processor.crop_element(image, cand.bbox)
            pixelated = self.patch_processor.pixelate_patch(patch, pixel_size=pixel_size)
            
            result = self.verifier.verify_candidate_patch(
                patch=patch,
                pixelated_patch=pixelated,
                element_type=cand.element_type
            )
            
            if "error" in result:
                cand.status = "ERROR"
                cand.damage_type = str(result.get("damage_type", "Error"))
                cand.visual_evidence = str(result.get("visual_evidence") or result.get("error", "Error"))
                cand.recommended_action = str(result.get("recommended_action", "Check API key or network connection"))
                cand.is_verified_broken = False
                cand.verified_confidence = 0.0
            else:
                is_broken = bool(result.get("is_broken", False))
                cand.is_verified_broken = is_broken
                cand.verified_confidence = float(result.get("confidence", 0.9))
                cand.damage_type = str(result.get("damage_type", "None"))
                cand.visual_evidence = str(result.get("visual_evidence", ""))
                cand.recommended_action = str(result.get("recommended_action", ""))
                
                if is_broken:
                    cand.status = "VERIFIED_BROKEN"
                else:
                    cand.status = "FALSE_POSITIVE_UNBROKEN"

        if progress_callback:
            progress_callback(1.0, "Verification complete.")

        return candidates

    def render_annotated_image(
        self,
        image: Image.Image,
        candidates: List[CandidateElement],
        show_raw: bool = False
    ) -> Image.Image:
        """
        Draws high-contrast bounding boxes and status labels:
        - If show_raw=True: Draws all candidates in warning orange/red (raw initial predictions).
        - If show_raw=False (Verified):
            * Red Bounding Box = Genuine Verified Broken
            * Green Bounding Box = Verified Unbroken (False Positive Cleared)
            * Gray/Yellow = Unverified / Error
        """
        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)
        
        try:
            # Attempt to load clean font, fallback to default
            font = ImageFont.truetype("arial.ttf", size=max(14, int(image.height * 0.022)))
            small_font = ImageFont.truetype("arial.ttf", size=max(11, int(image.height * 0.016)))
        except Exception:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        for cand in candidates:
            xmin, ymin, xmax, ymax = cand.bbox
            
            if show_raw:
                # Raw flagged mode
                box_color = (235, 90, 30)  # Bright Orange-Red
                label_text = f"FLAGGED BROKEN (#{cand.element_id}) [{cand.raw_confidence:.0%}]"
                sub_text = "Unverified Raw Detection"
            else:
                # Post-verification mode
                if cand.status == "VERIFIED_BROKEN":
                    box_color = (220, 35, 35)  # Crimson Red
                    label_text = f"🔴 BROKEN (#{cand.element_id}) [{cand.verified_confidence:.0%}]"
                    sub_text = cand.damage_type
                elif cand.status == "FALSE_POSITIVE_UNBROKEN":
                    box_color = (30, 185, 60)  # Emerald Green
                    label_text = f"🟢 INTACT / UNBROKEN (#{cand.element_id}) [{cand.verified_confidence:.0%}]"
                    sub_text = "False Positive Cleared"
                else:
                    box_color = (160, 160, 160)
                    label_text = f"⚪ #{cand.element_id} Pending"
                    sub_text = cand.status

            # Draw thick bounding box
            box_width = max(3, int(image.height * 0.004))
            for i in range(box_width):
                draw.rectangle([xmin - i, ymin - i, xmax + i, ymax + i], outline=box_color)

            # Draw label banner
            text_bbox = draw.textbbox((xmin, ymin), label_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            banner_h = text_h + 16
            banner_w = max(text_w + 14, 180)
            
            banner_ymin = max(0, ymin - banner_h)
            banner_ymax = banner_ymin + banner_h
            
            # Fill background banner
            draw.rectangle([xmin, banner_ymin, xmin + banner_w, banner_ymax], fill=box_color)
            draw.text((xmin + 6, banner_ymin + 2), label_text, fill=(255, 255, 255), font=font)
            draw.text((xmin + 6, banner_ymin + text_h + 3), sub_text, fill=(240, 240, 240), font=small_font)

        return annotated


def create_demo_building_image() -> Image.Image:
    """Generates a high-resolution synthetic building facade with reflective unbroken windows and 1 broken window."""
    w, h = 1000, 700
    img = Image.new('RGB', (w, h), color=(220, 225, 230))
    draw = ImageDraw.Draw(img)
    
    # Building brick/concrete facade
    draw.rectangle([60, 40, 940, 660], fill=(70, 80, 95), outline=(40, 45, 55), width=4)
    
    # 3x3 Window Grid
    for row in range(3):
        for col in range(3):
            wx1 = 120 + col * 270
            wy1 = 80 + row * 180
            wx2 = wx1 + 220
            wy2 = wy1 + 140
            
            # Window Frame
            draw.rectangle([wx1 - 8, wy1 - 8, wx2 + 8, wy2 + 8], fill=(30, 35, 42))
            
            is_damaged_window = (row == 1 and col == 1)  # Center window is actually broken
            
            if is_damaged_window:
                # Broken window: dark background + jagged fracture lines + missing glass hole
                draw.rectangle([wx1, wy1, wx2, wy2], fill=(25, 30, 40))
                # Radiating jagged cracks
                center_x, center_y = (wx1 + wx2) // 2, (wy1 + wy2) // 2
                draw.polygon([(center_x - 30, center_y - 20), (center_x + 35, center_y - 15), 
                              (center_x + 20, center_y + 35), (center_x - 25, center_y + 25)], fill=(10, 10, 15))
                # White/cyan crack lines
                draw.line([(wx1 + 20, wy1 + 10), (center_x - 15, center_y - 10), (wx2 - 10, wy2 - 20)], fill=(240, 245, 255), width=3)
                draw.line([(center_x, center_y), (wx1 + 40, wy2 - 10)], fill=(230, 240, 255), width=2)
                draw.line([(center_x + 10, center_y - 5), (wx2 - 20, wy1 + 15)], fill=(250, 250, 255), width=3)
                draw.line([(center_x - 10, center_y + 15), (wx1 + 10, wy1 + 80)], fill=(220, 230, 245), width=2)
            else:
                # Intact window with natural reflection (clouds, tree branch shadow, window mullion grid)
                draw.rectangle([wx1, wy1, wx2, wy2], fill=(135, 185, 220))
                # White cloud reflection streak
                draw.polygon([(wx1 + 30, wy1), (wx1 + 110, wy1), (wx1 + 40, wy2), (wx1, wy2)], fill=(185, 220, 245))
                # Tree branch shadow (often misclassified as cracks by simple edge detectors!)
                if row == 0:
                    draw.line([(wx1 + 10, wy1 + 20), (wx1 + 80, wy1 + 60), (wx1 + 160, wy1 + 50)], fill=(95, 130, 150), width=3)
                    draw.line([(wx1 + 80, wy1 + 60), (wx1 + 120, wy1 + 110)], fill=(95, 130, 150), width=2)
                # Window mullion divider
                mid_x = (wx1 + wx2) // 2
                mid_y = (wy1 + wy2) // 2
                draw.line([(mid_x, wy1), (mid_x, wy2)], fill=(45, 55, 65), width=4)
                draw.line([(wx1, mid_y), (wx2, mid_y)], fill=(45, 55, 65), width=4)

    return img

