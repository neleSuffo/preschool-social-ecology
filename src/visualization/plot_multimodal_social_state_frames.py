import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from supervision import Detections
from ultralytics import YOLO

from constants import Analysis, FaceDetection, PersonDetection
from models.proximity.estimate_proximity import calculate_proximity

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Standardized Pastel Palette (BGR format) ---
COLOR_AUDIO_BGR = (209, 192, 242)   # Soft Rose
COLOR_FACE_BGR = (195, 175, 140)    # Slate / Teal Gray
COLOR_PERSON_BGR = (175, 215, 185)  # Light Sage Green

# UI styling
COLOR_BOX_BACKING = (20, 20, 22)
COLOR_TEXT_DARK = (25, 25, 25)
COLOR_TEXT_LIGHT = (245, 245, 245)
COLOR_CARD_BG = (26, 26, 28)

BOX_THICKNESS = 8
BACKING_OFFSET = 4
PAD = 8
TEXT_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_csv_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    try:
        df = pd.read_csv(csv_path, sep=";")
        if len(df.columns) <= 1:
            df = pd.read_csv(csv_path, sep=",")
    except Exception:
        df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def parse_filename_identifiers(
    image_path: Path,
) -> Tuple[Optional[str], Optional[int]]:
    stem = image_path.stem
    match = re.search(r"^(.*?)(?:_frame|_)?_(\d+)$", stem)
    if match:
        video_name = match.group(1).strip()
        frame_idx = int(match.group(2))
        return video_name, frame_idx
    return None, None


def lookup_segment_social_state(
    segments_df: pd.DataFrame, video_name: str, frame_idx: int
) -> str:
    seg_sub = segments_df[
        segments_df["video_name"].astype(str).str.strip() == video_name
    ].copy()

    if seg_sub.empty:
        clean_vid = re.sub(r"\.[^.]+$", "", video_name)
        seg_sub = segments_df[
            segments_df["video_name"].astype(str).str.strip().str.replace(
                r"\.[^.]+$", "", regex=True
            ) == clean_vid
        ].copy()

    if seg_sub.empty:
        return "Alone"

    matched_seg = seg_sub[
        (seg_sub["segment_start"] <= frame_idx) & (seg_sub["segment_end"] >= frame_idx)
    ]

    if matched_seg.empty:
        return "Alone"

    raw_type = matched_seg.iloc[0].get("interaction_type", "Alone")
    state_map = {
        1: "Interacting",
        2: "Available",
        3: "Alone",
        "1": "Interacting",
        "2": "Available",
        "3": "Alone",
        "interacting": "Interacting",
        "available": "Available",
        "alone": "Alone",
    }
    return state_map.get(raw_type, str(raw_type).capitalize())


def run_model_inference(
    model: YOLO, image_path: Path
) -> Tuple[np.ndarray, Detections]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image from: {image_path}")
    output = model(Image.open(image_path))
    detections = Detections.from_ultralytics(output[0])
    return image, detections


def draw_labeled_box(
    image: np.ndarray,
    bbox: np.ndarray,
    label: str,
    box_color: Tuple[int, int, int],
    font_scale: float = 1.1,
    place_above: bool = True,
) -> np.ndarray:
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    # 1. Dark backing outline + pastel stroke
    cv2.rectangle(
        annotated, (x1, y1), (x2, y2), COLOR_BOX_BACKING, BOX_THICKNESS + BACKING_OFFSET
    )
    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)

    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, font_scale, TEXT_THICKNESS)
    text_x = max(PAD + 4, min(x1 + 4, img_w - text_w - PAD - 6))

    if place_above:
        text_y = y1 - baseline - PAD - BOX_THICKNESS
        if text_y - text_h - PAD < 0:
            text_y = y1 + text_h + PAD + BOX_THICKNESS + 4
    else:
        text_y = y2 + text_h + PAD + BOX_THICKNESS + 4
        if text_y + baseline + PAD > img_h:
            text_y = y2 - baseline - PAD - BOX_THICKNESS - 4

    bg_p1 = (text_x - PAD, text_y - text_h - PAD)
    bg_p2 = (text_x + text_w + PAD, text_y + baseline + PAD)

    # 2. Solid pastel badge with dark text
    cv2.rectangle(annotated, bg_p1, bg_p2, box_color, -1)
    cv2.rectangle(annotated, bg_p1, bg_p2, COLOR_BOX_BACKING, 2)
    cv2.putText(
        annotated,
        label,
        (text_x, text_y),
        FONT,
        font_scale,
        COLOR_TEXT_DARK,
        TEXT_THICKNESS,
        cv2.LINE_AA,
    )
    return annotated


def draw_rule_criteria_card_top_right(
    image: np.ndarray,
    rules: List[Dict[str, any]],
    has_kcds: bool,
    has_kcs: bool,
    has_ohs: bool,
    alpha: float = 0.90,
    margin: int = 24,
) -> np.ndarray:
    """Renders the state classification criteria HUD on the top right without naming the state."""
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    COLOR_ACTIVE_CHECK = (50, 215, 50)
    COLOR_INACTIVE = (120, 120, 125)

    header_scale = 1.15
    rule_scale = 1.00
    detail_scale = 0.85
    ui_thick = 2
    pad_x = 24
    pad_y = 22

    header_text = "Classification Criteria"
    rendered_lines = []

    # 1. State Heuristic Rules
    for r in rules:
        status_sym = "[v]" if r["active"] else "[ ]"
        rule_str = f"{status_sym} {r['name']}"
        rendered_lines.append(
            {"text": rule_str, "scale": rule_scale, "active": r["active"], "is_detail": False, "is_audio": False}
        )
        if r.get("detail"):
            detail_str = f"    {r['detail']}"
            rendered_lines.append(
                {"text": detail_str, "scale": detail_scale, "active": r["active"], "is_detail": True, "is_audio": False}
            )

    # 2. Raw Audio Inputs
    audio_items = [
        ("KCDS", has_kcds),
        ("KCS", has_kcs),
        ("OHS", has_ohs),
    ]

    # Calculate optimal width
    max_w = cv2.getTextSize(header_text, FONT, header_scale, ui_thick)[0][0]
    for line in rendered_lines:
        w = cv2.getTextSize(line["text"], FONT, line["scale"], ui_thick)[0][0]
        max_w = max(max_w, w)

    # Audio row text width estimate
    audio_row_w = 400
    max_w = max(max_w, audio_row_w)
    card_w = max_w + (pad_x * 2)

    # Calculate height
    line_gap = 10
    total_h = pad_y + 30
    for line in rendered_lines:
        h = cv2.getTextSize(line["text"], FONT, line["scale"], ui_thick)[0][1]
        total_h += h + line_gap
    total_h += 64 + pad_y
    card_h = total_h

    # PINNED TO TOP-RIGHT
    x2 = img_w - margin
    x1 = x2 - card_w
    y1 = margin
    y2 = y1 + card_h

    overlay = annotated.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_CARD_BG, -1)
    cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (85, 85, 90), 2)

    # 1. Header
    curr_y = y1 + pad_y + 20
    cv2.putText(
        annotated, header_text, (x1 + pad_x, curr_y),
        FONT, header_scale, COLOR_TEXT_LIGHT, ui_thick, cv2.LINE_AA
    )
    curr_y += 14
    cv2.line(
        annotated, (x1 + pad_x, curr_y), (x2 - pad_x, curr_y),
        (85, 85, 90), 1, cv2.LINE_AA
    )
    curr_y += 24

    # 2. Rule Checklist Rows
    for line in rendered_lines:
        text = line["text"]
        scale = line["scale"]
        active = line["active"]
        is_detail = line["is_detail"]
        _, h = cv2.getTextSize(text, FONT, scale, ui_thick)[0]

        if is_detail:
            text_color = COLOR_TEXT_LIGHT if active else COLOR_INACTIVE
            cv2.putText(
                annotated, text, (x1 + pad_x, curr_y),
                FONT, scale, text_color, 1, cv2.LINE_AA
            )
        else:
            sym = text[:3]
            label = text[3:]
            sym_color = COLOR_ACTIVE_CHECK if active else COLOR_INACTIVE
            text_color = COLOR_TEXT_LIGHT if active else COLOR_INACTIVE

            cv2.putText(
                annotated, sym, (x1 + pad_x, curr_y),
                FONT, scale, sym_color, ui_thick, cv2.LINE_AA
            )
            sym_w = cv2.getTextSize(sym, FONT, scale, ui_thick)[0][0]
            cv2.putText(
                annotated, label, (x1 + pad_x + sym_w, curr_y),
                FONT, scale, text_color, ui_thick, cv2.LINE_AA
            )

        curr_y += h + line_gap

    # 3. Audio Sub-Section
    curr_y += 6
    cv2.line(
        annotated, (x1 + pad_x, curr_y), (x2 - pad_x, curr_y),
        (85, 85, 90), 1, cv2.LINE_AA
    )
    curr_y += 24

    cv2.putText(
        annotated, "Audio:", (x1 + pad_x, curr_y),
        FONT, 0.85, (200, 200, 205), 1, cv2.LINE_AA
    )

    offset_x = x1 + pad_x + 85
    for name, is_active in audio_items:
        sym = "[v]" if is_active else "[ ]"
        pill_str = f"{sym} {name}"
        col = COLOR_AUDIO_BGR if is_active else COLOR_INACTIVE
        txt_col = COLOR_TEXT_LIGHT if is_active else COLOR_INACTIVE

        cv2.putText(
            annotated, sym, (offset_x, curr_y),
            FONT, 0.85, col, ui_thick, cv2.LINE_AA
        )
        sw = cv2.getTextSize(sym, FONT, 0.85, ui_thick)[0][0]
        cv2.putText(
            annotated, f" {name}", (offset_x + sw, curr_y),
            FONT, 0.85, txt_col, 1, cv2.LINE_AA
        )
        offset_x += 105

    return annotated


def build_rules_checklist_for_segment_state(
    segment_state: str, row: Optional[pd.Series]
) -> List[Dict[str, any]]:
    def to_bool(val) -> bool:
        if pd.isna(val):
            return False
        return bool(int(val)) if isinstance(val, (int, float, np.number)) else bool(val)

    r1_turn_taking = to_bool(row.get("rule1_turn_taking", False)) if row is not None else False
    r2_proximity = to_bool(row.get("rule2_close_proximity", False)) if row is not None else False
    r3_sustained_kcds = to_bool(row.get("rule3_kcds_speaking", False)) if row is not None else False

    r4_persistence = to_bool(row.get("person_seen_recently", False)) if row is not None else False
    presence_score = row.get("presence_score", 0.0) if row is not None else 0.0
    is_visual_anchor = presence_score >= 0.15
    r5_intermittent_speech = (
        to_bool(row.get("is_sustained_ohs", False)) and is_visual_anchor
    ) if row is not None else False

    prox_val = row.get("proximity", np.nan) if row is not None else np.nan

    if segment_state == "Interacting":
        prox_detail = f"px = {prox_val:.2f}" if (pd.notna(prox_val) and prox_val > 0) else ""
        r1_detail = "KCS <-> KCDS dyad" if r1_turn_taking else ""
        r3_detail = "Gated KCDS active" if r3_sustained_kcds else ""

        return [
            {"name": "R1: Turn-Taking", "active": r1_turn_taking, "detail": r1_detail},
            {"name": "R2: Face Proximity", "active": r2_proximity, "detail": prox_detail},
            {"name": "R3: Sustained KCDS", "active": r3_sustained_kcds, "detail": r3_detail},
        ]

    elif segment_state == "Available":
        has_face = to_bool(row.get("has_face", False)) if row is not None else False
        has_person = to_bool(row.get("has_person", False)) if row is not None else False
        r4_detail = "Person/Face visible" if (has_face or has_person) else (
            "Partner in memory" if r4_persistence else ""
        )
        r5_detail = "Gated OHS active" if r5_intermittent_speech else ""

        return [
            {"name": "R4: Visual Persistence", "active": r4_persistence, "detail": r4_detail},
            {"name": "R5: Intermittent OHS", "active": r5_intermittent_speech, "detail": r5_detail},
        ]

    else:  # Alone
        return [
            {"name": "R1-R3: Interacting", "active": False, "detail": "No interaction cues"},
            {"name": "R4-R5: Available", "active": False, "detail": "No partner cues"},
            {"name": "Baseline State", "active": True, "detail": "Default: Alone"},
        ]


def process_image(
    image_path: Path,
    output_dir: Path,
    frames_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    face_model: YOLO,
    person_model: YOLO,
):
    video_name, video_frame = parse_filename_identifiers(image_path)
    if video_name is None or video_frame is None:
        raise ValueError(f"Could not infer video_name or video_frame from {image_path.name}")

    # 1. Overarching segment state
    segment_state = lookup_segment_social_state(segments_df, video_name, video_frame)

    # 2. Extract frame-level empirical indicators
    subset_df = frames_df[frames_df["video_name"] == video_name]
    frame_col = "video_frame" if "video_frame" in subset_df.columns else "frame_number"
    subset_df = subset_df[subset_df[frame_col] == int(video_frame)]
    frame_row = subset_df.iloc[0] if not subset_df.empty else None

    rules = build_rules_checklist_for_segment_state(segment_state, frame_row)

    def to_bool(val) -> bool:
        if pd.isna(val):
            return False
        return bool(int(val)) if isinstance(val, (int, float, np.number)) else bool(val)

    has_kcds = to_bool(frame_row.get("has_cds", False)) if frame_row is not None else False
    has_kcs = to_bool(frame_row.get("has_kchi", False)) if frame_row is not None else False
    has_ohs = to_bool(frame_row.get("has_ohs", False)) if frame_row is not None else False

    # 3. Model Inference & Standardized Bounding Boxes
    raw_img, face_results = run_model_inference(face_model, image_path)
    _, person_results = run_model_inference(person_model, image_path)
    annotated = raw_img.copy()

    for bbox, conf in zip(person_results.xyxy, person_results.confidence):
        annotated = draw_labeled_box(
            annotated, bbox, f"Person ({conf:.2f})", COLOR_PERSON_BGR, place_above=False
        )

    for bbox, conf, cls_id in zip(
        face_results.xyxy, face_results.confidence, face_results.class_id
    ):
        x1, y1, x2, y2 = map(int, bbox)
        prox = calculate_proximity([x1, y1, x2, y2], cls_id)
        if isinstance(prox, (list, tuple, np.ndarray)):
            prox = float(prox[0])
        annotated = draw_labeled_box(
            annotated, bbox, f"Face ({conf:.2f}), px: {prox:.2f}", COLOR_FACE_BGR, place_above=True
        )

    # 4. Top-Right Criteria HUD (no state banner)
    final_output = draw_rule_criteria_card_top_right(
        annotated, rules, has_kcds, has_kcs, has_ohs, margin=24
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}_multimodal_annotated.jpg"
    cv2.imwrite(str(out_path), final_output, [cv2.IMWRITE_JPEG_QUALITY, 95])
    logging.info(f"Saved annotated frame to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate Frame with Detections and Overarching Criteria HUD"
    )
    parser.add_argument("--image_path", type=str, required=True, help="Path to an image file or directory of frames.")
    parser.add_argument("--frame_csv", type=str, default=str(Analysis.FRAME_LEVEL_INTERACTIONS_CSV), help="Path to frame_level_social_interactions.csv.")
    parser.add_argument("--segments_csv", type=str, default=str(Analysis.INTERACTION_SEGMENTS_CSV), help="Path to interaction_segments.csv.")
    parser.add_argument("--output_dir", type=str, default=str(Analysis.FINAL_OUTPUT_FOLDER / "multimodal_state_examples"), help="Target output folder.")
    args = parser.parse_args()

    logging.info(f"Loading frame interactions from: {args.frame_csv}")
    frames_df = load_csv_data(Path(args.frame_csv))

    logging.info(f"Loading interaction segments from: {args.segments_csv}")
    segments_df = load_csv_data(Path(args.segments_csv))

    logging.info("Loading YOLO weights...")
    face_model = YOLO(FaceDetection.TRAINED_WEIGHTS_PATH)
    person_model = YOLO(PersonDetection.TRAINED_WEIGHTS_PATH)

    input_p = Path(args.image_path)
    output_p = Path(args.output_dir)

    if input_p.is_dir():
        imgs = (
            list(input_p.glob("*.jpg"))
            + list(input_p.glob("*.png"))
            + list(input_p.glob("*.PNG"))
        )
        for img in imgs:
            process_image(img, output_p, frames_df, segments_df, face_model, person_model)
    else:
        process_image(input_p, output_p, frames_df, segments_df, face_model, person_model)


if __name__ == "__main__":
    exit(main())