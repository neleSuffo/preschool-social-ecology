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

# --- Standardized Pastel Palette (BGR) ---
COLOR_AUDIO_BGR = (209, 192, 242)   # Soft Rose
COLOR_FACE_BGR = (195, 175, 140)    # Slate / Teal Gray
COLOR_PERSON_BGR = (175, 215, 185)  # Light Sage Green

# UI styling (White Cards with Dark Text)
COLOR_BOX_BACKING = (20, 20, 22)
COLOR_TEXT_DARK = (20, 20, 22)
COLOR_TEXT_LIGHT = (250, 250, 250)

# Card Inversion Variables
COLOR_CARD_BG = (255, 255, 255)     # Solid White Card Background
COLOR_CARD_BORDER = (180, 180, 185) # Crisp Outer Outline
COLOR_CARD_LINE = (210, 210, 215)   # Internal Divider Lines
COLOR_INACTIVE_DARK = (130, 130, 135)
COLOR_ACTIVE_CHECK = (35, 165, 35)  # Deeper green for contrast on white

# Bounding box dimensions
BOX_THICKNESS = 10
BACKING_OFFSET = 5
BADGE_PAD = 14
TEXT_THICKNESS = 3
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
    font_scale: float = 2.5,
    place_above: bool = True,
) -> np.ndarray:
    """Draws a prominent box with bold dark outlines and enlarged, filled pastel badges."""
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    # 1. Dark backing outline + pastel stroke
    cv2.rectangle(
        annotated, (x1, y1), (x2, y2), COLOR_BOX_BACKING, BOX_THICKNESS + BACKING_OFFSET
    )
    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)

    # 2. Badge size calculation
    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, font_scale, TEXT_THICKNESS)
    text_x = max(BADGE_PAD + 6, min(x1 + 4, img_w - text_w - BADGE_PAD - 8))

    if place_above:
        text_y = y1 - baseline - BADGE_PAD - BOX_THICKNESS
        if text_y - text_h - BADGE_PAD < 0:
            text_y = y1 + text_h + BADGE_PAD + BOX_THICKNESS + 6
    else:
        text_y = y2 + text_h + BADGE_PAD + BOX_THICKNESS + 6
        if text_y + baseline + BADGE_PAD > img_h:
            text_y = y2 - baseline - BADGE_PAD - BOX_THICKNESS - 6

    bg_p1 = (text_x - BADGE_PAD, text_y - text_h - BADGE_PAD)
    bg_p2 = (text_x + text_w + BADGE_PAD, text_y + baseline + BADGE_PAD)

    # 3. Solid pastel badge with dark edge outline
    cv2.rectangle(annotated, bg_p1, bg_p2, box_color, -1)
    cv2.rectangle(annotated, bg_p1, bg_p2, COLOR_BOX_BACKING, 3)
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


def draw_rule_criteria_card_bottom_right(
    image: np.ndarray,
    rules: List[Dict[str, any]],
    alpha: float = 0.94,
    margin: int = 30,
    bottom_clearance: int = 0,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Renders the white 'Classification Rules' HUD card and returns its coordinates."""
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    header_scale = 2.5
    rule_scale = 2.0
    ui_thick = 3

    pad_x = 48
    pad_y = 42
    rule_block_gap = 32
    detail_gap = 18
    line_vertical_gap = 30

    header_text = "Classification Rules"

    # Pre-render rule items
    rendered_items = []
    for r in rules:
        sym = "[v]" if r["active"] else "[ ]"
        rule_str = f"{sym} {r['name']}"
        rendered_items.append({
            "text": rule_str,
            "scale": rule_scale,
            "active": r["active"],
            "is_detail": False
        })

    # Dynamic dimensions
    max_w = cv2.getTextSize(header_text, FONT, header_scale, ui_thick)[0][0]
    for item in rendered_items:
        w = cv2.getTextSize(item["text"], FONT, item["scale"], ui_thick)[0][0]
        max_w = max(max_w, w)
    card_w = max_w + (pad_x * 2)

    header_h = cv2.getTextSize(header_text, FONT, header_scale, ui_thick)[0][1]
    total_h = pad_y + header_h + line_vertical_gap + line_vertical_gap

    for i, item in enumerate(rendered_items):
        h = cv2.getTextSize(item["text"], FONT, item["scale"], ui_thick)[0][1]
        is_last = (i == len(rendered_items) - 1)
        next_is_detail = (i + 1 < len(rendered_items) and rendered_items[i + 1]["is_detail"])
        gap = detail_gap if next_is_detail else (rule_block_gap if not is_last else 0)
        total_h += h + gap

    total_h += pad_y
    card_h = total_h

    # Card coordinates (PINNED BOTTOM-RIGHT with bottom_clearance for the audio badge)
    x2 = img_w - margin
    x1 = x2 - card_w
    y2 = img_h - margin - bottom_clearance
    y1 = y2 - card_h

    # White Background
    overlay = annotated.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_CARD_BG, -1)
    cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_CARD_BORDER, 3)

    # 1. Header
    curr_y = y1 + pad_y + header_h
    cv2.putText(
        annotated, header_text, (x1 + pad_x, curr_y),
        FONT, header_scale, COLOR_TEXT_DARK, ui_thick, cv2.LINE_AA
    )

    # Divider Line
    curr_y += line_vertical_gap
    cv2.line(
        annotated, (x1 + pad_x, curr_y), (x2 - pad_x, curr_y),
        COLOR_CARD_LINE, 2, cv2.LINE_AA
    )
    curr_y += line_vertical_gap

    # 2. Rule Checklist Rows
    for i, item in enumerate(rendered_items):
        text = item["text"]
        scale = item["scale"]
        active = item["active"]
        is_detail = item["is_detail"]
        _, h = cv2.getTextSize(text, FONT, scale, ui_thick)[0]
        curr_y += h

        if is_detail:
            text_color = COLOR_TEXT_DARK if active else COLOR_INACTIVE_DARK
            cv2.putText(
                annotated, text, (x1 + pad_x, curr_y),
                FONT, scale, text_color, 2, cv2.LINE_AA
            )
        else:
            sym = text[:3]
            label = text[3:]
            sym_color = COLOR_ACTIVE_CHECK if active else COLOR_INACTIVE_DARK
            text_color = COLOR_TEXT_DARK if active else COLOR_INACTIVE_DARK

            cv2.putText(
                annotated, sym, (x1 + pad_x, curr_y),
                FONT, scale, sym_color, ui_thick, cv2.LINE_AA
            )
            sym_w = cv2.getTextSize(sym, FONT, scale, ui_thick)[0][0]
            cv2.putText(
                annotated, label, (x1 + pad_x + sym_w, curr_y),
                FONT, scale, text_color, ui_thick, cv2.LINE_AA
            )

        is_last = (i == len(rendered_items) - 1)
        next_is_detail = (i + 1 < len(rendered_items) and rendered_items[i + 1]["is_detail"])
        curr_y += detail_gap if next_is_detail else (rule_block_gap if not is_last else 0)

    return annotated, (x1, y1, x2, y2)


def draw_audio_badge_below(
    image: np.ndarray,
    card_bounds: Tuple[int, int, int, int],
    has_kcds: bool,
    has_kcs: bool,
    has_ohs: bool,
    spacing: int = 14,
    font_scale: float = 2.5,
) -> np.ndarray:
    """Renders a standalone Rose audio badge below the classification card matching Person/Face badge styling."""
    annotated = image.copy()
    c_x1, _, c_x2, c_y2 = card_bounds

    # Build audio content
    audio_parts = []
    if has_kcds:
        audio_parts.append("KCDS")
    if has_kcs:
        audio_parts.append("KCS")
    if has_ohs:
        audio_parts.append("OHS")

    label = f"Audio: {' + '.join(audio_parts)}" if audio_parts else "Audio: None"

    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, font_scale, TEXT_THICKNESS)

    badge_w = text_w + (BADGE_PAD * 2)
    badge_h = text_h + baseline + (BADGE_PAD * 2)

    # Align badge with the right edge of the card above it
    x2 = c_x2
    x1 = x2 - badge_w
    y1 = c_y2 + spacing
    y2 = y1 + badge_h

    # Filled Soft Rose with Dark Outline (matching Person/Face badges)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_AUDIO_BGR, -1)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_BOX_BACKING, 3)

    text_x = x1 + BADGE_PAD
    text_y = y1 + BADGE_PAD + text_h

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

    if segment_state == "Interacting":
        return [
            {"name": "R1: Turn-Taking", "active": r1_turn_taking},
            {"name": "R2: Face Proximity", "active": r2_proximity},
            {"name": "R3: Sustained KCDS", "active": r3_sustained_kcds},
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
    else:
        return [
            {"name": "R1-R3: Interacting", "active": False, "detail": "No interaction cues"},
            {"name": "R4-R5: Available", "active": False, "detail": "No partner cues"},
            {"name": "Baseline State", "active": True, "detail": "Default: Alone"},
        ]


def draw_frame_number_badge(
    image: np.ndarray,
    frame_idx: int,
    margin: int = 30,
    font_scale: float = 2.5,
    pad_x: int = 24,
    pad_y: int = 16,
) -> np.ndarray:
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    text = f"Frame: {frame_idx}"
    ui_thick = 3

    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, font_scale, ui_thick)

    card_w = text_w + (pad_x * 2)
    card_h = text_h + (pad_y * 2)

    x2 = img_w - margin
    x1 = x2 - card_w
    y1 = margin
    y2 = y1 + card_h

    overlay = annotated.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_CARD_BG, -1)
    cv2.addWeighted(overlay, 0.94, annotated, 0.06, 0, annotated)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_CARD_BORDER, 3)

    text_x = x1 + pad_x
    text_y = y1 + pad_y + text_h
    cv2.putText(
        annotated,
        text,
        (text_x, text_y),
        FONT,
        font_scale,
        COLOR_TEXT_DARK,
        ui_thick,
        cv2.LINE_AA,
    )
    return annotated


def process_image(
    image_path: Path,
    output_dir: Path,
    frames_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    face_model: YOLO,
    person_model: YOLO,
) -> Optional[Dict]:
    video_name, video_frame = parse_filename_identifiers(image_path)
    if video_name is None or video_frame is None:
        raise ValueError(f"Could not infer video_name or video_frame from {image_path.name}")

    segment_state = lookup_segment_social_state(segments_df, video_name, video_frame)

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

    raw_img, face_results = run_model_inference(face_model, image_path)
    _, person_results = run_model_inference(person_model, image_path)
    annotated = raw_img.copy()

    # Draw Person Boxes
    for bbox, conf in zip(person_results.xyxy, person_results.confidence):
        annotated = draw_labeled_box(
            annotated,
            bbox,
            f"Person ({conf:.2f})",
            COLOR_PERSON_BGR,
            font_scale=2.5,
            place_above=False,
        )

    # Draw Face Boxes
    for bbox, conf, cls_id in zip(
        face_results.xyxy, face_results.confidence, face_results.class_id
    ):
        x1, y1, x2, y2 = map(int, bbox)
        prox = calculate_proximity([x1, y1, x2, y2], cls_id)
        if isinstance(prox, (list, tuple, np.ndarray)):
            prox = float(prox[0])
        annotated = draw_labeled_box(
            annotated,
            bbox,
            f"Face ({conf:.2f}), px: {prox:.2f}",
            COLOR_FACE_BGR,
            font_scale=2.5,
            place_above=True,
        )

    # 1. Frame Number Badge (Top-Right)
    annotated = draw_frame_number_badge(annotated, int(video_frame), margin=30)

    # 2. Classification Rules Card + Separate Rose Audio Badge (Bottom-Right)
    # Estimate clearance needed below the card for the audio badge + spacing
    sample_text = "Audio: KCDS + KCS + OHS"
    (text_w, text_h), baseline = cv2.getTextSize(sample_text, FONT, 2.0, TEXT_THICKNESS)
    audio_badge_h = text_h + baseline + (BADGE_PAD * 2)
    audio_spacing = 16
    bottom_clearance = audio_badge_h + audio_spacing

    annotated, card_bounds = draw_rule_criteria_card_bottom_right(
        annotated, rules, margin=30, bottom_clearance=bottom_clearance
    )
    final_output = draw_audio_badge_below(
        annotated, card_bounds, has_kcds, has_kcs, has_ohs, spacing=audio_spacing, font_scale=2.5
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}_multimodal_annotated.jpg"
    cv2.imwrite(str(out_path), final_output, [cv2.IMWRITE_JPEG_QUALITY, 95])
    logging.info(f"Saved annotated frame to: {out_path}")

    if frame_row is not None:
        row_dict = frame_row.to_dict()
        row_dict["image_file"] = image_path.name
        row_dict["segment_social_state"] = segment_state
        return row_dict
    else:
        return {
            "image_file": image_path.name,
            "video_name": video_name,
            "frame_number": video_frame,
            "segment_social_state": segment_state,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Annotate Frame with Large Text Detections, Clean Card, and Separate Audio Badge"
    )
    parser.add_argument("--image_path", type=str, required=True, help="Path to image file or folder.")
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
    output_p.mkdir(parents=True, exist_ok=True)

    extracted_records = []

    if input_p.is_dir():
        imgs = sorted(
            list(input_p.glob("*.jpg"))
            + list(input_p.glob("*.png"))
            + list(input_p.glob("*.PNG"))
        )
        for img in imgs:
            row_info = process_image(img, output_p, frames_df, segments_df, face_model, person_model)
            if row_info:
                extracted_records.append(row_info)
    else:
        row_info = process_image(input_p, output_p, frames_df, segments_df, face_model, person_model)
        if row_info:
            extracted_records.append(row_info)

    if extracted_records:
        audit_df = pd.DataFrame(extracted_records)
        lead_cols = ["image_file", "segment_social_state", "video_name", "frame_number"]
        existing_leads = [c for c in lead_cols if c in audit_df.columns]
        other_cols = [c for c in audit_df.columns if c not in existing_leads]
        audit_df = audit_df[existing_leads + other_cols]

        csv_out_path = output_p / "selected_frames_audit.csv"
        audit_df.to_csv(csv_out_path, sep=";", index=False)
        logging.info(f"Saved audit CSV to: {csv_out_path}")


if __name__ == "__main__":
    exit(main())