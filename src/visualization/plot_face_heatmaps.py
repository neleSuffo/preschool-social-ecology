import argparse
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from config import AnalysisConfig, DataConfig, FaceConfig
from constants import Analysis, DataPaths, Visualization

SAMPLED_FPS = DataConfig.FPS / AnalysisConfig.SAMPLE_RATE  # (30/10 = 3 fps)
EXCLUSION_FRAMES = AnalysisConfig.EXCLUSION_SECONDS * DataConfig.FPS  # 900 frames


def parse_time_str_to_seconds(val) -> float:
    """Converts HH:MM:SS, MM:SS, or numeric values to total seconds."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    parts = val_str.split(":")
    try:
        if len(parts) == 3:  # HH:MM:SS
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:  # MM:SS
            return float(parts[0]) * 60 + float(parts[1])
        return float(val_str)
    except ValueError:
        return np.nan


def filter_initial_setup_frames(
    df: pd.DataFrame,
    exclusion_frames: int = EXCLUSION_FRAMES,
    video_col: str = "video_name",
    frame_col: str = "frame_number",
) -> pd.DataFrame:
    """Excludes the first `exclusion_frames` (30s) ONLY for session-starter videos ending in '01'."""
    df_clean = df.copy()

    is_starter_video = (
        df_clean[video_col]
        .astype(str)
        .str.strip()
        .str.replace(r"\.[^.]+$", "", regex=True)
        .str.contains(r"(?:^|[_/-])01$", regex=True)
    )

    is_excluded_startup = is_starter_video & (
        df_clean[frame_col] < exclusion_frames
    )
    dropped_count = is_excluded_startup.sum()
    starter_vids_count = df_clean.loc[is_starter_video, video_col].nunique()

    print(
        f"Startup Filter: Evaluated {starter_vids_count} starter ('_01') videos. "
        f"Excluded {dropped_count:,} setup face detections (frames 0-{exclusion_frames - 1})."
    )

    return df_clean[~is_excluded_startup].copy()


def load_or_create_matched_full_data(
    segments_csv_path: Path = None,
    cache_path: Path = None,
    #db_path: Path = Path("/home/nele_pauline_suffo/outputs/quantex_inference/inference_first_submission.db"),
    db_path: Path = Path("/home/nele_pauline_suffo/outputs/quantex_inference/inference.db"),
    sample_rate: int = AnalysisConfig.SAMPLE_RATE,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Loads matched face detections from cache, or queries SQLite with book/illustration exclusions,
    applies starter-video setup trimming, joins interaction segments, and caches results.
    """
    if cache_path.exists() and not force_recompute:
        print(f"Loading cached full matched detections from: {cache_path}")
        return pd.read_pickle(cache_path)

    print(f"Reading interaction segments CSV from: {segments_csv_path}")
    try:
        df_segments = pd.read_csv(segments_csv_path, sep=";")
        if len(df_segments.columns) <= 1:
            df_segments = pd.read_csv(segments_csv_path, sep=",")
    except Exception:
        df_segments = pd.read_csv(segments_csv_path)

    # 1. Clean phantom delimiter columns ('Unnamed: ...')
    df_segments = df_segments.loc[:, ~df_segments.columns.str.contains('^Unnamed', na=False)]
    df_segments.columns = df_segments.columns.str.strip()

    # 2. Standardize video column name
    if "video_name" not in df_segments.columns and "video_id" in df_segments.columns:
        if df_segments["video_id"].dtype == object:
            df_segments = df_segments.rename(columns={"video_id": "video_name"})

    df_segments["video_name_clean"] = (
        df_segments["video_name"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.[^.]+$", "", regex=True)
    )

    # 3. Standardize segment_start and segment_end into integer frame numbers
    if "segment_start" not in df_segments.columns or "segment_end" not in df_segments.columns:
        if "start_time_min" in df_segments.columns and "end_time_min" in df_segments.columns:
            start_s = df_segments["start_time_min"].apply(parse_time_str_to_seconds)
            end_s = df_segments["end_time_min"].apply(parse_time_str_to_seconds)
        elif "start_time_sec" in df_segments.columns and "end_time_sec" in df_segments.columns:
            start_s = pd.to_numeric(df_segments["start_time_sec"], errors="coerce")
            end_s = pd.to_numeric(df_segments["end_time_sec"], errors="coerce")
        else:
            raise ValueError(
                f"Cannot determine segment boundaries in {segments_csv_path}. "
                f"Columns found: {list(df_segments.columns)}"
            )

        df_segments["segment_start"] = (start_s * DataConfig.FPS).round().astype("Int64")
        df_segments["segment_end"] = (end_s * DataConfig.FPS).round().astype("Int64")

    # Drop any segments with invalid timestamps/frame indices
    df_segments = df_segments.dropna(subset=["segment_start", "segment_end"]).copy()
    df_segments["segment_start"] = df_segments["segment_start"].astype(int)
    df_segments["segment_end"] = df_segments["segment_end"].astype(int)

    print(f"Connecting to database at: {db_path}")
    conn = sqlite3.connect(db_path)

    # Robust exclusion query guarding against NULL values
    face_query = f"""
        SELECT 
            v.video_name,
            f.detection_id,
            f.frame_number,
            f.x_min,
            f.y_min,
            f.x_max,
            f.y_max,
            f.proximity,
            f.confidence_score
        FROM FaceDetections f
        JOIN Videos v ON f.video_id = v.video_id
        WHERE f.detection_id NOT IN (
            SELECT detection_id 
            FROM PersistentExclusions 
            WHERE type = 'face' AND detection_id IS NOT NULL
        )
        AND f.frame_number % {sample_rate} = 0
    """
    try:
        df_faces = pd.read_sql_query(face_query, conn)
    except sqlite3.OperationalError:
        face_query_fallback = face_query.replace(
            "f.video_id = v.video_id", "f.video_id = v.id"
        )
        df_faces = pd.read_sql_query(face_query_fallback, conn)
    finally:
        conn.close()

    if df_faces.empty:
        print("No face detections found.")
        return pd.DataFrame()

    print(f"Retrieved {len(df_faces):,} valid face detections (books/illustrations excluded).")

    df_faces["video_name_clean"] = (
        df_faces["video_name"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.[^.]+$", "", regex=True)
    )

    # Exclude initial 30s setup frames on session-starter videos (_01)
    df_faces = filter_initial_setup_frames(
        df_faces,
        exclusion_frames=EXCLUSION_FRAMES,
        video_col="video_name_clean",
        frame_col="frame_number",
    )

    df_faces["x_center"] = (df_faces["x_min"] + df_faces["x_max"]) / 2.0
    df_faces["y_center"] = (df_faces["y_min"] + df_faces["y_max"]) / 2.0

    print("Executing in-memory SQL interval join with segments...")
    mem_conn = sqlite3.connect(":memory:")
    df_faces.to_sql("faces", mem_conn, index=False)
    df_segments.to_sql("segments", mem_conn, index=False)

    mem_conn.execute(
        "CREATE INDEX idx_faces ON faces(video_name_clean, frame_number)"
    )
    mem_conn.execute(
        "CREATE INDEX idx_segments ON segments(video_name_clean, segment_start, segment_end)"
    )

    join_query = """
        SELECT 
            f.video_name_clean AS video_name,
            f.detection_id,
            f.frame_number,
            f.x_min,
            f.y_min,
            f.x_max,
            f.y_max,
            f.x_center,
            f.y_center,
            f.proximity,
            f.confidence_score,
            s.interaction_type AS social_state
        FROM faces f
        INNER JOIN segments s 
            ON f.video_name_clean = s.video_name_clean 
            AND f.frame_number >= s.segment_start 
            AND f.frame_number <= s.segment_end
    """
    df_matched = pd.read_sql_query(join_query, mem_conn)
    mem_conn.close()

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
    df_matched["social_state"] = (
        df_matched["social_state"]
        .map(state_map)
        .fillna(df_matched["social_state"].astype(str).str.capitalize())
    )

    print(f"\nTotal detections matched: {len(df_matched):,}")
    print(df_matched["social_state"].value_counts())

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_matched.to_pickle(cache_path)
    print(f"Cached matched detections to: {cache_path}")

    return df_matched


def compute_calibrated_3d_coordinates(
    df: pd.DataFrame,
    d_min: float = FaceConfig.PROXIMITY_MIN_DISTANCE,
    d_max: float = FaceConfig.PROXIMITY_MAX_DISTANCE,
    fov_deg: float = DataConfig.HORIZONTAL_FOV_DEG,
) -> pd.DataFrame:
    """Inverts the logarithmic proximity equation to metric distance and projects
    bounding box centers into lateral (X) and depth (Z) ground-plane coordinates.
    """
    df_ego = df.copy()

    if "x_center" not in df_ego.columns:
        df_ego["x_center"] = (df_ego["x_min"] + df_ego["x_max"]) / 2.0

    # Azimuth angle theta (-70° to +70° across 140° FOV)
    norm_x = (df_ego["x_center"] - (DataConfig.FRAME_WIDTH / 2.0)) / (DataConfig.FRAME_WIDTH / 2.0)
    theta_rad = norm_x * np.radians(fov_deg / 2.0)

    # Exact exponential inversion: d = d_min * (d_max / d_min)**(1 - prox)
    prox_clipped = df_ego["proximity"].clip(0.0, 1.0)
    r_meters = d_min * np.power(d_max / d_min, 1.0 - prox_clipped)

    df_ego["distance_m"] = r_meters
    df_ego["ego_x"] = r_meters * np.sin(theta_rad)
    df_ego["ego_z"] = r_meters * np.cos(theta_rad)

    return df_ego


def plot_egocentric_social_space_3m_horizon(
    df_matched: pd.DataFrame,
    output_path: Path,
    dataset_title: str = "Full Dataset",
    df_stats: pd.DataFrame = None,
    plot_horizon_m: float = FaceConfig.PLOT_HORIZON_METERS,
    d_min: float = FaceConfig.PROXIMITY_MIN_DISTANCE,
    d_max: float = FaceConfig.PROXIMITY_MAX_DISTANCE,
    fov_deg: float = DataConfig.HORIZONTAL_FOV_DEG,
):
    states = ["Interacting", "Available", "Alone"]
    state_anchor_colors = {
        "Interacting": "#691633",  # Deep Burgundy
        "Available": "#8E7C3B",    # Muted Olive / Gold
        "Alone": "#69777B",        # Slate Gray
    }

    state_palettes = {
        "Interacting": LinearSegmentedColormap.from_list(
            "InteractingCmap", ["#ffffff", "#c75d7e", "#691633"]
        ),
        "Available": LinearSegmentedColormap.from_list(
            "AvailableCmap", ["#ffffff", "#cfbc76", "#8E7C3B"]
        ),
        "Alone": LinearSegmentedColormap.from_list(
            "AloneCmap", ["#ffffff", "#9faeb3", "#69777B"]
        ),
    }

    df_ego = compute_calibrated_3d_coordinates(
        df_matched, d_min=d_min, d_max=d_max, fov_deg=fov_deg
    )

    # Increased height to 6.8 to provide ample vertical room and avoid squishing
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 6.8), sharex=True, sharey=True, constrained_layout=True
    )

    half_fov_rad = np.radians(fov_deg / 2.0)
    angles_arc = np.linspace(-half_fov_rad, half_fov_rad, 150)

    cone_edge_x = plot_horizon_m * np.sin(half_fov_rad)
    cone_edge_z = plot_horizon_m * np.cos(half_fov_rad)

    for ax, state in zip(axes, states):
        sub = df_ego[df_ego["social_state"] == state]
        n_total = len(sub)

        in_horizon = sub[sub["distance_m"] <= plot_horizon_m]
        beyond_horizon = sub[sub["distance_m"] > plot_horizon_m]
        n_beyond = len(beyond_horizon)
        pct_beyond = (n_beyond / n_total * 100.0) if n_total > 0 else 0.0

        ax.set_facecolor("#fcfcfc")

        # 1. 2D KDE Contours
        if len(in_horizon) > 20:
            sample_size = min(35000, len(in_horizon))
            sample = in_horizon.sample(n=sample_size, random_state=42)

            sns.kdeplot(
                data=sample,
                x="ego_x",
                y="ego_z",
                ax=ax,
                cmap=state_palettes[state],
                fill=True,
                thresh=0.025,
                levels=14,
                alpha=0.95,
            )

        # 2. Camera Field-of-View Cone Boundaries (140°)
        ax.plot([0, -cone_edge_x], [0, cone_edge_z], color="#444444", linestyle="--", linewidth=1.1, alpha=0.6)
        ax.plot([0, cone_edge_x], [0, cone_edge_z], color="#444444", linestyle="--", linewidth=1.1, alpha=0.6)

        # 3. Softened Radial Distance Arcs up to 3m
        dist_rings = [
            (0.5, "0.5m"),
            (1.5, "1.5m"),
            (3.0, "3.0m"),
        ]
        for r, label in dist_rings:
            arc_x = r * np.sin(angles_arc)
            arc_z = r * np.cos(angles_arc)

            ax.plot(arc_x, arc_z, color="white", linestyle="-", linewidth=2.0, alpha=0.65, zorder=4)
            ax.plot(arc_x, arc_z, color="#555555", linestyle="-" if r == plot_horizon_m else "--", linewidth=1.0, alpha=0.55, zorder=5)

            ax.text(
                0.06,
                r - 0.09,
                label,
                color="#333333",
                fontsize=13,
                fontweight="bold",
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white",
                    ec="none",
                    alpha=0.8,
                ),
            )

        # 4. Child Origin Marker
        ax.plot(
            0,
            0,
            marker="o",
            markersize=11,
            color="#111111",
            markeredgecolor="white",
            markeredgewidth=1.8,
            zorder=10,
        )
        ax.annotate(
            "CHILD",
            xy=(0, -0.12),
            xycoords="data",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="#111111",
        )

        # 5. Distant Faces Reporting Box
        annotation_text = f">3.0m:\nn = {n_beyond:,}\n({pct_beyond:.1f}%)"
        ax.text(
            0.96,
            0.95,
            annotation_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=13,
            fontweight="semibold",
            color="#333333",
            bbox=dict(
                boxstyle="round,pad=0.35",
                fc="white",
                ec="#cccccc",
                alpha=0.9,
                linewidth=0.8,
            ),
        )

        # 6. Two-line subtitle below state header
        mean_dist = sub["distance_m"].mean() if n_total > 0 else 0.0
        mean_prox = sub["proximity"].mean() if n_total > 0 else 0.0

        fdfr_str = " | FDFR: 0.0%"
        if df_stats is not None and not df_stats.empty:
            stat_row = df_stats[df_stats["Social State"] == state]
            if not stat_row.empty:
                fdfr_val = stat_row["Face Presence Rate (%)"].values[0]
                fdfr_str = f" | FDFR: {fdfr_val:.1f}%"

        ax.set_title(
            f"{state}\n"
            f"n = {n_total:,}{fdfr_str}\n"
            rf"$\mathit{{M}}_{{\mathrm{{distance}}}}$ = {mean_dist:.2f}m | $\mathit{{M}}_{{\mathrm{{proximity}}}}$ = {mean_prox:.2f}",
            fontsize=16,
            fontweight="bold",
            color=state_anchor_colors[state],
            pad=10,
        )

        ax.tick_params(axis="both", which="major", labelsize=10.5)
        ax.set_xlim(-cone_edge_x - 0.25, cone_edge_x + 0.25)
        ax.set_ylim(-0.38, plot_horizon_m + 0.2)
        ax.set_aspect("equal")
        ax.set_xlabel("Horizontal Position (Left ← 0 → Right) [m]", fontsize=14, labelpad=6)

    axes[0].set_ylabel("Distance in Front of Child [m]", fontsize=14, labelpad=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")


def extract_alone_close_face_events(
    df_matched: pd.DataFrame,
    output_csv_path: Path,
    proximity_thresh: float = 0.60,
    fps: int = DataConfig.FPS,
) -> pd.DataFrame:
    """Exports both a per-frame detection audit and an aggregated continuous event summary
    for inspection of high-proximity faces in the 'Alone' state.
    """
    mask = (df_matched["social_state"].str.capitalize() == "Alone") & (
        df_matched["proximity"] >= proximity_thresh
    )
    df_close_alone = df_matched[mask].copy()

    if df_close_alone.empty:
        print(f"No detections found matching social_state == 'Alone' and proximity >= {proximity_thresh}")
        return df_close_alone

    df_close_alone["time_sec"] = (df_close_alone["frame_number"] / fps).round(2)
    df_close_alone["timestamp_hhmmss"] = pd.to_datetime(
        df_close_alone["time_sec"], unit="s"
    ).dt.strftime("%H:%M:%S")

    df_close_alone = df_close_alone.sort_values(
        by=["video_name", "frame_number"]
    ).reset_index(drop=True)

    cols_to_keep = [
        "video_name",
        "frame_number",
        "timestamp_hhmmss",
        "time_sec",
        "proximity",
        "confidence_score",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ]
    df_frames = df_close_alone[[c for c in cols_to_keep if c in df_close_alone.columns]]

    events = []
    for vid, group in df_frames.groupby("video_name"):
        group = group.copy()
        time_diff = group["time_sec"].diff()
        group["event_id"] = (time_diff > 2.0).cumsum()

        for eid, event_df in group.groupby("event_id"):
            events.append({
                "video_name": vid,
                "start_time": event_df["timestamp_hhmmss"].iloc[0],
                "end_time": event_df["timestamp_hhmmss"].iloc[-1],
                "start_frame": event_df["frame_number"].iloc[0],
                "end_frame": event_df["frame_number"].iloc[-1],
                "n_detections": len(event_df),
                "max_proximity": event_df["proximity"].max(),
                "mean_confidence": event_df["confidence_score"].mean(),
            })

    df_events = pd.DataFrame(events)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    events_csv = output_csv_path.with_name(output_csv_path.stem + "_events_summary.csv")

    df_events.to_csv(events_csv, index=False, sep=";")

    print(f"\nFound {len(df_frames):,} frames across {len(df_events)} distinct events in 'Alone'.")
    print(f"Event summary saved to:     {events_csv}")

    return df_frames


def compute_segment_and_detection_statistics(
    segments_csv_path: Path,
    df_matched: pd.DataFrame,
    output_summary_csv: Path,
    sampled_fps: float = SAMPLED_FPS,
    fps_video: int = DataConfig.FPS,
    exclusion_seconds: int = AnalysisConfig.EXCLUSION_SECONDS,
) -> pd.DataFrame:
    """Computes duration, sampled frame capacity, and presence statistics per social state,
    incorporating the initial 30-second exclusion for starter videos ending in '01'.
    """
    print(f"Reading segments from: {segments_csv_path}")
    try:
        df_seg = pd.read_csv(segments_csv_path, sep=";")
        if len(df_seg.columns) <= 1:
            df_seg = pd.read_csv(segments_csv_path, sep=",")
    except Exception:
        df_seg = pd.read_csv(segments_csv_path)

    # Standardize column naming
    if "video_name" not in df_seg.columns and "video_id" in df_seg.columns:
        if df_seg["video_id"].dtype == object:
            df_seg = df_seg.rename(columns={"video_id": "video_name"})

    df_seg["video_name_clean"] = (
        df_seg["video_name"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.[^.]+$", "", regex=True)
    )

    # Standardize social state column
    state_col = "interaction_type" if "interaction_type" in df_seg.columns else "social_state"
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
    df_seg["social_state"] = (
        df_seg[state_col]
        .map(state_map)
        .fillna(df_seg[state_col].astype(str).str.capitalize())
    )

    # Determine segment start & end times in seconds
    if "start_time_min" in df_seg.columns and "end_time_min" in df_seg.columns:
        df_seg["start_s"] = df_seg["start_time_min"].apply(parse_time_str_to_seconds)
        df_seg["end_s"] = df_seg["end_time_min"].apply(parse_time_str_to_seconds)
    elif "segment_start" in df_seg.columns and "segment_end" in df_seg.columns:
        df_seg["start_s"] = df_seg["segment_start"] / fps_video
        df_seg["end_s"] = df_seg["segment_end"] / fps_video
    elif "start_time_sec" in df_seg.columns and "end_time_sec" in df_seg.columns:
        df_seg["start_s"] = pd.to_numeric(df_seg["start_time_sec"], errors="coerce")
        df_seg["end_s"] = pd.to_numeric(df_seg["end_time_sec"], errors="coerce")
    elif "duration_sec" in df_seg.columns:
        df_seg["start_s"] = 0.0
        df_seg["end_s"] = pd.to_numeric(df_seg["duration_sec"], errors="coerce")
    else:
        raise ValueError(
            f"Segments CSV missing recognizable time columns. Columns found: {list(df_seg.columns)}"
        )

    # Drop any rows where timestamp conversion failed
    df_seg = df_seg.dropna(subset=["start_s", "end_s"])

    # Account for 30s exclusion on _01 starter videos in segment duration
    is_starter = df_seg["video_name_clean"].str.contains(r"(?:^|[_/-])01$", regex=True)

    # Adjust start times for starter videos
    adj_start = df_seg["start_s"].copy()
    adj_start[is_starter] = adj_start[is_starter].clip(lower=exclusion_seconds)

    # Calculate effective segment duration (discarding intervals that fell entirely in the first 30s)
    df_seg["effective_duration_s"] = (df_seg["end_s"] - adj_start).clip(lower=0.0)

    # Calculate statistics per state
    state_summary = []
    states = ["Interacting", "Available", "Alone"]

    for state in states:
        sub_seg = df_seg[df_seg["social_state"] == state]
        total_seconds = sub_seg["effective_duration_s"].sum()
        total_hours = total_seconds / 3600.0
        total_minutes = total_seconds / 60.0

        # Theoretical sampled frames evaluated across this duration
        total_sampled_frames = int(round(total_seconds * sampled_fps))

        # Face detections in this state
        sub_faces = df_matched[df_matched["social_state"] == state]
        total_face_dets = len(sub_faces)

        # Sampled frames containing >= 1 face
        if not sub_faces.empty and "video_name" in sub_faces.columns:
            frames_with_faces = (
                sub_faces.groupby(["video_name", "frame_number"])
                .size()
                .shape[0]
            )
        else:
            frames_with_faces = 0

        face_presence_rate_pct = (
            (frames_with_faces / total_sampled_frames * 100.0)
            if total_sampled_frames > 0
            else 0.0
        )
        dets_per_sampled_frame = (
            (total_face_dets / total_sampled_frames)
            if total_sampled_frames > 0
            else 0.0
        )
        dets_per_hour = (
            (total_face_dets / total_hours)
            if total_hours > 0
            else 0.0
        )

        state_summary.append({
            "Social State": state,
            "Total Segments": len(sub_seg),
            "Total Duration (Hours)": round(total_hours, 2),
            "Total Duration (Minutes)": round(total_minutes, 1),
            "Total Sampled Frames": total_sampled_frames,
            "Total Face Detections": total_face_dets,
            "Frames with ≥1 Face": frames_with_faces,
            "Face Presence Rate (%)": round(face_presence_rate_pct, 2),
            "Faces / Sampled Frame": round(dets_per_sampled_frame, 3),
            "Faces / Hour": round(dets_per_hour, 1),
        })

    df_stats = pd.DataFrame(state_summary)

    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(output_summary_csv, index=False, sep=";")
    print(f"\nSaved state statistics to: {output_summary_csv}")
    return df_stats


def process_single_dataset(dataset_mode: str, output_dir: Path, force_recompute: bool = False):
    """Loads, filters, and computes statistics for a given dataset ('gt' or 'full')."""
    if dataset_mode == "gt":
        segments_csv = Analysis.GROUND_TRUTH_SEGMENTS_CSV
        cache_path = Visualization.GT_MATCHED_FACES_DETECTIONS
        dataset_label = "Ground Truth"
        prefix = "gt_"
    else:
        segments_csv = Analysis.INTERACTION_SEGMENTS_CSV
        if not segments_csv.exists():
            segments_csv = output_dir / "interaction_segments.csv"
        cache_path = Visualization.FULL_MATCHED_FACES_DETECTIONS
        dataset_label = "Full Dataset"
        prefix = "full_"

    output_stats_csv = output_dir / f"{Visualization.HEATMAP_SUBFOLDER}/{prefix}face_heatmap_stats.csv"
    output_figure = output_dir / f"{Visualization.HEATMAP_SUBFOLDER}/{prefix}face_heatmap.png"

    if cache_path.exists() and not force_recompute:
        print(f"Loading cached face detections [{dataset_label}] from: {cache_path}")
        df_data = pd.read_pickle(cache_path)
    else:
        print(f"Recomputing matched face data for [{dataset_label}]...")
        df_data = load_or_create_matched_full_data(
            segments_csv_path=segments_csv,
            cache_path=cache_path,
            force_recompute=True,
        )

    if "video_name" not in df_data.columns and "video_name_clean" in df_data.columns:
        df_data["video_name"] = df_data["video_name_clean"]

    df_filtered = filter_initial_setup_frames(
        df_data,
        exclusion_frames=EXCLUSION_FRAMES,
        video_col="video_name",
        frame_col="frame_number",
    )

    df_stats = compute_segment_and_detection_statistics(
        segments_csv_path=segments_csv,
        df_matched=df_filtered,
        output_summary_csv=output_stats_csv,
        sampled_fps=SAMPLED_FPS,
        fps_video=DataConfig.FPS,
        exclusion_seconds=AnalysisConfig.EXCLUSION_SECONDS,
    )

    return df_filtered, df_stats, output_figure, dataset_label


def run_pipeline(
    dataset_mode: str,
    output_dir: Path,
    force_recompute: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    df_filtered, df_stats, output_figure, dataset_label = process_single_dataset(
        dataset_mode, output_dir, force_recompute
    )

    plot_egocentric_social_space_3m_horizon(
        df_matched=df_filtered,
        output_path=output_figure,
        dataset_title=dataset_label,
        df_stats=df_stats,
        plot_horizon_m=FaceConfig.PLOT_HORIZON_METERS,
        d_min=FaceConfig.PROXIMITY_MIN_DISTANCE,
        d_max=FaceConfig.PROXIMITY_MAX_DISTANCE,
        fov_deg=DataConfig.HORIZONTAL_FOV_DEG,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrated Egocentric Social Space Analysis")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gt", "full"],
        default="gt",
        help="Select dataset mode: 'gt' (ground truth) or 'full' (full inference dataset). Default is 'gt'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=Analysis.FINAL_OUTPUT_FOLDER,
        help="Path to output directory containing caches and storing figures/reports.",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force re-querying SQLite and rebuilding cache rather than reading existing pickle files.",
    )
    args = parser.parse_args()

    run_pipeline(
        dataset_mode=args.dataset,
        output_dir=Path(args.output_dir),
        force_recompute=args.force_recompute,
    )