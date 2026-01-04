import pandas as pd
import numpy as np
from pathlib import Path
from constants import DataPaths, Inference, AudioClassification
from config import InferenceConfig

def parse_rttm(file_path: Path = AudioClassification.VTC_RTTM_FILE) -> pd.DataFrame:
    """Parses RTTM file into a DataFrame."""
    data = []
    if not file_path.exists():
        return pd.DataFrame()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                data.append({
                    'video_name': parts[1],
                    'start_time_seconds': float(parts[3]),
                    'duration': float(parts[4]),
                    'end_time_seconds': float(parts[3]) + float(parts[4]),
                    'speaker': parts[7]
                })
    return pd.DataFrame(data)

def process_block(block):
    """
    Categorizes the block into one of four types:
    1. Successful Initiation (Child starts, turn occurs)
    2. Successful Response (Adult starts, turn occurs)
    3. Unanswered Child Bid (Child starts, no turn)
    4. Unanswered Adult Prompt (Adult starts, no turn)
    """
    if not block:
        return 0, 0, 0, 0, 0
    
    turns = 0
    for i in range(1, len(block)):
        if block[i]['speaker'] != block[i-1]['speaker']:
            turns += 1
            
    # Initialize all counters
    succ_init, succ_resp, fail_init, fail_resp = 0, 0, 0, 0
    first_speaker = block[0]['speaker']
    
    if turns > 0:
        if first_speaker == 'KCHI':
            succ_init = 1  # Successful exchange led by child
        else:
            succ_resp = 1  # Successful exchange led by adult
    else:
        if first_speaker == 'KCHI':
            fail_init = 1  # Child spoke, but no one replied
        else:
            fail_resp = 1  # Adult spoke, but child didn't reply
            
    return turns, succ_init, succ_resp, fail_init, fail_resp

def count_directional_turns(vocalizations, segments_df):
    results = []
    MAX_TURN_GAP = InferenceConfig.MAX_TURN_TAKING_GAP_SEC
    MAX_SAME_GAP = InferenceConfig.MAX_SAME_SPEAKER_GAP_SEC

    for _, seg in segments_df.iterrows():
        # Only analyze 'Interacting' segments to maintain focused context
        if seg['interaction_type'] != 'Interacting':
            continue
            
        # Overlap Filter to capture all relevant speech within segment bounds
        seg_vocs = vocalizations[
            (vocalizations['video_name'] == seg['video_name']) & 
            (vocalizations['start_time_seconds'] < seg['end_time_sec']) & 
            (vocalizations['end_time_seconds'] > seg['start_time_sec'])
        ].copy().sort_values('start_time_seconds').reset_index(drop=True)
        
        # Filter for the Dyad (KCHI and KCDS)
        seg_vocs = seg_vocs[seg_vocs['speaker'].isin(['KCHI', 'KCDS'])].reset_index(drop=True)

        total_turns = 0
        s_init, s_resp, f_init, f_resp = 0, 0, 0, 0
        
        if not seg_vocs.empty:
            current_block = [seg_vocs.iloc[0]]
            for i in range(1, len(seg_vocs)):
                prev = seg_vocs.iloc[i-1]
                curr = seg_vocs.iloc[i]
                gap = curr['start_time_seconds'] - prev['end_time_seconds']
                
                threshold = MAX_SAME_GAP if curr['speaker'] == prev['speaker'] else MAX_TURN_GAP
                
                if gap <= threshold:
                    current_block.append(curr)
                else:
                    t, si, sr, fi, fr = process_block(current_block)
                    total_turns += t
                    s_init += si; s_resp += sr; f_init += fi; f_resp += fr
                    current_block = [curr]
            
            # Process final block
            t, si, sr, fi, fr = process_block(current_block)
            total_turns += t
            s_init += si; s_resp += sr; f_init += fi; f_resp += fr

        results.append({
            'child_id': seg['child_id'],
            'video_name': seg['video_name'],
            'age_at_recording': seg['age_at_recording'],
            'segment_start': seg['start_time_sec'],
            'segment_end': seg['end_time_sec'],
            'duration_sec': seg['duration_sec'],
            'total_turns': total_turns,
            'successful_initiations': s_init,
            'successful_responses': s_resp,
            'unanswered_child_bids': f_init,
            'unanswered_adult_prompts': f_resp,
            'segment_duration_minutes': seg['duration_sec'] / 60
        })
        
    return pd.DataFrame(results)

def main():
    print("🗣️ RESEARCH QUESTION 3: MULTIDIMENSIONAL SOCIAL DYNAMICS")
    print("=" * 70)
    
    # 1. Load Segments and Vocalizations
    segments_df = pd.read_csv(Inference.INTERACTION_SEGMENTS_CSV)
    all_vocalizations = parse_rttm()
    
    # 2. Categorize Social Blocks
    final_df = count_directional_turns(all_vocalizations, segments_df)
    
    # 3. Calculate Global Totals and Proportions
    final_df['total_attempts'] = (
        final_df['successful_initiations'] + final_df['successful_responses'] +
        final_df['unanswered_child_bids'] + final_df['unanswered_adult_prompts']
    )
    
    # Calculate percentages of the total social landscape
    for col in ['successful_initiations', 'successful_responses', 'unanswered_child_bids', 'unanswered_adult_prompts']:
        final_df[f'pct_{col}'] = (final_df[col] / final_df['total_attempts']).fillna(0)
    
    # Standard volume metric
    final_df['turns_per_minute'] = (final_df['total_turns'] / final_df['segment_duration_minutes']).fillna(0)
    
    # 4. Metadata Cleanup and Sort
    final_df['age_at_recording'] = (
        final_df['age_at_recording'].astype(str).str.replace('"', '').str.replace(',', '.').str.strip()
    )
    final_output = final_df.sort_values(['video_name', 'segment_start'])
    
    # 5. Save Results
    final_output.to_csv(Inference.TURN_TAKING_CSV, index=False)
    print(f"✅ Full four-category analysis saved to {Inference.TURN_TAKING_CSV}")

if __name__ == "__main__":
    main()