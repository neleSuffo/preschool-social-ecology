import pandas as pd

def find_segments_to_recheck(file_a, file_b, min_length=3, output_filename='/home/nele_pauline_suffo/outputs/segment_evaluation/segments_to_recheck.csv'):
    """
    Identifies non-matching annotation segments of a specified maximum length and aggregates them 
    into start/end seconds.

    Parameters:
    ----------
    file_a : str
        Path to the first annotator's CSV file.
    file_b : str
        Path to the second annotator's CSV file.
    min_length : int
        Maximum length of segments (in seconds) to consider for rechecking.
    output_filename : str
        Path to save the output CSV file.
        
    Returns:
    -------
    pd.DataFrame
        DataFrame containing segments to recheck with columns: video_name, start_second, end_second, annotator a, annotator b.
    """
    # 1. Load the data
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)

    # 2. Merge and Identify Non-Matching Seconds
    df_merged = pd.merge(df_a, df_b, on=['video_name', 'second'], suffixes=('_clara', '_lotta'))
    df_merged['mismatch'] = df_merged['interaction_type_clara'] != df_merged['interaction_type_lotta']
    df_mismatches = df_merged[df_merged['mismatch']].copy()

    # 3. Group Consecutive Mismatches into Segments
    # A new segment_group starts whenever the 'second' column jumps by more than 1
    # within the same 'video_name'.
    df_mismatches['segment_group'] = df_mismatches.groupby('video_name')['second'].transform(
        lambda x: (x.diff().fillna(1) > 1).cumsum()
    )

    # 4. Filter Segments by Length
    segment_sizes = df_mismatches.groupby(['video_name', 'segment_group']).size()
    segments_to_keep_index = segment_sizes[segment_sizes >= min_length].index
    df_final_seconds = df_mismatches.set_index(['video_name', 'segment_group']).loc[segments_to_keep_index].reset_index()

    # 5. Aggregate to Segment View
    agg_functions = {
        'second': ['min', 'max'],
        'interaction_type_clara': lambda x: ', '.join(sorted(x.astype(str).unique())),
        'interaction_type_lotta': lambda x: ', '.join(sorted(x.astype(str).unique())),
    }

    # Group by video_name and the segment_group identifier, then aggregate
    df_segments = df_final_seconds.groupby(['video_name', 'segment_group']).agg(agg_functions).reset_index()

    # Flatten the multi-level column index and rename
    df_segments.columns = ['video_name', 'segment_group', 'start_second', 'end_second', 'interaction_type_clara', 'interaction_type_lotta']

    # Final output DataFrame: video_name, start_second, end_second, annotator a, annotator b
    output_df = df_segments[['video_name', 'start_second', 'end_second', 'interaction_type_clara', 'interaction_type_lotta']]

    # Save the result
    output_df.to_csv(output_filename, index=False)
    
    return output_df

# Example of how to use the function:
# # Assuming your files are named 'annotator_A.csv' and 'annotator_B.csv'
seconds_to_recheck = find_segments_to_recheck('/home/nele_pauline_suffo/outputs/segment_evaluation/gt_ann1_secondwise.csv', '/home/nele_pauline_suffo/outputs/segment_evaluation/gt_ann2_secondwise.csv', min_length=3)
print(seconds_to_recheck.head())