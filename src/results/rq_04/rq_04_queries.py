import re
import pandas as pd
from pathlib import Path
import sys

# Get the src directory (2 levels up from current notebook location)
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import ResearchQuestions

def extract_child_id(video_name):
    match = re.search(r'id(\d{6})', video_name)
    return match.group(1) if match else None

def extract_presence_counts(csv_file: Path=ResearchQuestions.FRAME_LEVEL_INTERACTIONS_CSV):
    print("=" * 60)
    print("RQ 04: EXTRACTING PRESENCE COUNTS")
    print("=" * 60)
    
    # Load data
    print(f"📁 Loading data from: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {csv_file}")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    # Extract child_id from video_name and add as new column
    df['child_id'] = df['video_name'].apply(extract_child_id)
    
    # Check extraction success
    valid_child_ids = df['child_id'].notna().sum()
    total_frames = len(df)
    
    if valid_child_ids == 0:
        print("❌ Warning: No child IDs could be extracted. Check video name format.")
        return None
    
    unique_children = df['child_id'].nunique()
    
    # Group by child (video_id or video_name) and age_at_recording
    print("📊 Grouping data by child and age...")
    grouped = (
        df.groupby(['child_id', 'age_at_recording'])
        .agg(
            frames_with_adult_present=('adult_present', 'sum'),
            frames_with_child_present=('child_present', 'sum'),
            total_frames=('frame_number', 'count')
        )
        .reset_index()
    )
        
    # Create the pivoted structure
    pivot_data = []
    
    for _, row in grouped.iterrows():
        child_id = row['child_id']
        age_at_recording = row['age_at_recording']
        total_frames = row['total_frames']
        
        # Add Adult row
        pivot_data.append({
            'child_id': child_id,
            'age_at_recording': age_at_recording,
            'total_frames': total_frames,
            'category': 'Adult',
            'frames_present': row['frames_with_adult_present']
        })
        
        # Add Child row
        pivot_data.append({
            'child_id': child_id,
            'age_at_recording': age_at_recording,
            'total_frames': total_frames,
            'category': 'Child',
            'frames_present': row['frames_with_child_present']
        })
    
    # Convert to DataFrame
    results = pd.DataFrame(pivot_data)
    print(f"✅ Created pivoted dataset with {len(results)} rows")
    
    # Save results
    output_file = ResearchQuestions.PRESENCE_COUNTS_CSV 
    try:
        results.to_csv(output_file, index=False)
        print(f"💾 Successfully saved results to: {output_file}")        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return None
    
    return results

if __name__ == "__main__":
    # Run the analysis when script is executed directly
    results = extract_presence_counts()
    
    if results is not None:
        print(f"\n🎉 Analysis completed!")
    else:
        print("\n❌ Analysis failed. Check error messages above.")