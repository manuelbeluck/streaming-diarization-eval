from datasets import load_dataset
import soundfile as sf
from pathlib import Path

# ====== CONFIGURATION ======
DATASET_NAME = "eng"  # Language code: "eng", "deu", etc.
SAMPLE_INDEX = 1      # Which sample to extract
OUTPUT_DIR = "callhome_samples"  # Output directory

def create_rttm(timestamps_start, timestamps_end, speakers, file_id="audio"):
    """
    Create RTTM content from timestamps and speaker labels.
    
    RTTM format:
    SPEAKER <file_id> 1 <start_time> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    """
    rttm_lines = []
    
    for start, end, speaker in zip(timestamps_start, timestamps_end, speakers):
        duration = end - start
        # RTTM format: SPEAKER file_id channel start duration <NA> <NA> speaker_id <NA> <NA>
        line = f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        rttm_lines.append(line)
    
    return "\n".join(rttm_lines)


def extract_sample(dataset_name="eng", sample_index=0, output_dir="callhome_samples"):
    """
    Extract a sample from the CallHome dataset and save audio + RTTM to disk.
    
    Configure parameters at the top of the file:
    - DATASET_NAME: Language code (e.g., "eng", "deu")
    - SAMPLE_INDEX: Index of the sample to extract
    - OUTPUT_DIR: Directory to save the files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Loading CallHome {dataset_name.upper()} dataset...")
    ds = load_dataset("talkbank/callhome", dataset_name, split="data")
    
    if sample_index >= len(ds):
        print(f"Error: Sample index {sample_index} out of range. Dataset has {len(ds)} samples.")
        return
    
    sample = ds[sample_index]
    
    # Get audio data
    audio = sample["audio"]
    audio_array = audio["array"]
    sample_rate = audio["sampling_rate"]
    
    # Get metadata
    file_id = sample.get("uid", f"callhome_{dataset_name}_{sample_index:04d}")
    starts = sample["timestamps_start"]
    ends = sample["timestamps_end"]
    speakers = sample["speakers"]
    
    # Save audio file
    audio_filename = f"{file_id}.wav"
    audio_path = output_path / audio_filename
    sf.write(audio_path, audio_array, sample_rate)
    print(f"✓ Saved audio to: {audio_path}")
    
    # Create and save RTTM file
    rttm_content = create_rttm(starts, ends, speakers, file_id=file_id)
    rttm_filename = f"{file_id}.rttm"
    rttm_path = output_path / rttm_filename
    with open(rttm_path, "w") as f:
        f.write(rttm_content)
    print(f"✓ Saved RTTM to: {rttm_path}")
    
    # Print summary
    print(f"\nSample Summary:")
    print(f"  File ID: {file_id}")
    print(f"  Duration: {max(ends):.2f} seconds")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Number of speakers: {len(set(speakers))}")
    print(f"  Number of segments: {len(starts)}")
    print(f"  Speakers: {sorted(set(speakers))}")


if __name__ == "__main__":
    extract_sample(
        dataset_name=DATASET_NAME,
        sample_index=SAMPLE_INDEX,
        output_dir=OUTPUT_DIR
    )
