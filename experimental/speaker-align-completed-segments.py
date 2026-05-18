import json
import csv

def load_rttm(rttm_path):
    """Parses RTTM into a list of tuples: (start, end, speaker_id)"""
    diarization = []
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 8:
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                diarization.append((start, start + duration, speaker))
    return diarization

def get_speaker(start, end, diarization):
    """Finds speaker active at the midpoint of the given time range."""
    midpoint = (start + end) / 2
    for s_start, s_end, speaker in diarization:
        if s_start <= midpoint <= s_end:
            return speaker
    return "unknown"

def process_live_json_to_csv(input_file, rttm_file, output_file):
    completed_segments = {}
    diarization_data = load_rttm(rttm_file)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)

                # Only look at lines containing transcription segments
                if "segments" in data:
                    for segment in data["segments"]:
                        # Only grab completed ones
                        if segment.get("completed") is True:
                            start = segment["start"]
                            # Use start time as key to automatically handle duplicates
                            # The latest mention of the same start time will overwrite
                            # (usually they are identical anyway)
                            completed_segments[start] = {
                                "start": start,
                                "end": segment["end"],
                                "text": segment["text"].strip()
                            }
            except json.JSONDecodeError:
                continue # Skip lines that aren't valid JSON (like server ready)

    # Sort segments by start time
    sorted_segments = sorted(completed_segments.values(), key=lambda x: float(x['start']))

    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['start', 'end', 'speaker', 'text'])
        for seg in sorted_segments:
            speaker = get_speaker(float(seg['start']), float(seg['end']), diarization_data)
            writer.writerow([seg['start'], seg['end'], speaker, seg['text']])

# Usage
process_live_json_to_csv('round4_all-segments.txt', 'round4_16k_mono.rttm','round4_final_segments_aligned.csv')
print("CSV generated with unique completed segments and speaker alignment.")