import os
import random
from pydub import AudioSegment

#### LibriSpeech dataset preprocessing
#### transfer flac files to wav files
def process_audio_folders(input_folder, output_folder, max_duration_seconds=36000):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all `b` folders inside the `a` folder
    b_folders = [os.path.join(input_folder, b) for b in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, b))]

    # Counter for naming the output WAV files
    file_counter = 1
    total_duration = 0  # Total duration in seconds

    # Process each `b` folder
    for b_folder in b_folders:
        # Get all `c` folders inside the current `b` folder
        c_folders = [os.path.join(b_folder, c) for c in os.listdir(b_folder) if os.path.isdir(os.path.join(b_folder, c))]

        # Randomly select one `c` folder
        if not c_folders:
            print(f"No subfolders in {b_folder}, skipping.")
            continue

        selected_c_folder = random.choice(c_folders)
        print(f"Selected folder: {selected_c_folder}")

        # Initialize an empty audio segment
        combined_audio = AudioSegment.silent(duration=0)

        # Read and combine all FLAC files in the selected `c` folder
        for file_name in sorted(os.listdir(selected_c_folder)):
            if file_name.endswith(".flac"):
                file_path = os.path.join(selected_c_folder, file_name)
                print(f"Processing file: {file_path}")
                audio = AudioSegment.from_file(file_path, format="flac")
                combined_audio += audio

        # Check the duration of the combined audio
        combined_duration = len(combined_audio) / 1000  # Duration in seconds
        if total_duration + combined_duration > max_duration_seconds:
            print("Reached the maximum duration limit. Stopping.")
            break

        # Save the combined audio as a WAV file in the output folder
        output_file_name = f"{file_counter}.wav"
        output_file_path = os.path.join(output_folder, output_file_name)
        combined_audio.export(output_file_path, format="wav")
        print(f"Saved combined audio to: {output_file_path}")

        # Update counters
        total_duration += combined_duration
        file_counter += 1

    print(f"Total duration of generated files: {total_duration / 3600:.2f} hours")

input_folder = "/home/hsun/Datasets/MedleyDB/train-clean-100"
output_folder = "/home/hsun/Datasets/MedleyDB/wav_clean"
process_audio_folders(input_folder, output_folder)