import os
import soundfile as sf
import numpy as np


def count_wav_files(folder_path):
    # Initialize count
    count = 0
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a WAV file
        if (not file_name.startswith('.')) and file_name.lower().endswith('.wav'):
            count += 1
    return count

def audio_segment(input_folder: str,
                  output_folder:str,
                  duration: int):

    os.makedirs(output_folder, exist_ok=True)

    total_songs = count_wav_files(input_folder)
    print(f"{total_songs} songs allocated")
    print("Start Processing...\n")

    count = 0
    for file_name in os.listdir(input_folder):
        if (not file_name.startswith('.')) and file_name.lower().endswith('.wav'):

            print(f"Processing {file_name}")
            file_path = os.path.join(input_folder, file_name)
            # Load the audio
            audio_data, fs = sf.read(file_path)
            info = sf.info(file_path)

            # Convert to mono
            if info.channels != 1:
                audio_data = np.mean(audio_data, axis=1)

            # Calculate the number of 5-second segments
            num_segments = int(len(audio_data) // (fs * duration))

            # Loop through each segment and save it as a new WAV file
            for i in range(num_segments):
                start_time = int(i * fs * duration)
                end_time = int((i + 1) * fs * duration)
                segment = audio_data[start_time:end_time]

                E = 10 * np.log10(np.sum(segment ** 2))
                if E < -30:
                    continue

                max_abs = np.max(np.abs(segment))
                if max_abs == 0:
                    continue

                # normalization
                # segment = segment - np.mean(segment)
                # segment = segment / max_abs

                count += 1
                output_file = os.path.join(output_folder, f"{count}.wav")
                sf.write(output_file, segment, fs)
    print("Done!")
    print(f"A total of {count} audio clips are generated.")

if __name__ == "__main__":
    folder_MedleyDB = "/home/hsun/Datasets/MedleyDB/raw_musics_30"
    output_folder = "/home/hsun/Datasets/MedleyDB/seg_30songs"

    audio_segment(folder_MedleyDB, output_folder, 5)