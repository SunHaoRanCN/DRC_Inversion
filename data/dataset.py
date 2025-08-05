import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base_dir))

from models.DECOMP import compressor
import os
import soundfile as sf
import numpy as np
import random
import pickle
import shutil
import librosa


def add_noise(audio_signal, snr=20):
    # Calculate signal power
    signal_power = np.mean(audio_signal ** 2)

    # Convert SNR from dB to linear scale
    # SNR = 20 dB means signal power is 100 times noise power
    snr_linear = 10 ** (snr / 10)

    # Calculate required noise power
    noise_power = signal_power / snr_linear

    # Generate Gaussian noise with calculated power
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio_signal))

    # Add noise to signal
    noisy_signal = audio_signal + noise

    max_val = max(abs(np.max(noisy_signal)), abs(np.min(noisy_signal)))
    if max_val > 1.0:
        noisy_signal = noisy_signal / max_val

    return noisy_signal


def count_files_in_classes(folder):
    # Dictionary to store the count of files by class
    class_count = {}

    # Get all files in the folder
    wav_files = [f for f in os.listdir(folder) if f.endswith('.wav')]

    # Group and count files by class
    for wav_file in wav_files:
        class_name = wav_file.split('_')[-1].replace('.wav', '')
        if class_name not in class_count:
            class_count[class_name] = 0
        class_count[class_name] += 1

    return class_count


def print_class_distribution(class_count, folder_name):
    print(f"Class distribution in {folder_name}:")
    for class_name, count in sorted(class_count.items()):
        print(f"Class {class_name}: {count} files")
    print(f"Total classes: {len(class_count)}")
    print(f"Total files: {sum(class_count.values())}\n")


def generate_dataset(
        source_folder,
        root_folder,
        compressors,
        target_number=None,
        duration=5,
        audio_resample=False,
        target_fs=None,
        split_ratio=0.8
):
    count = 0
    total_num = 0

    all_folder = os.path.join(root_folder, "all")
    os.makedirs(all_folder, exist_ok=True)

    for file_name in os.listdir(source_folder):
        if total_num >= target_number:
            break

        if file_name.endswith('.wav'):
            count += 1
            file_path = os.path.join(source_folder, file_name)
            audio_data, fs = sf.read(file_path)
            info = sf.info(file_path)

            # Convert stereo to mono if needed
            if info.channels != 1:
                audio_data = np.mean(audio_data, axis=1)

            # Resample if required
            if audio_resample:
                audio_data = librosa.resample(audio_data, orig_sr=fs, target_sr=target_fs)
                fs = target_fs

            # Calculate the number of segments
            num_segments = int(len(audio_data) // (fs * duration))

            segment_count = 0
            for i in range(num_segments):
                start_time = int(i * fs * duration)
                end_time = int((i + 1) * fs * duration)
                segment = audio_data[start_time:end_time]

                # Skip silent segments
                if np.all(segment == 0):
                    continue

                # Skip low energy segments
                E = 10 * np.log10(np.sum(segment ** 2))
                if E < -30:
                    continue

                # Normalize segment
                segment = segment - np.mean(segment)
                segment = segment / np.max(np.abs(segment))

                segment_count += 1
                output_path = os.path.join(all_folder, f"{count}_{segment_count}_0.wav")
                sf.write(output_path, segment, fs)

                total_num += 1
                if total_num >= target_number:
                    break

                # Apply compressors
                for label, parameters in compressors.items():
                    # print(count, segment_count, label)
                    y = compressor(
                        segment,
                        fs,
                        parameters.get('param1'),
                        parameters.get('param2'),
                        parameters.get('param3'),
                        parameters.get('param4'),
                        parameters.get('param5'),
                        parameters.get('param6'),
                        parameters.get('param7')
                    )
                    output_path = os.path.join(all_folder, f"{count}_{segment_count}_{label}.wav")
                    sf.write(output_path, y, fs)

                    total_num += 1
                    if total_num >= target_number:
                        break

    train_folder = os.path.join(root_folder, "train")
    test_folder = os.path.join(root_folder, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get all files with the specified extension
    all_files = [f for f in os.listdir(all_folder) if f.endswith(".wav")]

    if not all_files:
        return {"error": "No wav files found !"}

    # Dictionary to store files by class
    files_by_class = {}

    for wav_file in all_files:
        class_name = wav_file.split('_')[-1].replace('.wav', '')
        if class_name not in files_by_class:
            files_by_class[class_name] = []
        files_by_class[class_name].append(wav_file)

    # Split files into train and test sets for each class
    for class_name, files in files_by_class.items():
        random.shuffle(files)
        split_index = int(len(files) * split_ratio)

        train_files = files[:split_index]
        test_files = files[split_index:]

        for file in train_files:
            source_path = os.path.join(all_folder, file)
            dest_path = os.path.join(train_folder, file)
            shutil.copy(source_path, dest_path)

        # Copy/move test files
        for file in test_files:
            source_path = os.path.join(all_folder, file)
            dest_path = os.path.join(test_folder, file)
            shutil.copy(source_path, dest_path)

    train_class_count = count_files_in_classes(train_folder)
    test_class_count = count_files_in_classes(test_folder)

    return train_class_count, test_class_count


def generate_multi_dataset(
        source_folder,
        root_folder,  # path to "all", "train, "test"
        compressors,
        target_number=None,
        duration=5,
        data_augmentation=True,
        snr=20,
        audio_resample=False,
        target_fs=16000,
        split_ratio=0.8
):
    count = 0
    total_num = 0

    all_folder = os.path.join(root_folder, "all")
    os.makedirs(all_folder, exist_ok=True)

    for file_name in os.listdir(source_folder):
        if total_num >= target_number:
            break

        if file_name.endswith('.wav'):
            # count += 1
            file_path = os.path.join(source_folder, file_name)
            audio_data, fs = sf.read(file_path)
            info = sf.info(file_path)

            # Convert stereo to mono if needed
            if info.channels != 1:
                audio_data = np.mean(audio_data, axis=1)

            # Resample if required
            if audio_resample:
                audio_data = librosa.resample(audio_data, orig_sr=fs, target_sr=target_fs)
                fs = target_fs

            # Apply noise augmentation if enabled
            if data_augmentation:
                noised_audio = add_noise(audio_data, snr)

            # Calculate the number of segments
            num_segments = int(len(audio_data) // (fs * duration))

            # segment_count = 0
            for i in range(num_segments):

                start_time = int(i * fs * duration)
                end_time = int((i + 1) * fs * duration)
                segment = audio_data[start_time:end_time]

                # Skip silent segments
                if np.all(segment == 0):
                    continue

                # Skip low energy segments
                E = 10 * np.log10(np.sum(segment ** 2))
                if E < -30:
                    continue

                # Normalize segment
                segment = segment - np.mean(segment)
                segment = segment / np.max(np.abs(segment))

                # segment_count += 1
                count += 1
                output_path = os.path.join(all_folder, f"{count}_0.wav")
                sf.write(output_path, segment, fs)
                total_num += 1

                # Apply compressors
                for label, parameters in compressors.items():
                    # print(count, segment_count, label)
                    y = compressor(
                        segment,
                        fs,
                        parameters.get('param1'),
                        parameters.get('param2'),
                        parameters.get('param3'),
                        parameters.get('param4'),
                        parameters.get('param5'),
                        parameters.get('param6'),
                        parameters.get('param7')
                    )
                    output_path = os.path.join(all_folder, f"{count}_{label}.wav")
                    sf.write(output_path, y, fs)

                    total_num += 1

                # Process noised segment if data augmentation is enabled
                if data_augmentation:
                    noised_segment = noised_audio[start_time:end_time]

                    # Normalize noised segment
                    noised_segment = noised_segment - np.mean(noised_segment)
                    noised_segment = noised_segment / np.max(np.abs(noised_segment))

                    # segment_count += 1
                    output_path = os.path.join(all_folder, f"{count}_n_0.wav")
                    sf.write(output_path, noised_segment, fs)
                    total_num += 1

                    # Apply compressors to noised segment
                    for label, parameters in compressors.items():
                        # print(count, segment_count, label)
                        y = compressor(
                            noised_segment,
                            fs,
                            parameters.get('param1'),
                            parameters.get('param2'),
                            parameters.get('param3'),
                            parameters.get('param4'),
                            parameters.get('param5'),
                            parameters.get('param6'),
                            parameters.get('param7')
                        )
                        output_path = os.path.join(all_folder, f"{count}_n_{label}.wav")
                        sf.write(output_path, y, fs)

                        total_num += 1
                        if total_num >= target_number:
                            break

    train_folder = os.path.join(root_folder, "train")
    test_folder = os.path.join(root_folder, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get all files with the specified extension
    all_files = [f for f in os.listdir(all_folder) if f.endswith(".wav")]

    if not all_files:
        return {"error": "No wav files found !"}

    # Dictionary to store files by class
    files_by_class = {}

    for wav_file in all_files:
        class_name = wav_file.split('_')[-1].replace('.wav', '')
        if class_name not in files_by_class:
            files_by_class[class_name] = []
        files_by_class[class_name].append(wav_file)

    # Split files into train and test sets for each class
    for class_name, files in files_by_class.items():
        random.shuffle(files)
        split_index = int(len(files) * split_ratio)

        train_files = files[:split_index]
        test_files = files[split_index:]

        for file in train_files:
            source_path = os.path.join(all_folder, file)
            dest_path = os.path.join(train_folder, file)
            shutil.copy(source_path, dest_path)

        # Copy/move test files
        for file in test_files:
            source_path = os.path.join(all_folder, file)
            dest_path = os.path.join(test_folder, file)
            shutil.copy(source_path, dest_path)

    train_class_count = count_files_in_classes(train_folder)
    test_class_count = count_files_in_classes(test_folder)

    return train_class_count, test_class_count


source_folder = "/Volumes/hsun_exFAT/Datasets/MedleyDB/raw_musics_30"
root_path = "/Volumes/hsun_exFAT/Datasets/MedleyDB/30profiles"

with open('/Users/sunhaoran/Projects/TASLP/30profiles.pkl', 'rb') as file:
    compressors = pickle.load(file)
duration = 5
SNR = np.arange(20, 70, 5)
# SNR = np.array([20])
for snr in SNR:
    root_folder = os.path.join(root_path, str(snr) + "dB")
    os.makedirs(root_folder, exist_ok=True)
    print(r"Generating dataset with SNR {snr} dB.")
    train_class_count, test_class_count = generate_multi_dataset(source_folder,
                                                                 root_folder,
                                                                 compressors,
                                                                 target_number=int(35867*2),
                                                                 duration=5,
                                                                 data_augmentation=True,
                                                                 snr=snr,
                                                                 split_ratio=0.8
                                                                 )
    print_class_distribution(train_class_count, "train folder")
    print_class_distribution(test_class_count, "test folder")
