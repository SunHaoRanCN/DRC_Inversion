import os
import numpy as np
import soundfile as sf
import time
import sys
import wave
import warnings
import gc
import psutil
import tracemalloc
from pydub import AudioSegment
from pydub.utils import mediainfo


def is_valid_wav(file_path):
    """检查文件是否为有效的WAV文件"""
    try:
        with wave.open(file_path, 'rb') as wf:
            # 简单的有效性检查
            if wf.getnchannels() == 0 or wf.getframerate() == 0:
                return False
        return True
    except:
        return False


def clear_memory():
    """显式清理内存"""
    # 删除大对象
    if 'all_data' in globals():
        del globals()['all_data']
    if 'combined' in globals():
        del globals()['combined']

    gc.collect()

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)


def merge_wavs(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    print("Start Processing...\n")

    for B_folder in os.listdir(input_folder):
        B_path = os.path.join(input_folder, B_folder)

        if B_folder.startswith('.') or not os.path.isdir(B_path):
            continue

        raw_folders = []
        for f in os.listdir(B_path):
            f_path = os.path.join(B_path, f)
            if os.path.isdir(f_path) and not f.startswith('.') and f.endswith('_RAW'):
                raw_folders.append(f)

        if not raw_folders:
            print(f"Warning: {B_folder} does not contain a raw music subfolder !")
            continue

        C_folder = os.path.join(B_path, raw_folders[0])

        wav_files = []
        for f in os.listdir(C_folder):
            file_path = os.path.join(C_folder, f)
            if (not f.startswith('.')) and f.lower().endswith('.wav'):
                if not is_valid_wav(file_path):
                    continue
                wav_files.append(file_path)

        if not wav_files:
            print(f"Warning: {C_folder} does not contain any wav file !")
            continue

        wav_files.sort()

        output_file = os.path.join(output_folder, f"{B_folder}.wav")

        print(f"Combining: {B_folder}")
        print(f"  contains {len(wav_files)} wav files")

        try:
            # 读取第一个文件的参数作为基准
            first_data, sample_rate = sf.read(wav_files[0])
            sum_data = np.zeros_like(first_data)

            for wav_file in wav_files:
                data, sr = sf.read(wav_file)
                if sr != sample_rate:
                    raise ValueError(f"Sample rate mismatch in {wav_file}")
                if len(data) != len(sum_data):
                    raise ValueError(f"Length mismatch in {wav_file}")
                sum_data += data

            sum_data = sum_data - np.max(sum_data)
            sum_data = sum_data / np.max(np.abs(sum_data))

            sf.write(output_file, sum_data, sample_rate)

            print(f"Created: {os.path.basename(output_file)}\n")

        except Exception as e:
            print(f"Error while processing {B_folder}: {str(e)}\n")

        finally:
            gc.collect()

    print(f"\nDone!")


if __name__ == "__main__":
    folder_MedleyDB = "/media/hsun/hsun_exFAT/Datasets/MedleyDB/MedleyDB"
    output_folder = "/media/hsun/hsun_exFAT/Datasets/MedleyDB/MedleyDB_all"

    merge_wavs(folder_MedleyDB, output_folder)