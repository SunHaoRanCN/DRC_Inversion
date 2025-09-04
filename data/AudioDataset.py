import soundfile as sf
import librosa
import numpy as np
from torch.utils.data import Dataset
import os
import torch
import random


class get_2D_dataset(Dataset):
    def __init__(self, root_dir, t_dim, f_dim, resample=False, target_fs=16000, transform=None):
        self.audio_folder = root_dir
        self.target_shape = (t_dim, f_dim)
        self.resample = resample
        self.target_fs = target_fs
        self.transform = transform
        self.valid_compressor_types = {'0', 'O'}
        self.audio_list = self.find_paths()

    def find_paths(self):
        all_files = []
        for file_name in os.listdir(self.audio_folder):
            if file_name.endswith(".wav"):
                file_path = os.path.join(self.audio_folder, file_name)
                all_files.append(file_path)
        return all_files

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        audio_path = self.audio_list[idx]
        audio, sr = sf.read(audio_path)
        parts = audio_path.split('_')
        compressor_type = int(parts[-1].split('.')[0])
        # if compressor_type == 'O':
        #     compressor_type = 0
        # elif compressor_type == 'A':
        #     compressor_type = 1
        # elif compressor_type == 'B':
        #     compressor_type = 2
        # elif compressor_type == 'C':
        #     compressor_type = 3
        # elif compressor_type == 'D':
        #     compressor_type = 4
        # elif compressor_type == 'E':
        #     compressor_type = 5
        # else:
        #     compressor_type = int(compressor_type)

        audio_name = audio_path.split("/")[-1]

        modified_path = audio_path.replace('train', 'all').replace('test', 'all')
        base_name, extension = modified_path.rsplit('.', 1)
        parts = base_name.split('_')
        parts[-1] = '0'
        raw_audio_path = '_'.join(parts) + '.' + extension

        raw_audio, _ = sf.read(raw_audio_path)

        if self.resample:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_fs)
            sr = self.target_fs

        total_samples = 5 * sr
        n_fft = 2 * (self.target_shape[0] - 1)
        hop_length = (total_samples - n_fft) // (self.target_shape[1] - 1)
        feature = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
        feature = librosa.amplitude_to_db(np.abs(feature))
        feature = feature.T

        if self.transform:
            feature = self.transform(feature)

        return audio, raw_audio, feature, compressor_type, audio_name


def norm_params(control_ranges, p):
    m = control_ranges[:, 0].reshape(1, -1)
    M = control_ranges[:, 1].reshape(1, -1)
    pp = (p - m) / (M - m)
    return pp.flatten()


class get_dataset(Dataset):
    def __init__(self, root_dir, compressors, control_ranges):
        self.audio_folder = root_dir
        self.compressors = compressors
        self.control_ranges = control_ranges
        self.audio_list = self.find_paths()

    def find_paths(self):
        all_files = []
        for file_name in os.listdir(self.audio_folder):
            if file_name.endswith(".wav"):
                file_path = os.path.join(self.audio_folder, file_name)
                all_files.append(file_path)
        return all_files

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        audio_path = self.audio_list[idx]
        audio, sr = sf.read(audio_path)

        parts = audio_path.split('_')
        compressor_type = parts[-1].split('.')[0]

        audio_name = audio_path.split("/")[-1]

        modified_path = audio_path.replace('train', 'all').replace('test', 'all')
        base_name, extension = modified_path.rsplit('.', 1)
        parts = base_name.split('_')
        parts[-1] = '0'
        raw_audio_path = '_'.join(parts) + '.' + extension
        raw_audio, _ = sf.read(raw_audio_path)

        if compressor_type == '0':
            real_p = np.array([0, 0, 0, 0, 0, 0])
        else:
            real_p = list(self.compressors[compressor_type].values())
            real_p.pop()
            real_p = np.array(real_p)
        norm_p = norm_params(self.control_ranges, real_p)

        return audio, raw_audio, compressor_type, real_p, norm_p, audio_name
