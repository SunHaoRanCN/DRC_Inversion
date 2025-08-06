import soundfile as sf
import librosa
import numpy as np
from torch.utils.data import Dataset
import os
import torch
import random


def add_noise_with_snr_numpy(audio: np.ndarray,
                             snr_db: float) -> np.ndarray:
    if np.isinf(snr_db):
        return audio.copy()

    signal_power = np.mean(audio ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(*audio.shape) * np.sqrt(noise_power)
    noisy_audio = np.clip(audio + noise, -1.0, 1.0)

    return noisy_audio.astype(audio.dtype)


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
        compressor_type = parts[-1].split('.')[0]
        if compressor_type == 'O':
            compressor_type = 0
        elif compressor_type == 'A':
            compressor_type = 1
        elif compressor_type == 'B':
            compressor_type = 2
        elif compressor_type == 'C':
            compressor_type = 3
        elif compressor_type == 'D':
            compressor_type = 4
        elif compressor_type == 'E':
            compressor_type = 5
        else:
            compressor_type = int(compressor_type)

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
        # mel = librosa.feature.melspectrogram(y=audio, sr=0.5*sr, n_fft=2048, hop_length=128, win_length=2048, n_mels=128)  # 5s audio LibriSpeech
        # mel = librosa.feature.melspectrogram(y=audio, sr=22050, n_fft=2048, hop_length=205, win_length=2048, n_mels=64)  # 2s audio LibriSpeech
        # mel = librosa.feature.melspectrogram(y=audio, sr=22050, n_fft=2048, hop_length=512, win_length=2048, n_mels=128)
        ###stft
        # mel = librosa.stft(y=audio,n_fft=254, hop_length=512) # (128, 431) 44100
        feature = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)  # (64, 431) 44100
        feature = librosa.amplitude_to_db(np.abs(feature))
        feature = feature.T

        if self.transform:
            feature = self.transform(feature)

        return audio, raw_audio, feature, compressor_type, audio_name


class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        type = self.labels[idx]

        audio_name = audio_path.split("/")[-1]
        modified_path = audio_path.replace('_train', '').replace('_test', '')
        base_name, extension = modified_path.rsplit('.', 1)
        parts = base_name.split('_')
        parts[-1] = '0'
        raw_audio_path = '_'.join(parts) + '.' + extension

        real_signal, sr = sf.read(raw_audio_path)
        target_signal, sr = sf.read(audio_path)
        mel = librosa.feature.melspectrogram(y=target_signal, sr=22050, n_fft=2048, hop_length=512, win_length=2048, window='hann', n_mels=64)
        # mel = librosa.feature.melspectrogram(y=target_signal, sr=22050, n_fft=2048, hop_length=372, win_length=2048, window='hann', n_mels=128)
        # mel = librosa.feature.melspectrogram(y=target_signal, sr=22050, n_fft=2048, hop_length=186, win_length=2048, window='hann', n_mels=64)
        mel = mel.T
        # mel = imageProcess(mel, (154*3, 12*3))
        # Apply transform if specified
        if self.transform:
            mel = self.transform(mel)

        return real_signal, target_signal, mel, type, audio_name



def norm_params(control_ranges, p):
    m = control_ranges[:, 0].reshape(1, -1)
    M = control_ranges[:, 1].reshape(1, -1)
    pp = (p - m) / (M - m)
    return pp.flatten()


class get_dataset(Dataset):
    def __init__(self, root_dir, compressors):
        self.audio_folder = root_dir
        self.compressors = compressors
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
        control_ranges = np.array([[-60, 0],
                                   [0, 15],
                                   [0, 130],
                                   [0, 130],
                                   [0, 500],
                                   [0, 2000]])
        # control_ranges = np.array([[-38, 0],
        #                            [0, 7.3],
        #                            [0, 5],
        #                            [0, 5],
        #                            [0, 13],
        #                            [0, 705]])

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
        norm_p = norm_params(control_ranges, real_p)

        return audio, raw_audio, compressor_type, real_p, norm_p, audio_name


def add_noise_with_snr(audio: torch.Tensor,
                       snr_db: float) -> torch.Tensor:
    if torch.isinf(torch.tensor(snr_db)):
        return audio.clone()

    signal_power = torch.mean(audio ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(audio) * torch.sqrt(noise_power)

    noisy_audio = torch.clamp(audio + noise, -1.0, 1.0)
    return noisy_audio


class CurriculumDataset(Dataset):
    def __init__(self, base_dataset, snr_scheduler):
        self.base_dataset = base_dataset
        self.snr_scheduler = snr_scheduler

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        processed_audio, raw_audio, compressor_type, real_p, norm_p, audio_name = self.base_dataset[idx]

        processed_audio = torch.from_numpy(processed_audio.copy()).float()
        raw_audio = torch.from_numpy(raw_audio.copy()).float()

        current_snr = self.snr_scheduler.get_current_snr()
        noisy_audio = add_noise_with_snr(processed_audio, current_snr)

        return (
            noisy_audio.numpy(),  # 保持与原数据集相同的numpy格式
            raw_audio.numpy(),
            compressor_type,
            real_p,
            norm_p,
            audio_name
        )


class AdversarialDataset(Dataset):
    def __init__(self, base_dataset,
                 snr_scheduler,
                 ):
        self.base_dataset = base_dataset
        self.snr_scheduler = snr_scheduler
        # 初始化一个随机掩码，用于确保每个epoch中每个样本只被处理一次
        self.noise_mask = np.zeros(len(base_dataset), dtype=bool)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        processed_audio, raw_audio, compressor_type, real_p, norm_p, audio_name = self.base_dataset[idx]
        processed_audio = torch.from_numpy(processed_audio.copy()).float()
        raw_audio = torch.from_numpy(raw_audio.copy()).float()

        # 若该样本被标记为需要添加噪声
        # if self.noise_mask[idx]:
        #     current_snr = self.snr_scheduler.get_current_snr()
        #     processed_audio = add_noise_with_snr(processed_audio, current_snr)

        return (
            # processed_audio.numpy(),
            # raw_audio.numpy(),
            processed_audio,
            raw_audio,
            compressor_type,
            real_p,
            norm_p,
            audio_name
        )

    # def reset_noise_mask(self):
    #     """在每个epoch开始时调用，重新生成噪声掩码"""
    #     # 随机选择一半的样本添加噪声
    #     mask_indices = random.sample(range(len(self.base_dataset)), k=len(self.base_dataset) // 2)
    #     self.noise_mask = np.zeros(len(self.base_dataset), dtype=bool)
    #     self.noise_mask[mask_indices] = True


def create_collate_fn(snr_scheduler):
    """创建动态添加噪声的collate_fn"""

    def collate_fn(batch):
        # 解包batch数据
        processed_audios, raw_audios, compressor_types, real_ps, norm_ps, audio_names = zip(*batch)

        # 转换为Tensor
        processed_audios = torch.stack(processed_audios)
        raw_audios = torch.stack(raw_audios)

        # 动态获取当前SNR值
        current_snr = snr_scheduler.get_current_snr()

        # 随机选择一半样本添加噪声
        batch_size = len(batch)
        num_noise = batch_size // 2
        noise_indices = torch.randperm(batch_size)[:num_noise]

        # 对选中样本添加噪声
        noisy_audios = processed_audios.clone()
        for idx in noise_indices:
            noisy_audios[idx] = add_noise_with_snr(noisy_audios[idx], current_snr)

        # 保持numpy格式输出（如原始需求）
        return (
            # noisy_audios.numpy(),
            # raw_audios.numpy(),
            noisy_audios,
            raw_audios,
            compressor_types,
            torch.tensor(real_ps),
            torch.tensor(norm_ps),
            audio_names
        )

    return collate_fn



if __name__ == '__main__':

    audio, fs = sf.read("/data/hsun/Projects/aquarius/wav examples/original_medleydb.wav")

    snr = 60

    noised_audio = add_noise_with_snr_numpy(audio, snr)

    sf.write("/data/hsun/Projects/aquarius/wav examples/60dBsnr.wav",
             noised_audio, fs)
