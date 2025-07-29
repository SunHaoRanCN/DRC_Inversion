import torch
import torchaudio
import numpy as np
import nnAudio.features
import librosa
from torch_stft import STFT

class TimeCQT_Encoder(torch.nn.Module):
    def __init__(
            self,
            samplerate: int = 16000,
            n_bins: int = 113,
            conv_class=torch.nn.Conv2d,
            NL_class=torch.nn.PReLU,
            out_channels=64,
            kernel_size=[1, 128],
            device=None,
            eps=1e-6,
            compute_representation=True
    ):
        super(TimeCQT_Encoder, self).__init__()
        self.n_bins = n_bins

        self.eps = eps

        self.out_dim = out_channels
        self.compute_representation = compute_representation
        self.name = "ENC_TimeCQT"
        if compute_representation:
            self.cqt = nnAudio.features.CQT1992v2(
                n_bins=n_bins,
                sr=samplerate
            )
            print(f"CQT kernel width : {self.cqt.kernel_width}")

        self.conv1 = torch.nn.Sequential(
            conv_class(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
            torch.nn.BatchNorm2d(num_features=out_channels),
            NL_class()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=31   # 31
            ),
            torch.nn.BatchNorm1d(num_features=out_channels),
            NL_class()
        )

    def forward(self, x):
        batch_size = x.size(0)

        if self.compute_representation:
            x = self.cqt(x).reshape(batch_size, 1, self.n_bins, -1)
            x = torch.log(x + self.eps)

        x = self.conv1(x)
        x, _ = torch.max(x, dim=2)
        # x = torch.mean(x, dim=2)
        x = self.conv2(x)
        x = x.mean(2)  # Mean across time frames
        x = x.reshape(batch_size, self.out_dim)  # Mean across frequency bins
        return x


class FrequencyCQT_Encoder(torch.nn.Module):
    def __init__(
            self,
            samplerate: int = 16000,
            n_bins: int = 113,
            conv_class=torch.nn.Conv2d,
            NL_class=torch.nn.PReLU,
            out_channels=64,
            kernel_size=[37, 1],
            device=None,
            eps=1e-6,
            compute_representation=True
    ):
        super(FrequencyCQT_Encoder, self).__init__()
        self.n_bins = n_bins

        self.eps = eps

        self.out_dim = out_channels
        self.compute_representation = compute_representation
        self.name = "ENC_FrequencyCQT"

        if compute_representation:
            self.cqt = nnAudio.features.CQT1992v2(
                n_bins=n_bins,
                sr=samplerate
            )
            print(f"CQT kernel width : {self.cqt.kernel_width}")

        self.conv1 = torch.nn.Sequential(
            conv_class(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
            torch.nn.BatchNorm2d(num_features=out_channels),
            NL_class()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=n_bins - kernel_size[0] + 1
            ),
            torch.nn.BatchNorm1d(num_features=out_channels),
            NL_class()
        )

    def forward(self, x):
        batch_size = x.size(0)

        if self.compute_representation:
            x = self.cqt(x).reshape(batch_size, 1, self.n_bins, -1)
            x = torch.log(x + self.eps)

        x = self.conv1(x)
        x, _ = torch.max(x, dim=3)
        # x = torch.mean(x, dim=3)
        # x = x.mean(2) # Mean across time frames
        x = self.conv2(x)
        x = x.reshape(batch_size, self.out_dim)  # Mean across frequency bins
        return x


class TFE(torch.nn.Module):
    def __init__(
            self,
            samplerate: int = 16000,
            f_dim: int = 64,
            t_dim: int = 431,
            label_dim: int = 6,
            conv_class=torch.nn.Conv2d,
            NL_class=torch.nn.PReLU,
            device=None,
            eps=1e-6,
            compute_representation=True
    ):
        super(TFE, self).__init__()
        self.samplerate = samplerate
        self.f_dim = f_dim
        self.t_dim = t_dim
        self.label_dim = label_dim
        self.eps = eps

        self.compute_representation = compute_representation

        # if self.compute_representation:
        #     total_samples = 5 * samplerate
        #     n_fft = 2 * (f_dim - 1)
        #     hop_length = (total_samples - n_fft) // (t_dim - 1)
        #     self.stft = torch.stft(
        #         n_fft=n_fft,
        #         hop_length=hop_length,
        #     )

        self.time = TimeCQT_Encoder(samplerate=self.samplerate, n_bins=self.f_dim, NL_class=NL_class, compute_representation=False)
        self.frequency = FrequencyCQT_Encoder(samplerate=self.samplerate, n_bins=self.f_dim, NL_class=NL_class, compute_representation=False)
        self.out_dim = self.time.out_dim + self.frequency.out_dim
        self.mlp = torch.nn.Linear(self.out_dim, self.label_dim)

    def forward(self,x):

        batch_size = x.size(0)

        if self.compute_representation:
            total_samples = 5 * self.samplerate
            n_fft = 2 * (self.f_dim - 1)
            hop_length = (total_samples - n_fft) // (self.t_dim - 1)
            # self.stft = torch.stft(
            #     n_fft=n_fft,
            #     hop_length=hop_length,
            # )
            x = x.squeeze(1)
            x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, return_complex=True)
            # x = x.reshape(batch_size, 1, self.n_bins, -1)
            x = torch.abs(x)
            x = torch.log(x + self.eps)
            x = x.unsqueeze(1)

        out = torch.zeros((batch_size, self.out_dim), device=x.device)

        out[:, :self.time.out_dim] = self.time(x)
        out[:, self.time.out_dim:] = self.frequency(x)
        out = self.mlp(out)

        return out

class TimeFrequencyCQT_Encoder(torch.nn.Module):
    def __init__(
            self,
            samplerate: int = 16000,
            n_bins: int = 64,
            label_dim = 6,
            conv_class=torch.nn.Conv2d,
            NL_class=torch.nn.PReLU,
            device=None,
            eps=1e-6,
            compute_representation=True
    ):
        super(TimeFrequencyCQT_Encoder, self).__init__()
        self.n_bins = n_bins
        self.label_dim = label_dim
        self.eps = eps

        self.compute_representation = compute_representation
        self.name = "ENC_TimeFrequencyCQT"

        if self.compute_representation:
            self.cqt = nnAudio.features.CQT1992v2(
                n_bins=n_bins,
                sr=samplerate
            )
            print(f"CQT kernel width : {self.cqt.kernel_width}")

        self.time = TimeCQT_Encoder(n_bins=self.n_bins, NL_class=NL_class, compute_representation=False)
        self.frequency = FrequencyCQT_Encoder(n_bins=self.n_bins, NL_class=NL_class, compute_representation=False)
        self.out_dim = self.time.out_dim + self.frequency.out_dim
        self.mlp = torch.nn.Linear(self.out_dim, self.label_dim)

    def forward(self, x):
        batch_size = x.size(0)

        if self.compute_representation:
            x = self.cqt(x).reshape(batch_size, 1, self.n_bins, -1)
            x = torch.log(x + self.eps)

        out = torch.zeros((batch_size, self.out_dim), device=x.device)

        out[:, :self.time.out_dim] = self.time(x)
        out[:, self.time.out_dim:] = self.frequency(x)
        out = self.mlp(out)

        return out

if __name__ == '__main__':

    batch_size = 12
    sr = 16000
    input_length = sr * 5
    f_dim = 64
    t_dim = 431

    input = torch.rand(batch_size, 1, input_length)
    print(f"Input Shape : {input.shape}\n")

    network = TimeFrequencyCQT_Encoder(samplerate=sr, n_bins=113)
    # network = TFE(samplerate=sr, f_dim=f_dim, t_dim=t_dim, label_dim=6)

    # model inference
    output = network(input)
    print(f"Output Shape : z={output.shape}")
