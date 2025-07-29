import numpy as np
import librosa
# from pesq import pesq
# from algorithmLib import compute_audio_quality
from scipy.io.wavfile import write
import os
from scipy.signal import stft
from scipy.fft import fft


def loss_mse(x, x_hat):
    # Compute squared differences
    squared_diff = (x - x_hat) ** 2
    # Compute mean of squared differences
    mse = np.mean(squared_diff)
    return mse


### Root Mean Square Error in dB
def loss_rmse_db(x, x_hat):
    error = x_hat - x
    N = len(error)
    e = 20 * np.log10(np.sqrt(np.sum(np.square(error)) / N))
    return e


### log scale Mel spectrogram
def loss_log_mel(x_hat, x, fs=22050, n_fft=2048, hop_length=512, win_length=2048, window='hann', n_mels=128):
    mel_spec1 = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                               n_mels=n_mels, window=window)
    log_mel_spec1 = librosa.power_to_db(mel_spec1, ref=np.max)
    mel_spec2 = librosa.feature.melspectrogram(y=x_hat, sr=fs, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                               n_mels=n_mels, window=window)
    log_mel_spec2 = librosa.power_to_db(mel_spec2, ref=np.max)
    l1_norm = np.sum(np.abs(log_mel_spec2) - np.abs(log_mel_spec1))
    return l1_norm


class MelSTFTLoss:
    def __init__(self,
                 sample_rate=22050,
                 n_fft=2048,
                 hop_length=512,
                 win_length=2048,
                 n_mels=128,
                 fmin=0.0,
                 fmax=None):
        """
        Initialize Mel STFT Loss calculation

        Parameters:
        -----------
        sample_rate : int, optional
            Sampling rate of the audio signal
        n_fft : int, optional
            Number of FFT components
        hop_length : int, optional
            Number of samples between successive frames
        win_length : int, optional
            Each frame of audio is windowed by window of length win_length
        n_mels : int, optional
            Number of Mel filterbank channels
        fmin : float, optional
            Lowest frequency to include in Mel filterbank
        fmax : float or None, optional
            Highest frequency to include in Mel filterbank
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2

        # Create Mel filterbank
        self.mel_basis = self._create_mel_filterbank()

    def _create_mel_filterbank(self):
        """
        Create Mel filterbank matrix

        Returns:
        --------
        mel_basis : numpy.ndarray
            Mel filterbank transformation matrix
        """
        # Frequency points for FFT
        fft_freqs = np.linspace(0, self.sample_rate / 2, self.n_fft // 2 + 1)

        # Convert Hz to Mel scale
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        # Create Mel points
        mel_low = hz_to_mel(self.fmin)
        mel_high = hz_to_mel(self.fmax)
        mel_points = np.linspace(mel_low, mel_high, self.n_mels + 2)

        # Convert Mel points back to Hz
        hz_points = mel_to_hz(mel_points)

        # Create filterbank
        mel_basis = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for m in range(1, self.n_mels + 1):
            left_hz = hz_points[m - 1]
            center_hz = hz_points[m]
            right_hz = hz_points[m + 1]

            for k, freq in enumerate(fft_freqs):
                # Triangular filter
                if left_hz <= freq <= center_hz:
                    mel_basis[m - 1, k] = (freq - left_hz) / (center_hz - left_hz)
                elif center_hz <= freq <= right_hz:
                    mel_basis[m - 1, k] = (right_hz - freq) / (right_hz - center_hz)

        return mel_basis

    def _compute_stft(self, audio):
        """
        Compute Short-time Fourier Transform

        Parameters:
        -----------
        audio : numpy.ndarray
            Input audio signal

        Returns:
        --------
        magnitude_spectrogram : numpy.ndarray
            Magnitude spectrogram of the input audio
        """
        # Compute STFT
        f, t, Zxx = stft(audio,
                         fs=self.sample_rate,
                         nperseg=self.n_fft,
                         noverlap=self.n_fft - self.hop_length,
                         window='hann')

        # Compute magnitude spectrogram
        return np.abs(Zxx)

    def _mel_spectrogram(self, magnitude_spec):
        """
        Convert magnitude spectrogram to Mel spectrogram

        Parameters:
        -----------
        magnitude_spec : numpy.ndarray
            Magnitude spectrogram

        Returns:
        --------
        mel_spec : numpy.ndarray
            Mel spectrogram
        """
        # Apply Mel filterbank to magnitude spectrogram
        mel_spec = np.dot(self.mel_basis, magnitude_spec[:self.n_fft // 2 + 1, :])

        # Add small epsilon to avoid log(0)
        mel_spec = np.log(mel_spec + 1e-10)

        return mel_spec

    def compute_loss(self, pred_audio, target_audio):
        """
        Compute Mel STFT Loss between two audio signals

        Parameters:
        -----------
        pred_audio : numpy.ndarray
            Predicted audio signal
        target_audio : numpy.ndarray
            Target audio signal

        Returns:
        --------
        loss : float
            Mel STFT loss value
        """
        # Compute magnitude spectrograms
        pred_mag_spec = self._compute_stft(pred_audio)
        target_mag_spec = self._compute_stft(target_audio)

        # Convert to Mel spectrograms
        pred_mel_spec = self._mel_spectrogram(pred_mag_spec)
        target_mel_spec = self._mel_spectrogram(target_mag_spec)

        # Compute Mean Absolute Error (MAE) between Mel spectrograms
        loss = np.mean(np.abs(pred_mel_spec - target_mel_spec))

        return loss


def loss_SISDR(estimate, reference):
    eps = np.finfo(float).eps
    alpha = np.dot(estimate.T, reference) / (np.dot(estimate.T, estimate) + eps)

    molecular = ((alpha * reference) ** 2).sum()
    denominator = ((alpha * reference - estimate) ** 2).sum()

    return 10 * np.log10((molecular) / (denominator+eps))


### Perceptual Evaluation of Speech Quality (PESQ)
def loss_PESQ(x, x_hat, fs=16000, band='wb'):
    """
    ref: numpy 1D array, reference audio signal
        deg: numpy 1D array, degraded audio signal
        fs:  integer, sampling rate
        mode: 'wb' (wide-band) or 'nb' (narrow-band)
        on_error: error-handling behavior, it could be PesqError.RETURN_VALUES or PesqError.RAISE_EXCEPTION by default
    Returns:
        pesq_score: float, P.862.2 Prediction (MOS-LQO)
    """
    return pesq(fs, x, x_hat, band)


### Perceptual Evaluation of Audio Quality (PEAQ)
def loss_PEAQ(x, x_hat, fs):
    write('audio_ref.wav', fs, x)
    write('audio_test.wav', fs, x_hat)
    score = compute_audio_quality('PEAQ', 'audio_ref.wav', 'audio_test.wav')
    os.remove('audio_ref.wav')
    os.remove('audio_test.wav')
    return score