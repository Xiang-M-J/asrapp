import random

import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Vol


# 音量调节
class Gain:
    def __init__(self, min_gain: float = -20.0, max_gain: float = 20.0):
        self.min_gain = min_gain
        self.max_gain = max_gain

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        gain = random.uniform(self.min_gain, self.max_gain)
        audio = Vol(gain, gain_type="db")(audio)
        return audio


# 回声效果
class Delay:
    def __init__(
            self,
            sample_rate=16000,
            volume_factor=0.5,
            min_delay=200,
            max_delay=500,
            delay_interval=50,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.volume_factor = volume_factor
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.delay_interval = delay_interval

    def calc_offset(self, ms):
        return int(ms * (self.sample_rate / 1000))

    def __call__(self, audio):
        ms = random.choice(
            np.arange(self.min_delay, self.max_delay, self.delay_interval)
        )

        offset = self.calc_offset(ms)
        beginning = torch.zeros(audio.shape[0], offset).to(audio.device)
        end = audio[:, :-offset]
        delayed_signal = torch.cat((beginning, end), dim=1)
        delayed_signal = delayed_signal * self.volume_factor
        audio = (audio + delayed_signal) / 2
        return audio


# 加入混响
class Reverb:
    def __init__(self):
        rir, sr = torchaudio.load("reverberation.wav")
        rir = rir[:, int(sr * 1.01): int(sr * 1.3)]
        self.rir = rir / torch.linalg.vector_norm(rir, ord=2)

    def __call__(self, audio):
        audio = torchaudio.functional.fftconvolve(audio, self.rir)
        return audio


# 加噪声
class Noise:
    def __init__(self, min_snr=10, max_snr=30):
        """
        :param min_snr: Minimum signal-to-noise ratio
        :param max_snr: Maximum signal-to-noise ratio
        """
        self.min_snr = min_snr
        self.max_snr = max_snr

    def __call__(self, audio):
        snr = torch.randint(self.min_snr, self.max_snr, [1])
        # snr = torch.from_numpy(np.array([10]))
        noise_audio = torchaudio.functional.add_noise(audio, torch.randn(audio.shape), snr)
        return noise_audio


# 改变音调(很慢慎用)（修改自 librosa.effects.pitch_shift）
class PitchShift:
    def __init__(self, pitch_shift_min=-4, pitch_shift_max=4):
        self.pitch_shift_min = pitch_shift_min * 10
        self.pitch_shift_max = pitch_shift_max * 10

    def pitch_shift(self, audio, n_steps):
        rate = 2.0 ** (-float(n_steps) / 12)
        return torchaudio.functional.resample(self.time_stretch(audio, rate), int(16000 / rate), 16000)

    def time_stretch(self, audio, rate):
        stft = torch.stft(audio, 2048, return_complex=True, window=torch.hann_window(2048), center=True)

        # Stretch by phase vocoding
        stft_stretch = self.phase_vocoder(
            stft.numpy(),
            rate=rate,
            hop_length=None,
            n_fft=None,
        )

        # Predict the length of y_stretch
        len_stretch = int(round(audio.shape[-1] / rate))

        # Invert the STFT
        y_stretch = torch.istft(stft_stretch, n_fft=2048, window=torch.hann_window(2048), length=len_stretch)

        return y_stretch

    def phasor_angles(self, x) -> np.complex_:  # pragma: no cover
        return np.cos(x) + 1j * np.sin(x)  #

    def phasor(self, angles, mag=None):
        z = self.phasor_angles(angles)
        if mag is not None:
            z *= mag

        return z

    def phase_vocoder(self, D: np.ndarray, rate, hop_length, n_fft):
        if n_fft is None:
            n_fft = 2 * (D.shape[-2] - 1)

        if hop_length is None:
            hop_length = int(n_fft // 4)

        time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)

        # Create an empty output array
        shape = list(D.shape)
        shape[-1] = len(time_steps)
        d_stretch = np.zeros_like(D, shape=shape)

        # Expected phase advance in each bin
        phi_advance = np.linspace(0, np.pi * hop_length, D.shape[-2])

        # Phase accumulator; initialize to the first sample
        phase_acc = np.angle(D[..., 0])

        # Pad 0 columns to simplify boundary logic
        padding = [(0, 0) for _ in D.shape]
        padding[-1] = (0, 2)
        D = np.pad(D, padding, mode="constant")

        for t, step in enumerate(time_steps):
            columns = D[..., int(step): int(step + 2)]

            # Weighting for linear magnitude interpolation
            alpha = np.mod(step, 1.0)
            mag = (1.0 - alpha) * np.abs(columns[..., 0]) + alpha * np.abs(columns[..., 1])

            # Store to output array
            d_stretch[..., t] = self.phasor(phase_acc, mag=mag)

            # Compute phase advance
            dphase = np.angle(columns[..., 1]) - np.angle(columns[..., 0]) - phi_advance

            # Wrap to -pi:pi range
            dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))

            # Accumulate phase
            phase_acc += phi_advance + dphase

        return torch.from_numpy(d_stretch)

    def __call__(self, audio):
        pitch = random.randint(self.pitch_shift_min, self.pitch_shift_max)
        pitch /= 10
        audio = self.pitch_shift(audio, float(pitch))
        return audio


# 随机裁剪音频两端信息
class RandomResizedCrop():
    def __init__(self, max_crop_samples=32000, min_crop_samples=16000):
        self.max_crop_samples = max_crop_samples
        self.min_crop_samples = min_crop_samples

    def __call__(self, audio):
        random_crop_samples = random.randint(self.min_crop_samples, self.max_crop_samples)
        max_samples = audio.shape[-1]
        if random_crop_samples > max_samples / 2:
            random_crop_samples = int(max_samples / 2)

        start_idx = random.randint(0, max_samples - random_crop_samples)
        audio = audio[..., start_idx: start_idx + random_crop_samples]
        return audio


# 随机裁剪音频中间信息
class InnerCrop():
    def __init__(self, max_crop_samples=48000, min_crop_samples=16000):
        self.max_crop_samples = max_crop_samples
        self.min_crop_samples = min_crop_samples

    def __call__(self, audio):
        random_crop_samples = random.randint(self.min_crop_samples, self.max_crop_samples)
        max_samples = audio.shape[-1]
        if random_crop_samples > max_samples / 2:
            random_crop_samples = int(max_samples / 2)
        start_idx = random.randint(500, int(max_samples / 2) - 1) - 100
        return torch.cat([audio[:, 0:start_idx], audio[:, start_idx + random_crop_samples:]], dim=-1)


# 翻转音频
class Reverse:
    def __init__(self):
        pass

    def __call__(self, audio):
        return torch.flip(audio, dims=[-1])


# 更改速度
class Speed:
    def __init__(self, min_speed: int = 80, max_speed: int = 120):
        self.min_speed = min_speed
        self.max_speed = max_speed

    def __call__(self, audio: torch.Tensor):
        speed = random.randint(self.min_speed, self.max_speed)
        speed = speed / 100
        audio, _ = torchaudio.functional.speed(audio, 16000, speed)
        return audio


class Compose:
    """Data augmentation module that transforms any given data example with a chain of audio augmentations."""

    def __init__(self, num_transform=1):
        self.num_transform = num_transform
        self.transforms = [Gain(), Delay(), Reverb(), Noise(), RandomResizedCrop(), InnerCrop(), Reverse(), Speed()]

    def __call__(self, x):
        x = self.transform(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "\t{0}".format(t)
        format_string += "\n)"
        return format_string

    def transform(self, x):
        choice = random.random()
        if choice < 0.8:
            self.num_transform = 1
        elif 0.8 < choice < 0.9:
            self.num_transform = 2
        else:
            self.num_transform = 3
        transforms = random.sample(self.transforms, self.num_transform)
        for t in transforms:
            x = t(x)
        return x


if __name__ == "__main__":
    audio, _ = torchaudio.load("A2_10.wav")
    agm = Compose()
    audio = agm(audio)
    torchaudio.save("p.wav", audio, 16000)
