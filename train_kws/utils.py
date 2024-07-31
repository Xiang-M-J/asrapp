import os.path
import random

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
import glob
import torchaudio.compliance.kaldi as kaldi
chunk_len = 16000 * 2


def process(wav_path):
    wav, sr = torchaudio.load(wav_path)
    # wav = wav.squeeze()
    choice = np.random.rand()
    if choice < 0.5:
        snr = torch.randint(5, 10, [1])
        wav = torchaudio.functional.add_noise(wav, torch.randn(wav.shape), snr)
    elif 0.5 <= choice < 0.7:
        snr = torch.randint(10, 15, [1])
        wav = torchaudio.functional.add_noise(wav, torch.randn(wav.shape), snr)
        # wav = torchaudio.sox_effects.apply_effects_tensor(wav, 16000, [["speed", "1.1"], ["rate", "16000"]])
    elif 0.7 <= choice < 0.9:
        snr = torch.randint(15, 20, [1])
        wav = torchaudio.functional.add_noise(wav, torch.randn(wav.shape), snr)
        # wav = torchaudio.sox_effects.apply_effects_tensor(wav, 16000, [["speed", "1.1"], ["rate", "16000"]])
    # elif 0.7 <= choice < 0.8:
    #     snr = torch.randint(5, 15, [1])
    #     wav = torchaudio.functional.add_noise(wav, torch.randn(wav.shape), snr)
    #     wav = torchaudio.sox_effects.apply_effects_tensor(wav, 16000, [["speed", "1.1"], ["rate", "16000"]])
    # elif 0.8 <= choice < 0.9:
    #     snr = torch.randint(5, 15, [1])
    #     wav = torchaudio.functional.add_noise(wav, torch.randn(wav.shape), snr)
    #     wav = torchaudio.sox_effects.apply_effects_tensor(wav, 16000, [["speed", "1.1"], ["rate", "16000"]])
    else:
        pass
    if wav.shape[-1] < chunk_len:
        wav = torch.nn.functional.pad(wav, (0, chunk_len - wav.shape[-1]))
    elif wav.shape[-1] > chunk_len:
        start = random.randint(0, wav.shape[-1] - chunk_len)
        end = start + chunk_len
        wav = wav[:, start:end]
    feats = kaldi.fbank(wav, num_mel_bins=80)
    feats = feats - feats.mean(0, keepdim=True)
    return feats


class SRDataset(Dataset):
    def __init__(self):
        path = "ST-CMDS-20170001_1-OS"
        files = glob.glob(path + "/*.metadata")
        self.wav_files = []
        self.labels = []
        self.now_label = 0
        count = 0
        for file in files:
            self.wav_files.append(file[:-8] + "wav")
            count += 1
            self.labels.append(self.now_label)
            if count % 120 == 0:
                self.now_label += 1

    def __getitem__(self, index):
        feats = process(self.wav_files[index])
        label = self.labels[index]
        return feats, torch.tensor(label, dtype=torch.int64)

    def __len__(self):
        return len(self.wav_files)


if __name__ == '__main__':
    sr = SRDataset()
