import glob
import random

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset
from augments import Speed, Reverb, Gain

chunk_len = 16000 * 2

# speed = Speed()
# reverb = Reverb()
# gain = Gain()


def process(wav_path, train=True):
    wav, sr = torchaudio.load(wav_path)
    # wav = wav.squeeze()
    if train:
        choice = np.random.rand()
        if choice < 0.5:
            snr = torch.randint(5, 10, [1])
            wav = torchaudio.functional.add_noise(wav, torch.randn(wav.shape), snr)
        elif 0.5 <= choice < 0.7:
            # wav = speed(wav)
            snr = torch.randint(10, 15, [1])
            wav = torchaudio.functional.add_noise(wav, torch.randn(wav.shape), snr)
            # wav = pitch_shift(wav)
            # wav = torchaudio.sox_effects.apply_effects_tensor(wav, 16000, [["speed", "1.1"], ["rate", "16000"]])
        # elif 0.5 <= choice < 0.7:
        #     wav = reverb(wav)
        elif 0.7 <= choice < 0.9:
            # wav = gain(wav)
            snr = torch.randint(15, 20, [1])
            wav = torchaudio.functional.add_noise(wav, torch.randn(wav.shape), snr)
            # wav = pitch_shift(wav)
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
        # min_start = 4800 if wav.shape[-1] - chunk_len > 4800 else 0
        start = random.randint(0, wav.shape[-1] - chunk_len)
        end = start + chunk_len
        wav = wav[:, start:end]
    feats = kaldi.fbank(wav, num_mel_bins=80)
    feats = feats - feats.mean(0, keepdim=True)
    return feats


def get_datasets():
    path = "ST-CMDS-20170001_1-OS"
    files = glob.glob(path + "/*.metadata")
    wav_files = []
    labels = []
    now_label = 0
    count = 0
    for file in files:
        wav_files.append(file[:-8] + "wav")
        count += 1
        labels.append(now_label)
        if count % 120 == 0:
            now_label += 1
    index = [i for i in range(len(files))]
    random.shuffle(index)
    wav_files = np.array(wav_files)
    labels = np.array(labels)
    wav_files = wav_files[index]
    labels = labels[index]
    train_num = int(len(wav_files) * 0.8)
    train_dataset = SRDataset(wav_files[:train_num], labels[:train_num], now_label, True)
    valid_dataset = SRDataset(wav_files[train_num:], labels[train_num:], now_label, False)
    return train_dataset, valid_dataset


class SRDataset(Dataset):
    def __init__(self, wav_files, labels, now_label, train=True):
        self.wav_files = wav_files
        self.labels = labels
        self.now_label = now_label
        self.train = train

    def __getitem__(self, index):
        feats = process(self.wav_files[index], self.train)
        label = self.labels[index]
        return feats, torch.tensor(label, dtype=torch.int64)

    def __len__(self):
        return len(self.wav_files)


if __name__ == '__main__':
    sr = SRDataset()
