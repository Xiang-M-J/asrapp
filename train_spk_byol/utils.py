import glob
import random

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset
from augments import Compose

chunk_len = 16000 * 2


# speed = Speed()
# reverb = Reverb()
# gain = Gain()


def process(wav, chunk_len):
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
    path = "../train_kws/ST-CMDS-20170001_1-OS"
    files = glob.glob(path + "/*.metadata")
    # files = sorted(files)
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


class Augment:
    def __init__(self):
        self.compose = Compose()
        self.duration = 16000 * 2

    def __call__(self, audio):
        a1 = self.compose(audio)
        a2 = self.compose(audio)
        feats1 = process(a1, self.duration)
        feats2 = process(a2, self.duration)
        return feats1, feats2


class SRDataset(Dataset):
    def __init__(self, wav_files, labels, now_label, train=True):
        self.wav_files = wav_files
        self.labels = labels
        self.now_label = now_label
        self.train = train
        self.aug = Augment()

    def __getitem__(self, index):
        wav_file_path = self.wav_files[index]
        audio, _ = torchaudio.load(wav_file_path)
        feats1, feats2 = self.aug(audio)
        return feats1, feats2

    def __len__(self):
        return len(self.wav_files)


if __name__ == '__main__':
    pass
    # sr = SRDataset()
