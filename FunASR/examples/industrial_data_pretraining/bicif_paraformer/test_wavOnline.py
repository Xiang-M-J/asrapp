import numpy as np
import torch

from funasr.frontends.wav_frontend import WavFrontendOnline

args = {'cmvn_file': 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch\\am.mvn', 'dither': 0.0, 'frame_length': 25,
        'frame_shift': 10, 'fs': 16000, 'lfr_m': 5, 'lfr_n': 1, 'n_mels': 80, 'window': 'hamming'}
frontend = WavFrontendOnline(**args)


def load_audio(audio):
    data, data_len = frontend(audio, audio.shape[1])

    return data, audio


x = torch.zeros([1, 3200])
for i in range(3200):
    x[0, i] = 100 * np.sin(0.01 * i)

y = load_audio(x)[0]
print(y.shape)
