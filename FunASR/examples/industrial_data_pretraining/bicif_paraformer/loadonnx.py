import onnx
import onnxruntime
import torch
import numpy as np
import torchaudio

import tokenizer
from funasr.frontends.wav_frontend import apply_lfr, load_cmvn, apply_cmvn

onnx_model = onnx.load("BiCifParaformer.onnx")
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession("BiCifParaformer_quant.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def load_audio(path):
    audio, fs = torchaudio.load(path)
    audio = audio * (1 << 15)  # audio 为整数 大概为 15 16 ...
    fbank = torchaudio.compliance.kaldi.fbank(
        audio, num_mel_bins=80, frame_length=25, frame_shift=10, dither=1.0, energy_floor=0.0, window_type="hamming",
        sample_frequency=16000, snip_edges=True)
    fbank = apply_lfr(fbank, lfr_m=7, lfr_n=6)
    cmvn = load_cmvn(r"D:\work\asrapp\FunASR\examples\industrial_data_pretraining\bicif_paraformer\iic\speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch\am.mvn")
    fbank = apply_cmvn(fbank, cmvn)
    return fbank.unsqueeze(0)


x = load_audio(r"test.wav")

# x = load_audio(
#     r"D:\work\asrapp\FunASR\examples\industrial_data_pretraining\bicif_paraformer\iic\speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch\example\asr_example.wav")
l = torch.ones([1], dtype=torch.int32) * x.shape[1]

# compute ONNX Runtime output prediction
ort_inputs = {"speech": to_numpy(x), "speech_lengths": to_numpy(l)}

# ort_inputs = {ort_session.get_inputs()[0], "speech_lengths": to_numpy(torch.Tensor([128]))}
# print(ort_session.get_outputs())
# print(ort_session.get_inputs())
ort_outs = ort_session.run(None, ort_inputs)

am_scores = ort_outs[0]
yseq = am_scores.argmax(axis=-1)
if yseq.shape[0] == 1:
    yseq = yseq[0]

token_int = [s for s in yseq if s not in [0, 1, 2]]
# y_seq = am_scores[0]
token = tokenizer.Tokenizer(r"D:\work\asrapp\FunASR\examples\industrial_data_pretraining\bicif_paraformer\iic"
                            r"\speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch\tokens.json")
info = token.id2token(yseq)
print(info)
