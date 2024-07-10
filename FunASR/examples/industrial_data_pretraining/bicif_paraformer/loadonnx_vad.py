import onnx
import onnxruntime
import torch
import torchaudio
import onnx.helper as helper
from onnx.helper import TensorProto


from funasr.frontends.wav_frontend import WavFrontendOnline

model = onnx.load("silero_vad.onnx")
#
# reshape_node = helper.make_node(
#     'FSMNExport',
#     inputs=['feats', "waveform"],
#     outputs=['scores', "info"],
# )
#
# input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 48, 48])
# model.graph.node[0].CopyFrom(reshape_node)
# onnx.save_model(model, "fsmnVad_change.onnx")

ort_session = onnxruntime.InferenceSession("silero_vad.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# WavFrontendOnline
args = {'cmvn_file': 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch\\am.mvn', 'dither': 0.0, 'frame_length': 25,
        'frame_shift': 10, 'fs': 16000, 'lfr_m': 5, 'lfr_n': 1, 'n_mels': 80, 'window': 'hamming'}
frontend = WavFrontendOnline(**args)


def load_audio(path):
    audio, fs = torchaudio.load(path)
    # audio = audio * (1 << 15)  # audio 为整数 大概为 15 16 ...
    data, data_len = frontend(audio, audio.shape[1])
    # fbank = torchaudio.compliance.kaldi.fbank(
    #     audio, num_mel_bins=80, frame_length=25, frame_shift=10, dither=1.0, energy_floor=0.0, window_type="hamming",
    #     sample_frequency=16000, snip_edges=True)
    # fbank = apply_lfr(fbank, lfr_m=7, lfr_n=6)
    # cmvn = load_cmvn(r"D:\work\asrapp\FunASR\examples\industrial_data_pretraining\bicif_paraformer\iic\speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch\am.mvn")
    # fbank = apply_cmvn(fbank, cmvn)
    return data, audio


feat, waveform = load_audio(r"D:\work\asrapp\FunASR\examples/industrial_data_pretraining/bicif_paraformer/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav")

in_cache0 = torch.randn(1, 128, 19, 1)
in_cache1 = torch.randn(1, 128, 19, 1)
in_cache2 = torch.randn(1, 128, 19, 1)
in_cache3 = torch.randn(1, 128, 19, 1)

# x = load_audio(
#     r"D:\work\asrapp\FunASR\examples\industrial_data_pretraining\bicif_paraformer\iic\speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch\example\asr_example.wav")
# l = torch.ones([1], dtype=torch.int64) * x.shape[1]

# compute ONNX Runtime output prediction
# ort_inputs = {"speech": to_numpy(feat), "in_cache0": to_numpy(in_cache0), "in_cache1": to_numpy(in_cache1),
#               "in_cache2": to_numpy(in_cache2), "in_cache3": to_numpy(in_cache3)}
ort_inputs = {"feats": to_numpy(feat)}

# ort_inputs = {ort_session.get_inputs()[0], "speech_lengths": to_numpy(torch.Tensor([128]))}
ort_outs = ort_session.run(None, ort_inputs)

print(ort_outs[0])
print(ort_outs[1])
# [[[0, 1810], [2640, 7520], [7800, 11860]]]


# torch.load()


output_data_buf_offset = 0
segment_batch = []

# contain_seg_start_point  contain_seg_end_point start_ms end_ms
#     if len(cache["stats"].output_data_buf) > 0:
#         for i in range(
#                 output_data_buf_offset, len(cache["stats"].output_data_buf)
#         ):
#
#
#                 if not is_final and (
#                         not cache["stats"].output_data_buf[i].contain_seg_start_point
#                         or not cache["stats"].output_data_buf[i].contain_seg_end_point
#                 ):
#                     continue
#                 segment = [
#                     cache["stats"].output_data_buf[i].start_ms,
#                     cache["stats"].output_data_buf[i].end_ms,
#                 ]
#                 cache["stats"].output_data_buf_offset += 1  # need update this parameter
#
#             segment_batch.append(segment)
#
#     if segment_batch:
#         segments.append(segment_batch)