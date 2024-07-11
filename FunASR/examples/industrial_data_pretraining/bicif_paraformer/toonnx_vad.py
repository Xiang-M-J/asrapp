# feat: [1 416 400]
# waveform: [1 66800]
# isFinal True

# input_dim: 400  proj_dim: 128   lorder: 20   rorder: 0


import sys

sys.path.append("../../..")

import torch.onnx

from funasr import AutoModel

model = AutoModel(
    model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    # punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    # spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
)

vad_model = model.vad_model

sequence_length = 218
speech = torch.randn([1, sequence_length, 400])
speech_lengths = torch.ones([1], dtype=torch.int64) * sequence_length
vad_model = vad_model.to("cpu")
feats = torch.randn([1, sequence_length, 400])
waveform = torch.randn([1, 208832])
# result = vad_model(feats, waveform)

kwargs = model.vad_kwargs
nkwargs = {}
for k, v in kwargs.items():
    if k == "model":
        continue
    else:
        nkwargs[k] = v

vad_model = vad_model.export_my(**nkwargs)
in_cache0 = torch.randn(1, 128, 19, 1)
in_cache1 = torch.randn(1, 128, 19, 1)
in_cache2 = torch.randn(1, 128, 19, 1)
in_cache3 = torch.randn(1, 128, 19, 1)

inputs = (feats)
# script = torch.onnx.dynamo_export(vad_model, inputs, "fsmnVad.onnx", opset_version=16,
#                   input_names=["feats", "waveform"],
#                   output_names=["scores", "info"],
#                   dynamic_axes={'feats': {1: 'sequence_length'},
#                                 "waveform": {1: 'time_length'},
#                                 "info": {0: "seg_len"}
#                                 })
torch.onnx.export(vad_model, inputs, "fsmnVad.onnx", opset_version=16,
                  input_names=["feats"],
                  output_names=["scores"],
                  dynamic_axes={'feats': {1: 'sequence_length'},
                                })

# inputs = (feats, in_cache0, in_cache1, in_cache2, in_cache3)
# torch.onnx.export(vad_model, inputs, "model.onnx", export_params=True)
# torch.onnx.export(vad_model, inputs, "FSMN.onnx", export_params=True,
#                   input_names=["speech", "speech_lengths"],
#                   output_names=["results"],
#                   #   opset_version=15,
#                   dynamic_axes={'speech': {1: 'sequence_length'}, "speech_lengths": {0: 'sequence_length'}}
#                   )
