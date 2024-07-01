import sys

sys.path.append("../../..")

import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

from funasr import AutoModel

model = AutoModel(
    model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    # vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    # punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    # spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
)

main_model = model.model
punc_model = model.punc_model

vad_model = model.vad_model

# sequence_length = 218
# speech = torch.randn([1, sequence_length, 560])
# speech_lengths = torch.ones([1]) * sequence_length
# main_model = main_model.to("cpu")
# kwargs = model.kwargs
# nkwargs = {}
# for k, v in kwargs.items():
#     if k == "model":
#         continue
#     else:
#         nkwargs[k] = v
#
# main_model = main_model.export(**nkwargs)
# inputs = (speech, speech_lengths)
# torch.onnx.export(main_model, inputs, "BiCifParaformer.onnx", export_params=True,
#                   input_names=["speech", "speech_lengths"],
#                   output_names=["results"],
#                   #   opset_version=15,
#                   dynamic_axes={'speech': {1: 'sequence_length'}, "speech_lengths": {0: 'sequence_length'}}
#                   )

wav_len = 64000
speech = torch.randn([1, wav_len])
main_model = main_model.to("cpu")
kwargs = model.kwargs
nkwargs = {}
for k, v in kwargs.items():
    if k == "model":
        continue
    else:
        nkwargs[k] = v

main_model = main_model.export_wav(**nkwargs)
inputs = speech

onnx_program = torch.onnx.dynamo_export(main_model, inputs)
# torch.onnx.export(main_model, inputs, "BiCifParaformer_wav.onnx", export_params=True,
#                   input_names=["waveform"],
#                   output_names=["results"],
#                     opset_version=18,
#                   dynamic_axes={'waveform': {1: 'wav_len'}}
#                   )

onnx_program.save("BiCifParaformer_wav.onnx")
