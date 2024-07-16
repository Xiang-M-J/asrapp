#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)


# method1, inference from model hub


from funasr import AutoModel

model_path = r"D:\work\asrapp\FunASR-main\examples\industrial_data_pretraining\paraformer_streaming\iic\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"

model = AutoModel(
    model=model_path,
)

res = model.export(type="onnx", quantize=True)
print(res)

# method2, inference from local path
# from funasr import AutoModel
#
#
# model = AutoModel(
#     model="/Users/zhifu/.cache/modelscope/hub/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online"
# )
#
# res = model.export(type="onnx", quantize=False)
# print(res)
