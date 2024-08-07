#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common")

res = model.generate(
    input=r"D:\work\asrapp\FunASR\examples\industrial_data_pretraining\bicif_paraformer\test.wav"
)
print(res)
