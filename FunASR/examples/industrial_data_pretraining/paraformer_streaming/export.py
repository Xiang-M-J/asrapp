#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
import torch

# method1, inference from model hub


from funasr import AutoModel

model_path = r"D:\work\asrapp\FunASR-main\examples\industrial_data_pretraining\paraformer_streaming\iic\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"

model = AutoModel(
    model=model_path,
)


def _onnx(
        model,
        quantize: bool = False,
        opset_version: int = 14,
        **kwargs,
):
    do_constant_folding = kwargs.get("do_constant_folding", True)
    keep_initializers_as_inputs = kwargs.get("keep_initializers_as_inputs", None)
    export_modules_as_functions = kwargs.get("export_modules_as_functions", False)
    dummy_input = model.export_dummy_inputs()

    verbose = kwargs.get("verbose", False)

    export_name = model.export_name()
    if export_modules_as_functions:
        opset_version = 16
    torch.onnx.export(
        model,
        dummy_input,
        export_name,
        verbose=verbose,
        opset_version=opset_version,
        input_names=model.export_input_names(),
        output_names=model.export_output_names(),
        dynamic_axes=model.export_dynamic_axes(),
        do_constant_folding=do_constant_folding, keep_initializers_as_inputs=keep_initializers_as_inputs,
        export_modules_as_functions=export_modules_as_functions,
    )

    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        import onnx

        quant_model_path = export_name.replace(".onnx", "_quant.onnx")

        onnx_model = onnx.load(export_name)
        nodes = [n.name for n in onnx_model.graph.node]
        nodes_to_exclude = [
            m for m in nodes if "output" in m or "bias_encoder" in m or "bias_decoder" in m
        ]
        quantize_dynamic(
            model_input=model_path,
            model_output=quant_model_path,
            op_types_to_quantize=["MatMul"],
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
            nodes_to_exclude=nodes_to_exclude,
        )


model = model.model.to(device='cpu')
model = model.export()
inputs = model.export_dummy_inputs()
model(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8])
model.eval()

script_module = torch.jit.script(model)

_onnx(script_module, quantize=True)

# res = model.export(type="onnx", quantize=True)
# print(res)

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
