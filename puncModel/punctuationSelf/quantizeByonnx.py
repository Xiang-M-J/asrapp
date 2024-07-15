import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'models/pun_models/model.onnx'
model_quant = 'models/pun_models/model.quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant,
                                   op_types_to_quantize=["MatMul"],
                                   per_channel=True,
                                   reduce_range=False,
                                   )
