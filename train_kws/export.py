import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
import onnx

model = torch.load("exp/cam++0801/model_final.pth")
model_path = "model_1.onnx"

model = model.encoder.to("cpu")
model.eval()
inputs = torch.randn(1, 128, 80)

torch.onnx.export(model, inputs, model_path, input_names=["feats"], output_names=["embedding"],
                  dynamic_axes={"feats": {1: "seq_len"}})

quant_model_path = model_path.replace(".onnx", "_quant.onnx")
onnx_model = onnx.load(model_path)
nodes = [n.name for n in onnx_model.graph.node]
nodes_to_exclude = [
    m for m in nodes if "output" in m or "bias_encoder" in m or "bias_decoder" in m
]
quantize_dynamic(
    model_input=model_path,
    model_output=quant_model_path,
    # op_types_to_quantize=["MatMul"],
    per_channel=True,
    reduce_range=False,
    weight_type=QuantType.QUInt8,
    nodes_to_exclude=nodes_to_exclude,
)
