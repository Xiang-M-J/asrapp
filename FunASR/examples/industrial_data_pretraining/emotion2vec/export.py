import os

from funasr import AutoModel
import torch
# model="iic/emotion2vec_base"
# model="iic/emotion2vec_base_finetuned"
# model="iic/emotion2vec_plus_seed"
# model="iic/emotion2vec_plus_base"
model = r"D:\work\asrapp\FunASR-main\examples\industrial_data_pretraining\emotion2vec\emotion2vec_base_finetuned"

model = AutoModel(
    model=model,
    # vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    # vad_model_revision="master",
    # vad_kwargs={"max_single_segment_time": 2000},
)

emotion_model = model.model
emotion_model = emotion_model.to("cpu")
inputs = torch.randn([1, 16000])

torch.onnx.export(emotion_model, inputs, "model.onnx", input_names=["speech"], output_names=["x"], dynamic_axes={"speech": {1: "speech_length"}})

from onnxruntime.quantization import QuantType, quantize_dynamic
import onnx

model_path = "model.onnx"
quant_model_path = model_path.replace(".onnx", "_quant.onnx")
if not os.path.exists(quant_model_path):
    onnx_model = onnx.load(model_path)
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

