import onnxruntime
import torch
import torchaudio

ort_session = onnxruntime.InferenceSession('model_quant.onnx')
model_path = r"D:\work\asrapp\FunASR-main\examples\industrial_data_pretraining\emotion2vec\emotion2vec_base_finetuned"
wav_file = f"{model_path}/example/sad_1_015.wav"


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


import time

t1 = time.time()
x, sr = torchaudio.load(wav_file)
x = (x - torch.mean(x)) / torch.sqrt(torch.var(x) + 1e-5)
y = ort_session.run(None, {ort_session.get_inputs()[0].name: to_numpy(x)})
t2 = time.time()
print(t2 - t1)
print(y)
