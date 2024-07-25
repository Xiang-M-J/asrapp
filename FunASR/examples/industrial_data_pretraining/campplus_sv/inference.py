import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as Kaldi
ort_session = onnxruntime.InferenceSession("model.onnx")

audio, sr = torchaudio.load(r"D:\work\asrapp\FunASR\examples\industrial_data_pretraining\bicif_paraformer\test.wav")

fbank = Kaldi.fbank(audio, num_mel_bins=80)
fbank = fbank - torch.mean(fbank, dim=0, keepdim=True)
fbank.unsqueeze_(0)
y = ort_session.run(None, {"feats": fbank.numpy()})
print(y)
