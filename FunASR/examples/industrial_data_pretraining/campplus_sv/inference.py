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
audio_list = ["A2_1.wav", "A2_2.wav", "A2_3.wav","D7_858.wav", "D7_859.wav", "D7_860.wav"]

embedding = []
for audio in audio_list:
    wav, _ = torchaudio.load(fr"D:\work\asrapp\FunASR\examples\industrial_data_pretraining\campplus\{audio}")
    fbank = Kaldi.fbank(wav, num_mel_bins=80)
    fbank = fbank - torch.mean(fbank, dim=0, keepdim=True)
    fbank.unsqueeze_(0)
    y = ort_session.run(None, {"feats": fbank.numpy()})
    embedding.append(y)

sim = torch.cosine_similarity(embedding[0], embedding[1])