import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

model = torch.load("exp/cam++0801/model_final.pth")
model.eval()
encoder = model.encoder
encoder = encoder.to("cpu")


def extract_audio(path: str):
    wav, _ = torchaudio.load(path)
    feats = kaldi.fbank(wav, num_mel_bins=80)
    feats = feats - feats.mean(0, keepdim=True)
    return feats


def cosine_similarity(vec1, vec2):
    ab = 0
    for i in range(vec1.shape[-1]):
        ab += vec1[i] * vec2[i]
    a_s = torch.norm(vec1)
    b_s = torch.norm(vec2)
    print(ab / (a_s * b_s + 1e-5))


# audio1 = r"D:\work\asrapp\FunASR-main\examples\industrial_data_pretraining\bicif_paraformer\asr_example.wav"
# audio2 = r"D:\work\asrapp\FunASR-main\examples\industrial_data_pretraining\bicif_paraformer\test.wav"
audio1 = r"audios/A2_2.wav"
audio2 = r"audios/D7_858.wav"
e1 = encoder(extract_audio(audio1).unsqueeze(0))
e2 = encoder(extract_audio(audio2).unsqueeze(0))
cosine_similarity(e1.squeeze(0), e2.squeeze(0))
