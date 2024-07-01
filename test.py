import torch
import torchaudio
import onnx
import onnxruntime

# 加载预训练的 wav2vec2 模型
# model = onnx.load("wav2vec2-base-960h.onnx")
# model.eval()

# 加载音频并预处理
waveform, sample_rate = torchaudio.load(r'D:\work\asrapp\onnx\assets\audio\96.wav')
assert sample_rate == 16000, "Wav2Vec 2.0 模型需要 16kHz 的音频"


ort_session = onnxruntime.InferenceSession("wav2vec2-base-960h.onnx")

ort_inputs = {ort_session.get_inputs()[0].name: waveform.numpy()}
ort_outs = ort_session.run(None, ort_inputs)
ort_out = ort_outs[0][0]

# 使用贪婪解码将模型输出转换为文本
def greedy_decode(logits, labels):
    predicted_ids = torch.argmax(logits, dim=-1)
    tokens = [labels[id] for id in predicted_ids]
    # 移除连续重复的标签
    decoded_output = []
    prev_token = None
    for token in tokens:
        if token != prev_token:
            decoded_output.append(token)
            prev_token = token
    # 移除特殊标记
    decoded_output = [token for token in decoded_output if token not in ('<s>', '<pad>', '</s>', '<unk>')]
    return ''.join(decoded_output)

# 定义标签
labels = [
  "<s>", "<pad>", "</s>", "<unk>", "|", "e", "t", "a", "o", "n", "i", "h", "s", "r", "d", "l", "u", "m", "w", "c", "f",
  "g", "y", "p", "b", "v", "k", "'", "x", "j", "q", "z",
]

# 解码输出
transcription = greedy_decode(torch.from_numpy(ort_out), labels)
print(transcription)
