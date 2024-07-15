import onnxruntime
import numpy as np

ort_session = onnxruntime.InferenceSession("models/pun_models/model.quant.onnx", providers=["CPUExecutionProvider"])

text = "近几年不但我用书给女儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书"

punc_list = ["", "，", "。", "？", "！", "、"]
with open("ernie-3.0-nano-zh/vocab.txt", 'r', encoding="utf-8") as f:
    vocab = f.read().splitlines()
text_list = [1]

for i in range(len(text)):
    idx = vocab.index(text[i])
    if idx == -1:
        text_list.append(len(vocab) - 1)
    else:
        text_list.append(idx)
text_list.append(2)
# print(text_list)
text_list = np.array(text_list, dtype=np.int64)

text_list = np.expand_dims(text_list, axis=0)
token_ids = np.zeros([1, len(text_list)], np.int64)
inputs = {"input_ids": text_list, "token_type_ids": token_ids}

output = ort_session.run(None, inputs)


def decode(text_list, output):
    decoded_text = ""
    for i in range(len(text_list)):
        decoded_text += vocab[text_list[i]]
        decoded_text += punc_list[output[i]]
    return decoded_text


print(decode(text_list[0], output[0]))
