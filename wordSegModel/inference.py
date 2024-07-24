import numpy as np
import torch
from utils.model import ErnieLinear

text = "近几年不但我用书给女儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书"

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
inputs = torch.from_numpy(text_list)
inputs.unsqueeze_(0)
model_dict = torch.load("models/checkpoint/model.pt")
model = ErnieLinear(2)
model.load_state_dict(model_dict)
model.eval()
y = model(inputs)
logits = y[1].detach().numpy()
res = np.argmax(logits, axis=-1)[1:-1]
for i in range(len(res)):
    if res[i] == 1:
        print(text[i]+" ", end="")
    else:
        print(text[i], end="")
# print(res)
import jieba
seg_list  = (jieba.cut(text))
print(" ".join(seg_list))