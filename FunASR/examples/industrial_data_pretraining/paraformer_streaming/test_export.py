import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class TestModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

    def forward(self, x, cache):
        if torch.any(cache):
            x = x + cache
        else:
            x = x + 1
        # x1 = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x1))
        return x, cache


script_module = torch.jit.script(TestModel())

test_model = TestModel()
test_model.eval()

# input = {"x": torch.randn(1, 1, 28, 28), "cache": torch.zeros(1, 1, 28, 28)}
# torch.onnx.export(script_module, input, "test.onnx", input_names=['x', "cache"], output_names=['output', "newcache"])

import onnxruntime

ort_session = onnxruntime.InferenceSession("test.onnx")

r = ort_session.run(None, {"x": 2* np.ones([1, 1, 28, 28], dtype=np.float32), "cache": np.ones([1, 1, 28, 28], dtype=np.float32)})
print(r)
