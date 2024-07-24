import torch
import torch.nn as nn
from transformers import ErnieForTokenClassification


class ErnieLinear(nn.Module):
    def __init__(self,
                 num_classes,
                 pretrained_token='ernie-3.0-nano-zh'):
        super(ErnieLinear, self).__init__()
        self.ernie = ErnieForTokenClassification.from_pretrained(pretrained_token)
        self.ernie.classifier = nn.Linear(in_features=312, out_features=num_classes, bias=True)
        # self.ernie.num_labels = num_classes
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, token_type_ids=None):
        y = self.ernie(input_ids, token_type_ids=token_type_ids).logits

        y = torch.reshape(y, shape=[-1, self.num_classes])
        logits = self.softmax(y)

        return y, logits


class ErnieLinearExport(nn.Module):
    def __init__(self,
                 num_classes,
                 pretrained_token='ernie-3.0-medium-zh'):
        super(ErnieLinearExport, self).__init__()
        self.ernie = ErnieForTokenClassification.from_pretrained(pretrained_token)
        self.ernie.classifier = nn.Linear(in_features=312, out_features=num_classes, bias=True)
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, token_type_ids=None):
        y = self.ernie(input_ids, token_type_ids=token_type_ids).logits

        y = torch.reshape(y, shape=[-1, self.num_classes])
        logits = self.softmax(y)

        preds = torch.argmax(logits, dim=-1)

        return preds
