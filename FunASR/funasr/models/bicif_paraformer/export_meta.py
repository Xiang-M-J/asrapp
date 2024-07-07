#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch
import types

from funasr.register import tables
from funasr.utils.load_utils import extract_fbank


def export_rebuild_model(model, **kwargs):
    is_onnx = kwargs.get("type", "onnx") == "onnx"
    encoder_class = tables.encoder_classes.get(kwargs["encoder"] + "Export")
    model.encoder = encoder_class(model.encoder, onnx=is_onnx)

    predictor_class = tables.predictor_classes.get(kwargs["predictor"] + "Export")
    model.predictor = predictor_class(model.predictor, onnx=is_onnx)

    decoder_class = tables.decoder_classes.get(kwargs["decoder"] + "Export")
    model.decoder = decoder_class(model.decoder, onnx=is_onnx)

    from funasr.utils.torch_function import sequence_mask

    model.make_pad_mask = sequence_mask(kwargs["max_seq_len"], flip=False)

    model.forward = types.MethodType(export_forward, model)
    model.export_dummy_inputs = types.MethodType(export_dummy_inputs, model)
    model.export_input_names = types.MethodType(export_input_names, model)
    model.export_output_names = types.MethodType(export_output_names, model)
    model.export_dynamic_axes = types.MethodType(export_dynamic_axes, model)

    model.export_name = "model"

    return model


def export_rebuild_model_wav(model, **kwargs):
    is_onnx = kwargs.get("type", "onnx") == "onnx"
    encoder_class = tables.encoder_classes.get(kwargs["encoder"] + "Export")
    model.encoder = encoder_class(model.encoder, onnx=is_onnx)

    predictor_class = tables.predictor_classes.get(kwargs["predictor"] + "Export")
    model.predictor = predictor_class(model.predictor, onnx=is_onnx)

    decoder_class = tables.decoder_classes.get(kwargs["decoder"] + "Export")
    model.decoder = decoder_class(model.decoder, onnx=is_onnx)

    from funasr.utils.torch_function import sequence_mask

    model.make_pad_mask = sequence_mask(kwargs["max_seq_len"], flip=False)

    model.forward = types.MethodType(export_forward_wav, model)
    model.export_dummy_inputs = types.MethodType(export_dummy_inputs_wav, model)
    model.export_input_names = types.MethodType(export_input_names_wav, model)
    model.export_output_names = types.MethodType(export_output_names_wav, model)
    model.export_dynamic_axes = types.MethodType(export_dynamic_axes_wav, model)

    model.export_name = "model"

    return model


def export_forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
):
    # a. To device
    batch = {"speech": speech, "speech_lengths": speech_lengths}

    enc, enc_len = self.encoder(**batch)
    mask = self.make_pad_mask(enc_len)[:, None, :]
    pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(enc, mask)
    pre_token_length = pre_token_length.round().type(torch.int32)

    decoder_out, _ = self.decoder(enc, enc_len, pre_acoustic_embeds, pre_token_length)
    decoder_out = torch.log_softmax(decoder_out, dim=-1)

    # get predicted timestamps
    us_alphas, us_cif_peak = self.predictor.get_upsample_timestmap(enc, mask, pre_token_length)

    results = []
    b, n, d = decoder_out.size()
    for i in range(b):
        am_scores = decoder_out[i]
        yseq = am_scores.argmax(dim=-1)
        if yseq.shape[0] == 1:
            yseq = yseq[0]

        token_int = [s for s in yseq if s not in [0, 1, 2]]
        results.append(token_int)
    return decoder_out


def export_forward_wav(
        self,
        waveform: torch.Tensor,
        **kwargs
):
    # a. To device
    # batch = {"waveform": waveform}
    frontend = "WavFrontend"
    kwargs["input_size"] = None
    frontend_class = tables.frontend_classes.get(frontend)
    frontend_conf = {'cmvn_file': 'iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch\\am.mvn', 'frame_length': 25, 'frame_shift': 10, 'fs': 16000, 'lfr_m': 7, 'lfr_n': 6, 'n_mels': 80, 'window': 'hamming'}
    frontend = frontend_class(**frontend_conf)
    kwargs["input_size"] = (
        frontend.output_size() if hasattr(frontend, "output_size") else None
    )

    speech, speech_lengths = extract_fbank(
        waveform, data_type=kwargs.get("data_type", "sound"), frontend=frontend
    )

    speech = speech.to(waveform.device)
    speech_lengths = speech_lengths.to(waveform.device)
    enc, enc_len = self.encoder(speech, speech_lengths)
    mask = self.make_pad_mask(enc_len)[:, None, :]
    pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(enc, mask)
    pre_token_length = pre_token_length.round().type(torch.int32)

    decoder_out, _ = self.decoder(enc, enc_len, pre_acoustic_embeds, pre_token_length)
    decoder_out = torch.log_softmax(decoder_out, dim=-1)

    # get predicted timestamps
    us_alphas, us_cif_peak = self.predictor.get_upsample_timestmap(enc, mask, pre_token_length)

    results = []
    b, n, d = decoder_out.size()
    for i in range(b):
        am_scores = decoder_out[i, : pre_token_length[i], :]
        yseq = am_scores.argmax(dim=-1)
        token_int = list(
            filter(
                lambda x: x != self.eos and x != self.sos and x != self.blank_id, yseq
            )
        )

        results.append(token_int)
    return results

def export_dummy_inputs(self):
    speech = torch.randn(2, 30, 560)
    speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
    return (speech, speech_lengths)


def export_input_names(self):
    return ["speech", "speech_lengths"]


def export_dummy_inputs_wav(self):
    speech = torch.randn(1, 16000)
    return speech


def export_input_names_wav(self):
    return ["waveform"]

def export_output_names_wav(self):
    return ["results"]


def export_dynamic_axes_wav(self):
    return {
        "waveform": {0: "batch_size", 1: "feats_length"},
    }

def export_output_names(self):
    return ["logits", "token_num", "us_alphas", "us_cif_peak"]


def export_dynamic_axes(self):
    return {
        "speech": {0: "batch_size", 1: "feats_length"},
        "speech_lengths": {
            0: "batch_size",
        },
        "logits": {0: "batch_size", 1: "logits_length"},
        "us_alphas": {0: "batch_size", 1: "alphas_length"},
        "us_cif_peak": {0: "batch_size", 1: "alphas_length"},
    }


def export_name(self):
    return "model.onnx"
