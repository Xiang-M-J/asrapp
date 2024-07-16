#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import types
import torch
from funasr.register import tables


def export_rebuild_model(model, **kwargs):
    model.device = "cpu"
    is_onnx = "onnx"
    # model_conf = {'ctc_weight': 0.0, 'length_normalized_loss': True, 'lsm_weight': 0.1, 'predictor_bias': 1, 'predictor_weight': 1.0,
    #  'sampling_ratio': 0.75}
    # specaug_conf = {'apply_freq_mask': True, 'apply_time_mask': True, 'apply_time_warp': False, 'freq_mask_width_range': [0, 30],
    #  'lfr_rate': 6, 'num_freq_mask': 1, 'num_time_mask': 1, 'time_mask_width_range': [0, 12],
    #  'time_warp_mode': 'bicubic', 'time_warp_window': 5}
    # decoder_conf = {'att_layer_num': 16, 'attention_heads': 4, 'dropout_rate': 0.1, 'kernel_size': 11, 'linear_units': 2048, 'num_blocks': 16, 'positional_dropout_rate': 0.1, 'sanm_shfit': 5, 'self_attention_dropout_rate': 0.1, 'src_attention_dropout_rate': 0.1}
    # predictor_conf = {'idim': 512, 'l_order': 1, 'r_order': 1, 'tail_threshold': 0.45, 'threshold': 1.0}
    # encoder_class = tables.encoder_classes.get(kwargs["encoder"])
    # model.encoder = encoder_class(model.encoder, onnx=is_onnx)
    #
    # predictor_class = tables.predictor_classes.get(kwargs["predictor"])
    # model.predictor = predictor_class(model.predictor, onnx=is_onnx)
    # decoder_class = tables.decoder_classes.get(kwargs["decoder"])
    # model.decoder = decoder_class(model.decoder, onnx=is_onnx)

    # from funasr.utils.torch_function import sequence_mask
    #
    # model.make_pad_mask = sequence_mask(kwargs["max_seq_len"], flip=False)

    model.forward = types.MethodType(export_forward, model)
    model.export_dummy_inputs = types.MethodType(export_dummy_inputs, model)
    model.export_input_names = types.MethodType(export_input_names, model)
    model.export_output_names = types.MethodType(export_output_names, model)
    model.export_dynamic_axes = types.MethodType(export_dynamic_axes, model)
    model.export_name = types.MethodType(export_name, model)

    return model


def export_forward(self, speech: torch.Tensor,
                   speech_lengths: torch.Tensor, feats, cif_hidden, cif_alphas):

    return self.forward_export(speech, speech_lengths, feats, cif_hidden, cif_alphas)


def export_dummy_inputs(self):
    speech = torch.randn(1, 10, 560)
    speech_lengths = torch.tensor([10], dtype=torch.int32)
    feats = torch.zeros(1, 5, 560)
    cif_hidden = torch.randn(1, 1, 512)
    cif_alphas = torch.randn(1, 1)
    # flag = torch.ones([6], dtype=torch.int32)
    # encoder_opt = torch.randn(50, 2, 2, 4, 10, 128)
    # decoder_opt = torch.randn(16, 2, 2, 4, 10, 128)
    # decode_fsmn = torch.randn(16, 2, 512, 12)
    return speech, speech_lengths, feats, cif_hidden, cif_alphas


def export_input_names(self):
    return ["speech", "speech_lengths", "feats", "cif_hidden", "cif_alphas"]


def export_dynamic_axes(self):
    return {
        "speech": {0: "batch_size", 1: "feats_length"},
        "speech_lengths": {
            0: "batch_size",
        },
        "feats": {0: "batch_size"},
        "cif_hidden": {0: "batch_size"},
        "cif_alphas": {0: "batch_size"},
        # "encoder_opt": {2: "batch_size"},
        # "decoder_opt": {2: "batch_size"},
        # "decode_fsmn": {1: "batch_size", 3: "fsmn_length"},
        "decoder_out": {0: "batch_size"},
        "pre_token_length": {0: "batch_size"},
        "feats_o": {0: "batch_size", 1: "feats_length"},
        "cif_hidden_o": {0: "batch_size"},
        "cif_alphas_o": {0: "batch_size"},
        # "encoder_opt_o": {2: "batch_size"},
        # "decoder_opt_o": {2: "batch_size"},
        # "decode_fsmn_o": {1: "batch_size"},
    }


def export_name(self):
    return "totalModel.onnx"


def export_output_names(self):
    return ["decoder_out", "pre_token_length", "feats_o", "cif_hidden_o", "cif_alphas_o"]


# def export_rebuild_model(model, **kwargs):
#     # self.device = kwargs.get("device")
#     is_onnx = kwargs.get("type", "onnx") == "onnx"
#     encoder_class = tables.encoder_classes.get(kwargs["encoder"] + "Export")
#     model.encoder = encoder_class(model.encoder, onnx=is_onnx)
#
#     predictor_class = tables.predictor_classes.get(kwargs["predictor"] + "Export")
#     model.predictor = predictor_class(model.predictor, onnx=is_onnx)
#
#     if kwargs["decoder"] == "ParaformerSANMDecoder":
#         kwargs["decoder"] = "ParaformerSANMDecoderOnline"
#     decoder_class = tables.decoder_classes.get(kwargs["decoder"] + "Export")
#     model.decoder = decoder_class(model.decoder, onnx=is_onnx)
#
#     from funasr.utils.torch_function import sequence_mask
#
#     model.make_pad_mask = sequence_mask(max_seq_len=None, flip=False)
#
#     import copy
#     import types
#
#     encoder_model = copy.copy(model)
#     decoder_model = copy.copy(model)
#
#     # encoder
#     encoder_model.forward = types.MethodType(export_encoder_forward, encoder_model)
#     encoder_model.export_dummy_inputs = types.MethodType(export_encoder_dummy_inputs, encoder_model)
#     encoder_model.export_input_names = types.MethodType(export_encoder_input_names, encoder_model)
#     encoder_model.export_output_names = types.MethodType(export_encoder_output_names, encoder_model)
#     encoder_model.export_dynamic_axes = types.MethodType(export_encoder_dynamic_axes, encoder_model)
#     encoder_model.export_name = types.MethodType(export_encoder_name, encoder_model)
#
#     # decoder
#     decoder_model.forward = types.MethodType(export_decoder_forward, decoder_model)
#     decoder_model.export_dummy_inputs = types.MethodType(export_decoder_dummy_inputs, decoder_model)
#     decoder_model.export_input_names = types.MethodType(export_decoder_input_names, decoder_model)
#     decoder_model.export_output_names = types.MethodType(export_decoder_output_names, decoder_model)
#     decoder_model.export_dynamic_axes = types.MethodType(export_decoder_dynamic_axes, decoder_model)
#     decoder_model.export_name = types.MethodType(export_decoder_name, decoder_model)
#
#     return encoder_model, decoder_model


def export_encoder_forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
):
    # a. To device
    batch = {"speech": speech, "speech_lengths": speech_lengths, "online": True}
    # batch = to_device(batch, device=self.device)

    enc, enc_len = self.encoder(**batch)
    mask = self.make_pad_mask(enc_len)[:, None, :]
    alphas, _ = self.predictor.forward_cnn(enc, mask)

    return enc, enc_len, alphas


def export_encoder_dummy_inputs(self):
    speech = torch.randn(2, 30, 560)
    speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
    return (speech, speech_lengths)


def export_encoder_input_names(self):
    return ["speech", "speech_lengths"]


def export_encoder_output_names(self):
    return ["enc", "enc_len", "alphas"]


def export_encoder_dynamic_axes(self):
    return {
        "speech": {0: "batch_size", 1: "feats_length"},
        "speech_lengths": {
            0: "batch_size",
        },
        "enc": {0: "batch_size", 1: "feats_length"},
        "enc_len": {
            0: "batch_size",
        },
        "alphas": {0: "batch_size", 1: "feats_length"},
    }


def export_encoder_name(self):
    return "model.onnx"


def export_decoder_forward(
        self,
        enc: torch.Tensor,
        enc_len: torch.Tensor,
        acoustic_embeds: torch.Tensor,
        acoustic_embeds_len: torch.Tensor,
        *args,
):
    decoder_out, out_caches = self.decoder(
        enc, enc_len, acoustic_embeds, acoustic_embeds_len, *args
    )
    sample_ids = decoder_out.argmax(dim=-1)

    return decoder_out, sample_ids, out_caches


def export_decoder_dummy_inputs(self):
    dummy_inputs = self.decoder.get_dummy_inputs(enc_size=self.encoder._output_size)
    return dummy_inputs


def export_decoder_input_names(self):
    return self.decoder.get_input_names()


def export_decoder_output_names(self):
    return self.decoder.get_output_names()


def export_decoder_dynamic_axes(self):
    return self.decoder.get_dynamic_axes()


def export_decoder_name(self):
    return "decoder.onnx"
