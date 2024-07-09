#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import types
import torch
from funasr.register import tables


def export_rebuild_model(model, **kwargs):
    is_onnx = kwargs.get("type", "onnx") == "onnx"
    encoder_class = tables.encoder_classes.get(kwargs["encoder"] + "Export")
    model.encoder = encoder_class(model.encoder, onnx=is_onnx)

    model.forward = types.MethodType(export_forward, model)
    model.export_dummy_inputs = types.MethodType(export_dummy_inputs, model)
    model.export_input_names = types.MethodType(export_input_names, model)
    model.export_output_names = types.MethodType(export_output_names, model)
    model.export_dynamic_axes = types.MethodType(export_dynamic_axes, model)
    model.export_name = types.MethodType(export_name, model)

    return model


def export_rebuild_model_my(model, **kwargs):
    is_onnx = kwargs.get("type", "onnx") == "onnx"
    encoder_class = tables.encoder_classes.get(kwargs["encoder"] + "Export")
    model.encoder = encoder_class(model.encoder, onnx=is_onnx)

    model.forward = types.MethodType(export_forward, model)
    model.export_dummy_inputs = types.MethodType(export_dummy_inputs, model)
    model.export_input_names = types.MethodType(export_input_names, model)
    model.export_output_names = types.MethodType(export_output_names, model)
    model.export_dynamic_axes = types.MethodType(export_dynamic_axes, model)
    model.export_name = types.MethodType(export_name, model)

    return model


def export_forward(self, feats: torch.Tensor, *args, **kwargs):
    scores, out_caches = self.encoder(feats, *args)

    return scores, out_caches


def export_forward_my(self, feats, waveform):
    is_final = False
    cache = {}
    if len(cache) == 0:
        self.init_cache(cache)
    cache["stats"].waveform = waveform
    is_streaming_input = False
    self.ComputeDecibel(cache=cache)
    self.ComputeScores(feats, cache=cache)
    self.DetectLastFrames(cache=cache)
    segments = []
    for batch_num in range(0, feats.shape[0]):  # only support batch_size = 1 now
        segment_batch = []
        if len(cache["stats"].output_data_buf) > 0:
            for i in range(
                    cache["stats"].output_data_buf_offset, len(cache["stats"].output_data_buf)
            ):
                if (
                        is_streaming_input
                ):  # in this case, return [beg, -1], [], [-1, end], [beg, end]
                    if not cache["stats"].output_data_buf[i].contain_seg_start_point:
                        continue
                    if (
                            not cache["stats"].next_seg
                            and not cache["stats"].output_data_buf[i].contain_seg_end_point
                    ):
                        continue
                    start_ms = (
                        cache["stats"].output_data_buf[i].start_ms
                        if cache["stats"].next_seg
                        else -1
                    )
                    if cache["stats"].output_data_buf[i].contain_seg_end_point:
                        end_ms = cache["stats"].output_data_buf[i].end_ms
                        cache["stats"].next_seg = True
                        cache["stats"].output_data_buf_offset += 1
                    else:
                        end_ms = -1
                        cache["stats"].next_seg = False
                    segment = [start_ms, end_ms]

                else:  # in this case, return [beg, end]

                    if not is_final and (
                            not cache["stats"].output_data_buf[i].contain_seg_start_point
                            or not cache["stats"].output_data_buf[i].contain_seg_end_point
                    ):
                        continue
                    segment = [
                        cache["stats"].output_data_buf[i].start_ms,
                        cache["stats"].output_data_buf[i].end_ms,
                    ]
                    cache["stats"].output_data_buf_offset += 1  # need update this parameter

                segment_batch.append(segment)

        if segment_batch:
            segments.append(segment_batch)
    return torch.tensor(segments, dtype=torch.int64)


def export_dummy_inputs(self, data_in=None, frame=30):
    if data_in is None:
        speech = torch.randn(1, frame, self.encoder_conf.get("input_dim"))
    else:
        speech = None  # Undo

    cache_frames = self.encoder_conf.get("lorder") + self.encoder_conf.get("rorder") - 1
    in_cache0 = torch.randn(1, self.encoder_conf.get("proj_dim"), cache_frames, 1)
    in_cache1 = torch.randn(1, self.encoder_conf.get("proj_dim"), cache_frames, 1)
    in_cache2 = torch.randn(1, self.encoder_conf.get("proj_dim"), cache_frames, 1)
    in_cache3 = torch.randn(1, self.encoder_conf.get("proj_dim"), cache_frames, 1)

    return (speech, in_cache0, in_cache1, in_cache2, in_cache3)

def export_dummy_inputs

def export_input_names(self):
    return ["speech", "in_cache0", "in_cache1", "in_cache2", "in_cache3"]


def export_output_names(self):
    return ["logits", "out_cache0", "out_cache1", "out_cache2", "out_cache3"]


def export_dynamic_axes(self):
    return {
        "speech": {1: "feats_length"},
    }


def export_name(
        self,
):
    return "model.onnx"
