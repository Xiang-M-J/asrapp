import math
from typing import Tuple, Dict
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from funasr.models.fsmn_vad_streaming.model import WindowDetector, Stats, VadStateMachine, FrameState, E2EVadFrameProb, \
    VadDetectMode, AudioChangeState, VADXOptions, E2EVadSpeechBufWithDoa
from funasr.register import tables


class LinearTransform(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, input):
        output = self.linear(input)

        return output


class AffineTransform(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(AffineTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        output = self.linear(input)

        return output


class RectifiedLinear(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(RectifiedLinear, self).__init__()
        self.dim = input_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        out = self.relu(input)
        return out


class FSMNBlock(nn.Module):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            lorder=None,
            rorder=None,
            lstride=1,
            rstride=1,
    ):
        super(FSMNBlock, self).__init__()

        self.dim = input_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride

        self.conv_left = nn.Conv2d(
            self.dim, self.dim, [lorder, 1], dilation=[lstride, 1], groups=self.dim, bias=False
        )

        if self.rorder > 0:
            self.conv_right = nn.Conv2d(
                self.dim, self.dim, [rorder, 1], dilation=[rstride, 1], groups=self.dim, bias=False
            )
        else:
            self.conv_right = None

    def forward(self, input: torch.Tensor, cache: torch.Tensor):
        x = torch.unsqueeze(input, 1)
        x_per = x.permute(0, 3, 2, 1)  # B D T C

        cache = cache.to(x_per.device)
        y_left = torch.cat((cache, x_per), dim=2)
        cache = y_left[:, :, -(self.lorder - 1) * self.lstride:, :]
        y_left = self.conv_left(y_left)
        out = x_per + y_left

        if self.conv_right is not None:
            # maybe need to check
            y_right = F.pad(x_per, [0, 0, 0, self.rorder * self.rstride])
            y_right = y_right[:, :, self.rstride:, :]
            y_right = self.conv_right(y_right)
            out += y_right

        out_per = out.permute(0, 3, 2, 1)
        output = out_per.squeeze(1)

        return output, cache


class BasicBlock(nn.Module):
    def __init__(
            self,
            linear_dim: int,
            proj_dim: int,
            lorder: int,
            rorder: int,
            lstride: int,
            rstride: int,
            stack_layer: int,
    ):
        super(BasicBlock, self).__init__()
        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride
        self.stack_layer = stack_layer
        self.linear = LinearTransform(linear_dim, proj_dim)
        self.fsmn_block = FSMNBlock(proj_dim, proj_dim, lorder, rorder, lstride, rstride)
        self.affine = AffineTransform(proj_dim, linear_dim)
        self.relu = RectifiedLinear(linear_dim, linear_dim)

    def forward(self, input: torch.Tensor, cache: Dict[str, torch.Tensor]):
        x1 = self.linear(input)  # B T D
        cache_layer_name = "cache_layer_{}".format(self.stack_layer)
        if cache_layer_name not in cache:
            cache[cache_layer_name] = torch.zeros(
                x1.shape[0], x1.shape[-1], (self.lorder - 1) * self.lstride, 1
            )
        x2, cache[cache_layer_name] = self.fsmn_block(x1, cache[cache_layer_name])
        x3 = self.affine(x2)
        x4 = self.relu(x3)
        return x4


class BasicBlock_export(nn.Module):
    def __init__(
            self,
            model,
    ):
        super(BasicBlock_export, self).__init__()
        self.linear = model.linear
        self.fsmn_block = model.fsmn_block
        self.affine = model.affine
        self.relu = model.relu

    def forward(self, input: torch.Tensor, in_cache: torch.Tensor):
        x = self.linear(input)  # B T D
        # cache_layer_name = 'cache_layer_{}'.format(self.stack_layer)
        # if cache_layer_name not in in_cache:
        #     in_cache[cache_layer_name] = torch.zeros(x1.shape[0], x1.shape[-1], (self.lorder - 1) * self.lstride, 1)
        x, out_cache = self.fsmn_block(x, in_cache)
        x = self.affine(x)
        x = self.relu(x)
        return x, out_cache


class FsmnStack(nn.Sequential):
    def __init__(self, *args):
        super(FsmnStack, self).__init__(*args)

    def forward(self, input: torch.Tensor, cache: Dict[str, torch.Tensor]):
        x = input
        for module in self._modules.values():
            x = module(x, cache)
        return x


"""
FSMN net for keyword spotting
input_dim:              feats dimension
linear_dim:             fsmn feats dimensionll
proj_dim:               fsmn projection dimension
lorder:                 fsmn left order
rorder:                 fsmn right order
num_syn:                output dimension
fsmn_layers:            no. of sequential fsmn layers
"""


@tables.register("encoder_classes", "FSMN")
class FSMN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            input_affine_dim: int,
            fsmn_layers: int,
            linear_dim: int,
            proj_dim: int,
            lorder: int,
            rorder: int,
            lstride: int,
            rstride: int,
            output_affine_dim: int,
            output_dim: int,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.input_affine_dim = input_affine_dim
        self.fsmn_layers = fsmn_layers
        self.linear_dim = linear_dim
        self.proj_dim = proj_dim
        self.output_affine_dim = output_affine_dim
        self.output_dim = output_dim

        self.in_linear1 = AffineTransform(input_dim, input_affine_dim)
        self.in_linear2 = AffineTransform(input_affine_dim, linear_dim)
        self.relu = RectifiedLinear(linear_dim, linear_dim)
        self.fsmn = FsmnStack(
            *[
                BasicBlock(linear_dim, proj_dim, lorder, rorder, lstride, rstride, i)
                for i in range(fsmn_layers)
            ]
        )
        self.out_linear1 = AffineTransform(linear_dim, output_affine_dim)
        self.out_linear2 = AffineTransform(output_affine_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def fuse_modules(self):
        pass

    def forward(
            self, input: torch.Tensor, cache: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            input (torch.Tensor): Input tensor (B, T, D)
            cache: when cache is not None, the forward is in streaming. The type of cache is a dict, egs,
            {'cache_layer_1': torch.Tensor(B, T1, D)}, T1 is equal to self.lorder. It is {} for the 1st frame
        """

        x1 = self.in_linear1(input)
        x2 = self.in_linear2(x1)
        x3 = self.relu(x2)
        x4 = self.fsmn(x3, cache)  # self.cache will update automatically in self.fsmn
        x5 = self.out_linear1(x4)
        x6 = self.out_linear2(x5)
        x7 = self.softmax(x6)

        return x7


@tables.register("encoder_classes", "FSMNExport")
class FSMNExport(nn.Module):
    def __init__(
            self,
            model,
            **kwargs,
    ):
        super().__init__()
        self.vad_opts = VADXOptions()

        # self.input_dim = input_dim
        # self.input_affine_dim = input_affine_dim
        # self.fsmn_layers = fsmn_layers
        # self.linear_dim = linear_dim
        # self.proj_dim = proj_dim
        # self.output_affine_dim = output_affine_dim
        # self.output_dim = output_dim
        #
        # self.in_linear1 = AffineTransform(input_dim, input_affine_dim)
        # self.in_linear2 = AffineTransform(input_affine_dim, linear_dim)
        # self.relu = RectifiedLinear(linear_dim, linear_dim)
        # self.fsmn = FsmnStack(*[BasicBlock(linear_dim, proj_dim, lorder, rorder, lstride, rstride, i) for i in
        #                         range(fsmn_layers)])
        # self.out_linear1 = AffineTransform(linear_dim, output_affine_dim)
        # self.out_linear2 = AffineTransform(output_affine_dim, output_dim)
        # self.softmax = nn.Softmax(dim=-1)
        self.in_linear1 = model.in_linear1
        self.in_linear2 = model.in_linear2
        self.relu = model.relu
        # self.fsmn = model.fsmn
        self.out_linear1 = model.out_linear1
        self.out_linear2 = model.out_linear2
        self.softmax = model.softmax
        self.fsmn = model.fsmn
        for i, d in enumerate(model.fsmn):
            if isinstance(d, BasicBlock):
                self.fsmn[i] = BasicBlock_export(d)

    def fuse_modules(self):
        pass

    def ResetDetection(self, cache: dict = {}):
        cache["stats"].continous_silence_frame_count = 0
        cache["stats"].latest_confirmed_speech_frame = 0
        cache["stats"].lastest_confirmed_silence_frame = -1
        cache["stats"].confirmed_start_frame = -1
        cache["stats"].confirmed_end_frame = -1
        cache["stats"].vad_state_machine = VadStateMachine.kVadInStateStartPointNotDetected
        cache["windows_detector"].Reset()
        cache["stats"].sil_frame = 0
        cache["stats"].frame_probs = []

        if cache["stats"].output_data_buf:
            assert cache["stats"].output_data_buf[-1].contain_seg_end_point == True
            drop_frames = int(cache["stats"].output_data_buf[-1].end_ms / self.vad_opts.frame_in_ms)
            real_drop_frames = drop_frames - cache["stats"].last_drop_frames
            cache["stats"].last_drop_frames = drop_frames
            cache["stats"].data_buf_all = cache["stats"].data_buf_all[
                                          real_drop_frames
                                          * int(self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000):
                                          ]
            cache["stats"].decibel = cache["stats"].decibel[real_drop_frames:]
            cache["stats"].scores = cache["stats"].scores[:, real_drop_frames:, :]

    def ComputeDecibel(self, cache: dict = {}) -> None:
        frame_sample_length = int(self.vad_opts.frame_length_ms * self.vad_opts.sample_rate / 1000)
        frame_shift_length = int(self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000)
        if cache["stats"].data_buf_all is None:
            cache["stats"].data_buf_all = cache["stats"].waveform[
                0
            ]  # cache["stats"].data_buf is pointed to cache["stats"].waveform[0]
            cache["stats"].data_buf = cache["stats"].data_buf_all
        else:
            cache["stats"].data_buf_all = torch.cat(
                (cache["stats"].data_buf_all, cache["stats"].waveform[0])
            )
        for offset in range(
                0, cache["stats"].waveform.shape[1] - frame_sample_length + 1, frame_shift_length
        ):
            cache["stats"].decibel.append(
                10
                * math.log10(
                    (cache["stats"].waveform[0][offset: offset + frame_sample_length])
                    .square()
                    .sum()
                    + 0.000001
                )
            )

    def OnSilenceDetected(self, valid_frame: int, cache: dict = {}):
        cache["stats"].lastest_confirmed_silence_frame = valid_frame
        if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
            self.PopDataBufTillFrame(valid_frame, cache=cache)

    # silence_detected_callback_
    # pass

    def OnVoiceDetected(self, valid_frame: int, cache: dict = {}) -> None:
        cache["stats"].latest_confirmed_speech_frame = valid_frame
        self.PopDataToOutputBuf(valid_frame, 1, False, False, False, cache=cache)

    def OnVoiceStart(self, start_frame: int, fake_result: bool = False, cache: dict = {}) -> None:
        if self.vad_opts.do_start_point_detection:
            pass
        if cache["stats"].confirmed_start_frame != -1:
            print("not reset vad properly\n")
        else:
            cache["stats"].confirmed_start_frame = start_frame

        if (
                not fake_result
                and cache["stats"].vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected
        ):
            self.PopDataToOutputBuf(
                cache["stats"].confirmed_start_frame, 1, True, False, False, cache=cache
            )

    def OnVoiceEnd(
            self, end_frame: int, fake_result: bool, is_last_frame: bool, cache: dict = {}
    ) -> None:
        for t in range(cache["stats"].latest_confirmed_speech_frame + 1, end_frame):
            self.OnVoiceDetected(t, cache=cache)
        if self.vad_opts.do_end_point_detection:
            pass
        if cache["stats"].confirmed_end_frame != -1:
            print("not reset vad properly\n")
        else:
            cache["stats"].confirmed_end_frame = end_frame
        if not fake_result:
            cache["stats"].sil_frame = 0
            self.PopDataToOutputBuf(
                cache["stats"].confirmed_end_frame, 1, False, True, is_last_frame, cache=cache
            )
        cache["stats"].number_end_time_detected += 1

    def MaybeOnVoiceEndIfLastFrame(
            self, is_final_frame: bool, cur_frm_idx: int, cache: dict = {}
    ) -> None:
        if is_final_frame:
            self.OnVoiceEnd(cur_frm_idx, False, True, cache=cache)
            cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected

    def GetLatency(self, cache: dict = {}) -> int:
        return int(self.LatencyFrmNumAtStartPoint(cache=cache) * self.vad_opts.frame_in_ms)

    def LatencyFrmNumAtStartPoint(self, cache: dict = {}) -> int:
        vad_latency = cache["windows_detector"].GetWinSize()
        if self.vad_opts.do_extend:
            vad_latency += int(self.vad_opts.lookback_time_start_point / self.vad_opts.frame_in_ms)
        return vad_latency

    def init_cache(self, cache: dict = {}):

        cache["frontend"] = {}
        cache["prev_samples"] = torch.empty(0)
        cache["encoder"] = {}

        windows_detector = WindowDetector(
            self.vad_opts.window_size_ms,
            self.vad_opts.sil_to_speech_time_thres,
            self.vad_opts.speech_to_sil_time_thres,
            self.vad_opts.frame_in_ms,
        )
        windows_detector.Reset()

        stats = Stats(
            sil_pdf_ids=self.vad_opts.sil_pdf_ids,
            max_end_sil_frame_cnt_thresh=self.vad_opts.max_end_silence_time
                                         - self.vad_opts.speech_to_sil_time_thres,
            speech_noise_thres=self.vad_opts.speech_noise_thres,
        )
        cache["windows_detector"] = windows_detector
        cache["stats"] = stats
        return cache

    def encode(self, feats, *args):
        x = self.in_linear1(feats)
        x = self.in_linear2(x)
        x = self.relu(x)
        # x4 = self.fsmn(x3, in_cache)  # self.in_cache will update automatically in self.fsmn
        # out_caches = list()
        for i, d in enumerate(self.fsmn):
            in_cache = args[i]
            x, out_cache = d(x, in_cache)
            # out_caches.append(out_cache)
        x = self.out_linear1(x)
        x = self.out_linear2(x)
        x = self.softmax(x)
        return x

    def DetectLastFrames(self, cache: dict = {}) -> int:
        if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateEndPointDetected:
            return 0
        for i in range(self.vad_opts.nn_eval_block_size - 1, -1, -1):
            frame_state = self.GetFrameState(
                cache["stats"].frm_cnt - 1 - i - cache["stats"].last_drop_frames, cache=cache
            )
            if i != 0:
                self.DetectOneFrame(frame_state, cache["stats"].frm_cnt - 1 - i, False, cache=cache)
            else:
                self.DetectOneFrame(frame_state, cache["stats"].frm_cnt - 1, True, cache=cache)

        return 0

    def GetFrameState(self, t: int, cache: dict = {}):
        frame_state = FrameState.kFrameStateInvalid
        cur_decibel = cache["stats"].decibel[t]
        cur_snr = cur_decibel - cache["stats"].noise_average_decibel
        # for each frame, calc log posterior probability of each state
        if cur_decibel < self.vad_opts.decibel_thres:
            frame_state = FrameState.kFrameStateSil
            self.DetectOneFrame(frame_state, t, False, cache=cache)
            return frame_state

        sum_score = 0.0
        noise_prob = 0.0
        assert len(cache["stats"].sil_pdf_ids) == self.vad_opts.silence_pdf_num
        if len(cache["stats"].sil_pdf_ids) > 0:
            assert len(cache["stats"].scores) == 1  # 只支持batch_size = 1的测试
            sil_pdf_scores = [
                cache["stats"].scores[0][t][sil_pdf_id] for sil_pdf_id in cache["stats"].sil_pdf_ids
            ]
            sum_score = sum(sil_pdf_scores)
            noise_prob = math.log(sum_score) * self.vad_opts.speech_2_noise_ratio
            total_score = 1.0
            sum_score = total_score - sum_score
        speech_prob = math.log(sum_score)
        if self.vad_opts.output_frame_probs:
            frame_prob = E2EVadFrameProb()
            frame_prob.noise_prob = noise_prob
            frame_prob.speech_prob = speech_prob
            frame_prob.score = sum_score
            frame_prob.frame_id = t
            cache["stats"].frame_probs.append(frame_prob)
        if math.exp(speech_prob) >= math.exp(noise_prob) + cache["stats"].speech_noise_thres:
            if cur_snr >= self.vad_opts.snr_thres and cur_decibel >= self.vad_opts.decibel_thres:
                frame_state = FrameState.kFrameStateSpeech
            else:
                frame_state = FrameState.kFrameStateSil
        else:
            frame_state = FrameState.kFrameStateSil
            if cache["stats"].noise_average_decibel < -99.9:
                cache["stats"].noise_average_decibel = cur_decibel
            else:
                cache["stats"].noise_average_decibel = (
                                                               cur_decibel
                                                               + cache["stats"].noise_average_decibel
                                                               * (self.vad_opts.noise_frame_num_used_for_snr - 1)
                                                       ) / self.vad_opts.noise_frame_num_used_for_snr

        return frame_state

    def DetectOneFrame(
            self, cur_frm_state: FrameState, cur_frm_idx: int, is_final_frame: bool, cache: dict = {}
    ) -> None:
        tmp_cur_frm_state = FrameState.kFrameStateInvalid
        if cur_frm_state == FrameState.kFrameStateSpeech:
            if math.fabs(1.0) > self.vad_opts.fe_prior_thres:
                tmp_cur_frm_state = FrameState.kFrameStateSpeech
            else:
                tmp_cur_frm_state = FrameState.kFrameStateSil
        elif cur_frm_state == FrameState.kFrameStateSil:
            tmp_cur_frm_state = FrameState.kFrameStateSil
        state_change = cache["windows_detector"].DetectOneFrame(
            tmp_cur_frm_state, cur_frm_idx, cache=cache
        )
        frm_shift_in_ms = self.vad_opts.frame_in_ms
        if AudioChangeState.kChangeStateSil2Speech == state_change:
            silence_frame_count = cache["stats"].continous_silence_frame_count
            cache["stats"].continous_silence_frame_count = 0
            cache["stats"].pre_end_silence_detected = False
            start_frame = 0
            if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
                start_frame = max(
                    cache["stats"].data_buf_start_frame,
                    cur_frm_idx - self.LatencyFrmNumAtStartPoint(cache=cache),
                )
                self.OnVoiceStart(start_frame, cache=cache)
                cache["stats"].vad_state_machine = VadStateMachine.kVadInStateInSpeechSegment
                for t in range(start_frame + 1, cur_frm_idx + 1):
                    self.OnVoiceDetected(t, cache=cache)
            elif cache["stats"].vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                for t in range(cache["stats"].latest_confirmed_speech_frame + 1, cur_frm_idx):
                    self.OnVoiceDetected(t, cache=cache)
                if (
                        cur_frm_idx - cache["stats"].confirmed_start_frame + 1
                        > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.OnVoiceEnd(cur_frm_idx, False, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif not is_final_frame:
                    self.OnVoiceDetected(cur_frm_idx, cache=cache)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx, cache=cache)
            else:
                pass
        elif AudioChangeState.kChangeStateSpeech2Sil == state_change:
            cache["stats"].continous_silence_frame_count = 0
            if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
                pass
            elif cache["stats"].vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if (
                        cur_frm_idx - cache["stats"].confirmed_start_frame + 1
                        > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.OnVoiceEnd(cur_frm_idx, False, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif not is_final_frame:
                    self.OnVoiceDetected(cur_frm_idx, cache=cache)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx, cache=cache)
            else:
                pass
        elif AudioChangeState.kChangeStateSpeech2Speech == state_change:
            cache["stats"].continous_silence_frame_count = 0
            if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if (
                        cur_frm_idx - cache["stats"].confirmed_start_frame + 1
                        > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    cache["stats"].max_time_out = True
                    self.OnVoiceEnd(cur_frm_idx, False, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif not is_final_frame:
                    self.OnVoiceDetected(cur_frm_idx, cache=cache)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx, cache=cache)
            else:
                pass
        elif AudioChangeState.kChangeStateSil2Sil == state_change:
            cache["stats"].continous_silence_frame_count += 1
            if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
                # silence timeout, return zero length decision
                if (
                        (self.vad_opts.detect_mode == VadDetectMode.kVadSingleUtteranceDetectMode.value)
                        and (
                                cache["stats"].continous_silence_frame_count * frm_shift_in_ms
                                > self.vad_opts.max_start_silence_time
                        )
                ) or (is_final_frame and cache["stats"].number_end_time_detected == 0):
                    for t in range(cache["stats"].lastest_confirmed_silence_frame + 1, cur_frm_idx):
                        self.OnSilenceDetected(t, cache=cache)
                    self.OnVoiceStart(0, True, cache=cache)
                    self.OnVoiceEnd(0, True, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                else:
                    if cur_frm_idx >= self.LatencyFrmNumAtStartPoint(cache=cache):
                        self.OnSilenceDetected(
                            cur_frm_idx - self.LatencyFrmNumAtStartPoint(cache=cache), cache=cache
                        )
            elif cache["stats"].vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if (
                        cache["stats"].continous_silence_frame_count * frm_shift_in_ms
                        >= cache["stats"].max_end_sil_frame_cnt_thresh
                ):
                    lookback_frame = int(
                        cache["stats"].max_end_sil_frame_cnt_thresh / frm_shift_in_ms
                    )
                    if self.vad_opts.do_extend:
                        lookback_frame -= int(
                            self.vad_opts.lookahead_time_end_point / frm_shift_in_ms
                        )
                        lookback_frame -= 1
                        lookback_frame = max(0, lookback_frame)
                    self.OnVoiceEnd(cur_frm_idx - lookback_frame, False, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif (
                        cur_frm_idx - cache["stats"].confirmed_start_frame + 1
                        > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.OnVoiceEnd(cur_frm_idx, False, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif self.vad_opts.do_extend and not is_final_frame:
                    if cache["stats"].continous_silence_frame_count <= int(
                            self.vad_opts.lookahead_time_end_point / frm_shift_in_ms
                    ):
                        self.OnVoiceDetected(cur_frm_idx, cache=cache)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx, cache=cache)
            else:
                pass

        if (
                cache["stats"].vad_state_machine == VadStateMachine.kVadInStateEndPointDetected
                and self.vad_opts.detect_mode == VadDetectMode.kVadMutipleUtteranceDetectMode.value
        ):
            self.ResetDetection(cache=cache)

    def PopDataBufTillFrame(self, frame_idx: int, cache: dict = {}) -> None:  # need check again
        while cache["stats"].data_buf_start_frame < frame_idx:
            if len(cache["stats"].data_buf) >= int(
                    self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000
            ):
                cache["stats"].data_buf_start_frame += 1
                cache["stats"].data_buf = cache["stats"].data_buf_all[
                                          (cache["stats"].data_buf_start_frame - cache["stats"].last_drop_frames)
                                          * int(self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000):
                                          ]

    def PopDataToOutputBuf(
            self,
            start_frm: int,
            frm_cnt: int,
            first_frm_is_start_point: bool,
            last_frm_is_end_point: bool,
            end_point_is_sent_end: bool,
            cache: dict = {},
    ) -> None:
        self.PopDataBufTillFrame(start_frm, cache=cache)
        expected_sample_number = int(
            frm_cnt * self.vad_opts.sample_rate * self.vad_opts.frame_in_ms / 1000
        )
        if last_frm_is_end_point:
            extra_sample = max(
                0,
                int(
                    self.vad_opts.frame_length_ms * self.vad_opts.sample_rate / 1000
                    - self.vad_opts.sample_rate * self.vad_opts.frame_in_ms / 1000
                ),
            )
            expected_sample_number += int(extra_sample)
        if end_point_is_sent_end:
            expected_sample_number = max(expected_sample_number, len(cache["stats"].data_buf))
        if len(cache["stats"].data_buf) < expected_sample_number:
            print("error in calling pop data_buf\n")

        if len(cache["stats"].output_data_buf) == 0 or first_frm_is_start_point:
            cache["stats"].output_data_buf.append(E2EVadSpeechBufWithDoa())
            cache["stats"].output_data_buf[-1].Reset()
            cache["stats"].output_data_buf[-1].start_ms = start_frm * self.vad_opts.frame_in_ms
            cache["stats"].output_data_buf[-1].end_ms = cache["stats"].output_data_buf[-1].start_ms
            cache["stats"].output_data_buf[-1].doa = 0
        cur_seg = cache["stats"].output_data_buf[-1]
        if cur_seg.end_ms != start_frm * self.vad_opts.frame_in_ms:
            print("warning\n")
        out_pos = len(cur_seg.buffer)  # cur_seg.buff现在没做任何操作
        data_to_pop = 0
        if end_point_is_sent_end:
            data_to_pop = expected_sample_number
        else:
            data_to_pop = int(
                frm_cnt * self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000
            )
        if data_to_pop > len(cache["stats"].data_buf):
            print('VAD data_to_pop is bigger than cache["stats"].data_buf.size()!!!\n')
            data_to_pop = len(cache["stats"].data_buf)
            expected_sample_number = len(cache["stats"].data_buf)

        cur_seg.doa = 0
        for sample_cpy_out in range(0, data_to_pop):
            # cur_seg.buffer[out_pos ++] = data_buf_.back();
            out_pos += 1
        for sample_cpy_out in range(data_to_pop, expected_sample_number):
            # cur_seg.buffer[out_pos++] = data_buf_.back()
            out_pos += 1
        if cur_seg.end_ms != start_frm * self.vad_opts.frame_in_ms:
            print("Something wrong with the VAD algorithm\n")
        cache["stats"].data_buf_start_frame += frm_cnt
        cur_seg.end_ms = (start_frm + frm_cnt) * self.vad_opts.frame_in_ms
        if first_frm_is_start_point:
            cur_seg.contain_seg_start_point = True
        if last_frm_is_end_point:
            cur_seg.contain_seg_end_point = True

    def forward(
            self,
            feats: torch.Tensor,

    ):
        """
        Args:
            feats (torch.Tensor): Input tensor (B, T, D)
            waveform (torch.Tensor): Input tensor (B, T)
            in_cache: when in_cache is not None, the forward is in streaming. The type of in_cache is a dict, egs,
            {'cache_layer_1': torch.Tensor(B, T1, D)}, T1 is equal to self.lorder. It is {} for the 1st frame
        """

        args = []
        for i in range(4):
            args.append(torch.zeros(1, 128, 19, 1))

        scores = self.encode(feats, *args)
        return scores


"""
one deep fsmn layer
dimproj:                projection dimension, feats and output dimension of memory blocks
dimlinear:              dimension of mapping layer
lorder:                 left order
rorder:                 right order
lstride:                left stride
rstride:                right stride
"""


class VADHelp(nn.Module):
    def __init__(self):
        super(VADHelp, self).__init__()
        self.vad_opts = VADXOptions()

    def ResetDetection(self, cache: dict = {}):
        cache["stats"].continous_silence_frame_count = 0
        cache["stats"].latest_confirmed_speech_frame = 0
        cache["stats"].lastest_confirmed_silence_frame = -1
        cache["stats"].confirmed_start_frame = -1
        cache["stats"].confirmed_end_frame = -1
        cache["stats"].vad_state_machine = VadStateMachine.kVadInStateStartPointNotDetected
        cache["windows_detector"].Reset()
        cache["stats"].sil_frame = 0
        cache["stats"].frame_probs = []

        if cache["stats"].output_data_buf:
            assert cache["stats"].output_data_buf[-1].contain_seg_end_point == True
            drop_frames = int(cache["stats"].output_data_buf[-1].end_ms / self.vad_opts.frame_in_ms)
            real_drop_frames = drop_frames - cache["stats"].last_drop_frames
            cache["stats"].last_drop_frames = drop_frames
            cache["stats"].data_buf_all = cache["stats"].data_buf_all[
                                          real_drop_frames
                                          * int(self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000):
                                          ]
            cache["stats"].decibel = cache["stats"].decibel[real_drop_frames:]
            cache["stats"].scores = cache["stats"].scores[:, real_drop_frames:, :]

    def ComputeDecibel(self, cache: dict = {}) -> None:
        frame_sample_length = int(self.vad_opts.frame_length_ms * self.vad_opts.sample_rate / 1000)
        frame_shift_length = int(self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000)
        if cache["stats"].data_buf_all is None:
            cache["stats"].data_buf_all = cache["stats"].waveform[
                0
            ]  # cache["stats"].data_buf is pointed to cache["stats"].waveform[0]
            cache["stats"].data_buf = cache["stats"].data_buf_all
        else:
            cache["stats"].data_buf_all = torch.cat(
                (cache["stats"].data_buf_all, cache["stats"].waveform[0])
            )
        for offset in range(
                0, cache["stats"].waveform.shape[1] - frame_sample_length + 1, frame_shift_length
        ):
            cache["stats"].decibel.append(
                10
                * math.log10(
                    (cache["stats"].waveform[0][offset: offset + frame_sample_length])
                    .square()
                    .sum()
                    + 0.000001
                )
            )

    def OnSilenceDetected(self, valid_frame: int, cache: dict = {}):
        cache["stats"].lastest_confirmed_silence_frame = valid_frame
        if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
            self.PopDataBufTillFrame(valid_frame, cache=cache)

    # silence_detected_callback_
    # pass

    def OnVoiceDetected(self, valid_frame: int, cache: dict = {}) -> None:
        cache["stats"].latest_confirmed_speech_frame = valid_frame
        self.PopDataToOutputBuf(valid_frame, 1, False, False, False, cache=cache)

    def OnVoiceStart(self, start_frame: int, fake_result: bool = False, cache: dict = {}) -> None:
        if self.vad_opts.do_start_point_detection:
            pass
        if cache["stats"].confirmed_start_frame != -1:
            print("not reset vad properly\n")
        else:
            cache["stats"].confirmed_start_frame = start_frame

        if (
                not fake_result
                and cache["stats"].vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected
        ):
            self.PopDataToOutputBuf(
                cache["stats"].confirmed_start_frame, 1, True, False, False, cache=cache
            )

    def OnVoiceEnd(
            self, end_frame: int, fake_result: bool, is_last_frame: bool, cache: dict = {}
    ) -> None:
        for t in range(cache["stats"].latest_confirmed_speech_frame + 1, end_frame):
            self.OnVoiceDetected(t, cache=cache)
        if self.vad_opts.do_end_point_detection:
            pass
        if cache["stats"].confirmed_end_frame != -1:
            print("not reset vad properly\n")
        else:
            cache["stats"].confirmed_end_frame = end_frame
        if not fake_result:
            cache["stats"].sil_frame = 0
            self.PopDataToOutputBuf(
                cache["stats"].confirmed_end_frame, 1, False, True, is_last_frame, cache=cache
            )
        cache["stats"].number_end_time_detected += 1

    def MaybeOnVoiceEndIfLastFrame(
            self, is_final_frame: bool, cur_frm_idx: int, cache: dict = {}
    ) -> None:
        if is_final_frame:
            self.OnVoiceEnd(cur_frm_idx, False, True, cache=cache)
            cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected

    def GetLatency(self, cache: dict = {}) -> int:
        return int(self.LatencyFrmNumAtStartPoint(cache=cache) * self.vad_opts.frame_in_ms)

    def LatencyFrmNumAtStartPoint(self, cache: dict = {}) -> int:
        vad_latency = cache["windows_detector"].GetWinSize()
        if self.vad_opts.do_extend:
            vad_latency += int(self.vad_opts.lookback_time_start_point / self.vad_opts.frame_in_ms)
        return vad_latency

    def init_cache(self, cache: dict = {}):

        cache["frontend"] = {}
        cache["prev_samples"] = torch.empty(0)
        cache["encoder"] = {}

        windows_detector = WindowDetector(
            self.vad_opts.window_size_ms,
            self.vad_opts.sil_to_speech_time_thres,
            self.vad_opts.speech_to_sil_time_thres,
            self.vad_opts.frame_in_ms,
        )
        windows_detector.Reset()

        stats = Stats(
            sil_pdf_ids=self.vad_opts.sil_pdf_ids,
            max_end_sil_frame_cnt_thresh=self.vad_opts.max_end_silence_time
                                         - self.vad_opts.speech_to_sil_time_thres,
            speech_noise_thres=self.vad_opts.speech_noise_thres,
        )
        cache["windows_detector"] = windows_detector
        cache["stats"] = stats
        return cache

    def encode(self, feats, *args):
        x = self.in_linear1(feats)
        x = self.in_linear2(x)
        x = self.relu(x)
        # x4 = self.fsmn(x3, in_cache)  # self.in_cache will update automatically in self.fsmn
        # out_caches = list()
        for i, d in enumerate(self.fsmn):
            in_cache = args[i]
            x, out_cache = d(x, in_cache)
            # out_caches.append(out_cache)
        x = self.out_linear1(x)
        x = self.out_linear2(x)
        x = self.softmax(x)
        return x

    def DetectLastFrames(self, cache: dict = {}) -> int:
        if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateEndPointDetected:
            return 0
        for i in range(self.vad_opts.nn_eval_block_size - 1, -1, -1):
            frame_state = self.GetFrameState(
                cache["stats"].frm_cnt - 1 - i - cache["stats"].last_drop_frames, cache=cache
            )
            if i != 0:
                self.DetectOneFrame(frame_state, cache["stats"].frm_cnt - 1 - i, False, cache=cache)
            else:
                self.DetectOneFrame(frame_state, cache["stats"].frm_cnt - 1, True, cache=cache)

        return 0

    def GetFrameState(self, t: int, cache: dict = {}):
        frame_state = FrameState.kFrameStateInvalid
        cur_decibel = cache["stats"].decibel[t]
        cur_snr = cur_decibel - cache["stats"].noise_average_decibel
        # for each frame, calc log posterior probability of each state
        if cur_decibel < self.vad_opts.decibel_thres:
            frame_state = FrameState.kFrameStateSil
            self.DetectOneFrame(frame_state, t, False, cache=cache)
            return frame_state

        sum_score = 0.0
        noise_prob = 0.0
        assert len(cache["stats"].sil_pdf_ids) == self.vad_opts.silence_pdf_num
        if len(cache["stats"].sil_pdf_ids) > 0:
            assert len(cache["stats"].scores) == 1  # 只支持batch_size = 1的测试
            sil_pdf_scores = [
                cache["stats"].scores[0][t][sil_pdf_id] for sil_pdf_id in cache["stats"].sil_pdf_ids
            ]
            sum_score = sum(sil_pdf_scores)
            noise_prob = math.log(sum_score) * self.vad_opts.speech_2_noise_ratio
            total_score = 1.0
            sum_score = total_score - sum_score
        speech_prob = math.log(sum_score)
        if self.vad_opts.output_frame_probs:
            frame_prob = E2EVadFrameProb()
            frame_prob.noise_prob = noise_prob
            frame_prob.speech_prob = speech_prob
            frame_prob.score = sum_score
            frame_prob.frame_id = t
            cache["stats"].frame_probs.append(frame_prob)
        if math.exp(speech_prob) >= math.exp(noise_prob) + cache["stats"].speech_noise_thres:
            if cur_snr >= self.vad_opts.snr_thres and cur_decibel >= self.vad_opts.decibel_thres:
                frame_state = FrameState.kFrameStateSpeech
            else:
                frame_state = FrameState.kFrameStateSil
        else:
            frame_state = FrameState.kFrameStateSil
            if cache["stats"].noise_average_decibel < -99.9:
                cache["stats"].noise_average_decibel = cur_decibel
            else:
                cache["stats"].noise_average_decibel = (
                                                               cur_decibel
                                                               + cache["stats"].noise_average_decibel
                                                               * (self.vad_opts.noise_frame_num_used_for_snr - 1)
                                                       ) / self.vad_opts.noise_frame_num_used_for_snr

        return frame_state

    def DetectOneFrame(
            self, cur_frm_state: FrameState, cur_frm_idx: int, is_final_frame: bool, cache: dict = {}
    ) -> None:
        tmp_cur_frm_state = FrameState.kFrameStateInvalid
        if cur_frm_state == FrameState.kFrameStateSpeech:
            if math.fabs(1.0) > self.vad_opts.fe_prior_thres:
                tmp_cur_frm_state = FrameState.kFrameStateSpeech
            else:
                tmp_cur_frm_state = FrameState.kFrameStateSil
        elif cur_frm_state == FrameState.kFrameStateSil:
            tmp_cur_frm_state = FrameState.kFrameStateSil
        state_change = cache["windows_detector"].DetectOneFrame(
            tmp_cur_frm_state, cur_frm_idx, cache=cache
        )
        frm_shift_in_ms = self.vad_opts.frame_in_ms
        if AudioChangeState.kChangeStateSil2Speech == state_change:
            silence_frame_count = cache["stats"].continous_silence_frame_count
            cache["stats"].continous_silence_frame_count = 0
            cache["stats"].pre_end_silence_detected = False
            start_frame = 0
            if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
                start_frame = max(
                    cache["stats"].data_buf_start_frame,
                    cur_frm_idx - self.LatencyFrmNumAtStartPoint(cache=cache),
                )
                self.OnVoiceStart(start_frame, cache=cache)
                cache["stats"].vad_state_machine = VadStateMachine.kVadInStateInSpeechSegment
                for t in range(start_frame + 1, cur_frm_idx + 1):
                    self.OnVoiceDetected(t, cache=cache)
            elif cache["stats"].vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                for t in range(cache["stats"].latest_confirmed_speech_frame + 1, cur_frm_idx):
                    self.OnVoiceDetected(t, cache=cache)
                if (
                        cur_frm_idx - cache["stats"].confirmed_start_frame + 1
                        > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.OnVoiceEnd(cur_frm_idx, False, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif not is_final_frame:
                    self.OnVoiceDetected(cur_frm_idx, cache=cache)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx, cache=cache)
            else:
                pass
        elif AudioChangeState.kChangeStateSpeech2Sil == state_change:
            cache["stats"].continous_silence_frame_count = 0
            if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
                pass
            elif cache["stats"].vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if (
                        cur_frm_idx - cache["stats"].confirmed_start_frame + 1
                        > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.OnVoiceEnd(cur_frm_idx, False, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif not is_final_frame:
                    self.OnVoiceDetected(cur_frm_idx, cache=cache)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx, cache=cache)
            else:
                pass
        elif AudioChangeState.kChangeStateSpeech2Speech == state_change:
            cache["stats"].continous_silence_frame_count = 0
            if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if (
                        cur_frm_idx - cache["stats"].confirmed_start_frame + 1
                        > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    cache["stats"].max_time_out = True
                    self.OnVoiceEnd(cur_frm_idx, False, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif not is_final_frame:
                    self.OnVoiceDetected(cur_frm_idx, cache=cache)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx, cache=cache)
            else:
                pass
        elif AudioChangeState.kChangeStateSil2Sil == state_change:
            cache["stats"].continous_silence_frame_count += 1
            if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
                # silence timeout, return zero length decision
                if (
                        (self.vad_opts.detect_mode == VadDetectMode.kVadSingleUtteranceDetectMode.value)
                        and (
                                cache["stats"].continous_silence_frame_count * frm_shift_in_ms
                                > self.vad_opts.max_start_silence_time
                        )
                ) or (is_final_frame and cache["stats"].number_end_time_detected == 0):
                    for t in range(cache["stats"].lastest_confirmed_silence_frame + 1, cur_frm_idx):
                        self.OnSilenceDetected(t, cache=cache)
                    self.OnVoiceStart(0, True, cache=cache)
                    self.OnVoiceEnd(0, True, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                else:
                    if cur_frm_idx >= self.LatencyFrmNumAtStartPoint(cache=cache):
                        self.OnSilenceDetected(
                            cur_frm_idx - self.LatencyFrmNumAtStartPoint(cache=cache), cache=cache
                        )
            elif cache["stats"].vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if (
                        cache["stats"].continous_silence_frame_count * frm_shift_in_ms
                        >= cache["stats"].max_end_sil_frame_cnt_thresh
                ):
                    lookback_frame = int(
                        cache["stats"].max_end_sil_frame_cnt_thresh / frm_shift_in_ms
                    )
                    if self.vad_opts.do_extend:
                        lookback_frame -= int(
                            self.vad_opts.lookahead_time_end_point / frm_shift_in_ms
                        )
                        lookback_frame -= 1
                        lookback_frame = max(0, lookback_frame)
                    self.OnVoiceEnd(cur_frm_idx - lookback_frame, False, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif (
                        cur_frm_idx - cache["stats"].confirmed_start_frame + 1
                        > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.OnVoiceEnd(cur_frm_idx, False, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif self.vad_opts.do_extend and not is_final_frame:
                    if cache["stats"].continous_silence_frame_count <= int(
                            self.vad_opts.lookahead_time_end_point / frm_shift_in_ms
                    ):
                        self.OnVoiceDetected(cur_frm_idx, cache=cache)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx, cache=cache)
            else:
                pass

        if (
                cache["stats"].vad_state_machine == VadStateMachine.kVadInStateEndPointDetected
                and self.vad_opts.detect_mode == VadDetectMode.kVadMutipleUtteranceDetectMode.value
        ):
            self.ResetDetection(cache=cache)

    def PopDataBufTillFrame(self, frame_idx: int, cache: dict = {}) -> None:  # need check again
        while cache["stats"].data_buf_start_frame < frame_idx:
            if len(cache["stats"].data_buf) >= int(
                    self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000
            ):
                cache["stats"].data_buf_start_frame += 1
                cache["stats"].data_buf = cache["stats"].data_buf_all[
                                          (cache["stats"].data_buf_start_frame - cache["stats"].last_drop_frames)
                                          * int(self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000):
                                          ]

    def PopDataToOutputBuf(
            self,
            start_frm: int,
            frm_cnt: int,
            first_frm_is_start_point: bool,
            last_frm_is_end_point: bool,
            end_point_is_sent_end: bool,
            cache: dict = {},
    ) -> None:
        self.PopDataBufTillFrame(start_frm, cache=cache)
        expected_sample_number = int(
            frm_cnt * self.vad_opts.sample_rate * self.vad_opts.frame_in_ms / 1000
        )
        if last_frm_is_end_point:
            extra_sample = max(
                0,
                int(
                    self.vad_opts.frame_length_ms * self.vad_opts.sample_rate / 1000
                    - self.vad_opts.sample_rate * self.vad_opts.frame_in_ms / 1000
                ),
            )
            expected_sample_number += int(extra_sample)
        if end_point_is_sent_end:
            expected_sample_number = max(expected_sample_number, len(cache["stats"].data_buf))
        if len(cache["stats"].data_buf) < expected_sample_number:
            print("error in calling pop data_buf\n")

        if len(cache["stats"].output_data_buf) == 0 or first_frm_is_start_point:
            cache["stats"].output_data_buf.append(E2EVadSpeechBufWithDoa())
            cache["stats"].output_data_buf[-1].Reset()
            cache["stats"].output_data_buf[-1].start_ms = start_frm * self.vad_opts.frame_in_ms
            cache["stats"].output_data_buf[-1].end_ms = cache["stats"].output_data_buf[-1].start_ms
            cache["stats"].output_data_buf[-1].doa = 0
        cur_seg = cache["stats"].output_data_buf[-1]
        if cur_seg.end_ms != start_frm * self.vad_opts.frame_in_ms:
            print("warning\n")
        out_pos = len(cur_seg.buffer)  # cur_seg.buff现在没做任何操作
        data_to_pop = 0
        if end_point_is_sent_end:
            data_to_pop = expected_sample_number
        else:
            data_to_pop = int(
                frm_cnt * self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000
            )
        if data_to_pop > len(cache["stats"].data_buf):
            print('VAD data_to_pop is bigger than cache["stats"].data_buf.size()!!!\n')
            data_to_pop = len(cache["stats"].data_buf)
            expected_sample_number = len(cache["stats"].data_buf)

        cur_seg.doa = 0
        for sample_cpy_out in range(0, data_to_pop):
            # cur_seg.buffer[out_pos ++] = data_buf_.back();
            out_pos += 1
        for sample_cpy_out in range(data_to_pop, expected_sample_number):
            # cur_seg.buffer[out_pos++] = data_buf_.back()
            out_pos += 1
        if cur_seg.end_ms != start_frm * self.vad_opts.frame_in_ms:
            print("Something wrong with the VAD algorithm\n")
        cache["stats"].data_buf_start_frame += frm_cnt
        cur_seg.end_ms = (start_frm + frm_cnt) * self.vad_opts.frame_in_ms
        if first_frm_is_start_point:
            cur_seg.contain_seg_start_point = True
        if last_frm_is_end_point:
            cur_seg.contain_seg_end_point = True

    def forward(self, scores, waveform):
        is_final = False
        cache = {}
        if len(cache) == 0:
            self.init_cache(cache)

        cache["stats"].waveform = waveform
        is_streaming_input = False
        self.ComputeDecibel(cache=cache)
        self.vad_opts.nn_eval_block_size = scores.shape[1]
        cache["stats"].frm_cnt += scores.shape[1]  # count total frames
        if cache["stats"].scores is None:
            cache["stats"].scores = scores  # the first calculation
        else:
            cache["stats"].scores = torch.cat((cache["stats"].scores, scores), dim=1)

        self.DetectLastFrames(cache=cache)
        segments = []
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


@tables.register("encoder_classes", "DFSMN")
class DFSMN(nn.Module):

    def __init__(self, dimproj=64, dimlinear=128, lorder=20, rorder=1, lstride=1, rstride=1):
        super(DFSMN, self).__init__()

        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride

        self.expand = AffineTransform(dimproj, dimlinear)
        self.shrink = LinearTransform(dimlinear, dimproj)

        self.conv_left = nn.Conv2d(
            dimproj, dimproj, [lorder, 1], dilation=[lstride, 1], groups=dimproj, bias=False
        )

        if rorder > 0:
            self.conv_right = nn.Conv2d(
                dimproj, dimproj, [rorder, 1], dilation=[rstride, 1], groups=dimproj, bias=False
            )
        else:
            self.conv_right = None

    def forward(self, input):
        f1 = F.relu(self.expand(input))
        p1 = self.shrink(f1)

        x = torch.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)

        y_left = F.pad(x_per, [0, 0, (self.lorder - 1) * self.lstride, 0])

        if self.conv_right is not None:
            y_right = F.pad(x_per, [0, 0, 0, (self.rorder) * self.rstride])
            y_right = y_right[:, :, self.rstride:, :]
            out = x_per + self.conv_left(y_left) + self.conv_right(y_right)
        else:
            out = x_per + self.conv_left(y_left)

        out1 = out.permute(0, 3, 2, 1)
        output = input + out1.squeeze(1)

        return output


"""
build stacked dfsmn layers
"""


def buildDFSMNRepeats(linear_dim=128, proj_dim=64, lorder=20, rorder=1, fsmn_layers=6):
    repeats = [
        nn.Sequential(DFSMN(proj_dim, linear_dim, lorder, rorder, 1, 1)) for i in range(fsmn_layers)
    ]

    return nn.Sequential(*repeats)


if __name__ == "__main__":
    fsmn = FSMN(400, 140, 4, 250, 128, 10, 2, 1, 1, 140, 2599)
    print(fsmn)

    num_params = sum(p.numel() for p in fsmn.parameters())
    print("the number of model params: {}".format(num_params))
    x = torch.zeros(128, 200, 400)  # batch-size * time * dim
    y, _ = fsmn(x)  # batch-size * time * dim
    print("feats shape: {}".format(x.shape))
    print("output shape: {}".format(y.shape))

    print(fsmn.to_kaldi_net())
