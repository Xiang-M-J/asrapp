import 'dart:math';

import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:mainapp/utils/feature_utils.dart';
import 'package:onnxruntime/onnxruntime.dart';

class VadStateMachine {
  static int kVadInStateStartPointNotDetected = 1;
  static int kVadInStateInSpeechSegment = 2;
  static int kVadInStateEndPointDetected = 3;
}

class FrameState {
  static int kFrameStateInvalid = -1;
  static int kFrameStateSpeech = 1;
  static int kFrameStateSil = 0;
}

class AudioChangeState {
  static int kChangeStateSpeech2Speech = 0;
  static int kChangeStateSpeech2Sil = 1;
  static int kChangeStateSil2Sil = 2;
  static int kChangeStateSil2Speech = 3;
  static int kChangeStateNoBegin = 4;
  static int kChangeStateInvalid = 5;
}

class VadDetectMode {
  static int kVadSingleUtteranceDetectMode = 0;
  static int kVadMutipleUtteranceDetectMode = 1;
}

class VADXOptions {
  int sample_rate = 16000;
  int detect_mode = VadDetectMode.kVadMutipleUtteranceDetectMode;
  int snr_mode = 0;
  int max_end_silence_time = 800;
  int max_start_silence_time = 3000;
  bool do_start_point_detection = true;
  bool do_end_point_detection = true;
  int window_size_ms = 200;
  int sil_to_speech_time_thres = 150;
  int speech_to_sil_time_thres = 150;
  double speech_2_noise_ratio = 1.0;
  int do_extend = 1;
  int lookback_time_start_point = 200;
  int lookahead_time_end_point = 100;
  int max_single_segment_time = 60000;
  int nn_eval_block_size = 8;
  int dcd_block_size = 4;
  int snr_thres = -100;
  int noise_frame_num_used_for_snr = 100;
  int decibel_thres = -100;
  double speech_noise_thres = 0.6;
  double fe_prior_thres = 1e-4;
  int silence_pdf_num = 1;
  List<int> sil_pdf_ids = [0];
  double speech_noise_thresh_low = -0.1;
  double speech_noise_thresh_high = 0.3;
  bool output_frame_probs = false;
  int frame_in_ms = 10;
  int frame_length_ms = 25;
}

class E2EVadSpeechBufWithDoa {
  int start_ms = 0;
  int end_ms = 0;
  List<double> buffer = List.empty(growable: true);
  bool contain_seg_start_point = false;
  bool contain_seg_end_point = false;
  int doa = 0;
  E2EVadSpeechBufWithDoa();

  Reset() {
    start_ms = 0;
    end_ms = 0;
    buffer = [];
    contain_seg_start_point = false;
    contain_seg_end_point = false;
    doa = 0;
  }
}

class E2EVadFrameProb {
  double noise_prob = 0.0;
  double speech_prob = 0.0;
  double score = 0.0;
  int frame_id = 0;
  int frm_state = 0;
}

class Stats {
  int data_buf_start_frame = 0;
  int frm_cnt = 0;
  int latest_confirmed_speech_frame = 0;
  int lastest_confirmed_silence_frame = -1;
  int continous_silence_frame_count = 0;
  int vad_state_machine = VadStateMachine.kVadInStateStartPointNotDetected;
  int confirmed_start_frame = -1;
  int confirmed_end_frame = -1;
  int number_end_time_detected = 0;
  int sil_frame = 0;
  List<int> sil_pdf_ids;
  double noise_average_decibel = -100.0;
  bool pre_end_silence_detected = false;
  bool next_seg = true;
  List<E2EVadSpeechBufWithDoa> output_data_buf = [];
  int output_data_buf_offset = 0;
  List<double> frame_probs = [];
  double max_end_sil_frame_cnt_thresh;
  double speech_noise_thres;
  List<List<double>>? scores;
  bool max_time_out = false;
  List<double> decibel = [];
  List<double>? data_buf;
  List<double>? data_buf_all;
  List<double>? waveform;
  int last_drop_frames = 0;
  Stats(this.sil_pdf_ids, this.max_end_sil_frame_cnt_thresh,
      this.speech_noise_thres);
}

class WindowDetector {
  late int window_size_ms;
  late int sil_to_speech_time;
  late int speech_to_sil_time;
  late int frame_size_ms;
  late int win_size_frame;
  double win_sum = 0;
  late List<int> win_state;
  int cur_win_pos = 0;
  int pre_frame_state = FrameState.kFrameStateSil;
  int cur_frame_state = FrameState.kFrameStateSil;
  late int sil_to_speech_frmcnt_thres;
  late int speech_to_sil_frmcnt_thres;
  int voice_last_frame_count = 0;
  int noise_last_frame_count = 0;
  int hydre_frame_count = 0;

  WindowDetector(this.window_size_ms, this.sil_to_speech_time,
      this.speech_to_sil_time, this.frame_size_ms) {
    win_size_frame = window_size_ms ~/ frame_size_ms;
    win_state = List.filled(win_size_frame, 0);

    sil_to_speech_frmcnt_thres = (sil_to_speech_time ~/ frame_size_ms);
    speech_to_sil_frmcnt_thres = (speech_to_sil_time ~/ frame_size_ms);
  }

  void Reset() {
    cur_win_pos = 0;
    win_sum = 0;
    win_state = List.filled(win_size_frame, 0);
    pre_frame_state = FrameState.kFrameStateSil;
    cur_frame_state = FrameState.kFrameStateSil;
    voice_last_frame_count = 0;
    noise_last_frame_count = 0;
    hydre_frame_count = 0;
  }

  int GetWinSize() {
    return (win_size_frame);
  }

  DetectOneFrame(int frameState, int frameCount, Map cache) {
    int curFrameState = FrameState.kFrameStateSil;
    if (frameState == FrameState.kFrameStateSpeech) {
      curFrameState = 1;
    } else if (frameState == FrameState.kFrameStateSil) {
      curFrameState = 0;
    } else {
      return AudioChangeState.kChangeStateInvalid;
    }

    win_sum -= win_state[cur_win_pos];
    win_sum += curFrameState;
    win_state[cur_win_pos] = curFrameState;
    cur_win_pos = (cur_win_pos + 1) % win_size_frame;

    if (pre_frame_state == FrameState.kFrameStateSil &&
        win_sum >= sil_to_speech_frmcnt_thres) {
      pre_frame_state = FrameState.kFrameStateSpeech;
      return AudioChangeState.kChangeStateSil2Speech;
    }

    if (pre_frame_state == FrameState.kFrameStateSpeech &&
        win_sum <= speech_to_sil_frmcnt_thres) {
      pre_frame_state = FrameState.kFrameStateSil;
      return AudioChangeState.kChangeStateSpeech2Sil;
    }

    if (pre_frame_state == FrameState.kFrameStateSil) {
      return AudioChangeState.kChangeStateSil2Sil;
    }

    if (pre_frame_state == FrameState.kFrameStateSpeech) {
      return AudioChangeState.kChangeStateSpeech2Speech;
    }

    return AudioChangeState.kChangeStateInvalid;
  }

  int FrameSizeMs() {
    return frame_size_ms;
  }
}

class FsmnVADStreaming {
  VADXOptions vad_opts = VADXOptions();

  Map encoder_conf = {
    'fsmn_layers': 4,
    'input_affine_dim': 140,
    'input_dim': 400,
    'linear_dim': 250,
    'lorder': 20,
    'lstride': 1,
    'output_affine_dim': 140,
    'output_dim': 248,
    'proj_dim': 128,
    'rorder': 0,
    'rstride': 0
  };

  void ResetDetection(Map cache) {
    cache["stats"].continous_silence_frame_count = 0;
    cache["stats"].latest_confirmed_speech_frame = 0;
    cache["stats"].lastest_confirmed_silence_frame = -1;
    cache["stats"].confirmed_start_frame = -1;
    cache["stats"].confirmed_end_frame = -1;
    cache["stats"].vad_state_machine =
        VadStateMachine.kVadInStateStartPointNotDetected;
    ;
    cache["windows_detector"].Reset();
    cache["stats"].sil_frame = 0;
    cache["stats"].frame_probs = List<double>.empty(growable: true);

    if (cache["stats"].output_data_buf.length != 0) {
      assert(cache["stats"].output_data_buf.last.contain_seg_end_point == true);
      int dropFrames =
          (cache["stats"].output_data_buf.last.end_ms ~/ vad_opts.frame_in_ms);
      int realDropFrames = dropFrames - (cache["stats"].last_drop_frames as int);
      cache["stats"].last_drop_frames = dropFrames;
      cache["stats"].data_buf_all = cache["stats"].data_buf_all.sublist(
          realDropFrames *
              (vad_opts.frame_in_ms * vad_opts.sample_rate ~/ 1000));
      cache["stats"].decibel = cache["stats"].decibel.sublist(realDropFrames);
      if (realDropFrames == cache["stats"].scores.length){
        cache["stats"].scores = List<List<double>>.empty(growable: true);
      }else{
        for(var i = 0; i<realDropFrames; i++){

          cache["stats"].scores.removeAt(i);
        }
      }

    }
  }

  void ComputeDecibel(Map cache) {
    int frameSampleLength =
        (vad_opts.frame_length_ms * vad_opts.sample_rate ~/ 1000);
    int frameShiftLength =
        (vad_opts.frame_in_ms * vad_opts.sample_rate ~/ 1000);
    cache["stats"].data_buf_all = cache["stats"].waveform;
    cache["stats"].data_buf = cache["stats"].data_buf_all;
    cache["stats"].decibel = List<double>.empty(growable: true);
    var frameDecibel = 0.00001;
    for (var offset = 0;
        offset < (cache["stats"].waveform.length - frameSampleLength + 1);
        offset += frameShiftLength) {
      frameDecibel = 0.00001;
      var frame =
          cache["stats"].waveform.sublist(offset, offset + frameSampleLength);
      for (var f in frame) {
        frameDecibel += sqrt(f);
      }
      cache["stats"].decibel.add(log(frameDecibel) / log(10));
    }
  }

  ComputeScores(List<List<double>> scores, Map cache) {
    vad_opts.nn_eval_block_size = scores.length;
    cache["stats"].frm_cnt += scores.length;

    cache["stats"].scores = scores;
  }

  void PopDataBufTillFrame(int frameIdx, Map cache) {
    while (cache["stats"].data_buf_start_frame < frameIdx) {
      if (cache["stats"].data_buf.length >=
          (vad_opts.frame_in_ms * vad_opts.sample_rate ~/ 1000)) {
        cache["stats"].data_buf_start_frame += 1;
        cache["stats"].data_buf = cache["stats"].data_buf_all.sublist(
            (cache["stats"].data_buf_start_frame -
                    cache["stats"].last_drop_frames) *
                (vad_opts.frame_in_ms * vad_opts.sample_rate ~/ 1000));
      }
    }
  }

  void PopDataToOutputBuf(
    int startFrm,
    int frmCnt,
    bool firstFrmIsStartPoint,
    bool lastFrmIsEndPoint,
    bool endPointIsSentEnd,
    Map cache,
  ) {
    PopDataBufTillFrame(startFrm, cache = cache);

    int expectedSampleNumber =
        (frmCnt * vad_opts.sample_rate * vad_opts.frame_in_ms ~/ 1000);
    if (lastFrmIsEndPoint) {
      int extraSample = max(
        0,
        (vad_opts.frame_length_ms * vad_opts.sample_rate ~/ 1000 -
            vad_opts.sample_rate * vad_opts.frame_in_ms ~/ 1000),
      );
      expectedSampleNumber += (extraSample);
      if (endPointIsSentEnd) {
        expectedSampleNumber =
            max(expectedSampleNumber, (cache["stats"].data_buf.length));
      }
      if (cache["stats"].data_buf.length < expectedSampleNumber) {
        print("error in calling pop data_buf\n");
      }

      if (cache["stats"].output_data_buf.isEmpty || firstFrmIsStartPoint) {
        cache["stats"].output_data_buf.add(E2EVadSpeechBufWithDoa());
        cache["stats"].output_data_buf.last.Reset();
        cache["stats"].output_data_buf.last.start_ms =
            startFrm * vad_opts.frame_in_ms;
        cache["stats"].output_data_buf.last.end_ms =
            cache["stats"].output_data_buf.last.start_ms;
        cache["stats"].output_data_buf.last.doa = 0;
      }
      var curSeg = cache["stats"].output_data_buf.last;
      if (curSeg.end_ms != startFrm * vad_opts.frame_in_ms) {
        print("warning\n");
      }
      var out_pos = curSeg.buffer.length; // cur_seg.buff现在没做任何操作
      var data_to_pop = 0;
      if (endPointIsSentEnd) {
        data_to_pop = expectedSampleNumber;
      } else {
        data_to_pop =
            (frmCnt * vad_opts.frame_in_ms * vad_opts.sample_rate ~/ 1000)
                .toInt();
      }
      if (data_to_pop > cache["stats"].data_buf.length) {
        print(
            'VAD data_to_pop is bigger than cache["stats"].data_buf.size()!!!\n');
        data_to_pop = cache["stats"].data_buf.length;
        expectedSampleNumber = cache["stats"].data_buf.length;
      }

      curSeg.doa = 0;
      for (var sampleCpyOut = 0; sampleCpyOut < data_to_pop; sampleCpyOut++) {
        out_pos++;
      }
      for (var sampleCpyOut = data_to_pop;
          sampleCpyOut < expectedSampleNumber;
          sampleCpyOut++) {
        out_pos++;
      }
      if (curSeg.end_ms != startFrm * vad_opts.frame_in_ms) {
        print("Something wrong with the VAD algorithm\n");
      }
      cache["stats"].data_buf_start_frame += frmCnt;
      curSeg.end_ms = (startFrm + frmCnt) * vad_opts.frame_in_ms;
      if (firstFrmIsStartPoint) {
        curSeg.contain_seg_start_point = true;
      }
      if (lastFrmIsEndPoint) {
        curSeg.contain_seg_end_point = true;
      }
    }
  }

  void OnSilenceDetected(int validFrame, [Map cache = const {}]) {
    cache["stats"]!.lastest_confirmed_silence_frame = validFrame;
    if (cache["stats"]!.vad_state_machine ==
        VadStateMachine.kVadInStateStartPointNotDetected) {
      PopDataBufTillFrame(validFrame, cache = cache);
    }
  }

  void OnVoiceDetected(int validFrame, [Map cache = const {}]) {
    cache["stats"]!.latest_confirmed_speech_frame = validFrame;
    PopDataToOutputBuf(validFrame, 1, false, false, false, cache = cache);
  }

  void OnVoiceStart(int startFrame, Map cache, bool fakeResult) {
    if (vad_opts.do_start_point_detection) {
      // pass
    }
    if (cache["stats"]!.confirmed_start_frame != -1) {
      print("not reset vad properly\n");
    } else {
      cache["stats"]!.confirmed_start_frame = startFrame;
    }

    if (!fakeResult &&
        cache["stats"]!.vad_state_machine ==
            VadStateMachine.kVadInStateStartPointNotDetected) {
      PopDataToOutputBuf(cache["stats"]!.confirmed_start_frame, 1, true, false,
          false, cache = cache);
    }
  }

  void OnVoiceEnd(int endFrame, bool fakeResult, bool isLastFrame,
      [Map cache = const {}]) {
    for (int t = cache["stats"]!.latest_confirmed_speech_frame + 1;
        t < endFrame;
        t++) {
      OnVoiceDetected(t, cache);
    }
    if (vad_opts.do_end_point_detection) {
      // pass
    }
    if (cache["stats"]!.confirmed_end_frame != -1) {
      print("not reset vad properly\n");
    } else {
      cache["stats"]!.confirmed_end_frame = endFrame;
    }
    if (!fakeResult) {
      cache["stats"]!.sil_frame = 0;
      PopDataToOutputBuf(cache["stats"]!.confirmed_end_frame, 1, false, true,
          isLastFrame, cache);
    }
    cache["stats"]!.number_end_time_detected++;
  }

  void MaybeOnVoiceEndIfLastFrame(bool isFinalFrame, int curFrmIdx,
      Map cache) {
    if (isFinalFrame) {
      OnVoiceEnd(curFrmIdx, false, true, cache);
      cache["stats"]!.vad_state_machine =
          VadStateMachine.kVadInStateEndPointDetected;
    }
  }

  int GetLatency([Map cache = const {}]) {
    return (LatencyFrmNumAtStartPoint(cache) * vad_opts.frame_in_ms).toInt();
  }

  int LatencyFrmNumAtStartPoint(Map cache) {
    int vad_latency = cache["windows_detector"].GetWinSize();
    if (vad_opts.do_extend == 1) {
      vad_latency += (vad_opts.lookback_time_start_point ~/ vad_opts.frame_in_ms);
    }
    return vad_latency;
  }

  int GetFrameState(int t, Map cache) {
    int frameState = FrameState.kFrameStateInvalid;
    double curDecibel = cache["stats"].decibel[t];
    double curSnr = curDecibel - cache["stats"].noise_average_decibel;

// for each frame, calc log posterior probability of each state
    if (curDecibel < vad_opts.decibel_thres) {
      frameState = FrameState.kFrameStateSil;
      DetectOneFrame(frameState, t, false, cache = cache);
      return frameState;
    }

    double sum_score = 0.0;
    double noise_prob = 0.0;
    assert(cache["stats"].sil_pdf_ids.length == vad_opts.silence_pdf_num);
    if (cache["stats"].sil_pdf_ids.isNotEmpty) {
      List<double> silPdfScores = [
        for (int sil_pdf_id in cache["stats"].sil_pdf_ids)
          cache["stats"].scores[t][sil_pdf_id]
      ];
      sum_score = silPdfScores.reduce((value, element) => value + element);
      noise_prob = log(sum_score) * vad_opts.speech_2_noise_ratio;
      double totalScore = 1.0;
      sum_score = totalScore - sum_score;
    }
    double speech_prob = log(sum_score);
    if (vad_opts.output_frame_probs) {
      E2EVadFrameProb frameProb = E2EVadFrameProb();
      frameProb.noise_prob = noise_prob;
      frameProb.speech_prob = speech_prob;
      frameProb.score = sum_score;
      frameProb.frame_id = t;
      cache["stats"].frame_probs.add(frameProb);
    }
    if (exp(speech_prob) >=
        exp(noise_prob) + cache["stats"].speech_noise_thres) {
      if (curSnr >= vad_opts.snr_thres &&
          curDecibel >= vad_opts.decibel_thres) {
        frameState = FrameState.kFrameStateSpeech;
      } else {
        frameState = FrameState.kFrameStateSil;
      }
    } else {
      frameState = FrameState.kFrameStateSil;
      if (cache["stats"].noise_average_decibel < -99.9) {
        cache["stats"].noise_average_decibel = curDecibel;
      } else {
        cache["stats"].noise_average_decibel = (curDecibel +
                cache["stats"].noise_average_decibel *
                    (vad_opts.noise_frame_num_used_for_snr - 1)) /
            vad_opts.noise_frame_num_used_for_snr;
      }
    }
    return frameState;
  }

  Map init_cache(Map cache) {
    cache["frontend"] = {};
    cache["prev_samples"] = List.empty(growable: true);
    cache["encoder"] = {};
    WindowDetector windowsDetector = WindowDetector(
      vad_opts.window_size_ms,
      vad_opts.sil_to_speech_time_thres,
      vad_opts.speech_to_sil_time_thres,
      vad_opts.frame_in_ms,
    );
    windowsDetector.Reset();
    Stats stats = Stats(
        vad_opts.sil_pdf_ids,
        (vad_opts.max_end_silence_time - vad_opts.speech_to_sil_time_thres)
            .toDouble(),
        vad_opts.speech_noise_thres);
    cache["windows_detector"] = windowsDetector;
    cache["stats"] = stats;
    return cache;
  }

  List<List<int>> forward(
    List<List<double>> scores,
    List<double> waveform,
  ) {
    bool isFinal = false;
    Map cache = init_cache({});

    cache["stats"].waveform = waveform;
    ComputeDecibel(cache = cache);
    ComputeScores(scores, cache = cache);

    DetectLastFrames(cache = cache);

    List<List<int>> segments = [];
    if (cache["stats"].output_data_buf.isNotEmpty) {
      for (int i = cache["stats"].output_data_buf_offset,
              len = cache["stats"].output_data_buf.length;
          i < len;
          i++) {
        if (!isFinal &&
            (!cache["stats"].output_data_buf[i].contain_seg_start_point ||
                !cache["stats"].output_data_buf[i].contain_seg_end_point)) {
          continue;
        }
        List<int> segment = [
          cache["stats"].output_data_buf[i].startMs,
          cache["stats"].output_data_buf[i].endMs,
        ];
        cache["stats"].output_data_buf_offset++; // need update this parameter
        segments.add(segment);
      }
    }
    // if (isFinal) {
    //   // reset class variables and clear the dict for the next query
    //     AllResetDetection();
    // }
    return segments;
  }

  int DetectLastFrames(Map cache) {
    if (cache["stats"].vad_state_machine ==
        VadStateMachine.kVadInStateEndPointDetected) {
      return 0;
    }

    for (int i = vad_opts.nn_eval_block_size - 1; i >= 0; i--) {
      int frameState = FrameState.kFrameStateInvalid;
      frameState = GetFrameState(
        cache["stats"].frm_cnt - 1 - i - cache["stats"].last_drop_frames,
        cache = cache,
      );

      if (i != 0) {
        DetectOneFrame(
            frameState, cache["stats"].frm_cnt - 1 - i, false, cache = cache);
      } else {
        DetectOneFrame(
            frameState, cache["stats"].frm_cnt - 1, true, cache = cache);
      }
    }

    return 0;
  }

  void DetectOneFrame(
      int curFrmState, int curFrmIdx, bool isFinalFrame, Map cache) {
    int tmpCurFrmState = FrameState.kFrameStateInvalid;
    if (curFrmState == FrameState.kFrameStateSpeech) {
      if (1 > vad_opts.fe_prior_thres) {
        tmpCurFrmState = FrameState.kFrameStateSpeech;
      } else {
        tmpCurFrmState = FrameState.kFrameStateSil;
      }
    } else if (curFrmState == FrameState.kFrameStateSil) {
      tmpCurFrmState = FrameState.kFrameStateSil;
    }

    int stateChange = cache["windows_detector"]
        .DetectOneFrame(tmpCurFrmState, curFrmIdx, cache = cache);

    int frmShiftInMs = vad_opts.frame_in_ms;
    if (AudioChangeState.kChangeStateSil2Speech == stateChange) {
      cache["stats"].continous_silence_frame_count = 0;
      cache["stats"].pre_end_silence_detected = false;
      int silence_frame_count = cache["stats"].continous_silence_frame_count;
      int start_frame = 0;
      if (cache["stats"].vad_state_machine ==
          VadStateMachine.kVadInStateStartPointNotDetected) {
        start_frame = max(
          cache["stats"].data_buf_start_frame,
          curFrmIdx - LatencyFrmNumAtStartPoint(cache = cache),
        );
        OnVoiceStart(start_frame, cache = cache, false);
        cache["stats"].vad_state_machine =
            VadStateMachine.kVadInStateInSpeechSegment;
        for (var i = start_frame + 1; i < curFrmIdx + 1; i++) {
          OnVoiceDetected(i, cache = cache);
        }
      } else if (cache["stats"].vad_state_machine ==
          VadStateMachine.kVadInStateInSpeechSegment) {
        for (var t = cache["stats"].latest_confirmed_speech_frame + 1;
            t < curFrmIdx; t ++) {
          OnVoiceDetected(t, cache = cache);
        }
        if (curFrmIdx - cache["stats"].confirmed_start_frame + 1 >
            vad_opts.max_single_segment_time / frmShiftInMs) {
          OnVoiceEnd(curFrmIdx, false, false, cache = cache);
          cache["stats"].vad_state_machine =
              VadStateMachine.kVadInStateEndPointDetected;
        } else if (!isFinalFrame) {
            OnVoiceDetected(curFrmIdx, cache = cache);
        } else {
            MaybeOnVoiceEndIfLastFrame(
              isFinalFrame, curFrmIdx, cache = cache);
        }
      }
    } else if (AudioChangeState.kChangeStateSpeech2Sil == stateChange) {
      cache["stats"].continous_silence_frame_count = 0;
      if (cache["stats"].vad_state_machine ==
          VadStateMachine.kVadInStateStartPointNotDetected) {
      } else if (cache["stats"].vad_state_machine ==
          VadStateMachine.kVadInStateInSpeechSegment) {
        if (curFrmIdx - cache["stats"].confirmed_start_frame + 1 >
              vad_opts.max_single_segment_time / frmShiftInMs) {
            OnVoiceEnd(curFrmIdx, false, false, cache = cache);
          cache["stats"].vad_state_machine =
              VadStateMachine.kVadInStateEndPointDetected;
        } else if (!isFinalFrame) {
          OnVoiceDetected(curFrmIdx, cache = cache);
        } else {
          MaybeOnVoiceEndIfLastFrame(
              isFinalFrame, curFrmIdx, cache = cache);
        }
      }
    } else if (AudioChangeState.kChangeStateSpeech2Speech == stateChange) {
      cache["stats"].continous_silence_frame_count = 0;
      if (cache["stats"].vad_state_machine ==
          VadStateMachine.kVadInStateInSpeechSegment) {
        if (curFrmIdx - cache["stats"].confirmed_start_frame + 1 >
              vad_opts.max_single_segment_time / frmShiftInMs) {
          cache["stats"].max_time_out = true;
            OnVoiceEnd(curFrmIdx, false, false, cache = cache);
          cache["stats"].vad_state_machine =
              VadStateMachine.kVadInStateEndPointDetected;
        } else if (!isFinalFrame) {
            OnVoiceDetected(curFrmIdx, cache = cache);
        } else {
            MaybeOnVoiceEndIfLastFrame(
              isFinalFrame, curFrmIdx, cache = cache);
        }
      }
    } else if (AudioChangeState.kChangeStateSil2Sil == stateChange) {
      cache["stats"].continous_silence_frame_count += 1;
      if (cache["stats"].vad_state_machine ==
          VadStateMachine.kVadInStateStartPointNotDetected) {
        // silence timeout, return zero length decision
        if (  vad_opts.detect_mode ==
                    VadDetectMode.kVadSingleUtteranceDetectMode &&
                cache["stats"].continous_silence_frame_count * frmShiftInMs >
                      vad_opts.max_start_silence_time ||
            isFinalFrame && cache["stats"].number_end_time_detected == 0) {
          for (var t = cache["stats"].lastest_confirmed_silence_frame + 1;
              t <
                  curFrmIdx -
                      (cache["stats"].lastest_confirmed_silence_frame + 1);) {
              OnSilenceDetected(t, cache = cache);
          }
          OnVoiceStart(0, cache = cache, true);
          OnVoiceEnd(0, true, false, cache = cache);
          cache["stats"].vad_state_machine =
              VadStateMachine.kVadInStateEndPointDetected;
        } else{
          if (curFrmIdx >=
              LatencyFrmNumAtStartPoint(cache = cache)) {
            OnSilenceDetected(
                curFrmIdx -   LatencyFrmNumAtStartPoint(cache = cache),
                cache = cache);
          }
        }
      } else if (cache["stats"].vad_state_machine ==
          VadStateMachine.kVadInStateInSpeechSegment) {
        if (cache["stats"].continous_silence_frame_count * frmShiftInMs >=
            cache["stats"].max_end_sil_frame_cnt_thresh) {
          int lookbackFrame =
              (cache["stats"].max_end_sil_frame_cnt_thresh ~/ frmShiftInMs);
          if ( vad_opts.do_extend == 1) {
            lookbackFrame -=
                (vad_opts.lookahead_time_end_point ~/ frmShiftInMs);
            lookbackFrame -= 1;
            lookbackFrame = max(0, lookbackFrame);
          }
            OnVoiceEnd(
              curFrmIdx - lookbackFrame, false, false, cache = cache);
          cache["stats"].vad_state_machine =
              VadStateMachine.kVadInStateEndPointDetected;
        } else if (curFrmIdx - cache["stats"].confirmed_start_frame + 1 >
              vad_opts.max_single_segment_time / frmShiftInMs) {
            OnVoiceEnd(curFrmIdx, false, false, cache = cache);
          cache["stats"].vad_state_machine =
              VadStateMachine.kVadInStateEndPointDetected;
        } else if ((  vad_opts.do_extend == 1) && !isFinalFrame) {
          if (cache["stats"].continous_silence_frame_count <=
              (  vad_opts.lookahead_time_end_point ~/ frmShiftInMs)) {
              OnVoiceDetected(curFrmIdx, cache = cache);
          }
        } else {
            MaybeOnVoiceEndIfLastFrame(
              isFinalFrame, curFrmIdx, cache = cache);
        }
      }
    }
    if (cache["stats"].vad_state_machine ==
            VadStateMachine.kVadInStateEndPointDetected &&
          vad_opts.detect_mode ==
            VadDetectMode.kVadMutipleUtteranceDetectMode) {
        ResetDetection(cache = cache);
    }
  }
}


class FsmnVaDetector {

  OrtSessionOptions? _sessionOptions;
  OrtSession? _session;
  bool isInitialed = false;
  /// model states
  var _triggered = false;
  FsmnVADStreaming? streaming = FsmnVADStreaming();

  FsmnVaDetector();

  reset() {
    _triggered = false;

  }

  release() {
    _sessionOptions?.release();
    _sessionOptions = null;
    _session?.release();
    _session = null;
    streaming = null;
  }

  Future<bool> initModel(String path) async {
    _sessionOptions = OrtSessionOptions()
      ..setInterOpNumThreads(1)
      ..setIntraOpNumThreads(1)
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);
    const assetFileName = 'assets/models/fsmn_vad.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    _session = OrtSession.fromBuffer(bytes, _sessionOptions!);
    isInitialed = true;
    return true;
  }

  Future<bool> initModelWrapper(String path){
    return compute(initModel, path);
  }
  
  List<double> int2double(List<int> intData){
    List<double> doubleData = intData.map((e) => e / 32768).toList();
    return doubleData;
  }

  doubleList2FloatList(List<List<double>> data){
    List<Float32List> out = List.empty(growable: true);
    for (var i = 0; i < data.length; i++) {
      var flist = Float32List.fromList(data[i]);
      out.add(flist);
    }
    return out;
  }

  Future<List<List<int>>?> predict(List<int> intData) async {
    final feature = extractFbankOnline(intData);
    final inputOrt = OrtValueTensor.createTensorWithDataList(doubleList2FloatList(feature), [1, feature.length, 400]);
    final runOptions = OrtRunOptions();
    final inputs = {'feats': inputOrt};
    final List<OrtValue?>? outputs;

    outputs = await _session?.runAsync(runOptions, inputs);
    inputOrt.release();

    runOptions.release();
    /// Output probability & update h,c recursively
    final output = (outputs?[0]?.value as List<List<List<double>>>)[0];

    outputs?.forEach((element) {
      element?.release();
    });
    List<List<int>>? segments = streaming?.forward(output, int2double(intData));

    return segments;
  }
}


// void main(){
//   FsmnVADStreaming streaming = FsmnVADStreaming();
//   for(var i = 0; i < 100; i++){
//     List<double> waveform = List<double>.generate(190000, (i) => Random().nextInt(255).toDouble(), growable: true);
//     List<List<double>> scores = List<List<double>>.filled(1186, List<double>.generate(248, (i) =>  Random().nextDouble()), growable: true);
//     final segments = streaming.forward(scores, waveform);
//     print(segments);
//     scores.removeAt(0);
//   }
// }