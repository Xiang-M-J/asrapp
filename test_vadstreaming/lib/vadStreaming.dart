class VadStateMachine {
  static int kVadInStateStartPointNotDetected = 1;
  int kVadInStateInSpeechSegment = 2;
  int kVadInStateEndPointDetected = 3;
}

class FrameState {
  int kFrameStateInvalid = -1;
  int kFrameStateSpeech = 1;
  int kFrameStateSil = 0;
}

class AudioChangeState {
  int kChangeStateSpeech2Speech = 0;
  int kChangeStateSpeech2Sil = 1;
  int kChangeStateSil2Sil = 2;
  int kChangeStateSil2Speech = 3;
  int kChangeStateNoBegin = 4;
  int kChangeStateInvalid = 5;
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
  int sil_pdf_ids;
  double noise_average_decibel = -100.0;
  bool pre_end_silence_detected = false;
  bool next_seg = true;
  List<double> output_data_buf = [];
  int output_data_buf_offset = 0;
  List<double> frame_probs = [];
  double max_end_sil_frame_cnt_thresh;
  double speech_noise_thres;
  List<double>? scores;
  bool max_time_out = false;
  List<double> decibel = [];
  List<double>? data_buf;
  List<double>? data_buf_all;
  List<double>? waveform;
  int last_drop_frames = 0;
  Stats(this.sil_pdf_ids, this.max_end_sil_frame_cnt_thresh,
      this.speech_noise_thres);
}


class WindowDetector{
  int window_size_ms;
  int sil_to_speech_time;
  int speech_to_sil_time;
  int frame_size_ms;

  WindowDetector(this.window_size_ms, this.sil_to_speech_time, this.speech_to_sil_time, this.frame_size_ms){
    int win_size_frame = int(window_size_ms / frame_size_ms);

  }
}
   
    def __init__(
        self,
        window_size_ms: int,
        sil_to_speech_time: int,
        speech_to_sil_time: int,
        frame_size_ms: int,
    ):
        self.window_size_ms = window_size_ms
        self.sil_to_speech_time = sil_to_speech_time
        self.speech_to_sil_time = speech_to_sil_time
        self.frame_size_ms = frame_size_ms

        self.win_size_frame = int(window_size_ms / frame_size_ms)
        self.win_sum = 0
        self.win_state = [0] * self.win_size_frame  # 初始化窗

        self.cur_win_pos = 0
        self.pre_frame_state = FrameState.kFrameStateSil
        self.cur_frame_state = FrameState.kFrameStateSil
        self.sil_to_speech_frmcnt_thres = int(sil_to_speech_time / frame_size_ms)
        self.speech_to_sil_frmcnt_thres = int(speech_to_sil_time / frame_size_ms)

        self.voice_last_frame_count = 0
        self.noise_last_frame_count = 0
        self.hydre_frame_count = 0

    def Reset(self) -> None:
        self.cur_win_pos = 0
        self.win_sum = 0
        self.win_state = [0] * self.win_size_frame
        self.pre_frame_state = FrameState.kFrameStateSil
        self.cur_frame_state = FrameState.kFrameStateSil
        self.voice_last_frame_count = 0
        self.noise_last_frame_count = 0
        self.hydre_frame_count = 0

    def GetWinSize(self) -> int:
        return int(self.win_size_frame)

    def DetectOneFrame(
        self, frameState: FrameState, frame_count: int, cache: dict = {}
    ) -> AudioChangeState:
        cur_frame_state = FrameState.kFrameStateSil
        if frameState == FrameState.kFrameStateSpeech:
            cur_frame_state = 1
        elif frameState == FrameState.kFrameStateSil:
            cur_frame_state = 0
        else:
            return AudioChangeState.kChangeStateInvalid
        self.win_sum -= self.win_state[self.cur_win_pos]
        self.win_sum += cur_frame_state
        self.win_state[self.cur_win_pos] = cur_frame_state
        self.cur_win_pos = (self.cur_win_pos + 1) % self.win_size_frame

        if (
            self.pre_frame_state == FrameState.kFrameStateSil
            and self.win_sum >= self.sil_to_speech_frmcnt_thres
        ):
            self.pre_frame_state = FrameState.kFrameStateSpeech
            return AudioChangeState.kChangeStateSil2Speech

        if (
            self.pre_frame_state == FrameState.kFrameStateSpeech
            and self.win_sum <= self.speech_to_sil_frmcnt_thres
        ):
            self.pre_frame_state = FrameState.kFrameStateSil
            return AudioChangeState.kChangeStateSpeech2Sil

        if self.pre_frame_state == FrameState.kFrameStateSil:
            return AudioChangeState.kChangeStateSil2Sil
        if self.pre_frame_state == FrameState.kFrameStateSpeech:
            return AudioChangeState.kChangeStateSpeech2Speech
        return AudioChangeState.kChangeStateInvalid

    def FrameSizeMs(self) -> int:
        return int(self.frame_size_ms)