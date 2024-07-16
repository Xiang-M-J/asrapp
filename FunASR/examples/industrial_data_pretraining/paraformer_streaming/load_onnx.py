import onnxruntime
import torch
import torchaudio

from funasr.frontends.wav_frontend import WavFrontendOnline

base_path = (r"D:\work\asrapp\FunASR-main\examples\industrial_data_pretraining\paraformer_streaming\iic"
             r"\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online\\")

encoder_session = onnxruntime.InferenceSession(base_path + "model.quant.onnx")
decoder_session = onnxruntime.InferenceSession(base_path + "decoder.quant.onnx")

model_session = onnxruntime.InferenceSession(base_path + "totalModel_quant.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# WavFrontendOnline
args = {'cmvn_file': base_path + 'am.mvn', 'dither': 0.0, 'frame_length': 25, 'frame_shift': 10, 'fs': 16000,
        'lfr_m': 7, 'lfr_n': 6, 'n_mels': 80, 'window': 'hamming'}
frontend = WavFrontendOnline(**args)


def load_audio(wave, cache, is_final):
    # audio = audio * (1 << 15)  # audio 为整数 大概为 15 16 ...
    data, data_len = frontend(wave, wave.shape[-1], is_final=is_final, cache=cache)

    return data, data_len


speech, sr = torchaudio.load("test.wav")

chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1
chunk_stride = chunk_size[1] * 960  # 600ms、480ms

cache = {}
total_chunk_num = int(((speech.shape[-1]) - 1) / chunk_stride + 1)

is_first = True


def init_cache(cache: dict = {}):
    chunk_size = [0, 10, 5]
    encoder_chunk_look_back = 4
    decoder_chunk_look_back = 1
    batch_size = 1

    enc_output_size = 512
    feats_dims = 80 * 7
    cache_encoder = {
        "start_idx": 0,
        "cif_hidden": torch.zeros((batch_size, 1, enc_output_size)),
        "cif_alphas": torch.zeros((batch_size, 1)),
        "chunk_size": chunk_size,
        "encoder_chunk_look_back": encoder_chunk_look_back,
        "last_chunk": False,
        "opt": None,
        "feats": torch.zeros((batch_size, chunk_size[0] + chunk_size[2], feats_dims)),
        "tail_chunk": False,
    }
    cache["encoder"] = cache_encoder

    cache_decoder = {
        "decode_fsmn": None,
        "decoder_chunk_look_back": decoder_chunk_look_back,
        "opt": None,
        "chunk_size": chunk_size,
    }
    cache["decoder"] = cache_decoder
    cache["frontend"] = {}
    cache["prev_samples"] = torch.empty(0)

    return cache


init_cache(cache)
n_feats = None
cif_hidden = None
cif_alphas = None

for i in range(total_chunk_num):
    speech_chunk = speech[0, i * chunk_stride: (i + 1) * chunk_stride]
    audio_sample = torch.cat((cache["prev_samples"], speech_chunk))
    is_final = i == total_chunk_num - 1
    chunk_stride_samples = int(chunk_size[1] * 960)
    n = int((audio_sample.shape[-1]) // chunk_stride_samples + int(is_final))
    m = int((audio_sample.shape[-1]) % chunk_stride_samples * (1 - int(is_final)))
    for j in range(n):
        is_final = is_final and j == n - 1
        audio_sample_i = audio_sample[j * chunk_stride_samples: (j + 1) * chunk_stride_samples]
        if is_final and len(audio_sample_i) < 960:
            cache["encoder"]["tail_chunk"] = True
            feats = cache["encoder"]["feats"]
            feats_len = torch.tensor([speech.shape[1]], dtype=torch.int64).to(
                speech.device
            )
        else:
            feats, feats_len = load_audio(torch.unsqueeze(audio_sample_i, 0), is_final=is_final,
                                          cache=cache["frontend"])
            feats_len = torch.tensor(feats_len, dtype=torch.int32)
        # ort_inputs = {"speech": to_numpy(feats), "speech_lengths": to_numpy(feats_len)}
        # # ort_inputs = {ort_session.get_inputs()[0], "speech_lengths": to_numpy(torch.Tensor([128]))}
        # enc, enc_len, alphas = encoder_session.run(None, ort_inputs)
        # acoustic_embeds = torch.randn(2, 10, 512).type(torch.float32)
        # acoustic_embeds_len = torch.tensor([5, 10], dtype=torch.int32)
        # ort_inputs1 = {"enc": enc, "enc_len": enc_len, "acoustic_embeds": to_numpy(acoustic_embeds),
        #                "acoustic_embeds_len": to_numpy(acoustic_embeds_len)}
        # cache_num = 16
        # for v in range(cache_num):
        #     ort_inputs1.update({f"in_cache_{v}": to_numpy(torch.zeros(2, 512, 10))})
        # decode_outputs = decoder_session.run(None, ort_inputs1)
        # logits = decode_outputs[0]
        # ids = decode_outputs[1]

        if is_first:
            ort_inputs = {"speech": to_numpy(feats), "feats": to_numpy(torch.zeros(1, 5, 560)),
                          "cif_hidden": to_numpy(torch.randn(1, 1, 512)), "cif_alphas": to_numpy(torch.randn(1, 1))}
        else:
            ort_inputs = {"speech": to_numpy(feats), "feats": n_feats, "cif_hidden": cif_hidden,
                          "cif_alphas": cif_alphas}

        output = model_session.run(None, ort_inputs)
        # decode_outputs = decoder_session.run(None, ort_inputs1)
        logits = output[0]
        ids = output[1]
        n_feats = output[2]
        cif_hidden = output[3]
        cif_alphas = output[4]
        is_first = False
        print(ids)
    cache["prev_samples"] = audio_sample[:-m]
