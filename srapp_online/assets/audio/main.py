import soundfile as sf

wav, sr = sf.read("asr_example.wav")
wav = wav * (1 << 15)
print(wav.shape)
print(sr)