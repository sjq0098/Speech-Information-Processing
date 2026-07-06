# coding=utf-8
"""HW1: baseline synthesis + parameter-effect demos (measure duration vs speed, etc.)."""
import os, json, numpy as np, torch, soundfile as sf
OUT = r"d:/Speech Information Processing/exp4/outputs"; os.makedirs(OUT, exist_ok=True)
from melo.api import TTS
dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
m = TTS(language='ZH', device=dev); spk = m.hps.data.spk2id['ZH']; sr = m.hps.data.sampling_rate

def synth(name, **kw):
    a = m.tts_to_file(text, spk, None, quiet=True, **kw)
    sf.write(os.path.join(OUT, name), a, sr)
    return len(a)/sr

rows = {}
rows['baseline (speed1.0,sdp0.2,ns0.6,nsw0.8)'] = synth('hw1_baseline.wav')
rows['speed=0.7']  = synth('hw1_speed07.wav', speed=0.7)
rows['speed=1.4']  = synth('hw1_speed14.wav', speed=1.4)
rows['sdp_ratio=0.0'] = synth('hw1_sdp0.wav', sdp_ratio=0.0)
rows['sdp_ratio=1.0'] = synth('hw1_sdp1.wav', sdp_ratio=1.0)
rows['noise_scale=0.1'] = synth('hw1_ns01.wav', noise_scale=0.1)
rows['noise_scale=1.0'] = synth('hw1_ns10.wav', noise_scale=1.0)
for k,v in rows.items(): print(f"{k:45s} dur={v:.3f}s")
json.dump(rows, open(os.path.join(OUT,'hw1_durations.json'),'w'), indent=1)
print("DONE HW1")
