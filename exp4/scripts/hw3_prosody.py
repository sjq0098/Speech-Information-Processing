# coding=utf-8
"""HW3: 韵律控制 - 通过线性缩放 w_ceil 让语速从 1x 线性降到约 0.33x (时长 1x->3x).
生成 original (uniform) 与 homework_3.wav (linearly slowed), 并画 w_ceil / 波形对比图.
"""
import os, math, json
import numpy as np
import torch

OUT = r"d:/Speech Information Processing/exp4/outputs"
FIG = r"d:/Speech Information Processing/exp4/figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

from melo.api import TTS

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
text = "你听到了吗？这句话我会说得越来越慢。"
model = TTS(language='ZH', device=device)
spk = model.hps.data.spk2id['ZH']
sr = model.hps.data.sampling_rate
hop = model.hps.data.hop_length
print("sr", sr, "hop", hop, "unit(ms)", 1000*hop/sr)

# deterministic duration prediction
w_ceil_list, phone_list, tone_list = model.get_original_w_ceil(
    text, spk, speed=1.0, sdp_ratio=0, noise_scale=0, noise_scale_w=0, quiet=True)

# flatten per piece
orig = [w.squeeze().int().tolist() for w in w_ceil_list]
orig = [[o] if isinstance(o, int) else o for o in orig]
lengths = [len(o) for o in orig]
N = sum(lengths)
print("pieces:", len(orig), "phone counts:", lengths, "total:", N)

# global linear multiplier 1.0 -> 3.0 over the whole utterance
START, END = 1.0, 3.0
modified = []
mult_curve = []
gidx = 0
for o in orig:
    m = []
    for v in o:
        factor = START + (END - START) * (gidx / (N - 1))
        mult_curve.append(factor)
        m.append(max(1, int(math.ceil(v * factor))))
        gidx += 1
    modified.append(m)

orig_flat = [v for o in orig for v in o]
mod_flat = [v for m in modified for v in m]
print("orig total frames:", sum(orig_flat), "-> modified total frames:", sum(mod_flat))
print("orig dur (s):", sum(orig_flat)*hop/sr, " modified dur (s):", sum(mod_flat)*hop/sr)

# original uniform-speed audio (deterministic)
model.tts_to_file(text, spk, os.path.join(OUT,'hw3_original.wav'),
                  speed=1.0, sdp_ratio=0, noise_scale=0, noise_scale_w=0, quiet=True)
# slowed audio via custom duration
model.tts_to_file_custom_duration(text, spk, os.path.join(OUT,'homework_3.wav'),
                  speed=1.0, sdp_ratio=0, noise_scale=0, noise_scale_w=0,
                  w_ceil_customized=modified, quiet=True)
print("audios saved")

# symbols for x labels
s2i = model.symbol_to_id; i2s = {v:k for k,v in s2i.items()}
syms = [i2s.get(i,'') for p in phone_list for i in p.flatten().tolist()]

# figure
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
fig = plt.figure(figsize=(13,8))
ax1 = plt.subplot(3,1,1)
x = np.arange(N)
ax1.bar(x-0.2, orig_flat, width=0.4, label='original w_ceil', color='#8fb7e0')
ax1.bar(x+0.2, mod_flat, width=0.4, label='modified w_ceil', color='#d67b7b')
ax1.set_ylabel('duration (frames)'); ax1.legend(loc='upper left'); ax1.set_title('HW3: per-phone duration (w_ceil), original vs. linearly slowed')
ax1b = ax1.twinx(); ax1b.plot(x, mult_curve, 'k--', lw=1.2, label='speed-down factor'); ax1b.set_ylabel('duration x-factor'); ax1b.set_ylim(0.8,3.2)
ax1.set_xticks(x); ax1.set_xticklabels(syms, fontsize=5, rotation=90)

wa,_ = librosa.load(os.path.join(OUT,'hw3_original.wav'), sr=sr)
wb,_ = librosa.load(os.path.join(OUT,'homework_3.wav'), sr=sr)
ax2 = plt.subplot(3,1,2); ax2.plot(np.arange(len(wa))/sr, wa, lw=0.4, color='#1f77b4'); ax2.set_title('original (uniform speed) waveform'); ax2.set_xlabel('time (s)')
ax3 = plt.subplot(3,1,3); ax3.plot(np.arange(len(wb))/sr, wb, lw=0.4, color='#c0392b'); ax3.set_title('homework_3.wav: gradually slowing down (1x -> ~0.33x)'); ax3.set_xlabel('time (s)')
# align x-limits
xmax = max(len(wa),len(wb))/sr
ax2.set_xlim(0,xmax); ax3.set_xlim(0,xmax)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'hw3_prosody.png'), dpi=130)
print("figure saved")

json.dump({'text':text,'phones':syms,'w_ceil_original':orig_flat,'w_ceil_modified':mod_flat,
           'mult_curve':[round(m,3) for m in mult_curve],
           'orig_dur_s':sum(orig_flat)*hop/sr,'mod_dur_s':sum(mod_flat)*hop/sr},
          open(os.path.join(OUT,'hw3_data.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=1)
print("DONE HW3")
