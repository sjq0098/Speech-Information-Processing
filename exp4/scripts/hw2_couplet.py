# coding=utf-8
"""HW2: 多音字对联合成 (郭沫若读法).
Baseline (wrong) -> zh.wav ; custom frontend (correct) -> homework_2.wav.
Also dumps per-piece phones/tones/word2ph and a waveform/spectrogram comparison figure.
Run from MeloTTS-Homework dir with the speechmelotts env.
"""
import os, sys, json
import numpy as np
import torch

OUT = r"d:/Speech Information Processing/exp4/outputs"
FIG = r"d:/Speech Information Processing/exp4/figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

from melo.api import TTS
from melo.text.cleaner import clean_text

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
text = "海水朝，朝朝朝，朝朝朝落；浮云长，长长长，长长长消。"
model = TTS(language='ZH', device=device)
spk = model.hps.data.spk2id['ZH']
lang = model.language  # ZH_MIX_EN
print("model.language =", lang)

# 1) baseline wrong reading
model.tts_to_file(text, spk, os.path.join(OUT, 'hw2_baseline_wrong.wav'), speed=0.8, quiet=True)
print("baseline (wrong) saved")

# 2) split into sub-sentences
pieces = TTS.split_sentences_into_pieces(text, language='ZH', quiet=True)
print("pieces:", pieces, "n=", len(pieces))

# target 郭沫若 readings, consumed in order across the whole couplet
# 朝: cháo=(2,'ch'), zhāo=(1,'zh')   长: cháng=(2,'ch'), zhǎng=(3,'zh')
zhao_q = [(2,'ch'),(1,'zh'),(1,'zh'),(2,'ch'),(1,'zh'),(2,'ch'),(1,'zh')]  # 海水[朝] [朝朝朝] [朝朝朝]落
chang_q = [(3,'zh'),(2,'ch'),(2,'ch'),(3,'zh'),(2,'ch'),(3,'zh'),(2,'ch')] # 浮云[长] [长长长] [长长长]消
zi = ci = 0

phones_customized, tones_customized, word2ph_customized = [], [], []
align_rows = []  # (piece, char, pinyin, phones, tone) for the report
for pi, piece in enumerate(pieces):
    norm, ph, tn, w2p = clean_text(piece, lang)
    ph = list(ph); tn = list(tn); w2p = list(w2p)
    # word2ph aligns to BERT tokens = ['_'] + [non-space chars of norm] + ['_']
    chars = ['_'] + [c for c in norm if c != ' '] + ['_']
    assert len(chars) == len(w2p), (len(chars), len(w2p), chars)
    p = 0
    for j, c in enumerate(chars):
        cnt = w2p[j]
        if c == '朝':
            t, init = zhao_q[zi]; zi += 1
            ph[p:p+cnt] = [init, 'ao']; tn[p:p+cnt] = [t]*cnt
            align_rows.append((pi, c, ('cháo' if t==2 else 'zhāo'), ph[p:p+cnt], t))
        elif c == '长':
            t, init = chang_q[ci]; ci += 1
            ph[p:p+cnt] = [init, 'ang']; tn[p:p+cnt] = [t]*cnt
            align_rows.append((pi, c, ('cháng' if t==2 else 'zhǎng'), ph[p:p+cnt], t))
        p += cnt
    assert sum(w2p) == len(ph) == len(tn)
    phones_customized.append(list(ph)); tones_customized.append(list(tn)); word2ph_customized.append(list(w2p))
    print(f"--- piece {pi}: {piece} | norm={norm}")
    print("chars  :", chars)
    print("phones :", ph)
    print("tones  :", tn)
    print("word2ph:", w2p)

print("consumed 朝:", zi, " 长:", ci)
for r in align_rows: print("  poly:", r)

# save the readable lists BEFORE synthesis mutates word2ph in place
json.dump({'text':text,'pieces':pieces,
           'phones':[list(p) for p in phones_customized],
           'tones':[list(t) for t in tones_customized],
           'word2ph':[list(w) for w in word2ph_customized],
           'align_rows':align_rows},
          open(os.path.join(OUT,'hw2_lists.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=1)

# 3) synth with custom frontend
model.tts_to_file_custom_frontend(
    text, spk, os.path.join(OUT, 'homework_2.wav'), speed=0.8,
    phones_customized=phones_customized, tones_customized=tones_customized,
    word2ph_customized=word2ph_customized, quiet=True)
print("homework_2.wav saved")

# 4) figure: waveform + spectrogram, wrong vs correct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa, librosa.display
sr = model.hps.data.sampling_rate
wa,_ = librosa.load(os.path.join(OUT,'hw2_baseline_wrong.wav'), sr=sr)
wb,_ = librosa.load(os.path.join(OUT,'homework_2.wav'), sr=sr)
fig, ax = plt.subplots(2,2, figsize=(12,6))
for col,(w,name) in enumerate([(wa,'MeloTTS default (all cháo / cháng)'),(wb,"Guo Moruo reading (custom frontend)")]):
    t = np.arange(len(w))/sr
    ax[0,col].plot(t, w, lw=0.4, color='#1f77b4'); ax[0,col].set_title(name, fontsize=10)
    ax[0,col].set_xlabel('time (s)'); ax[0,col].set_ylabel('amp')
    S = librosa.amplitude_to_db(np.abs(librosa.stft(w, n_fft=1024, hop_length=256)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, hop_length=256, x_axis='time', y_axis='hz', ax=ax[1,col], cmap='magma')
    ax[1,col].set_ylim(0,8000)
fig.suptitle('HW2: Couplet synthesis - default vs. Guo Moruo polyphone reading', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG,'hw2_couplet_compare.png'), dpi=130)
print("figure saved")

print("DONE HW2")
