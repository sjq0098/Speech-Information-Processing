# coding=utf-8
"""HW4 part B: quality eval on the 50+50 wavs from hw4_bench.
CER via Whisper-small (ASR), objective MOS via DNSMOS (speechmos).
"""
import os, json, re, numpy as np, librosa, whisper
from speechmos import dnsmos

OUT = r"d:/Speech Information Processing/exp4/outputs"
FIG = r"d:/Speech Information Processing/exp4/figures"
QP = os.path.join(OUT,"quality","pytorch"); QO = os.path.join(OUT,"quality","onnx")
SENTS = json.load(open(os.path.join(OUT,"hw4_rtf.json")))["sentences"]

def hanzi(s): return re.sub(r"[^一-鿿]", "", s)
def cer(ref, hyp):
    r, h = list(ref), list(hyp)
    dp = list(range(len(h)+1))
    for i in range(1, len(r)+1):
        prev = dp[0]; dp[0] = i
        for j in range(1, len(h)+1):
            cur = dp[j]
            dp[j] = min(dp[j]+1, dp[j-1]+1, prev + (r[i-1]!=h[j-1]))
            prev = cur
    return dp[len(h)] / max(1, len(r))

print("loading whisper small (cpu)...")
asr = whisper.load_model("small", device="cpu")

def eval_dir(d, name):
    cers=[]; ov=[]; sg=[]; bk=[]; p8=[]
    for i in range(len(SENTS)):
        wav = os.path.join(d, f"{i:03d}.wav")
        if not os.path.exists(wav): continue
        # load 16k mono float32 (avoid whisper's ffmpeg dependency)
        y, _ = librosa.load(wav, sr=16000, mono=True)
        y = y.astype(np.float32)
        # CER
        txt = asr.transcribe(y, language="zh", fp16=False, verbose=False)["text"]
        c = cer(hanzi(SENTS[i]), hanzi(txt)); cers.append(c)
        # DNSMOS (16k mono)
        r = dnsmos.run(y, sr=16000, return_df=False)
        ov.append(r["ovrl_mos"]); sg.append(r["sig_mos"]); bk.append(r["bak_mos"]); p8.append(r["p808_mos"])
        if i < 3: print(f"  [{name}] {i}: CER={c:.3f} ovrl={r['ovrl_mos']:.3f} | ref={hanzi(SENTS[i])} | hyp={hanzi(txt)}")
    res = dict(n=len(cers), CER=float(np.mean(cers)),
               DNSMOS_OVRL=float(np.mean(ov)), DNSMOS_SIG=float(np.mean(sg)),
               DNSMOS_BAK=float(np.mean(bk)), DNSMOS_P808=float(np.mean(p8)))
    print(f"[{name}] n={res['n']} CER={res['CER']:.4f} OVRL={res['DNSMOS_OVRL']:.3f} "
          f"SIG={res['DNSMOS_SIG']:.3f} BAK={res['DNSMOS_BAK']:.3f} P808={res['DNSMOS_P808']:.3f}")
    return res

rp = eval_dir(QP, "PyTorch")
ro = eval_dir(QO, "ONNX")
json.dump({"pytorch":rp,"onnx":ro}, open(os.path.join(OUT,"hw4_quality.json"),"w"), indent=1)

# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2, figsize=(11,4.2))
ax[0].bar(["PyTorch","ONNX"], [rp["CER"],ro["CER"]], color=["#c0392b","#2471a3"])
ax[0].set_title("CER (Whisper-small, lower=better)"); ax[0].set_ylabel("CER")
for i,v in enumerate([rp["CER"],ro["CER"]]): ax[0].text(i, v, f"{v:.3f}", ha="center", va="bottom")
labels=["OVRL","SIG","BAK","P808"]; x=np.arange(len(labels)); w=0.35
pv=[rp["DNSMOS_OVRL"],rp["DNSMOS_SIG"],rp["DNSMOS_BAK"],rp["DNSMOS_P808"]]
ov=[ro["DNSMOS_OVRL"],ro["DNSMOS_SIG"],ro["DNSMOS_BAK"],ro["DNSMOS_P808"]]
ax[1].bar(x-w/2, pv, w, label="PyTorch", color="#c0392b")
ax[1].bar(x+w/2, ov, w, label="ONNX", color="#2471a3")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].set_title("DNSMOS (higher=better)"); ax[1].legend(); ax[1].set_ylim(0,5)
fig.suptitle("HW4: synthesis quality, PyTorch vs ONNX (50 samples each)")
fig.tight_layout(); fig.savefig(os.path.join(FIG,"hw4_quality.png"), dpi=130)
print("DONE QUALITY")
