# coding=utf-8
"""HW4 part A: RTF benchmark, PyTorch(melo, CPU) vs ONNX(sherpa-onnx, CPU),
across num_threads in {1,2,4}, over 50 sentences. Saves audio at threads=4 for
quality eval, plus an RTF-vs-threads figure and JSON.
Run with env torch prepended (PYTHONPATH=<env site-packages>).
"""
import os, time, json, numpy as np, soundfile as sf, torch
OUT = r"d:/Speech Information Processing/exp4/outputs"
FIG = r"d:/Speech Information Processing/exp4/figures"
QP = os.path.join(OUT, "quality", "pytorch"); QO = os.path.join(OUT, "quality", "onnx")
for d in (QP, QO, FIG): os.makedirs(d, exist_ok=True)
BUND = r"d:/Speech Information Processing/exp4/onnx_export/vits-melo-tts-zh_en"
MY = r"d:/Speech Information Processing/exp4/onnx_export/mymodel"
THREADS = [1, 2, 4]

SENTS = [
 "今天天气很好，我们一起去公园散步吧。","人工智能正在改变我们的生活方式。","请在下午三点半到会议室开会。",
 "这家餐厅的红烧肉做得非常地道。","科学技术是第一生产力。","他每天早上六点起床跑步锻炼身体。",
 "语音合成技术近年来取得了长足的进步。","北京是中国的首都，也是一座历史文化名城。","春天来了，公园里的花都开了。",
 "我们需要在周五之前完成这个项目。","深度学习模型的训练需要大量的数据。","这本书详细介绍了机器学习的基本原理。",
 "长城是世界文化遗产之一。","请记得带上你的身份证和准考证。","这个季度公司的销售额增长了百分之二十。",
 "音乐能够陶冶人的情操。","孩子们在操场上快乐地玩耍。","黄河是中华民族的母亲河。",
 "他用三种语言流利地进行了演讲。","健康的生活习惯有助于延年益寿。","这幅画的色彩搭配十分和谐。",
 "我打算明年去欧洲旅行一个月。","环境保护是每个公民的责任。","这道数学题的解法非常巧妙。",
 "秋天的枫叶红得像火一样。","他终于实现了自己多年的梦想。","这款手机的续航能力很强。",
 "老师耐心地为同学们讲解难题。","西湖的风景秀丽，令人流连忘返。","坚持每天阅读能够开阔视野。",
 "这次实验的结果超出了我们的预期。","请把音量调小一点，谢谢。","飞机将于晚上八点准时起飞。",
 "他是一位经验丰富的软件工程师。","这条河的水质近年来明显改善了。","博物馆里陈列着许多珍贵的文物。",
 "我们应该珍惜时间，努力学习。","这台电脑的运行速度快得惊人。","樱花盛开的季节吸引了众多游客。",
 "医生建议他多休息，少熬夜。","这首诗表达了作者对故乡的思念。","团队合作是成功的重要因素。",
 "新能源汽车越来越受到消费者的欢迎。","请在表格中填写你的姓名和联系方式。","这场比赛的结果实在让人意外。",
 "海边的日落景色美得像一幅油画。","知识的积累需要日复一日的坚持。","这个应用程序的界面设计很友好。",
 "他在演讲中引用了许多经典的例子。","冬天的早晨，窗户上结满了霜花。",
]
SENTS = SENTS[:50]
print("num sentences:", len(SENTS))

# ---------- PyTorch (melo, CPU) ----------
from melo.api import TTS
mel = TTS(language='ZH', device='cpu'); spk = mel.hps.data.spk2id['ZH']; sr = mel.hps.data.sampling_rate
def pt_synth(t):
    return mel.tts_to_file(t, spk, None, quiet=True)
pt_rtf = {}
for n in THREADS:
    torch.set_num_threads(n)
    for w in SENTS[:2]: pt_synth(w)      # warmup
    tot_i = tot_d = 0.0; rtfs = []
    for i, t in enumerate(SENTS):
        t0 = time.time(); a = pt_synth(t); el = time.time() - t0
        d = len(a) / sr; tot_i += el; tot_d += d; rtfs.append(el / d)
        if n == 4: sf.write(os.path.join(QP, f"{i:03d}.wav"), a, sr)
    pt_rtf[n] = dict(mean_rtf=float(np.mean(rtfs)), overall_rtf=tot_i/tot_d, tot_infer=tot_i, tot_dur=tot_d)
    print(f"[PyTorch] threads={n} mean_RTF={pt_rtf[n]['mean_rtf']:.4f} overall_RTF={pt_rtf[n]['overall_rtf']:.4f}")

# ---------- ONNX (sherpa-onnx, CPU) ----------
import sherpa_onnx
def build(nthr, model_dir):
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=os.path.join(model_dir, "model.onnx"),
                lexicon=os.path.join(model_dir, "lexicon.txt"),
                tokens=os.path.join(model_dir, "tokens.txt"),
                dict_dir=os.path.join(BUND, "dict")),
            provider="cpu", num_threads=nthr),
        rule_fsts=os.path.join(BUND,"date.fst")+","+os.path.join(BUND,"number.fst"),
        max_num_sentences=1)
    assert cfg.validate()
    return sherpa_onnx.OfflineTts(cfg)
on_rtf = {}
for n in THREADS:
    tts = build(n, MY)
    for w in SENTS[:2]: tts.generate(w, sid=0, speed=1.0)  # warmup
    tot_i = tot_d = 0.0; rtfs = []
    for i, t in enumerate(SENTS):
        t0 = time.time(); a = tts.generate(t, sid=0, speed=1.0); el = time.time() - t0
        samp = np.array(a.samples); d = len(samp) / a.sample_rate
        tot_i += el; tot_d += d; rtfs.append(el / d)
        if n == 4: sf.write(os.path.join(QO, f"{i:03d}.wav"), samp, a.sample_rate)
    on_rtf[n] = dict(mean_rtf=float(np.mean(rtfs)), overall_rtf=tot_i/tot_d, tot_infer=tot_i, tot_dur=tot_d)
    print(f"[ONNX]    threads={n} mean_RTF={on_rtf[n]['mean_rtf']:.4f} overall_RTF={on_rtf[n]['overall_rtf']:.4f}")

json.dump({'sentences':SENTS,'pytorch':pt_rtf,'onnx':on_rtf}, open(os.path.join(OUT,'hw4_rtf.json'),'w'), indent=1)

# figure: RTF vs threads
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7,4.5))
ax.plot(THREADS, [pt_rtf[n]['overall_rtf'] for n in THREADS], 'o-', label='PyTorch (melo, +BERT)', color='#c0392b')
ax.plot(THREADS, [on_rtf[n]['overall_rtf'] for n in THREADS], 's-', label='ONNX (sherpa-onnx)', color='#2471a3')
ax.axhline(1.0, ls='--', color='gray', lw=0.8); ax.text(THREADS[-1], 1.02, 'RTF=1 (real-time)', ha='right', fontsize=8, color='gray')
ax.set_xlabel('num_threads'); ax.set_ylabel('overall RTF (lower=faster)'); ax.set_xticks(THREADS)
ax.set_title('HW4: RTF vs. num_threads (CPU), PyTorch vs ONNX'); ax.legend(); ax.grid(alpha=0.3)
for n in THREADS:
    ax.annotate(f"{pt_rtf[n]['overall_rtf']:.2f}", (n, pt_rtf[n]['overall_rtf']), textcoords="offset points", xytext=(0,6), fontsize=8)
    ax.annotate(f"{on_rtf[n]['overall_rtf']:.3f}", (n, on_rtf[n]['overall_rtf']), textcoords="offset points", xytext=(0,-12), fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'hw4_rtf.png'), dpi=130)
print("DONE BENCH")
