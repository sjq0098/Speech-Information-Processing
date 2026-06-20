# coding=utf-8
"""
加分项：用独立的说话人验证模型(ECAPA-TDNN, 与实验五一致)客观评估音色转换是否成功。
思路：分别提取 音色源(my_voice)、内容源(speaker1)、转换结果(converted) 的说话人嵌入,
计算余弦相似度。若转换成功, converted 应当更接近 my_voice 而非 speaker1。
注意：这里用的是“独立”的 ECAPA 模型(192维), 而不是 FACodec 自带的 spk_embs, 避免循环论证。
模型权重已通过 HF 镜像下载到本地 asv_ckpt/, 这里直接离线加载 ECAPA-TDNN, 不依赖联网。
"""
import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF

# 兼容性补丁: speechbrain 1.1.0 调用 torch.amp.custom_fwd(device_type=...),
# 但本环境 torch 2.1.2 只有 torch.cuda.amp.custom_fwd(无 device_type), 这里做个垫片。
import torch.cuda.amp as _camp
if not hasattr(torch.amp, "custom_fwd"):
    torch.amp.custom_fwd = lambda fwd=None, *, device_type=None, cast_inputs=None: _camp.custom_fwd(fwd, cast_inputs=cast_inputs)
    torch.amp.custom_bwd = lambda bwd=None, *, device_type=None: _camp.custom_bwd(bwd)

from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

SR = 16000
CKPT = "asv_ckpt/spkrec-ecapa-cnceleb/embedding_model.ckpt"

# 按 hyperparams.yaml 的配置手动搭建并加载 ECAPA-TDNN(只取 embedding 部分)
fbank = Fbank(n_mels=80)
mvn = InputNormalization(norm_type="sentence", std_norm=False)
emb_model = ECAPA_TDNN(
    input_size=80,
    channels=[1024, 1024, 1024, 1024, 3072],
    kernel_sizes=[5, 3, 3, 3, 1],
    dilations=[1, 2, 3, 4, 1],
    attention_channels=128,
    lin_neurons=192,
)
emb_model.load_state_dict(torch.load(CKPT, map_location="cpu"))
emb_model.eval()


def embed(path):
    wav, sr = torchaudio.load(path)               # [channels, T]
    if wav.shape[0] > 1:                           # 多声道 -> 单声道
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:                                   # 统一重采样到 16k
        wav = AF.resample(wav, sr, SR)
    with torch.no_grad():
        feats = fbank(wav)                         # [1, T', 80]
        feats = mvn(feats, torch.ones(1))          # 句级均值方差归一化
        e = emb_model(feats).squeeze().numpy()     # [192]
    return e / (np.linalg.norm(e) + 1e-9)          # L2 归一化, 点积即余弦相似度


files = {
    "my_voice":      "samples/my_voice.wav",                  # 音色源(目标音色)
    "speaker1":      "samples/speaker1.wav",                  # 内容源(原音色)
    "converted":     "samples/speaker1_rec_with_new_spk.wav", # 音色转换结果
    "speaker2":      "samples/speaker2.wav",                  # 重构 sanity check 用
    "speaker2_rec":  "samples/speaker2_rec.wav",
}

E = {k: embed(v) for k, v in files.items()}
cos = lambda a, b: float(np.dot(E[a], E[b]))

print("\n================ 说话人相似度(余弦, 越大越像) ================")
print(f"converted  vs  my_voice (目标音色)   : {cos('converted','my_voice'):+.4f}")
print(f"converted  vs  speaker1 (原始音色)   : {cos('converted','speaker1'):+.4f}")
print(f"my_voice   vs  speaker1 (两人本底差异): {cos('my_voice','speaker1'):+.4f}")
print("---- sanity check: 普通重构应保持说话人 ----")
print(f"speaker2_rec vs speaker2             : {cos('speaker2_rec','speaker2'):+.4f}")

shift = cos("converted", "my_voice") - cos("converted", "speaker1")
print("\n================ 结论 ================")
print(f"音色迁移量 = sim(converted,my_voice) - sim(converted,speaker1) = {shift:+.4f}")
if shift > 0:
    print("=> 转换结果在说话人空间中更靠近你的音色, 音色转换方向正确。")
else:
    print("=> 转换结果仍更靠近 speaker1, 单参考零样本迁移的音色相似度有限(详见报告分析)。")

# ===== 消融实验: 不同特征组合下的音色转换(均使用你的音色) =====
import os
abl = {
    "A 韵律+内容+细节": "samples/vc_A_pro_con_det.wav",
    "B 韵律+内容":      "samples/vc_B_pro_con.wav",
    "C 仅内容":         "samples/vc_C_con.wav",
}
if all(os.path.exists(p) for p in abl.values()):
    for k, p in abl.items():
        E[k] = embed(p)
    print("\n================ 消融: 特征组合 vs 音色迁移 ================")
    print(f"{'组合':<16}{'sim(→你)':>10}{'sim(→spk1)':>12}{'迁移量':>10}")
    for k in abl:
        s_me, s_s1 = float(np.dot(E[k], E['my_voice'])), float(np.dot(E[k], E['speaker1']))
        print(f"{k:<16}{s_me:>+10.4f}{s_s1:>+12.4f}{s_me - s_s1:>+10.4f}")
