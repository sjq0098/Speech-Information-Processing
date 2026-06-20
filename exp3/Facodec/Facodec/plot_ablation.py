# coding=utf-8
"""绘制音色转换消融实验的自制分析图(数据来自 verify_timbre.py 的 ECAPA 评测)。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

combos = ["A\n韵律+内容+细节", "B\n韵律+内容", "C\n仅内容"]
sim_you = [0.082, 0.199, 0.138]   # 与“你的音色”的相似度
sim_s1 = [0.641, 0.399, 0.442]    # 与 speaker1 的相似度
shift = [y - s for y, s in zip(sim_you, sim_s1)]

x = np.arange(len(combos))
w = 0.36
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# 左图：两类说话人相似度对比
b1 = ax1.bar(x - w / 2, sim_you, w, label="→ 你的音色 (目标)", color="#4C9F70")
b2 = ax1.bar(x + w / 2, sim_s1, w, label="→ speaker1 (原音色)", color="#E08A7B")
ax1.bar_label(b1, fmt="%.3f", padding=2, fontsize=9)
ax1.bar_label(b2, fmt="%.3f", padding=2, fontsize=9)
ax1.set_xticks(x); ax1.set_xticklabels(combos)
ax1.set_ylabel("ECAPA 说话人余弦相似度")
ax1.set_ylim(0, 0.72)
ax1.set_title("不同特征组合下的说话人相似度")
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

# 右图：音色迁移量(越接近 0 越成功)
colors = ["#C0392B", "#27AE60", "#E67E22"]   # A 红 / B 绿(最好) / C 橙
b3 = ax2.bar(x, shift, 0.5, color=colors)
ax2.bar_label(b3, fmt="%.3f", padding=3, fontsize=10)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(combos)
ax2.set_ylabel("音色迁移量 = sim(→你) − sim(→spk1)")
ax2.set_ylim(-0.64, 0.06)
ax2.set_title("音色迁移量(越接近 0 越成功)")
ax2.grid(axis="y", alpha=0.3)

fig.suptitle("音色转换消融实验：去掉细节(Detail)码显著改善音色迁移", fontsize=14)
fig.tight_layout()
fig.savefig("figures/03_vc_ablation.png", dpi=150, bbox_inches="tight")
print("saved figures/03_vc_ablation.png")
