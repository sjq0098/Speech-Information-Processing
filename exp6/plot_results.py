"""
根据 ckpt/train_log.json 与 outputs/eval_test.json 画报告用图，保存到 images/。
图内文字用英文，避免 matplotlib 缺中文字体。
    images/res_train_curve.png   训练/验证 loss 与 dev WER 随 epoch 变化
    images/res_wer_breakdown.png dev/test 的 WER 及 S/D/I 组成
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)
log = json.load(open("ckpt/train_log.json", encoding="utf-8"))
epochs = [r["epoch"] for r in log]
train_loss = [r["train_loss"] for r in log]
dev_loss = [r["dev_loss"] for r in log]
dev_wer = [r["dev_wer"] * 100 for r in log]
best_i = min(range(len(log)), key=lambda k: dev_wer[k])

# ---------- 图 1：训练曲线 ----------
fig, ax1 = plt.subplots(figsize=(7, 4.2))
ax1.plot(epochs, train_loss, "o-", color="#1f77b4", label="train loss")
ax1.plot(epochs, dev_loss, "s-", color="#2ca02c", label="dev loss")
ax1.set_xlabel("epoch")
ax1.set_ylabel("CTC loss")
ax1.set_xticks(epochs)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(epochs, dev_wer, "^--", color="#d62728", label="dev WER")
ax2.scatter([epochs[best_i]], [dev_wer[best_i]], s=140, facecolors="none",
            edgecolors="#d62728", linewidths=2, zorder=5,
            label=f"best ep{epochs[best_i]} ({dev_wer[best_i]:.1f}%)")
ax2.set_ylabel("dev WER (%)", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
plt.title("Training / validation loss and dev WER")
plt.tight_layout()
plt.savefig("images/res_train_curve.png", dpi=160)
plt.close()
print("saved images/res_train_curve.png")

# ---------- 图 2：WER 组成 ----------
def load_stats(path):
    if os.path.exists(path):
        d = json.load(open(path, encoding="utf-8"))
        return d
    return None

dev_eval = load_stats("outputs/eval_dev.json")
test_eval = load_stats("outputs/eval_test.json")

names, Ss, Ds, Is, Ns = [], [], [], [], []
# dev 用最优 epoch 的训练日志（已有 S/D/I）
best = log[best_i]
names.append(f"dev (ep{best['epoch']})")
Ss.append(best["dev_S"]); Ds.append(best["dev_D"]); Is.append(best["dev_I"]); Ns.append(best["dev_N"])
if test_eval:
    names.append("test")
    Ss.append(test_eval["S"]); Ds.append(test_eval["D"]); Is.append(test_eval["I"]); Ns.append(test_eval["N"])

fig, ax = plt.subplots(figsize=(6, 4.2))
import numpy as np
x = np.arange(len(names))
bottom = np.zeros(len(names))
for vals, lab, col in [(Ss, "Sub", "#1f77b4"), (Ds, "Del", "#ff7f0e"), (Is, "Ins", "#2ca02c")]:
    pct = [100 * v / n for v, n in zip(vals, Ns)]
    ax.bar(x, pct, bottom=bottom, label=lab, color=col)
    bottom += np.array(pct)
for xi, n in zip(x, bottom):
    ax.text(xi, n + 0.3, f"WER {n:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("error rate (%)")
ax.set_title("WER composition (S / D / I)")
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("images/res_wer_breakdown.png", dpi=160)
plt.close()
print("saved images/res_wer_breakdown.png")

# ---------- 图 3：Post-LN / Pre-LN / 学习率 消融 ----------
def wer_series(path, n=6):
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    return [r["dev_wer"] * 100 for r in d[:n]]

post_hi = wer_series("ckpt_postln_baseline/train_log.json")   # Post-LN, lr=1e-3
post_lo = wer_series("ckpt_postln_lr1e4/train_log.json")      # Post-LN, lr=1e-4 (助教)
pre = wer_series("ckpt/train_log.json")                       # Pre-LN,  lr=1e-3 (本实验)
if pre is not None:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    if post_hi is not None:
        ax.plot(range(1, len(post_hi) + 1), post_hi, "o-", color="#d62728",
                label="Post-LN, lr=1e-3 (collapse)")
    if post_lo is not None:
        ax.plot(range(1, len(post_lo) + 1), post_lo, "^-", color="#ff7f0e",
                label="Post-LN, lr=1e-4 (slow)")
    ax.plot(range(1, len(pre) + 1), pre, "s-", color="#1f77b4",
            label="Pre-LN, lr=1e-3 (fixed, fast)")
    ax.axhline(100, color="#d62728", ls=":", alpha=0.4)
    ax.axhline(30, color="gray", ls="--", alpha=0.5)
    ax.text(len(pre), 31, "target 30%", ha="right", va="bottom", fontsize=8, color="gray")
    ax.set_xlabel("epoch")
    ax.set_ylabel("dev WER (%)")
    ax.set_xticks(range(1, max(len(pre), len(post_lo or [1])) + 1))
    ax.set_ylim(0, 108)
    ax.set_title("Post-LN is LR-fragile; Pre-LN is robust and fast")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/res_prenorm_ablation.png", dpi=160)
    plt.close()
    print("saved images/res_prenorm_ablation.png")
else:
    print("skip ablation figure (missing ckpt log)")
