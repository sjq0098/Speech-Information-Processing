"""
评测脚本：加载 dev 上最优的权重，在指定数据集(默认 test)上做贪心解码并计算 WER。

用法（exp6/ 目录，speechbrain 环境）：
    python evaluate.py                       # 用 ckpt/best.pt 评 test
    python evaluate.py --split dev           # 评 dev
    python evaluate.py --ckpt ckpt/epoch_8.pt

产物：
    outputs/eval_{split}.json    汇总指标 + 若干条预测样例
    outputs/hyp_{split}.txt      全部预测（id \t 拼音）
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataprocess.vocab import build_pinyin_list_from_text, Vocab
from dataprocess.dataset import ASRDataset, collate_fn
from model import ASRTransformerCTC
from wer import compute_corpus_wer, levenshtein
from train import greedy_decode, labels_to_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="ckpt/best.pt")
    ap.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--n-examples", type=int, default=12)
    args = ap.parse_args()

    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 词表与训练时一致（都来自 train 拼音，sorted 后确定）
    vocab = Vocab(build_pinyin_list_from_text("dataset/split/train/pinyin"))

    # 加载 checkpoint（best.pt 里含 config；epoch_x.pt 只有 state_dict，用默认配置）
    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        cfg = ckpt.get("config", None)
        state = ckpt["state_dict"]
        ep = ckpt.get("epoch", "?")
    else:
        cfg, state, ep = None, ckpt, "?"
    if cfg is None:
        cfg = dict(input_dim=80, d_model=256, nhead=4, num_encoder_layers=6,
                   dim_feedforward=1024, dropout=0.1, vocab_size=vocab.vocab_size)
    model = ASRTransformerCTC(**cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"[loaded] {args.ckpt} (epoch={ep}) | eval on {args.split}")

    loader = DataLoader(
        ASRDataset(f"dataset/split/{args.split}/wav.scp", f"dataset/split/{args.split}/pinyin", vocab),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn)

    ctc_loss = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    total_loss, nb = 0.0, 0
    all_ref, all_hyp = [], []
    with torch.no_grad():
        for feats, feat_lens, labels, label_lens in loader:
            feats = feats.to(device)
            log_probs, out_lens = model(feats, feat_lens.to(device))
            total_loss += ctc_loss(log_probs, labels.to(device), out_lens, label_lens.to(device)).item()
            nb += 1
            all_hyp.extend(greedy_decode(log_probs.cpu(), out_lens.cpu(), vocab))
            all_ref.extend(labels_to_text(labels, label_lens, vocab))

    stats = compute_corpus_wer(all_ref, all_hyp)
    loss = total_loss / max(nb, 1)
    print(f"\n===== {args.split} 结果 =====")
    print(f"样本数 N_utt = {len(all_ref)}")
    print(f"CTC loss    = {loss:.4f}")
    print(f"WER         = {stats['wer'] * 100:.2f}%  (S={stats['S']} D={stats['D']} I={stats['I']} / N={stats['N']})")

    # 保存全部预测
    with open(f"outputs/hyp_{args.split}.txt", "w", encoding="utf-8") as f:
        for i, h in enumerate(all_hyp):
            f.write(f"{i}\t{h}\n")

    # 挑几条样例（含正确/错误各若干）用于报告
    examples = []
    for r, h in zip(all_ref, all_hyp):
        s, d, i = levenshtein(r.split(), h.split())
        examples.append({"ref": r, "hyp": h, "S": s, "D": d, "I": i, "err": s + d + i})
    correct = [e for e in examples if e["err"] == 0][:args.n_examples // 2]
    wrong = sorted([e for e in examples if e["err"] > 0], key=lambda x: x["err"])[:args.n_examples // 2]

    summary = {"split": args.split, "ckpt": args.ckpt, "epoch": ep, "n_utt": len(all_ref),
               "ctc_loss": loss, **stats,
               "examples_correct": correct, "examples_wrong": wrong}
    json.dump(summary, open(f"outputs/eval_{args.split}.json", "w"),
              ensure_ascii=False, indent=2)
    print(f"\n已写出 outputs/eval_{args.split}.json 和 outputs/hyp_{args.split}.txt")


if __name__ == "__main__":
    main()
