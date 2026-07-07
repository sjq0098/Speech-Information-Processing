"""
训练脚本：Transformer Encoder + CTC 的拼音语音识别。

用法（在 exp6/ 目录下，speechbrain 环境）：
    python train.py                      # 默认 10 epoch，自动用 GPU
    python train.py --epochs 1 --limit 256 --num-workers 0   # 快速冒烟测试

产物：
    ckpt/epoch_{i}.pt        每个 epoch 的权重
    ckpt/best.pt             dev WER 最优的权重（含配置）
    ckpt/train_log.json      每个 epoch 的 train_loss / dev_loss / dev_wer
"""
import os
import sys
import json
import time
import math
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)                       # 数据路径是相对 exp6/ 的
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataprocess.vocab import build_pinyin_list_from_text, Vocab
from dataprocess.dataset import ASRDataset, collate_fn
from model import ASRTransformerCTC
from wer import compute_corpus_wer


# ---------------------------------------------------------------
# 贪心 CTC 解码：对每帧取 argmax，再合并相邻重复并去 blank
# ---------------------------------------------------------------
def greedy_decode(log_probs, output_lengths, vocab):
    """
    :param log_probs: [T, B, V]
    :param output_lengths: [B]
    :return: list[str]，每条一个空格分隔的拼音串
    """
    preds = log_probs.argmax(dim=-1)          # [T, B]
    preds = preds.transpose(0, 1).contiguous()  # [B, T]
    results = []
    for i in range(preds.size(0)):
        L = int(output_lengths[i].item())
        ids = preds[i, :L].tolist()
        results.append(vocab.idx2text(ids, remove_blank=True))   # 合并重复 + 去 blank
    return results


def labels_to_text(labels, label_lengths, vocab):
    """把 padding 过的标签张量还原成拼音串（按真实长度截断）。"""
    refs = []
    for i in range(labels.size(0)):
        L = int(label_lengths[i].item())
        ids = labels[i, :L].tolist()
        refs.append(" ".join(vocab.itos[t] for t in ids))
    return refs


@torch.no_grad()
def evaluate(model, loader, vocab, device, ctc_loss):
    """在 dev/test 上算平均 CTC loss 和 WER。"""
    model.eval()
    total_loss, nb = 0.0, 0
    all_ref, all_hyp = [], []
    for feats, feat_lens, labels, label_lens in loader:
        feats = feats.to(device)
        log_probs, out_lens = model(feats, feat_lens.to(device))
        loss = ctc_loss(log_probs, labels.to(device), out_lens, label_lens.to(device))
        total_loss += loss.item()
        nb += 1
        all_hyp.extend(greedy_decode(log_probs.cpu(), out_lens.cpu(), vocab))
        all_ref.extend(labels_to_text(labels, label_lens, vocab))
    stats = compute_corpus_wer(all_ref, all_hyp)
    return total_loss / max(nb, 1), stats


def build_loader(wav_scp, text, vocab, batch_size, num_workers, shuffle, limit=0):
    ds = ASRDataset(wav_scp, text, vocab)
    if limit and limit > 0:
        ds = Subset(ds, list(range(min(limit, len(ds)))))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--norm-first", action="store_true", help="用 Pre-LN Transformer（从零训练更稳）")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条训练数据（冒烟测试用）")
    ap.add_argument("--ckpt-dir", type=str, default="ckpt")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} | torch {torch.__version__}")

    # ---- 词表：仅从训练集拼音构建 ----
    pinyin_list = build_pinyin_list_from_text("dataset/split/train/pinyin")
    vocab = Vocab(pinyin_list)
    print(f"[vocab] size={vocab.vocab_size} blank_id={vocab.blank_id}")

    # ---- 数据 ----
    train_loader = build_loader("dataset/split/train/wav.scp", "dataset/split/train/pinyin",
                                vocab, args.batch_size, args.num_workers, shuffle=True, limit=args.limit)
    dev_loader = build_loader("dataset/split/dev/wav.scp", "dataset/split/dev/pinyin",
                              vocab, args.batch_size, args.num_workers, shuffle=False,
                              limit=args.limit)

    # ---- 模型 ----
    cfg = dict(input_dim=80, d_model=args.d_model, nhead=args.nhead,
               num_encoder_layers=args.layers, dim_feedforward=args.ff,
               dropout=args.dropout, vocab_size=vocab.vocab_size,
               norm_first=args.norm_first)
    model = ASRTransformerCTC(**cfg).to(device)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {cfg} | params={n_param:.2f}M")

    # ---- 损失 / 优化器 / 学习率 warmup ----
    ctc_loss = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)   # zero_infinity 防 nan
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    warmup = max(args.warmup, 1)
    total_steps = len(train_loader) * args.epochs
    lr_floor = 0.05                                              # 退火下限（相对峰值）

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup                           # 线性 warmup
        prog = min(max((step - warmup) / max(total_steps - warmup, 1), 0.0), 1.0)
        return lr_floor + (1 - lr_floor) * 0.5 * (1 + math.cos(math.pi * prog))  # 余弦退火
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- 训练循环 ----
    log = []
    best_wer = float("inf")
    best_epoch = -1
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, nb = 0.0, 0
        t0 = time.time()
        for feats, feat_lens, labels, label_lens in train_loader:
            feats = feats.to(device)
            log_probs, out_lens = model(feats, feat_lens.to(device))
            loss = ctc_loss(log_probs, labels.to(device), out_lens, label_lens.to(device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            run_loss += loss.item()
            nb += 1
            global_step += 1
            if nb % 100 == 0:
                print(f"  ep{epoch} step{nb} loss={run_loss / nb:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        train_loss = run_loss / max(nb, 1)
        dev_loss, dev_stats = evaluate(model, dev_loader, vocab, device, ctc_loss)
        dt = time.time() - t0

        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"epoch_{epoch}.pt"))
        is_best = dev_stats["wer"] < best_wer
        if is_best:
            best_wer = dev_stats["wer"]
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "config": cfg, "vocab_size": vocab.vocab_size},
                       os.path.join(args.ckpt_dir, "best.pt"))

        rec = {"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss,
               "dev_wer": dev_stats["wer"], "dev_S": dev_stats["S"], "dev_D": dev_stats["D"],
               "dev_I": dev_stats["I"], "dev_N": dev_stats["N"], "sec": round(dt, 1)}
        log.append(rec)
        json.dump(log, open(os.path.join(args.ckpt_dir, "train_log.json"), "w"),
                  ensure_ascii=False, indent=2)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} "
              f"dev_WER={dev_stats['wer'] * 100:.2f}% ({dt:.0f}s){'  <-- best' if is_best else ''}")

    print(f"\n训练完成。最优 epoch={best_epoch}，dev WER={best_wer * 100:.2f}%")
    print(f"最优权重已保存到 {os.path.join(args.ckpt_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
