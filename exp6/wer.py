"""
WER (Word Error Rate) 评测工具。
本实验以“拼音”为识别单元，一个拼音算一个 word。
WER = (S + D + I) / N ，其中
  S: 替换错误 (substitution)
  D: 删除错误 (deletion)
  I: 插入错误 (insertion)
  N: 参考序列中的拼音总数
"""


def levenshtein(ref, hyp):
    """
    计算 ref -> hyp 的编辑距离，并回溯出 S/D/I 的具体次数。
    :param ref: list[str]，参考(正确)拼音序列
    :param hyp: list[str]，预测拼音序列
    :return: (S, D, I)
    """
    n, m = len(ref), len(hyp)
    # dp[i][j]: ref[:i] 变成 hyp[:j] 的最小编辑距离
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i          # 全部删除
    for j in range(m + 1):
        dp[0][j] = j          # 全部插入

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],   # 替换
                    dp[i - 1][j],       # 删除
                    dp[i][j - 1],       # 插入
                )

    # ---- 回溯统计 S / D / I ----
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1                        # 命中，无错误
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1
            i -= 1
            j -= 1                        # 替换
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1
            i -= 1                        # 删除
        else:
            I += 1
            j -= 1                        # 插入
    return S, D, I


def compute_corpus_wer(refs, hyps):
    """
    语料级 WER：把所有句子的 S/D/I/N 累加后再除。
    :param refs: list[str]，每个元素是空格分隔的拼音串
    :param hyps: list[str]，同上
    :return: dict(wer, S, D, I, N)
    """
    S = D = I = N = 0
    for r, h in zip(refs, hyps):
        rt = r.split()
        ht = h.split()
        s, d, i = levenshtein(rt, ht)
        S += s
        D += d
        I += i
        N += len(rt)
    wer = (S + D + I) / max(N, 1)
    return {"wer": wer, "S": S, "D": D, "I": I, "N": N}


if __name__ == "__main__":
    # 简单自测
    ref = "jin tian tian qi hen hao"
    hyp = "jin tian qi hen hao hao"   # 删一个 tian，末尾多一个 hao
    print(levenshtein(ref.split(), hyp.split()))     # 期望 (S,D,I) 大致 (0,1,1) 或 (1,0,1)
    print(compute_corpus_wer([ref], [hyp]))
