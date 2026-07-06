# coding=utf-8
"""Fill the two code TODOs directly in the homework notebooks (keep the # TODO line)."""
import json, io

def to_lines(s):
    lines = s.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

def patch(path, cell_id, new_src):
    nb = json.load(io.open(path, encoding="utf-8"))
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            c["source"] = to_lines(new_src)
            if c["cell_type"] == "code":
                c["outputs"] = []; c["execution_count"] = None
            break
    else:
        raise SystemExit(f"cell {cell_id} not found in {path}")
    json.dump(nb, io.open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("patched", cell_id, "in", path)

HW2 = r"d:/Speech Information Processing/exp4/MeloTTS-Homework/tts_homework_2.ipynb"
HW3 = r"d:/Speech Information Processing/exp4/MeloTTS-Homework/tts_homework_3.ipynb"

hw2_src = '''# TODO: 完成两个子句的 phones、tones、word2ph 列表
# 思路：先用正常前端 clean_text 跑出基线 phones/tones/word2ph，再只改“朝/长”两个多音字。
# word2ph 与 BERT token 对齐（[CLS] + 逐个非空格字符 + [SEP]），首尾 '_' 对应句子起止。
# 上联 海 水 朝(cháo) 朝(zhāo) 朝(zhāo) 朝(cháo) 朝(zhāo) 朝(cháo) 朝(zhāo) 落（“海水”三声连读变调，海读阳平）
phones1 = ['_', 'h', 'ai', 'sh', 'ui', 'ch', 'ao', ',', 'zh', 'ao', 'zh', 'ao', 'ch', 'ao', ',', 'zh', 'ao', 'ch', 'ao', 'zh', 'ao', 'l', 'uo', '.', '_']
tones1 = [0, 2, 2, 3, 3, 2, 2, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2, 2, 1, 1, 4, 4, 0, 0]
word2ph1 = [1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1]

# 下联 浮 云 长(zhǎng) 长(cháng) 长(cháng) 长(zhǎng) 长(cháng) 长(zhǎng) 长(cháng) 消
phones2 = ['_', 'f', 'u', 'y', 'vn', 'zh', 'ang', ',', 'ch', 'ang', 'ch', 'ang', 'zh', 'ang', ',', 'ch', 'ang', 'zh', 'ang', 'ch', 'ang', 'x', 'iao', '.', '_']
tones2 = [0, 2, 2, 2, 2, 3, 3, 0, 2, 2, 2, 2, 3, 3, 0, 2, 2, 3, 3, 2, 2, 1, 1, 0, 0]
word2ph2 = [1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1]

# 校验：sum(word2ph) 应等于 len(phones)
assert sum(word2ph1) == len(phones1) == len(tones1)
assert sum(word2ph2) == len(phones2) == len(tones2)'''

hw3_src = '''import math

# 问题三：语速由句首约 1 倍线性降至句尾约 0.33 倍（每个音素时长 1 倍 -> 3 倍）
text = "你听到了吗？这句话我会说得越来越慢。"
output_path = 'homework_3.wav'

# 关闭随机性，取得确定性的原始 w_ceil（逐音素时长）
w_ceil_list, phone_list, tone_list = model.get_original_w_ceil(
    text, speaker_ids['ZH'], output_path, speed=1, sdp_ratio=0, noise_scale=0, noise_scale_w=0)

# 展平为逐片段的 python 列表，并统计全句音素总数 N
orig = [w.squeeze().int().tolist() for w in w_ceil_list]
orig = [[v] if isinstance(v, int) else v for v in orig]
N = sum(len(seg) for seg in orig)

# 对全句第 k 个音素乘以线性因子 f(k) = 1 + 2*k/(N-1)（1.0 -> 3.0）
modified_w_ceil_list = []
k = 0
for seg in orig:
    new_seg = []
    for v in seg:
        factor = 1.0 + 2.0 * k / (N - 1)
        new_seg.append(max(1, int(math.ceil(v * factor))))
        k += 1
    modified_w_ceil_list.append(new_seg)

hop, sr = model.hps.data.hop_length, model.hps.data.sampling_rate
print(f'音素总数 N = {N}')
print(f'原始时长 ≈ {sum(sum(s) for s in orig) * hop / sr:.2f}s'
      f'  ->  调整后 ≈ {sum(sum(s) for s in modified_w_ceil_list) * hop / sr:.2f}s')

# 按调整后的 w_ceil 合成（关闭随机性以便对比）
model.tts_to_file_custom_duration(
    text, speaker_ids['ZH'], output_path, speed=1, sdp_ratio=0, noise_scale=0, noise_scale_w=0,
    w_ceil_customized=modified_w_ceil_list)

from IPython.display import Audio
Audio(output_path)'''

patch(HW2, "34fa112c", hw2_src)
patch(HW3, "e447773f", hw3_src)
print("done")
