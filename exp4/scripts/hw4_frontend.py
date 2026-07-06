# coding=utf-8
"""Worker: emit sherpa debug (with per-sentence markers) to stderr, melo TN to stdout.
Run as:  python hw4_frontend.py 1>melo.txt 2>sherpa_dbg.txt   then parse.
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sherpa_onnx
from melo.text.chinese_mix import text_normalize as melo_tn

BUND = r"d:/Speech Information Processing/exp4/onnx_export/vits-melo-tts-zh_en"
MY = r"d:/Speech Information Processing/exp4/onnx_export/mymodel"
SENTS = [
    "会议定在2025年3月15日上午9点。",
    "销售额同比增长了25.6%。",
    "圆周率约为3.14159。",
    "请拨打客服热线10086。",
    "这是第2届人工智能大会。",
    "决赛将在2008年8月8日举行。",
]
cfg = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=os.path.join(MY, "model.onnx"), lexicon=os.path.join(MY, "lexicon.txt"),
            tokens=os.path.join(MY, "tokens.txt"), dict_dir=os.path.join(BUND, "dict")),
        provider="cpu", num_threads=2, debug=True),
    rule_fsts=os.path.join(BUND, "date.fst")+","+os.path.join(BUND, "number.fst"),
    max_num_sentences=1)
assert cfg.validate()
tts = sherpa_onnx.OfflineTts(cfg)

for i, s in enumerate(SENTS):
    os.write(2, ("\n##MARK\t%d##\n" % i).encode("utf-8"))
    tts.generate(s, sid=0, speed=1.0)
    sys.stdout.write("MELO\t%d\t%s\t%s\n" % (i, s, melo_tn(s).rstrip(".")))
    sys.stdout.flush()
