# coding=utf-8
import sys, os, time, numpy as np, soundfile as sf
import sherpa_onnx
BUND = r"d:/Speech Information Processing/exp4/onnx_export/vits-melo-tts-zh_en"
model = sys.argv[1] if len(sys.argv)>1 else os.path.join(BUND,"model.onnx")
nthr = int(sys.argv[2]) if len(sys.argv)>2 else 2
cfg = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=model,
            lexicon=os.path.join(BUND,"lexicon.txt"),
            tokens=os.path.join(BUND,"tokens.txt"),
            dict_dir=os.path.join(BUND,"dict"),
        ),
        provider="cpu", num_threads=nthr, debug=False),
    rule_fsts=",".join([os.path.join(BUND,f) for f in ["date.fst","number.fst"]]),
    max_num_sentences=1)
assert cfg.validate(), "invalid config"
tts = sherpa_onnx.OfflineTts(cfg)
text = "今天天气很好，我们一起去公园散步吧。"
t=time.time(); a=tts.generate(text, sid=1, speed=1.0); el=time.time()-t
dur=len(a.samples)/a.sample_rate
print(f"threads={nthr} sr={a.sample_rate} dur={dur:.3f}s infer={el:.3f}s RTF={el/dur:.4f}")
sf.write(r"d:/Speech Information Processing/exp4/outputs/sherpa_smoke.wav", np.array(a.samples), a.sample_rate)
print("OK")
