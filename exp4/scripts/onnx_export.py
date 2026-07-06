# coding=utf-8
"""Export the local MeloTTS-ZH (myshell) checkpoint to ONNX for sherpa-onnx.
Re-implemented locally (logic follows sherpa-onnx scripts/melo-tts/export-onnx.py,
which I reviewed). Produces model.onnx + tokens.txt + lexicon.txt in OUTDIR.
Key point: the ONNX graph feeds BERT/JA-BERT as ZEROS (sherpa-onnx melo is BERT-free).
"""
import os, torch, onnx
from melo.api import TTS
from melo.text import language_id_map, language_tone_start_map
from melo.text.chinese import pinyin_to_symbol_map
from melo.text.english import eng_dict, refine_syllables
from pypinyin import Style, lazy_pinyin, phrases_dict, pinyin_dict

OUTDIR = r"d:/Speech Information Processing/exp4/onnx_export/mymodel"
os.makedirs(OUTDIR, exist_ok=True)

for k, v in list(pinyin_to_symbol_map.items()):
    if isinstance(v, list):
        break
    pinyin_to_symbol_map[k] = v.split()


def get_initial_final_tone(word):
    initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    finals = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
    ans_phone, ans_tone = [], []
    for c, v in zip(initials, finals):
        v_wo = v[:-1]
        try:
            tone = v[-1]
        except Exception:
            return [], []
        pinyin = c + v_wo
        if tone not in "12345":
            return [], []
        if c:
            rep = {"uei": "ui", "iou": "iu", "uen": "un"}
            if v_wo in rep: pinyin = c + rep[v_wo]
        else:
            rep = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}
            if pinyin in rep:
                pinyin = rep[pinyin]
            else:
                srep = {"v": "yu", "e": "e", "i": "y", "u": "w"}
                if pinyin and pinyin[0] in srep:
                    pinyin = srep[pinyin[0]] + pinyin[1:]
        if pinyin not in pinyin_to_symbol_map:
            continue
        phone = pinyin_to_symbol_map[pinyin]
        ans_phone += phone
        ans_tone += [tone] * len(phone)
    return ans_phone, ans_tone


def generate_tokens(symbols):
    with open(os.path.join(OUTDIR, "tokens.txt"), "w", encoding="utf-8") as f:
        for i, s in enumerate(symbols):
            f.write(f"{s} {i}\n")


def generate_lexicon():
    word_dict = pinyin_dict.pinyin_dict
    phrases = phrases_dict.phrases_dict
    with open(os.path.join(OUTDIR, "lexicon.txt"), "w", encoding="utf-8") as f:
        for word in eng_dict:
            phones, tones = refine_syllables(eng_dict[word])
            tones = [str(t + language_tone_start_map["EN"]) for t in tones]
            f.write(f"{word.lower()} {' '.join(phones)} {' '.join(tones)}\n")
        for key in word_dict:
            if not (0x4E00 <= key <= 0x9FA5):
                continue
            w = chr(key)
            phone, tone = get_initial_final_tone(w)
            if not phone: continue
            f.write(f"{w} {' '.join(phone)} {' '.join(tone)}\n")
        for w in phrases:
            phone, tone = get_initial_final_tone(w)
            if not phone or len(phone) != len(tone): continue
            f.write(f"{w} {' '.join(phone)} {' '.join(tone)}\n")


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lang_id = language_id_map[model.language]

    def forward(self, x, x_lengths, tones, sid, noise_scale, length_scale, noise_scale_w):
        bert = torch.zeros(x.shape[0], 1024, x.shape[1], dtype=torch.float32)
        ja_bert = torch.zeros(x.shape[0], 768, x.shape[1], dtype=torch.float32)
        lang_id = torch.zeros_like(x)
        lang_id[:, 1::2] = self.lang_id
        return self.model.model.infer(
            x=x, x_lengths=x_lengths, sid=sid, tone=tones, language=lang_id,
            bert=bert, ja_bert=ja_bert, noise_scale=noise_scale,
            noise_scale_w=noise_scale_w, length_scale=length_scale)[0]


def add_meta(fn, meta):
    m = onnx.load(fn)
    while len(m.metadata_props): m.metadata_props.pop()
    for k, v in meta.items():
        p = m.metadata_props.add(); p.key = k; p.value = str(v)
    onnx.save(m, fn)


def main():
    print("generating lexicon...")
    generate_lexicon()
    model = TTS(language="ZH", device="cpu")
    generate_tokens(model.hps["symbols"])
    wrapper = ModelWrapper(model)
    x = torch.randint(1, 10, size=(1, 60), dtype=torch.int64)
    x_lengths = torch.tensor([x.size(1)], dtype=torch.int64)
    sid = torch.tensor([list(model.hps.data.spk2id.values())[0]], dtype=torch.int64)
    tones = torch.zeros_like(x)
    ns = torch.tensor([0.6], dtype=torch.float32)
    ls = torch.tensor([1.0], dtype=torch.float32)
    nsw = torch.tensor([0.8], dtype=torch.float32)
    fn = os.path.join(OUTDIR, "model.onnx")
    print("exporting onnx...")
    torch.onnx.export(
        wrapper, (x, x_lengths, tones, sid, ns, ls, nsw), fn, opset_version=18,
        input_names=["x", "x_lengths", "tones", "sid", "noise_scale", "length_scale", "noise_scale_w"],
        output_names=["y"],
        dynamic_axes={"x": {0: "N", 1: "L"}, "x_lengths": {0: "N"},
                      "tones": {0: "N", 1: "L"}, "y": {0: "N", 1: "S", 2: "T"}})
    add_meta(fn, {
        "model_type": "melo-vits", "comment": "melo", "version": 2,
        "language": "Chinese + English", "add_blank": int(model.hps.data.add_blank),
        "n_speakers": 1, "jieba": 1, "sample_rate": model.hps.data.sampling_rate,
        "bert_dim": 1024, "ja_bert_dim": 768,
        "speaker_id": list(model.hps.data.spk2id.values())[0],
        "lang_id": language_id_map[model.language],
        "tone_start": language_tone_start_map[model.language],
        "url": "https://github.com/myshell-ai/MeloTTS", "license": "MIT license",
        "description": "MeloTTS ZH exported for sherpa-onnx"})
    print("size(MB):", os.path.getsize(fn)/1e6)
    print("DONE EXPORT")


if __name__ == "__main__":
    main()
