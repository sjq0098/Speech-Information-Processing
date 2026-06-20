# coding=utf-8
"""
加分项消融实验：探究“音色转换效果弱”的原因。
猜想：speaker1 的残差/细节(Acoustic Detail)码本泄漏了 speaker1 自身音色,
盖过了被替换的全局音色向量。因此对比三种特征组合下的音色转换结果:
  A) prosody + content + detail  (= 主程序的做法)
  B) prosody + content           (去掉 detail)
  C) content                     (只保留内容)
全部使用“你的音色”嵌入, 内容/语调来自 speaker1, 输出供 verify_timbre.py 客观评测。
"""
import torch
import librosa
import soundfile as sf
from ns3_codec import FACodecEncoder, FACodecDecoder

SR = 16000


def load_audio(p):
    wav = librosa.load(p, sr=SR)[0]
    return torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)


fa_encoder = FACodecEncoder(ngf=32, up_ratios=[2, 4, 5, 5], out_channels=256)
fa_decoder = FACodecDecoder(
    in_channels=256, upsample_initial_channel=1024, ngf=32, up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2, vq_num_q_p=1, vq_num_q_r=3, vq_dim=256, codebook_dim=8,
    codebook_size_prosody=10, codebook_size_content=10, codebook_size_residual=10,
    use_gr_x_timbre=True, use_gr_residual_f0=True, use_gr_residual_phone=True,
)
fa_encoder.load_state_dict(torch.load("./ckpt/ns3_facodec_encoder.bin"))
fa_decoder.load_state_dict(torch.load("./ckpt/ns3_facodec_decoder.bin"))
fa_encoder.eval()
fa_decoder.eval()

timbre_wav = load_audio("samples/my_voice.wav")     # 目标音色
content_wav = load_audio("samples/speaker1.wav")    # 内容/语调来源

with torch.no_grad():
    # 你的音色嵌入
    _, _, _, _, spk_embs_mine = fa_decoder(fa_encoder(timbre_wav), eval_vq=False, vq=True)
    # speaker1 的 韵律/内容/细节
    _, _, _, q = fa_decoder.quantize(fa_encoder(content_wav))
    prosody, content, detail = q[0], q[1], q[2]

    variants = {
        "A_pro_con_det": prosody + content + detail,
        "B_pro_con":     prosody + content,
        "C_con":         content,
    }
    for name, emb in variants.items():
        wav = fa_decoder.inference(emb, spk_embs_mine)
        out = f"samples/vc_{name}.wav"
        sf.write(out, wav[0][0].cpu().numpy(), SR)
        print("saved", out)

print("Ablation done.")
