# coding=utf-8

import os
import torch
import librosa
import librosa.display
import numpy as np
import soundfile as sf

import matplotlib
matplotlib.use("Agg")  # 无界面后端，脚本运行时直接把图保存为文件，便于放进实验报告
import matplotlib.pyplot as plt
# 让 matplotlib 正常显示中文（Windows 自带字体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from ns3_codec import FACodecEncoder, FACodecDecoder

SAMPLE_RATE = 16000      # FACodec 要求输入 16kHz
FIG_DIR = "figures"      # 对比图保存目录


#### Step2: 加载待处理的音频。
def load_audio(wav_path):
    wav = librosa.load(wav_path, sr=SAMPLE_RATE)[0]
    wav = torch.from_numpy(wav).float()
    wav = wav.unsqueeze(0).unsqueeze(0)
    return wav


def plot_wave_and_spec(pairs, save_path, suptitle):
    """使用实验一的工具(librosa + matplotlib)绘制 波形图 + 幅度谱图 对比。

    para:
        pairs: List[(子图标题, 一维 numpy 波形)]，每一项画成一列(波形图在上、幅度谱图在下)
        save_path: 图片保存路径
        suptitle: 整张图的总标题
    """
    n = len(pairs)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)
    for j, (title, y) in enumerate(pairs):
        y = np.asarray(y, dtype=np.float32)
        # 第一行：波形图（时域）
        times = np.arange(len(y)) / SAMPLE_RATE
        axes[0, j].plot(times, y, linewidth=0.5)
        axes[0, j].set_xlim(0, times[-1] if len(times) else 1)
        axes[0, j].set_title(f"{title}  波形图 Waveform")
        axes[0, j].set_xlabel("时间 Time (s)")
        axes[0, j].set_ylabel("幅度 Amplitude")
        # 第二行：幅度谱图（STFT 取模 -> 转 dB），与实验一课程一的做法一致
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(
            S_db, sr=SAMPLE_RATE, x_axis="time", y_axis="hz", ax=axes[1, j]
        )
        axes[1, j].set_title(f"{title}  幅度谱图 Spectrogram")
        fig.colorbar(img, ax=axes[1, j], format="%+2.0f dB")
    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure:", save_path)


#### 实验三：使用Facodec实现对语音的离散化、特征解耦和重构
def main():

    #### Step1: Facodec由FACodecEncoder和FACodecDecoder，两部分组成；
        ### 此处实例化fa_encoder、fa_decoder，并加载预训练后的模型权重。

    fa_encoder = FACodecEncoder(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=256,
    )

    fa_decoder = FACodecDecoder(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )

    fa_encoder.load_state_dict(torch.load("./ckpt/ns3_facodec_encoder.bin"))
    fa_decoder.load_state_dict(torch.load("./ckpt/ns3_facodec_decoder.bin"))

    fa_encoder.eval()
    fa_decoder.eval()

    #### Step2: 加载待处理的音频。
    test_wav_file = "speaker2"
    test_wav = load_audio(f"samples/{test_wav_file}.wav")
    print("Test Audio Shape: ", test_wav.shape)


    #### Step3: 利用Facodec对语音进行离散化和解耦处理。
    with torch.no_grad():

        #### Step3.1: 使用fa_encoder对音频进行嵌入；
        encoder_out = fa_encoder(test_wav)
        print("Encoder_out Shape: ", encoder_out.shape)

        #### Step3.2: 观察fa_decoder对音频进行离散化解耦的效果；
        _, vq_id, _, _, spk_embs = fa_decoder(encoder_out, eval_vq=False, vq=True)
        prosody_code = vq_id[:1]
        print("Prosody Code Shape:", prosody_code.shape)
        cotent_code = vq_id[1:3]
        print("Cotent Code Shape:", cotent_code.shape)
        detail_code = vq_id[3:]
        print("Residual Code Shape:", detail_code.shape) ### tips: 此处batch_size被调整到了第二维度

        #### Step3.3: 对比（离散化和连续特征）下fa_decoder对音频进行解耦的区别；
        _, _, _, _, spk_embs = fa_decoder(encoder_out, eval_vq=False, vq=True)
        _, _, _, quantized = fa_decoder.quantize(encoder_out)
        prosody = quantized[0]
        print("Prosody Embedding Shape:", prosody.shape)
        content = quantized[1]
        print("Content Embedding Shape:", content.shape)
        detail = quantized[2]
        print("Detail Embedding Shape:", detail.shape)
        spk_embs = spk_embs
        print("Speaker Embedding Shape:", spk_embs.shape)

        #### Step3.4: 组合解耦的各部分特征, 使用fa_decoder解码重构音频。
        all_embs = prosody + content + detail
        rec_wav = fa_decoder.inference(all_embs, spk_embs)


    #### Step4: 保存重构的音频。
    print("Reconstruct Audio Shape: ", rec_wav.shape)
    sf.write(f"samples/{test_wav_file}_rec.wav", rec_wav[0][0].cpu().numpy(), SAMPLE_RATE)
    print("Successfully Reconstruct!")

    #### 实验要求(2)(30%)：用实验一的工具画 原始音频 与 重构音频 的 波形图+频谱图 对比。
    plot_wave_and_spec(
        [
            ("原始音频 speaker2", test_wav[0, 0].cpu().numpy()),
            ("重构音频 reconstructed", rec_wav[0][0].cpu().numpy()),
        ],
        os.path.join(FIG_DIR, "01_reconstruct_compare.png"),
        "原始音频 vs 重构音频 对比 (Reconstruction)",
    )


    #### Step5（音色转换 / Voice Conversion，实验要求(4) 10%）:
    ###  用“自己录制的语音”提取音色 Timbre/Speaker，
    ###  搭配 speaker1.wav 的 语调Prosody + 内容Content + 细节Acoustic Detail 进行重构，
    ###  使重构音频的内容/语调来自 speaker1，但音色变成自己的。
    content_file = "speaker1"          # 语调/内容/细节 的来源
    timbre_file = "my_voice"           # TODO: 录制自己的语音并保存为 samples/my_voice.wav
    timbre_path = f"samples/{timbre_file}.wav"
    if not os.path.exists(timbre_path):
        # 还没录自己的语音时，先用 speaker2 占位，保证整条流程可跑通；录好后把文件放进 samples/ 即可。
        print(f"[Note] {timbre_path} not found; using speaker2.wav as a placeholder timbre source. "
              f"Record your own voice and replace it later.")
        timbre_file = "speaker2"
        timbre_path = f"samples/{timbre_file}.wav"

    timbre_wav = load_audio(timbre_path)
    content_wav = load_audio(f"samples/{content_file}.wav")

    with torch.no_grad():
        #### Step5.1: 从“音色源”音频中提取说话人音色嵌入 spk_embs；
        enc_timbre = fa_encoder(timbre_wav)
        _, _, _, _, spk_embs_timbre = fa_decoder(enc_timbre, eval_vq=False, vq=True)
        print("New Speaker Embedding Shape:", spk_embs_timbre.shape)

        #### Step5.2: 从 speaker1 中提取 语调/内容/细节 的连续解耦特征；
        enc_content = fa_encoder(content_wav)
        _, _, _, quantized_c = fa_decoder.quantize(enc_content)
        prosody_c, content_c, detail_c = quantized_c[0], quantized_c[1], quantized_c[2]

        #### Step5.3: 用 speaker1 的内容特征 + 自己的音色嵌入，重构出“换音色”的音频。
        all_embs_vc = prosody_c + content_c + detail_c
        vc_wav = fa_decoder.inference(all_embs_vc, spk_embs_timbre)

    vc_out_path = f"samples/{content_file}_rec_with_new_spk.wav"
    sf.write(vc_out_path, vc_wav[0][0].cpu().numpy(), SAMPLE_RATE)
    print("Voice Conversion Done! Saved:", vc_out_path)

    #### 实验要求(4)：画出 音色转换前(speaker1) 与 转换后 的 波形图+频谱图 对比。
    plot_wave_and_spec(
        [
            ("转换前 speaker1", content_wav[0, 0].cpu().numpy()),
            ("转换后 内容=speaker1/音色=新", vc_wav[0][0].cpu().numpy()),
        ],
        os.path.join(FIG_DIR, "02_timbre_convert_compare.png"),
        "音色转换前后对比 (Timbre Conversion)",
    )


if __name__ == "__main__":
    main()
