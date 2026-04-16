#coding=gbk

import torch
import librosa
import soundfile as sf
from ns3_codec import FACodecEncoder, FACodecDecoder


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

    
    #### Step2: 加载待处理的音频。
    def load_audio(wav_path):
        wav = librosa.load(wav_path, sr=16000)[0]
        wav = torch.from_numpy(wav).float()
        wav = wav.unsqueeze(0).unsqueeze(0)
        return wav
    
    test_wav_file = "speaker2"
    test_wav = load_audio(f"samples/{test_wav_file}.wav")
    print("Test Audio Shape: ", test_wav.shape)
    

    #### Step3: 利用Facodec对语音进行离散化和解耦处理。
    fa_encoder.eval()
    fa_decoder.eval()
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
    sf.write(f"samples/{test_wav_file}_rec.wav", rec_wav[0][0].cpu().numpy(), 16000)
    print("Successfully Reconstruct!")

if __name__ == "__main__":
    main()