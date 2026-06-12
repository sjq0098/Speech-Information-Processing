# 实验环境说明（conda + CUDA 11.8）

六个实验的依赖存在硬性冲突（librosa 0.10.1 vs 0.9.1、transformers 4.27.4 旧钉死、speechbrain/torchcodec 等），
无法用单一环境干净运行，因此收敛为 **3 个 conda 环境**。三者统一使用 **PyTorch 2.1.2 + CUDA 11.8**（RTX 4060 Laptop）。

| 环境 | 覆盖实验 | Python | 关键依赖 |
|------|----------|--------|----------|
| `speechcore`     | exp1 特征提取 / exp2 CharRNN / exp3 FACodec / exp6 ASR | 3.10 | librosa 0.10.1, transformers 4.36.2, pyworld, einops, opencc |
| `speechmelotts`  | exp4 MeloTTS | 3.9 | MeloTTS (editable)，transformers 4.27.4、librosa 0.9.1 等钉死依赖 |
| `speechbrain`    | exp5 说话人识别 | 3.10 | speechbrain (PyPI), scikit-learn |

## 一、创建环境

```powershell
# 1) speechcore（覆盖 4 个实验）
conda env create -f setup/speechcore.yml

# 2) speechmelotts（exp4）
conda env create -f setup/speechmelotts.yml
conda run -n speechmelotts pip install -e exp4/MeloTTS-Homework
# dicdir 已随仓库提供；若缺失再执行：
# conda run -n speechmelotts python -m unidic download

# 3) speechbrain（exp5）
conda env create -f setup/speechbrain.yml
```

## 二、注册 Jupyter 内核（在 VSCode / Jupyter 中可选择）

```powershell
conda run -n speechcore    python -m ipykernel install --user --name speechcore    --display-name "Python (speechcore)"
conda run -n speechmelotts python -m ipykernel install --user --name speechmelotts --display-name "Python (speechmelotts)"
conda run -n speechbrain   python -m ipykernel install --user --name speechbrain   --display-name "Python (speechbrain)"
```

## 三、各实验如何运行

| 实验 | 激活环境 | 入口 |
|------|----------|------|
| exp1 | `conda activate speechcore` | `exp1/exp1_2026.ipynb` |
| exp2 | `conda activate speechcore` | `exp2/lab2.ipynb`（先解压 `data(1).zip`） |
| exp3 | `conda activate speechcore` | `exp3/Facodec/Facodec/test.py` |
| exp4 | `conda activate speechmelotts` | `exp4/MeloTTS-Homework/tts_homework_*.ipynb` |
| exp5 | `conda activate speechbrain` | `exp5/lab5/0{1,2,3}_*.ipynb` |
| exp6 | `conda activate speechcore` | `exp6/`（基于 dataprocess/model/tokenizer 自行实现训练/推理/评测） |

## 四、验证 GPU 是否可用

```powershell
conda run -n speechcore python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
预期输出包含 `2.1.2+cu118 True NVIDIA GeForce RTX 4060 Laptop GPU`。

## 备注 / 偏差说明
- **speechbrain 未安装 torchcodec**：讲义依赖清单里有 torchcodec，但它在 Windows 上需要系统级 FFmpeg 共享库，易安装失败；
  notebooks 实际用 `torchaudio.load()`（soundfile 后端）即可读音频，因此省略以提升稳定性。
- **exp4 的 torch 不会被覆盖**：先装 cu118 torch，再 `pip install -e .`；MeloTTS 的 `requirements.txt` 中 torch 未钉版本，
  pip 检测到已满足不会重装，从而保留 GPU 版本。
- exp5 首次运行会从 HuggingFace 自动下载 `LanceaKing/spkrec-ecapa-cnceleb` checkpoint。
