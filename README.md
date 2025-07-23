# DDPM
A PyTorch-based diffusion model that generates custom emoji-style images using a U-Net architecture with attention and DDPM noise scheduling. Trained on the Subinium emoji dataset.

# 🌀 Emoji Diffusion Model

Generate emojis using a U-Net-based diffusion model, trained from scratch on the Subinium EmojiImage dataset.

---

## ✨ Features

- Trained on Apple emoji images (PNG format)
- U-Net architecture with skip connections
- Multi-head self-attention (MHSA)
- Time step positional embeddings
- DDPM-style noise scheduler (1000 steps)
- Mixed precision training with `torch.cuda.amp`

---

## 🗂️ Dataset

**Dataset:** [Kaggle - Subinium EmojiImage Dataset](https://www.kaggle.com/datasets/subinium/emojiimage-dataset)

### 📥 How to use in Colab:

```bash
# Upload your kaggle.json file first using Colab UI
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d subinium/emojiimage-dataset
!unzip -o emojiimage-dataset.zip -d emoji_dataset
