# DL_CSI_v2

FDD downlink CSI prediction using DeepSeek-R1-Distill-Qwen-1.5B + LoRA.

See [CLAUDE.md](CLAUDE.md) for the full project documentation, architecture,
setup instructions, and usage guide.

Quick start:

```bash
pip install -r requirements.txt
# Download DeepSeek-R1-Distill-Qwen-1.5B to ./models/deepseek-1_5b
python scripts/generate_data.py --config config.yaml --split train val test
python scripts/train_warmup.py --config config.yaml
python scripts/train_lora.py --config config.yaml --warmup-checkpoint outputs/checkpoints/best_warmup.pt
python scripts/evaluate.py --config config.yaml --checkpoint outputs/checkpoints/best_lora.pt --split test
```
