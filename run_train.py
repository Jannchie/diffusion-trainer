"""Stable Diffusion XL Finetuner."""

from diffusion_trainer.finetune.sdxl import SDXLTuner

if __name__ == "__main__":
    model_path = R"E:/webui_forge_cu121_torch21/webui/models/Stable-diffusion/creative-xl-0.9-b2.safetensors"
    tuner = SDXLTuner(
        model_path=model_path,
    )
    tuner()
