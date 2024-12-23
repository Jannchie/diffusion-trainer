import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae: AutoencoderKL = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)  # type: ignore
vae = vae.to(device)  # type: ignore
vae_scale_factor = 2 ** (len(vae.config.get("block_out_channels", 0)) - 1)
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)


@torch.no_grad()
def latents_to_image(latent: torch.Tensor) -> Image.Image:
    latents = latent.unsqueeze(0)
    sample = vae.decode(latents).sample  # type: ignore
    return image_processor.postprocess(sample)[0]  # type: ignore


npz_file = np.load("latents_1_0.npy")

latents = npz_file[0] / vae.config.get("scaling_factor", 1.0)

img = latents_to_image(torch.Tensor(latents).to(device, dtype=torch.float16))
img.save("reconstructed_image.png")
