"""Utilities for loading SDXL checkpoints."""

from logging import getLogger
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors.torch import load_file
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

from diffusion_trainer.lib import model_util, sdxl_original_unet

logger = getLogger("diffusion_trainer")


def convert_sdxl_text_encoder_2_checkpoint(checkpoint: dict) -> tuple[dict, None | torch.Tensor]:  # noqa: C901, PLR0915
    SDXL_KEY_PREFIX = "conditioner.embedders.1.model."  # noqa: N806

    # SD2のと、基本的には同じ。logit_scaleを後で使うので、それを追加で返す
    # logit_scaleはcheckpointの保存時に使用する
    def convert_key(key: str | None) -> None | str:  # noqa: C901, PLR0912
        # common conversion
        if key is None:
            return None
        key = key.replace(SDXL_KEY_PREFIX + "transformer.", "text_model.encoder.")
        key = key.replace(SDXL_KEY_PREFIX, "text_model.")
        if "resblocks" in key:
            # resblocks conversion
            key = key.replace(".resblocks.", ".layers.")
            if ".ln_" in key:
                key = key.replace(".ln_", ".layer_norm")
            elif ".mlp." in key:
                key = key.replace(".c_fc.", ".fc1.")
                key = key.replace(".c_proj.", ".fc2.")
            elif ".attn.out_proj" in key:
                key = key.replace(".attn.out_proj.", ".self_attn.out_proj.")
            elif ".attn.in_proj" in key:
                key = None  # 特殊なので後で処理する
            else:
                msg = f"unexpected key in SD: {key}"
                raise ValueError(msg)
        elif ".positional_embedding" in key:
            key = key.replace(".positional_embedding", ".embeddings.position_embedding.weight")
        elif ".text_projection" in key:
            key = key.replace("text_model.text_projection", "text_projection.weight")
        elif ".logit_scale" in key:
            key = None  # 後で処理する
        elif ".token_embedding" in key:
            key = key.replace(".token_embedding.weight", ".embeddings.token_embedding.weight")
        elif ".ln_final" in key:
            key = key.replace(".ln_final", ".final_layer_norm")
        # ckpt from comfy has this key: text_model.encoder.text_model.embeddings.position_ids
        elif ".embeddings.position_ids" in key:
            key = None  # remove this key: position_ids is not used in newer transformers
        return key

    keys = list(checkpoint.keys())
    new_sd = {}
    for key in keys:
        new_key = convert_key(key)
        if new_key is None:
            continue
        new_sd[new_key] = checkpoint[key]

    # attnの変換
    for key in keys:
        if ".resblocks" in key and ".attn.in_proj_" in key:
            # 三つに分割
            values = torch.chunk(checkpoint[key], 3)

            key_suffix = ".weight" if "weight" in key else ".bias"
            key_pfx = key.replace(SDXL_KEY_PREFIX + "transformer.resblocks.", "text_model.encoder.layers.")
            key_pfx = key_pfx.replace("_weight", "")
            key_pfx = key_pfx.replace("_bias", "")
            key_pfx = key_pfx.replace(".attn.in_proj", ".self_attn.")
            new_sd[key_pfx + "q_proj" + key_suffix] = values[0]
            new_sd[key_pfx + "k_proj" + key_suffix] = values[1]
            new_sd[key_pfx + "v_proj" + key_suffix] = values[2]

    # logit_scale はDiffusersには含まれないが、保存時に戻したいので別途返す
    logit_scale = checkpoint.get(SDXL_KEY_PREFIX + "logit_scale", None)

    # temporary workaround for text_projection.weight.weight for Playground-v2
    if "text_projection.weight.weight" in new_sd:
        logger.info("convert_sdxl_text_encoder_2_checkpoint: convert text_projection.weight.weight to text_projection.weight")
        new_sd["text_projection.weight"] = new_sd["text_projection.weight.weight"]
        del new_sd["text_projection.weight.weight"]

    return new_sd, logit_scale


# load state_dict without allocating new tensors
def _load_state_dict_on_device(model: torch.nn.Module, state_dict: dict, device: str, dtype: None | torch.dtype = None) -> str:
    # dtype will use fp32 as default
    missing_keys = list(model.state_dict().keys() - state_dict.keys())
    unexpected_keys = list(state_dict.keys() - model.state_dict().keys())

    # similar to model.load_state_dict()
    if not missing_keys and not unexpected_keys:
        for k in list(state_dict.keys()):
            set_module_tensor_to_device(model, k, device, value=state_dict.pop(k), dtype=dtype)
        return "<All keys matched successfully>"

    # error_msgs
    error_msgs: list[str] = []
    if missing_keys:
        error_msgs.insert(0, "Missing key(s) in state_dict: {}. ".format(", ".join(f'"{k}"' for k in missing_keys)))
    if unexpected_keys:
        error_msgs.insert(0, "Unexpected key(s) in state_dict: {}. ".format(", ".join(f'"{k}"' for k in unexpected_keys)))

    msg = "Error(s) in loading state_dict for {}:\n\t{}".format(model.__class__.__name__, "\n\t".join(error_msgs))
    raise RuntimeError(msg)


def load_models_from_sdxl_checkpoint(ckpt_path: Path, device: str, dtype: None | torch.dtype = None) -> tuple:
    """Load models from the SDXL checkpoint."""
    # dtype is used for full_fp16/bf16 integration. Text Encoder will remain fp32, because it runs on CPU when caching

    # Load the state dict
    if ckpt_path.suffix == ".safetensors":
        checkpoint = None
        state_dict = load_file(ckpt_path, device=device)
        epoch = None
        global_step = None
    else:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint.get("epoch", 0)
            global_step = checkpoint.get("global_step", 0)
        else:
            state_dict = checkpoint
            epoch = 0
            global_step = 0
        checkpoint = None

    # U-Net
    logger.info("building U-Net")
    with init_empty_weights():
        unet = sdxl_original_unet.SdxlUNet2DConditionModel()

    logger.info("loading U-Net from checkpoint")
    unet_sd = {}
    for k in list(state_dict.keys()):
        if k.startswith("model.diffusion_model."):
            unet_sd[k.replace("model.diffusion_model.", "")] = state_dict.pop(k)
    info = _load_state_dict_on_device(unet, unet_sd, device=device, dtype=dtype)
    logger.info("U-Net: %s", info)

    # Text Encoders
    logger.info("building text encoders")

    # Text Encoder 1 is same to Stability AI's SDXL
    text_model1_cfg = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=768,
        # torch_dtype="float32",
        # transformers_version="4.25.0.dev0",
    )
    with init_empty_weights():
        text_model1 = CLIPTextModel._from_config(text_model1_cfg)  # noqa: SLF001

    # Text Encoder 2 is different from Stability AI's SDXL. SDXL uses open clip, but we use the model from HuggingFace.
    # Note: Tokenizer from HuggingFace is different from SDXL. We must use open clip's tokenizer.
    text_model2_cfg = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=20,
        max_position_embeddings=77,
        hidden_act="gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=1280,
        # torch_dtype="float32",
        # transformers_version="4.25.0.dev0",
    )
    with init_empty_weights():
        text_model2 = CLIPTextModelWithProjection(text_model2_cfg)

    logger.info("loading text encoders from checkpoint")
    te1_sd = {}
    te2_sd = {}
    for k in list(state_dict.keys()):
        if k.startswith("conditioner.embedders.0.transformer."):
            te1_sd[k.replace("conditioner.embedders.0.transformer.", "")] = state_dict.pop(k)
        elif k.startswith("conditioner.embedders.1.model."):
            te2_sd[k] = state_dict.pop(k)

    # 最新の transformers では position_ids を含むとエラーになるので削除 / remove position_ids for latest transformers
    if "text_model.embeddings.position_ids" in te1_sd:
        te1_sd.pop("text_model.embeddings.position_ids")

    info_1 = _load_state_dict_on_device(text_model1, te1_sd, device=device)  # remain fp32
    logger.info("text encoder 1: %s", info_1)

    converted_sd, logit_scale = convert_sdxl_text_encoder_2_checkpoint(te2_sd)
    info_2 = _load_state_dict_on_device(text_model2, converted_sd, device=device)  # remain fp32
    logger.info("text encoder 2: %s", info_2)

    # prepare vae
    logger.info("building VAE")
    vae_config = model_util.create_vae_diffusers_config()
    with init_empty_weights():
        vae = AutoencoderKL(**vae_config)

    logger.info("loading VAE from checkpoint")
    converted_vae_checkpoint = model_util.convert_ldm_vae_checkpoint(state_dict, vae_config)
    info = _load_state_dict_on_device(vae, converted_vae_checkpoint, device=device, dtype=dtype)
    logger.info("VAE: %s", info)

    ckpt_info = (epoch, global_step) if epoch is not None else None
    return text_model1, text_model2, vae, unet, logit_scale, ckpt_info
