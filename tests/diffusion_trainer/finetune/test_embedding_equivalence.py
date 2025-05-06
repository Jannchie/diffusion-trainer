import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sd15


def get_native_embedding(
    pipe: StableDiffusionPipeline,
    prompt: str,
    device: torch.device | str,
) -> torch.Tensor:
    text_inputs = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs["input_ids"].to(device)
    prompt_embeds_output = pipe.text_encoder(
        text_input_ids,
        output_hidden_states=True,
    )
    return prompt_embeds_output.last_hidden_state


def get_sd_embed_embedding(
    pipe: StableDiffusionPipeline,
    prompt: str,
    device: torch.device | str,
) -> torch.Tensor:
    # 类型忽略，兼容所有 diffusers pipeline
    prompt_embeds, _ = get_weighted_text_embeddings_sd15(  # type: ignore
        pipe,
        prompt,
        "",
        pad_last_block=True,
    )
    return prompt_embeds.to(device)


def test_embedding_equivalence() -> None:
    model_id = "models/AOM3B2_orangemixs_fp16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    pipe = pipe.to(device)

    prompt = "a photo of a cat"
    native_emb = get_native_embedding(pipe, prompt, device)
    sd_embed_emb = get_sd_embed_embedding(pipe, prompt, device)

    assert native_emb.shape == sd_embed_emb.shape, f"Shape mismatch: {native_emb.shape} vs {sd_embed_emb.shape}"
    assert torch.allclose(native_emb, sd_embed_emb, atol=1e-5), "Embedding values are not close!"


def test_embedding_with_weight() -> None:
    model_id = "models/AOM3B2_orangemixs_fp16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    pipe = pipe.to(device)

    prompt = "a photo of a (cat:1.5)"
    native_emb = get_native_embedding(pipe, prompt, device)
    sd_embed_emb = get_sd_embed_embedding(pipe, prompt, device)

    assert native_emb.shape == sd_embed_emb.shape, f"Shape mismatch: {native_emb.shape} vs {sd_embed_emb.shape}"
    # 有权重时，embedding 应明显不同
    assert not torch.allclose(native_emb, sd_embed_emb, atol=1e-5), "Embedding with weight should be different!"
