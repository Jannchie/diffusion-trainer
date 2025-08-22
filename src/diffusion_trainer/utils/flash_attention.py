"""Flash Attention integration for memory-efficient training."""

import logging
from typing import Any, Protocol

import torch
from torch import nn

logger = logging.getLogger("diffusion_trainer")


class DiffusionPipelineProtocol(Protocol):
    """Protocol for diffusion pipelines."""

    unet: Any
    text_encoder: Any
    text_encoder_2: Any


def _try_import_xformers() -> tuple[Any, Any] | None:
    """Try to import xformers and return the modules if successful."""
    try:
        import xformers  # noqa: PLC0415
        import xformers.ops  # noqa: PLC0415

        return xformers, xformers.ops  # noqa: TRY300
    except ImportError:
        return None


def _try_import_xformers_processor() -> type[Any] | None:
    """Try to import XFormersAttnProcessor."""
    try:
        from diffusers.models.attention_processor import XFormersAttnProcessor  # noqa: PLC0415

        return XFormersAttnProcessor  # noqa: TRY300
    except ImportError:
        return None


def enable_flash_attention_for_model(model: nn.Module, model_type: str = "unknown") -> bool:
    """
    Enable Flash Attention (xformers) for a given model.

    Args:
        model: The model to enable Flash Attention for
        model_type: Type of model for logging purposes

    Returns:
        True if Flash Attention was successfully enabled, False otherwise
    """
    xformers_modules = _try_import_xformers()
    if xformers_modules is None:
        logger.warning("xformers not available - Flash Attention cannot be enabled")
        return False

    xformers, xformers_ops = xformers_modules

    try:
        logger.info("xformers version: %s", xformers.__version__)

        # Try different methods to enable Flash Attention
        success = _try_enable_flash_attention_methods(model, model_type)

        if not success:
            logger.warning("Could not enable Flash Attention for %s - no compatible method found", model_type)

        return success  # noqa: TRY300

    except Exception as e:
        logger.warning("Failed to enable Flash Attention for %s: %s", model_type, e)
        return False


def _try_enable_flash_attention_methods(model: nn.Module, model_type: str) -> bool:
    """Try different methods to enable Flash Attention for a model."""
    # Method 1: Native xformers support
    if hasattr(model, "enable_xformers_memory_efficient_attention"):
        model.enable_xformers_memory_efficient_attention()  # type: ignore
        logger.info("✓ Flash Attention enabled for %s", model_type)
        return True

    # Method 2: Legacy method
    if hasattr(model, "set_use_memory_efficient_attention_xformers"):
        model.set_use_memory_efficient_attention_xformers(True)  # type: ignore
        logger.info("✓ Flash Attention enabled for %s (legacy method)", model_type)
        return True

    # Method 3: Attention processors
    if hasattr(model, "set_attn_processor"):
        return _try_set_attention_processors(model, model_type)

    return False


def _try_set_attention_processors(model: nn.Module, model_type: str) -> bool:
    """Try to set xformers attention processors."""
    xformers_processor_cls = _try_import_xformers_processor()
    if xformers_processor_cls is None:
        logger.warning("XFormersAttnProcessor not available")
        return False

    # Get all attention processors
    attn_processors = {}
    for name, module in model.named_modules():
        if hasattr(module, "processor") and "attn" in name.lower():
            attn_processors[name] = xformers_processor_cls()

    if attn_processors:
        model.set_attn_processor(attn_processors)  # type: ignore
        logger.info("✓ Flash Attention enabled for %s using attention processors", model_type)
        return True

    return False


def disable_flash_attention_for_model(model: nn.Module, model_type: str = "unknown") -> bool:
    """
    Disable Flash Attention for a given model.

    Args:
        model: The model to disable Flash Attention for
        model_type: Type of model for logging purposes

    Returns:
        True if Flash Attention was successfully disabled, False otherwise
    """
    try:
        success = _try_disable_flash_attention_methods(model, model_type)

        if not success:
            logger.warning("Could not disable Flash Attention for %s - no compatible method found", model_type)

        return success  # noqa: TRY300

    except Exception as e:
        logger.warning("Failed to disable Flash Attention for %s: %s", model_type, e)
        return False


def _try_disable_flash_attention_methods(model: nn.Module, model_type: str) -> bool:
    """Try different methods to disable Flash Attention for a model."""
    # Method 1: Native disable method
    if hasattr(model, "disable_xformers_memory_efficient_attention"):
        model.disable_xformers_memory_efficient_attention()  # type: ignore
        logger.info("✓ Flash Attention disabled for %s", model_type)
        return True

    # Method 2: Legacy method
    if hasattr(model, "set_use_memory_efficient_attention_xformers"):
        model.set_use_memory_efficient_attention_xformers(False)  # type: ignore
        logger.info("✓ Flash Attention disabled for %s (legacy method)", model_type)
        return True

    # Method 3: Default attention processors
    if hasattr(model, "set_default_attn_processor"):
        model.set_default_attn_processor()  # type: ignore
        logger.info("✓ Flash Attention disabled for %s using default attention processors", model_type)
        return True

    return False


def is_flash_attention_available() -> bool:
    """
    Check if Flash Attention (xformers) is available.

    Returns:
        True if xformers is available and functional, False otherwise
    """
    xformers_modules = _try_import_xformers()
    if xformers_modules is None:
        return False

    xformers, xformers_ops = xformers_modules

    try:
        # Simple test to make sure xformers works
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        # Test with small tensors
        batch_size, seq_len, dim = 2, 64, 128
        query = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)
        key = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)
        value = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)

        # Try to run xformers attention
        result = xformers_ops.memory_efficient_attention(query, key, value)

        if result is not None:
            logger.info("✓ Flash Attention is available and functional")
            return True

    except Exception as e:
        logger.info("Flash Attention test failed: %s", e)

    return False


def get_attention_stats(model: nn.Module) -> dict[str, Any]:
    """
    Get statistics about attention mechanisms in the model.

    Args:
        model: The model to analyze

    Returns:
        Dictionary with attention statistics
    """
    stats = {
        "total_attention_layers": 0,
        "flash_attention_enabled": 0,
        "attention_types": {},
        "model_has_xformers_support": False,
    }

    # Check if model has xformers support
    stats["model_has_xformers_support"] = hasattr(model, "enable_xformers_memory_efficient_attention")

    # Count attention layers and their types
    for name, module in model.named_modules():
        if "attn" in name.lower() or "attention" in name.lower():
            stats["total_attention_layers"] += 1

            # Try to determine attention type
            if hasattr(module, "processor"):
                processor_type = type(module.processor).__name__
                stats["attention_types"][processor_type] = stats["attention_types"].get(processor_type, 0) + 1

                # Check if it's xformers
                if "xformers" in processor_type.lower() or "XFormers" in processor_type:
                    stats["flash_attention_enabled"] += 1

    return stats


def log_attention_info(model: nn.Module, model_name: str = "Model") -> None:
    """
    Log information about attention mechanisms in the model.

    Args:
        model: The model to analyze
        model_name: Name of the model for logging
    """
    stats = get_attention_stats(model)

    logger.info("%s attention statistics:", model_name)
    logger.info("  Total attention layers: %d", stats["total_attention_layers"])
    logger.info("  Flash Attention layers: %d", stats["flash_attention_enabled"])
    logger.info("  XFormers support: %s", stats["model_has_xformers_support"])

    if stats["attention_types"]:
        logger.info("  Attention processor types:")
        for proc_type, count in stats["attention_types"].items():
            logger.info("    - %s: %d layers", proc_type, count)

    if stats["flash_attention_enabled"] > 0:
        efficiency = (stats["flash_attention_enabled"] / max(stats["total_attention_layers"], 1)) * 100
        logger.info("  Flash Attention coverage: %.1f%%", efficiency)


def enable_flash_attention_pipeline(
    pipeline: Any,  # noqa: ANN401
    *,
    enable_unet: bool = True,
    enable_text_encoder: bool = True,
) -> dict[str, bool]:
    """
    Enable Flash Attention for all compatible components in a diffusion pipeline.

    Args:
        pipeline: The diffusion pipeline
        enable_unet: Whether to enable Flash Attention for UNet
        enable_text_encoder: Whether to enable Flash Attention for text encoders

    Returns:
        Dictionary with results for each component
    """
    results = {}

    if not is_flash_attention_available():
        logger.warning("Flash Attention is not available - skipping enablement")
        return results

    # Enable for UNet
    if enable_unet and hasattr(pipeline, "unet"):
        results["unet"] = enable_flash_attention_for_model(pipeline.unet, "UNet")
        log_attention_info(pipeline.unet, "UNet")

    # Enable for text encoder
    if enable_text_encoder and hasattr(pipeline, "text_encoder"):
        results["text_encoder"] = enable_flash_attention_for_model(pipeline.text_encoder, "Text Encoder")
        log_attention_info(pipeline.text_encoder, "Text Encoder")

    # Enable for text encoder 2 (SDXL)
    if enable_text_encoder and hasattr(pipeline, "text_encoder_2"):
        results["text_encoder_2"] = enable_flash_attention_for_model(pipeline.text_encoder_2, "Text Encoder 2")
        log_attention_info(pipeline.text_encoder_2, "Text Encoder 2")

    # Log summary
    enabled_components = [comp for comp, enabled in results.items() if enabled]
    if enabled_components:
        logger.info("✅ Flash Attention enabled for: %s", ", ".join(enabled_components))
    else:
        logger.warning("❌ Flash Attention could not be enabled for any component")

    return results
