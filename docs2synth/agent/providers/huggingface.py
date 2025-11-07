"""Hugging Face Transformers provider implementation for local models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore

from docs2synth.agent.base import BaseLLMProvider, LLMResponse
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face Transformers provider for local models.

    Loads models directly from Hugging Face Hub or local path.
    """

    def __init__(
        self,
        model: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        hf_token: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize Hugging Face provider.

        Args:
            model: Model identifier (e.g., 'meta-llama/Llama-2-7b-chat-hf')
            device: Device to use ('cuda', 'cpu', 'auto'). Auto-detects if None.
            load_in_8bit: Load model in 8-bit mode (requires bitsandbytes)
            load_in_4bit: Load model in 4-bit mode (requires bitsandbytes)
            **kwargs: Additional model loading parameters
        """
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError(
                "transformers package is required. Install with: pip install transformers torch"
            )

        # Split generation defaults from loading kwargs
        generation_keys = {
            "temperature",
            "top_p",
            "top_k",
            "max_new_tokens",
            "max_tokens",
            "do_sample",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "eos_token_id",
            "pad_token_id",
            # Filter out OpenAI-style args that should never reach HF loaders
            "stop",
            "logprobs",
            "logit_bias",
            "seed",
            # Ensure response_format never reaches loaders
            "response_format",
        }
        gen_defaults = {
            k: kwargs.pop(k) for k in list(kwargs.keys()) if k in generation_keys
        }
        # Persist only generation defaults into provider config
        super().__init__(model, **gen_defaults)

        # Auto-detect device
        if device is None:
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"

        # Convert to torch.device object for consistency
        self.device = torch.device(device) if isinstance(device, str) else device
        device_str = str(self.device)
        self.logger.info(f"Loading model {model} on {device_str}...")

        # Extract token if passed via kwargs for backward-compat
        if hf_token is None and "hf_token" in kwargs:
            hf_token = kwargs.pop("hf_token")
        token_kw = {"token": hf_token} if hf_token else {}

        # Load tokenizer (pass only loading kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model, **token_kw, **kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        load_kwargs = {
            "torch_dtype": torch.float16 if device_str == "cuda" else torch.float32
        }
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map=device_str if device_str != "cpu" else None,
            **load_kwargs,
            **token_kw,
            **kwargs,
        )

        if device_str == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        self.logger.info(f"Model {model} loaded successfully")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text using Hugging Face model."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Hugging Face JSON mode via prompt engineering
        if response_format == "json":
            full_prompt = f"{full_prompt}\n\nIMPORTANT: You must respond with valid JSON only, no additional text or markdown."

        # Tokenize input
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

        # Merge generation defaults from self.config
        # Note: HF doesn't support presence_penalty/frequency_penalty (OpenAI-specific)
        default_keys = {
            "temperature",
            "top_p",
            "top_k",
            "max_new_tokens",
            "do_sample",
            "repetition_penalty",
            "eos_token_id",
            "pad_token_id",
        }
        merged = {k: v for k, v in self.config.items() if k in default_keys}
        if temperature is not None:
            merged["temperature"] = temperature
        if max_tokens is not None:
            merged["max_new_tokens"] = max_tokens
        elif "max_new_tokens" not in merged:
            merged["max_new_tokens"] = 512
        if "pad_token_id" not in merged:
            merged["pad_token_id"] = self.tokenizer.pad_token_id
        if "eos_token_id" not in merged:
            merged["eos_token_id"] = self.tokenizer.eos_token_id
        if "do_sample" not in merged and merged.get("temperature", 0) > 0:
            merged["do_sample"] = True
        # Filter out OpenAI-specific params that HF doesn't support
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in {"presence_penalty", "frequency_penalty", "logprobs", "logit_bias"}
        }
        merged.update(filtered_kwargs)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **merged,
            )

        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Calculate token usage
        prompt_tokens = inputs["input_ids"].shape[1]
        completion_tokens = outputs[0].shape[0] - prompt_tokens

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": outputs[0].shape[0],
        }

        return LLMResponse(
            content=generated_text,
            model=(
                self.model.config.name
                if hasattr(self.model.config, "name")
                else self.model
            ),
            usage=usage,
            metadata={},
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat completion with message history."""
        # Format messages into prompt
        # This is a simple implementation - you may want to use model-specific chat templates
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n".join(prompt_parts) + "\nAssistant:"

        return self.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs,
        )
