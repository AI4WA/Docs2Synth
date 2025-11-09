"""vLLM provider implementation for local high-performance LLM inference."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore

from docs2synth.agent.base import BaseLLMProvider, LLMResponse
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


class VLLMProvider(BaseLLMProvider):
    """vLLM provider for local high-performance LLM inference.

    Supports two modes:
    1. "server" mode (default): Connects to a running vLLM OpenAI-compatible API server
    2. "direct" mode: Uses vLLM Python API directly (no server needed)

    Server mode:
        Start vLLM server: python -m vllm.entrypoints.openai.api_server --model <model_name>
        Default endpoint: http://localhost:8000/v1

    Direct mode:
        Directly loads and uses vLLM model (requires GPU and vllm package)

    See https://github.com/vllm-project/vllm for more information.
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-2-7b-chat-hf",
        mode: str = "server",
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize vLLM provider.

        Args:
            model: Model name/identifier
            mode: "server" (HTTP API) or "direct" (Python API). Default: "server"
            base_url: vLLM server URL (only used in server mode, default: http://localhost:8000/v1)
            api_key: API key (optional, only used in server mode)
            **kwargs: Additional parameters:
                - For server mode: OpenAI client parameters
                - For direct mode: vLLM LLM initialization parameters (tensor_parallel_size, etc.)
        """
        # Persist full provider config (defaults for generation)
        super().__init__(model, **kwargs)

        self.mode = mode.lower()
        if self.mode not in ("server", "direct"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'server' or 'direct'")

        if self.mode == "server":
            # Server mode: use OpenAI-compatible HTTP API
            if OpenAI is None:
                raise ImportError(
                    "openai package is required for server mode. Install with: pip install openai"
                )

            # Build OpenAI client with ONLY supported client options
            allowed_client_keys = {
                "api_key",
                "base_url",
                "organization",
                "project",
                "timeout",
                "http_client",
            }
            client_kwargs: Dict[str, Any] = {
                k: v for k, v in kwargs.items() if k in allowed_client_keys
            }
            # Set base_url (vLLM default endpoint)
            if "base_url" not in client_kwargs:
                client_kwargs["base_url"] = base_url
            # vLLM typically doesn't require API keys, but OpenAI client requires the parameter
            # Use provided api_key or a dummy value (vLLM server will ignore it)
            if "api_key" not in client_kwargs:
                client_kwargs["api_key"] = api_key if api_key else "EMPTY"

            logger.info(f"Connecting to vLLM server at {client_kwargs['base_url']}")
            self.client = OpenAI(**client_kwargs)
            self.llm = None
        else:
            # Direct mode: use vLLM Python API
            if LLM is None or SamplingParams is None:
                raise ImportError(
                    "vllm package is required for direct mode. Install with: pip install vllm"
                )

            # Extract vLLM-specific initialization parameters
            vllm_init_keys = {
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "trust_remote_code",
                "dtype",
                "max_model_len",
                "gpu_memory_utilization",
                "swap_space",
                "cpu_offload_gb",
                "quantization",
                "enforce_eager",
                "max_seq_len_to_capture",
                "disable_custom_all_reduce",  # Additional parameter that might help
                "enable_lora",  # For LoRA models
                "max_lora_rank",  # For LoRA models
            }
            vllm_kwargs = {k: v for k, v in kwargs.items() if k in vllm_init_keys}

            logger.info(f"Loading vLLM model {model} directly (no server needed)...")
            try:
                self.llm = LLM(model=model, **vllm_kwargs)
                logger.info(f"Successfully loaded vLLM model {model}")
            except Exception as e:
                import traceback

                error_details = traceback.format_exc()
                logger.error(
                    f"Failed to load vLLM model {model}\n"
                    f"Error: {str(e)}\n"
                    f"Model: {model}\n"
                    f"vLLM kwargs: {vllm_kwargs}\n"
                    f"Full traceback:\n{error_details}\n"
                    f"\nTroubleshooting tips:\n"
                    f"1. For vision-language models, ensure trust_remote_code=True\n"
                    f"2. Try setting enforce_eager=True if model architecture is not fully supported\n"
                    f"3. Adjust max_model_len (try 16384, 8192, or 4096) if GPU memory is insufficient\n"
                    f"4. Some models (like Qwen3-VL) may not be fully supported in direct mode - consider using server mode\n"
                    f"5. Check vLLM version compatibility: pip install --upgrade vllm"
                )
                raise
            self.client = None

    def _encode_image(self, image: Any) -> str:
        """Encode PIL Image to base64 string."""
        try:
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError(
                "PIL (Pillow) is required for image support. Install with: pip install Pillow"
            )

        if not isinstance(image, PILImage.Image):
            # If it's already a file path or URL, return as-is
            if isinstance(image, (str, bytes)):
                return str(image)
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Convert PIL Image to base64
        buffer = io.BytesIO()
        # Save as JPEG (most compatible format)
        if image.mode in ("RGBA", "LA", "P"):
            # Convert RGBA/LA/P to RGB for JPEG
            rgb_image = PILImage.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            rgb_image.paste(
                image, mask=image.split()[-1] if image.mode == "RGBA" else None
            )
            image = rgb_image
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"

    def _build_message_content(self, text: str, image: Optional[Any] = None) -> Any:
        """Build message content with optional image."""
        if image is None:
            return text

        # For vision models, content must be a list
        image_url = self._encode_image(image)
        return [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt string using vLLM's chat template."""
        if self.mode == "direct" and self.llm is not None:
            # Use vLLM's tokenizer chat template if available
            try:
                tokenizer = self.llm.get_tokenizer()
                if hasattr(tokenizer, "apply_chat_template"):
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    return prompt
            except Exception as e:
                logger.warning(
                    f"Failed to use chat template: {e}. Using simple format."
                )

        # Fallback: simple format
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extract text from content list
                text_parts = [
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                ]
                content = " ".join(text_parts)
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text using vLLM (server or direct mode)."""
        # Extract image from kwargs
        image = kwargs.pop("image", None)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_content = self._build_message_content(prompt, image)
        messages.append({"role": "user", "content": user_content})

        if self.mode == "server":
            # Server mode: use HTTP API
            # Merge default generation kwargs from provider config
            default_keys = {
                "temperature",
                "max_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "stop",
                "logprobs",
                "logit_bias",
                "seed",
            }
            merged_kwargs: Dict[str, Any] = {
                k: v
                for k, v in self.config.items()
                if k in default_keys and v is not None
            }
            # Call-time args override defaults
            if temperature is not None:
                merged_kwargs["temperature"] = temperature
            if max_tokens is not None:
                merged_kwargs["max_tokens"] = max_tokens
            # Filter out None values from kwargs
            merged_kwargs.update({k: v for k, v in kwargs.items() if v is not None})

            # Support JSON mode
            if response_format == "json":
                merged_kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **merged_kwargs,
            )

            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": (
                    response.usage.completion_tokens if response.usage else 0
                ),
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=self.model,
                usage=usage,
                metadata={"finish_reason": response.choices[0].finish_reason},
            )
        else:
            # Direct mode: use Python API
            # Convert messages to prompt
            full_prompt = self._messages_to_prompt(messages)

            # JSON mode via prompt engineering
            if response_format == "json":
                full_prompt = f"{full_prompt}\n\nIMPORTANT: You must respond with valid JSON only, no additional text or markdown."

            # Build SamplingParams
            default_keys = {
                "temperature",
                "top_p",
                "top_k",
                "max_tokens",
                "stop",
                "frequency_penalty",
                "presence_penalty",
                "seed",
            }
            sampling_kwargs: Dict[str, Any] = {
                k: v
                for k, v in self.config.items()
                if k in default_keys and v is not None
            }
            if temperature is not None:
                sampling_kwargs["temperature"] = temperature
            if max_tokens is not None:
                sampling_kwargs["max_tokens"] = max_tokens
            # vLLM uses stop as a list
            if "stop" in sampling_kwargs and isinstance(sampling_kwargs["stop"], str):
                sampling_kwargs["stop"] = [sampling_kwargs["stop"]]

            sampling_params = SamplingParams(**sampling_kwargs)

            # Generate
            outputs = self.llm.generate([full_prompt], sampling_params)
            output = outputs[0]

            # Extract usage info (vLLM provides token counts)
            usage = {
                "prompt_tokens": (
                    len(output.prompt_token_ids)
                    if hasattr(output, "prompt_token_ids")
                    else 0
                ),
                "completion_tokens": (
                    len(output.outputs[0].token_ids) if output.outputs else 0
                ),
                "total_tokens": 0,
            }
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

            finish_reason = (
                output.outputs[0].finish_reason if output.outputs else "stop"
            )

            return LLMResponse(
                content=output.outputs[0].text if output.outputs else "",
                model=self.model,
                usage=usage,
                metadata={"finish_reason": finish_reason},
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat completion with message history (server or direct mode)."""
        # Extract image from kwargs
        image = kwargs.pop("image", None)

        # Convert message format if needed
        formatted_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # If this is the last user message and we have an image, add image to content
            if (
                role == "user"
                and image is not None
                and msg == messages[-1]
                and isinstance(content, str)
            ):
                content = self._build_message_content(content, image)
            elif isinstance(content, list):
                # Content is already in the correct format (e.g., from a previous call)
                content = content
            else:
                # Regular text content
                content = content

            formatted_messages.append({"role": role, "content": content})

        if self.mode == "server":
            # Server mode: use HTTP API
            # Merge default generation kwargs from provider config
            default_keys = {
                "temperature",
                "max_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "stop",
                "logprobs",
                "logit_bias",
                "seed",
            }
            merged_kwargs: Dict[str, Any] = {
                k: v
                for k, v in self.config.items()
                if k in default_keys and v is not None
            }
            if temperature is not None:
                merged_kwargs["temperature"] = temperature
            if max_tokens is not None:
                merged_kwargs["max_tokens"] = max_tokens
            # Filter out None values from kwargs
            merged_kwargs.update({k: v for k, v in kwargs.items() if v is not None})

            # Support JSON mode
            if response_format == "json":
                merged_kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                **merged_kwargs,
            )

            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": (
                    response.usage.completion_tokens if response.usage else 0
                ),
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=self.model,
                usage=usage,
                metadata={"finish_reason": response.choices[0].finish_reason},
            )
        else:
            # Direct mode: use Python API
            # Convert messages to prompt
            full_prompt = self._messages_to_prompt(formatted_messages)

            # JSON mode via prompt engineering
            if response_format == "json":
                full_prompt = f"{full_prompt}\n\nIMPORTANT: You must respond with valid JSON only, no additional text or markdown."

            # Build SamplingParams
            default_keys = {
                "temperature",
                "top_p",
                "top_k",
                "max_tokens",
                "stop",
                "frequency_penalty",
                "presence_penalty",
                "seed",
            }
            sampling_kwargs: Dict[str, Any] = {
                k: v
                for k, v in self.config.items()
                if k in default_keys and v is not None
            }
            if temperature is not None:
                sampling_kwargs["temperature"] = temperature
            if max_tokens is not None:
                sampling_kwargs["max_tokens"] = max_tokens
            # vLLM uses stop as a list
            if "stop" in sampling_kwargs and isinstance(sampling_kwargs["stop"], str):
                sampling_kwargs["stop"] = [sampling_kwargs["stop"]]

            sampling_params = SamplingParams(**sampling_kwargs)

            # Generate
            outputs = self.llm.generate([full_prompt], sampling_params)
            output = outputs[0]

            # Extract usage info
            usage = {
                "prompt_tokens": (
                    len(output.prompt_token_ids)
                    if hasattr(output, "prompt_token_ids")
                    else 0
                ),
                "completion_tokens": (
                    len(output.outputs[0].token_ids) if output.outputs else 0
                ),
                "total_tokens": 0,
            }
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

            finish_reason = (
                output.outputs[0].finish_reason if output.outputs else "stop"
            )

            return LLMResponse(
                content=output.outputs[0].text if output.outputs else "",
                model=self.model,
                usage=usage,
                metadata={"finish_reason": finish_reason},
            )
