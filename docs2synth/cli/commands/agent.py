"""Agent commands for text generation and chat."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click

from docs2synth.cli.utils import (
    build_gen_kwargs,
    load_history_file,
    resolve_config_path,
    save_history_file,
)
from docs2synth.utils import get_logger

logger = get_logger(__name__)


@click.group("agent")
@click.pass_context
def agent_group(ctx: click.Context) -> None:
    """LLM agent commands for text generation and chat."""
    pass


@agent_group.command("generate")
@click.argument("prompt", type=str)
@click.option(
    "--provider",
    type=str,
    default="openai",
    show_default=True,
    help="Provider name (openai, anthropic, gemini, doubao, ollama, huggingface, vllm)",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model name (optional, uses provider default if not specified)",
)
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config.yml (optional, uses DOCS2SYNTH_CONFIG env var or ./config.yml if set)",
)
@click.option(
    "--system-prompt",
    type=str,
    default=None,
    help="System prompt for chat models",
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="Sampling temperature (0.0-2.0)",
)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    help="Maximum tokens to generate",
)
@click.option(
    "--response-format",
    type=click.Choice(["json", "text"]),
    default=None,
    help="Response format (json for JSON mode)",
)
@click.option(
    "--image",
    type=click.Path(exists=True),
    default=None,
    help="Path to image file (optional, for vision models)",
)
@click.pass_context
def agent_generate(
    ctx: click.Context,
    prompt: str,
    provider: str,
    model: str | None,
    config_path: str | None,
    system_prompt: str | None,
    temperature: float | None,
    max_tokens: int | None,
    response_format: str | None,
    image: str | None,
) -> None:
    """Generate text from a prompt using LLM agents.

    PROMPT: The text prompt to generate from.

    Examples:
        docs2synth agent generate "Explain quantum computing"
        docs2synth agent generate "List 3 items" --provider anthropic --response-format json
        docs2synth agent generate "What's in this image?" --image photo.jpg --provider openai --model gpt-4o
    """
    from PIL import Image as PILImage

    from docs2synth.agent import AgentWrapper

    try:
        # Build kwargs for AgentWrapper
        agent_kwargs: dict[str, Any] = {}
        # Resolve config_path default to ./config.yml if present
        if not config_path and Path("./config.yml").exists():
            config_path = "./config.yml"

        if model:
            agent_kwargs["model"] = model
        if config_path:
            agent_kwargs["config_path"] = config_path

        agent = AgentWrapper(provider=provider, **agent_kwargs)

        # Build generation kwargs
        gen_kwargs: dict[str, Any] = {}
        if system_prompt:
            gen_kwargs["system_prompt"] = system_prompt
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if max_tokens is not None:
            gen_kwargs["max_tokens"] = max_tokens
        if response_format:
            gen_kwargs["response_format"] = response_format
        if image:
            image_obj = PILImage.open(image)
            gen_kwargs["image"] = image_obj

        click.echo(click.style(f"Generating with {provider}...", fg="blue"))
        response = agent.generate(prompt, **gen_kwargs)

        click.echo(click.style("\nResponse:", fg="green", bold=True))
        click.echo(response.content)

        if response.usage:
            click.echo(
                click.style(f"\nToken usage: {response.usage}", fg="cyan", dim=True)
            )

    except Exception as e:
        logger.exception("Agent generate command failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@agent_group.command("chat")
@click.argument("message", type=str)
@click.option(
    "--provider",
    type=str,
    default="openai",
    show_default=True,
    help="Provider name (openai, anthropic, gemini, doubao, ollama, huggingface, vllm)",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model name (optional, uses provider default if not specified)",
)
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config.yml (optional, uses DOCS2SYNTH_CONFIG env var or ./config.yml if set)",
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="Sampling temperature (0.0-2.0)",
)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    help="Maximum tokens to generate",
)
@click.option(
    "--response-format",
    type=click.Choice(["json", "text"]),
    default=None,
    help="Response format (json for JSON mode)",
)
@click.option(
    "--history-file",
    type=click.Path(),
    default=None,
    help="Path to JSON file with chat history (optional)",
)
@click.option(
    "--image",
    type=click.Path(exists=True),
    default=None,
    help="Path to image file (optional, for vision models)",
)
@click.pass_context
def agent_chat(
    ctx: click.Context,
    message: str,
    provider: str,
    model: str | None,
    config_path: str | None,
    temperature: float | None,
    max_tokens: int | None,
    response_format: str | None,
    history_file: str | None,
    image: str | None,
) -> None:
    """Chat with LLM agents using message history.

    MESSAGE: The user message to send.

    Examples:
        docs2synth agent chat "What is Python?"
        docs2synth agent chat "Explain AI" --provider anthropic --model claude-3-5-sonnet-20241022
        docs2synth agent chat "What's in this image?" --image photo.jpg --provider openai --model gpt-4o
        docs2synth agent chat "Hello" --history-file chat.json
    """
    from PIL import Image as PILImage

    from docs2synth.agent import AgentWrapper

    try:
        config_path = resolve_config_path(config_path)

        agent_kwargs: dict[str, Any] = {}
        if model:
            agent_kwargs["model"] = model
        if config_path:
            agent_kwargs["config_path"] = config_path
        agent = AgentWrapper(provider=provider, **agent_kwargs)

        messages = load_history_file(history_file)
        messages.append({"role": "user", "content": message})

        gen_kwargs = build_gen_kwargs(temperature, max_tokens, response_format)
        if image:
            image_obj = PILImage.open(image)
            gen_kwargs["image"] = image_obj

        click.echo(click.style(f"Chatting with {provider}...", fg="blue"))
        response = agent.chat(messages, **gen_kwargs)

        click.echo(click.style("\nResponse:", fg="green", bold=True))
        click.echo(response.content)

        if response.usage:
            click.echo(
                click.style(f"\nToken usage: {response.usage}", fg="cyan", dim=True)
            )

        messages.append({"role": "assistant", "content": response.content})
        save_history_file(history_file, messages)

    except Exception as e:
        logger.exception("Agent chat command failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


def _load_vllm_config(config_path: str | None) -> dict[str, Any]:
    """Load vLLM configuration from config file."""
    import yaml

    vllm_config = {}
    if config_path:
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                vllm_config = config.get("agent", {}).get("vllm", {})
        except Exception as e:
            click.echo(
                click.style(f"⚠ Warning: Could not load config.yml: {e}", fg="yellow")
            )
    return vllm_config


def _build_vllm_command(
    model_name: str,
    host: str,
    port: int,
    vllm_config: dict[str, Any],
    trust_remote_code: bool | None,
    max_model_len: int | None,
    gpu_memory_utilization: float | None,
    tensor_parallel_size: int | None,
) -> list[str]:
    """Build vLLM server command with optional parameters."""
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--host",
        host,
        "--port",
        str(port),
    ]

    # Add optional parameters (CLI override > config)
    if trust_remote_code is not None:
        if trust_remote_code:
            cmd.append("--trust-remote-code")
    elif vllm_config.get("trust_remote_code"):
        cmd.append("--trust-remote-code")

    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])
    elif vllm_config.get("max_model_len"):
        cmd.extend(["--max-model-len", str(vllm_config["max_model_len"])])

    if gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    elif vllm_config.get("gpu_memory_utilization"):
        cmd.extend(
            [
                "--gpu-memory-utilization",
                str(vllm_config["gpu_memory_utilization"]),
            ]
        )

    if tensor_parallel_size is not None:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    elif vllm_config.get("tensor_parallel_size"):
        cmd.extend(["--tensor-parallel-size", str(vllm_config["tensor_parallel_size"])])

    return cmd


def _display_vllm_startup_info(
    model_name: str, host: str, port: int, cmd: list[str]
) -> None:
    """Display vLLM server startup information."""
    click.echo(click.style("🚀 Starting vLLM OpenAI API Server", fg="green", bold=True))
    click.echo(click.style(f"   Model: {model_name}", fg="cyan"))
    click.echo(click.style(f"   Endpoint: http://{host}:{port}/v1", fg="cyan"))
    click.echo()
    click.echo(click.style("Command:", fg="blue"))
    click.echo(f"   {' '.join(cmd)}")
    click.echo()
    click.echo(
        click.style(
            "💡 Tip: Keep this terminal open. Use Ctrl+C to stop the server.",
            fg="yellow",
        )
    )
    click.echo(
        click.style(
            "💡 Test: curl http://localhost:{}/health".format(port), fg="yellow"
        )
    )
    click.echo()


@agent_group.command("vllm-server")
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config.yml (optional, uses ./config.yml if exists)",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Override model from config (e.g., meta-llama/Llama-2-7b-chat-hf)",
)
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Host to bind the server (default: 0.0.0.0)",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Port to bind the server (default: 8000)",
)
@click.option(
    "--trust-remote-code",
    is_flag=True,
    default=None,
    help="Enable trust_remote_code (required for some models like Qwen)",
)
@click.option(
    "--max-model-len",
    type=int,
    default=None,
    help="Maximum model context length (default: model's default)",
)
@click.option(
    "--gpu-memory-utilization",
    type=float,
    default=None,
    help="GPU memory utilization (0.0-1.0, default: 0.9)",
)
@click.option(
    "--tensor-parallel-size",
    type=int,
    default=None,
    help="Number of GPUs for tensor parallelism (default: 1)",
)
@click.pass_context
def agent_vllm_server(
    ctx: click.Context,
    config_path: str | None,
    model: str | None,
    host: str,
    port: int,
    trust_remote_code: bool | None,
    max_model_len: int | None,
    gpu_memory_utilization: float | None,
    tensor_parallel_size: int | None,
) -> None:
    """Start a vLLM OpenAI-compatible API server.

    This command starts a vLLM server using configuration from config.yml
    or command-line options. The server provides an OpenAI-compatible API
    that can be used with the vLLM provider in server mode.

    Examples:
        # Start with config.yml settings
        docs2synth agent vllm-server

        # Override model
        docs2synth agent vllm-server --model meta-llama/Llama-2-7b-chat-hf

        # Custom port and GPU settings
        docs2synth agent vllm-server --port 8080 --gpu-memory-utilization 0.8

        # Multi-GPU setup
        docs2synth agent vllm-server --tensor-parallel-size 2
    """
    import subprocess

    try:
        config_path = resolve_config_path(config_path)
        vllm_config = _load_vllm_config(config_path)

        # Determine model (CLI override > config > error)
        model_name = model or vllm_config.get("model")
        if not model_name:
            click.echo(
                click.style(
                    "✗ Error: No model specified. Use --model or set agent.vllm.model in config.yml",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)

        cmd = _build_vllm_command(
            model_name,
            host,
            port,
            vllm_config,
            trust_remote_code,
            max_model_len,
            gpu_memory_utilization,
            tensor_parallel_size,
        )

        _display_vllm_startup_info(model_name, host, port, cmd)
        subprocess.run(cmd, check=True)

    except KeyboardInterrupt:
        click.echo()
        click.echo(click.style("✓ Server stopped by user", fg="green"))
        sys.exit(0)
    except FileNotFoundError:
        click.echo(
            click.style(
                "✗ Error: vLLM not installed. Install with: pip install vllm", fg="red"
            ),
            err=True,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        click.echo(click.style(f"✗ Error: vLLM server failed: {e}", fg="red"), err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("vLLM server command failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)
