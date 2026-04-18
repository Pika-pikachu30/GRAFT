import logging
import time
from typing import Any, Dict, Optional, Tuple

import litellm
from litellm import completion, cost_per_token
from rich.logging import RichHandler
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Set up litellm configuration to avoid spamming the console
litellm.suppress_debug_info = True

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Sets up a rich logger for pretty CLI output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, markup=True)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

logger = setup_logger(__name__)

import yaml
from litellm.exceptions import ServiceUnavailableError

def _get_default_llm() -> str:
    """Reads config.yaml for default llm_model or returns ollama_chat/llama3.2:latest."""
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            return config.get("indexing", {}).get("llm_model", "ollama_chat/llama3.2:latest")
    except Exception:
        return "ollama_chat/llama3.2:latest"

class LLMWrapper:
    """Wrapper around litellm to provide unified LLM calls, retries, and logging."""

    def __init__(self, default_model: Optional[str] = None):
        """
        Initializes the LLMWrapper.

        Args:
            default_model (Optional[str]): The default model to use. If None, falls back to config.yaml
        """
        self.default_model = default_model or _get_default_llm()
        
    @retry(
        wait=wait_exponential(multiplier=1, min=5, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((litellm.RateLimitError, litellm.APIConnectionError, litellm.APIError, ServiceUnavailableError)),
        reraise=True
    )
    def generate(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None, **kwargs) -> str:
        """
        Generates text using the chosen LLM with retry logic.
        
        Args:
            prompt (str): The user prompt.
            system_prompt (Optional[str]): System instructions.
            model (Optional[str]): Override default model.
            
        Returns:
            str: Generated text response.
        """
        active_model = model or self.default_model
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        try:
            # Support local Ollama
            api_base_kwargs = {}
            if active_model.startswith("ollama") or active_model.startswith("ollama_chat"):
                import os
                api_base = os.getenv("OPENAI_API_BASE", "http://localhost:11434").strip().rstrip("/")
                if api_base.endswith("/v1"):
                    api_base = api_base[:-3]
                api_base_kwargs["api_base"] = api_base

            response = completion(
                model=active_model,
                messages=messages,
                **api_base_kwargs,
                **kwargs
            )
        except Exception as e:
            logger.error(f"LLM Call failed: {e}")
            raise e
            
        latency = time.time() - start_time
        
        # Extract metrics
        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        
        # Calculate cost
        try:
            cost = litellm.cost_calculator.cost_per_token(
                model=active_model, 
                prompt_tokens=tokens_in, 
                completion_tokens=tokens_out
            )
            cost_usd = cost[0] + cost[1]
        except Exception:
            cost_usd = 0.0

        # Log calls
        logger.debug(
            f"LLM Call [{active_model}] | Latency: {latency:.2f}s | "
            f"Tokens: {tokens_in} in, {tokens_out} out | Cost: ${cost_usd:.6f}"
        )
        
        return response.choices[0].message.content or ""
