import argparse
import asyncio

from app.agent.manus import Manus
from app.config import config
from app.llm import LLM
from app.logger import logger


def build_runtime_llm(config_name: str, model_name: str | None) -> LLM:
    """Build an LLM instance from config and optional user-selected model."""
    selected_config = config.llm.get(config_name, config.llm["default"])

    if model_name:
        selected_config = selected_config.model_copy(update={"model": model_name})

    runtime_key = f"{config_name}:{selected_config.model}"
    runtime_llm_config = {
        "default": selected_config,
        runtime_key: selected_config,
    }
    return LLM(config_name=runtime_key, llm_config=runtime_llm_config)


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Manus agent with a prompt")
    llm_profile_names = list(config.llm.keys())
    parser.add_argument(
        "--prompt", type=str, required=False, help="Input prompt for the agent"
    )
    parser.add_argument(
        "--llm-config",
        type=str,
        default="default",
        choices=llm_profile_names,
        help="LLM configuration profile name from config.toml (default: default)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Override model name (for example: gemma3:1b)",
    )
    args = parser.parse_args()

    # Create and initialize Manus agent
    llm = build_runtime_llm(config_name=args.llm_config, model_name=args.model)
    agent = await Manus.create(llm=llm)
    try:
        # Use command line prompt if provided, otherwise ask for input
        prompt = args.prompt if args.prompt else input("Enter your prompt: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request...")
        await agent.run(prompt)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
