"""Qwen attention analysis example.

Usage:
    python run.py                        # uses config.yaml in same directory
    python run.py --config config.yaml   # explicit path
    python run.py --prompt "Your prompt here"
"""

import argparse
from pathlib import Path

import evalrx

DEFAULT_PROMPT = "The Eiffel Tower is located in the city of"
CONFIG = Path(__file__).parent / "config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    config = evalrx.load_config(args.config)
    result = evalrx.run(config, args.prompt)

    print(result.summary())


if __name__ == "__main__":
    main()