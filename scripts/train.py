"""Train a model from a config. Usage: python scripts/train.py --config configs/<exp>.yaml"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise NotImplementedError(args.config)


if __name__ == "__main__":
    main()
