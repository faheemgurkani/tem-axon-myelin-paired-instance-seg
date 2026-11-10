"""Score a prediction folder against GT with the official-metric replica and diagnostics.
Usage: python scripts/evaluate.py --pred <dir> --gt <dir> [--config configs/default.yaml]"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    raise NotImplementedError(args)


if __name__ == "__main__":
    main()
