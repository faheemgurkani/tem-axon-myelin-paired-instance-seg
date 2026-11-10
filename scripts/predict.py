"""Predict uint16 paired-instance masks for a folder of TEM images.
Usage: python scripts/predict.py --config <cfg> --input <dir> --output <dir>"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise NotImplementedError(args)


if __name__ == "__main__":
    main()
