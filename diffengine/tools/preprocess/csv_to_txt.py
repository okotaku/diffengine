import argparse

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a checkpoint to be published")
    parser.add_argument("input", help="Path to csv")
    parser.add_argument("out", help="Path to output txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    img_df = pd.read_csv(args.input)
    img_df.to_csv(args.out, header=False, index=False, sep=" ")

if __name__ == "__main__":
    main()
