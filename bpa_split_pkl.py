import argparse
import pickle
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split BPA pickle into train/val/test pickle files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/bpa_raw/bpa.pkl"),
        help="Input pickle file in process_geom format.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/bpa_raw"),
        help="Directory to save split pickle files.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split.",
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Disable shuffling before split.",
    )
    return parser.parse_args()


def save_pickle(path: Path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(
            f"train/val/test ratios must sum to 1.0, got {total_ratio:.6f}"
        )

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise TypeError(f"Expected list in input pickle, got {type(data)}")
    if len(data) == 0:
        raise ValueError("Input pickle is empty.")

    items = list(data)
    if not args.no_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(items)

    n_total = len(items)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    n_test = n_total - n_train - n_val

    train_data = items[:n_train]
    val_data = items[n_train : n_train + n_val]
    test_data = items[n_train + n_val :]

    if len(test_data) != n_test:
        raise RuntimeError("Split size mismatch, please check split logic.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_file = args.output_dir / "train_data.pkl"
    val_file = args.output_dir / "val_data.pkl"
    test_file = args.output_dir / "test_data.pkl"

    save_pickle(train_file, train_data)
    save_pickle(val_file, val_data)
    save_pickle(test_file, test_data)

    print(f"Input: {args.input}")
    print(f"Output dir: {args.output_dir}")
    print(f"Total groups: {n_total}")
    print(f"Train groups: {len(train_data)} -> {train_file}")
    print(f"Val groups: {len(val_data)} -> {val_file}")
    print(f"Test groups: {len(test_data)} -> {test_file}")
    print(
        "Ratios used: "
        f"train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}"
    )
    print(f"Shuffle: {not args.no_shuffle}, seed={args.seed}")


if __name__ == "__main__":
    main()
