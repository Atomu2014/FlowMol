import argparse
import pickle
from collections import OrderedDict
from pathlib import Path

from rdkit import Chem
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert all SDF files in a directory to process_geom.py split pickle format."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Directory containing SDF files (searched recursively).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sdf_raw.pkl"),
        help="Output pickle path.",
    )
    parser.add_argument(
        "--remove_hs",
        action="store_true",
        help="Remove hydrogens while reading SDF (default keeps H atoms).",
    )
    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Enable RDKit sanitization while reading (default off).",
    )
    return parser.parse_args()


def iter_sdf_molecules(sdf_path: Path, remove_hs: bool, sanitize: bool):
    supplier = Chem.SDMolSupplier(
        str(sdf_path), removeHs=remove_hs, sanitize=sanitize
    )
    for mol in supplier:
        if mol is None:
            continue
        yield mol


def canonical_smiles(mol: Chem.Mol):
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def main():
    args = parse_args()
    data_dir = args.data_dir
    output_file = args.output

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    sdf_files = sorted(data_dir.rglob("*.sdf"))
    if not sdf_files:
        raise FileNotFoundError(f"No .sdf files found under: {data_dir}")

    grouped = OrderedDict()
    total_read = 0
    total_skipped = 0

    for sdf_file in tqdm.tqdm(sdf_files, desc="Reading SDF files"):
        for mol in iter_sdf_molecules(
            sdf_file, remove_hs=args.remove_hs, sanitize=args.sanitize
        ):
            total_read += 1
            smiles = canonical_smiles(mol)
            if not smiles:
                total_skipped += 1
                continue
            grouped.setdefault(smiles, []).append(mol)

    output_data = [(smiles, conformers) for smiles, conformers in grouped.items()]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(output_data, f)

    print(f"SDF files scanned: {len(sdf_files)}")
    print(f"Molecules read: {total_read}")
    print(f"Molecules skipped (no valid smiles): {total_skipped}")
    print(f"Unique smiles groups: {len(output_data)}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
