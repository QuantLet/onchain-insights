"""Convert section 12 forecast pickles to compact, lossless NPZ artifacts.

Run from the repository root with the Python environment that can read the
source pickles:

    python "12. Evaluation of Probabilistic Forecasts/tables/compact_forecasts.py"

The original pickles are left in place. The table loader automatically prefers
the generated ``*.compact.npz`` files when they are available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from forecast_artifacts import (
    load_legacy_pickle,
    save_compact_archive,
    shared_candidate_keys,
    values_equal,
)


SECTION_DIR = Path(__file__).resolve().parents[1]


def discover_inputs(root):
    return sorted(
        path for path in root.rglob("*preds_test_set.pkl")
        if ".ipynb_checkpoints" not in path.parts
    )


def compact_path(source):
    return source.with_name(source.stem + ".compact.npz")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=SECTION_DIR,
        help="section 12 directory (default: this script's parent section)",
    )
    parser.add_argument(
        "--inputs", nargs="+", type=Path,
        help="specific source pickle files; by default discover them under --root",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="replace compact archives that already exist",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    sources = [p.resolve() for p in args.inputs] if args.inputs else discover_inputs(root)
    if not sources:
        parser.error("No *preds_test_set.pkl files found under {}".format(root))
    missing = [path for path in sources if not path.is_file()]
    if missing:
        parser.error("Input pickle does not exist: {}".format(missing[0]))

    outputs = [compact_path(path) for path in sources]
    shared_output = root / "compact_test_data.npz"
    if args.inputs and shared_output.exists():
        existing_compacts = set(root.rglob("*.compact.npz"))
        if existing_compacts.difference(outputs):
            parser.error(
                "A shared archive is already in use by other compact files. "
                "Rebuild the full set of source pickles together."
            )
    existing = [path for path in outputs + [shared_output] if path.exists()]
    if existing and not args.force:
        parser.error(
            "Compact output already exists: {} (use --force to replace it)".format(existing[0])
        )

    common = None
    for source in sources:
        try:
            artifact = load_legacy_pickle(source)
        except Exception as exc:
            raise RuntimeError(
                "Could not read {}. Run this converter in an environment compatible "
                "with the pickle's NumPy/Pandas versions. Original files were not changed. "
                "Loader error: {}".format(source, exc)
            ) from exc
        if not isinstance(artifact, dict):
            raise TypeError("Expected a dictionary in {}".format(source))

        if common is None:
            common = {
                key: artifact[key]
                for key in shared_candidate_keys()
                if key in artifact
            }
        else:
            for key in list(common):
                if key not in artifact or not values_equal(common[key], artifact[key]):
                    del common[key]
        print("Compared shared fields in {}".format(source.relative_to(root)))
        del artifact

    common = common or {}
    if common:
        save_compact_archive(shared_output, common)
        print("Saved shared fields {} -> {} ({:.1f} MiB)".format(
            ", ".join(common), shared_output,
            shared_output.stat().st_size / (1024.0 ** 2),
        ))
    elif shared_output.exists():
        shared_output.unlink()

    for source, output in zip(sources, outputs):
        artifact = load_legacy_pickle(source)
        model_data = {key: value for key, value in artifact.items() if key not in common}
        save_compact_archive(output, model_data)
        print("Saved {} ({:.1f} MiB)".format(
            output.relative_to(root), output.stat().st_size / (1024.0 ** 2),
        ))
        del artifact, model_data

    print("Converted {} forecast files. Source pickles were kept.".format(len(sources)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
