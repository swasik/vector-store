#!/usr/bin/env python3
"""Print an example VSS sizing report using default input parameters.

Usage::

    python3 example_sizing.py
"""

from vss_sizing import SizingInput, compute_sizing


def main() -> None:
    inp = SizingInput()
    result = compute_sizing(inp)

    print("=" * 60)
    print("  ScyllaDB VSS Sizing — Default Parameters")
    print("=" * 60)
    print()
    print(f"  Vectors        : {inp.num_vectors:>14,}")
    print(f"  Dimensions     : {inp.dimensions:>14,}")
    print(f"  Target QPS     : {inp.target_qps:>14,}")
    print(f"  Recall         : {inp.recall:>13}%")
    print(f"  K              : {inp.k:>14,}")
    print(f"  Quantization   : {inp.quantization.name:>14}")
    print(f"  Metadata/vec   : {inp.metadata_bytes_per_vector:>12} B")
    print(f"  Filter columns : {inp.filtering_columns:>14}")
    print()
    print(result.summary)
    print()


if __name__ == "__main__":
    main()
