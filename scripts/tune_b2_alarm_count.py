"""Compare all B2 variants on validation; print selections for protocol.yaml."""
import sys

from make_baseline_table import main

if __name__ == "__main__":
    sys.argv.append("--tune-b2")
    main()
