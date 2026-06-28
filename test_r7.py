from __future__ import annotations

import sys

from validate_r7 import main


if __name__ == "__main__":
    if "--split" not in sys.argv:
        sys.argv.extend(["--split", "test"])
    main()
