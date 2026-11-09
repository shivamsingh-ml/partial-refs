"""Run the review app on your own machine from an unpacked review bundle (listens on 127.0.0.1 only).

    python -m partialref.review.local path/to/review-bundle-<run>
Then open http://localhost:8501 . Logs are written to <bundle>/logs/<parse run>/review_log.jsonl (IDs and codes only).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    bundle = Path(argv[0]).resolve()
    info = json.loads((bundle / "bundle_info.json").read_text())
    env = {**os.environ, "REVIEW_BUNDLE": str(bundle), "REVIEW_LOCAL": "1",
           "REVIEW_PARSE_RUN": str(bundle / "parse" / info["parse_run"]), "REVIEW_MANIFEST": str(bundle),
           "REVIEW_LOG_ROOT": str(bundle / "logs"), "REVIEW_LANG": "en"}
    app = Path(__file__).with_name("app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.address", "127.0.0.1",
                            "--server.port", "8501", "--server.headless", "true",
                            "--browser.gatherUsageStats", "false"], env=env)


if __name__ == "__main__":
    sys.exit(main())
