from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from .config import load_config
from .models import Example
from .v2_runner import _make_v2_backend


def main() -> None:
    config = load_config(sys.argv[1])
    with redirect_stdout(sys.stderr):
        backend = _make_v2_backend(config, sys.argv[2], Path(sys.argv[3]))
    try:
        for line in sys.stdin:
            request = json.loads(line)
            try:
                example = Example.from_dict(request["example"])
                with redirect_stdout(sys.stderr):
                    if request.get("op") == "features":
                        features = backend.features(example, int(request["layer"]))  # type: ignore[attr-defined]
                        result = {"features": features}
                    else:
                        prediction = backend.predict(request["experiment_id"], example)
                        result = {"prediction": prediction.to_dict()}
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
