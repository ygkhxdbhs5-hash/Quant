"""Colab: refresh data for a random 500-ticker test sample.

Paste into ONE Colab cell:

    %cd /content
    !rm -rf Quant
    !git clone --branch cursor/universe-sample-500-cb1c https://github.com/ygkhxdbhs5-hash/Quant.git
    %cd Quant
    !pip -q install -r requirements.txt

    import os
    from google.colab import userdata
    os.environ["MASSIVE_API_KEY"] = userdata.get("MASSIVE_API_KEY")
    os.environ["PYTHONUNBUFFERED"] = "1"

    %run notebooks/colab_run_sample_500.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _api_key() -> str:
    key = os.environ.get("MASSIVE_API_KEY") or ""
    if not key:
        try:
            from google.colab import userdata  # type: ignore

            key = userdata.get("MASSIVE_API_KEY") or ""
        except Exception:
            key = ""
    if not key:
        import getpass

        key = getpass.getpass("MASSIVE_API_KEY: ").strip()
    if not key:
        raise RuntimeError("MASSIVE_API_KEY required")
    return key


def run(module: str) -> None:
    print("\n" + "=" * 72)
    print("RUN", module)
    print("=" * 72)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        [sys.executable, "-u", "-m", module, "--config", "config/config.yaml"],
        cwd=str(Path.cwd()),
        env=env,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    os.environ["MASSIVE_API_KEY"] = _api_key()
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Confirm sample setting
    print(subprocess.getoutput("grep -n universe_sample_size config/config.yaml"))

    cache = Path("cache")
    cache.mkdir(exist_ok=True)
    for p in cache.rglob("*"):
        if p.is_file() and p.name != ".gitkeep":
            p.unlink(missing_ok=True)
    (cache / "massive_cache").mkdir(parents=True, exist_ok=True)
    print("cache cleared")

    run("downloader.download_universe")
    run("downloader.download_prices")
    # Fundamentals optional for CMVS v3 — enable with: run("downloader.download_fundamentals")
    # run("downloader.download_fundamentals")

    import pickle

    uni = pickle.load(open("data/metadata/universe.pkl", "rb"))
    print(
        f"DONE tickers={len(uni['tickers'])} all_tickers={len(uni['all_tickers'])} "
        f"profiles={len(uni['profile_meta'])}"
    )


if __name__ == "__main__":
    main()
