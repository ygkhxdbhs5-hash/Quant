"""Colab: force the full-universe downloader, then run it.

Paste into ONE Colab cell:

    %cd /content
    !rm -rf Quant
    !git clone --branch cursor/universe-paginate-debug-cb1c https://github.com/ygkhxdbhs5-hash/Quant.git
    %cd Quant
    !pip -q install -r requirements.txt
    %run notebooks/colab_force_full_universe.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

CODE_MARK = "FULL_UNIVERSE_FORCE_V3"
REPO = Path.cwd()
TARGET = REPO / "downloader" / "download_universe.py"


FIXED_MAIN = textwrap.dedent(
    f'''
def main(argv: list[str] | None = None) -> int:
    """CODE_MARK={CODE_MARK} — tickers is ALWAYS all_tickers (no 30-name symbols.txt)."""
    print("=" * 72)
    print(f"CODE_MARK={CODE_MARK}")
    print("If you do not see this mark, Colab is running a different file.")
    print("=" * 72)

    parser = argparse.ArgumentParser(description="Download NASDAQ universe + profiles (Massive)")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    client = make_client(config)
    paths = config.get("paths", {{}})
    metadata_dir = Path(paths.get("metadata", "data/metadata"))

    all_tickers, delisted_meta = download_complete_nasdaq_universe(client)

    # NEVER use universe_limit or symbols.txt
    print(f">> Ignoring universe_limit={{config.get('universe_limit')!r}}")
    print(f">> Ignoring symbols.txt at {{metadata_dir / 'symbols.txt'}}")

    tickers = list(all_tickers)
    print(f">> ABOUT_TO_FETCH_PROFILES n={{len(tickers)}} (must be ~11088, not 30)")
    if len(tickers) < 1000:
        raise SystemExit(
            f"FATAL: only {{len(tickers)}} tickers after Massive pull — aborting"
        )

    profile_meta = fetch_profile_meta(client, tickers)
    print(f">> PROFILES_DONE n={{len(profile_meta)}}")
    save_universe(metadata_dir, all_tickers, delisted_meta, profile_meta, tickers)
    print(
        f">> SAVED tickers={{len(tickers)}} all_tickers={{len(all_tickers)}} "
        f"profiles={{len(profile_meta)}}"
    )
    return 0
'''
)


def _patch_download_universe() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"Missing {TARGET}. Did you %cd Quant?")

    text = TARGET.read_text(encoding="utf-8")
    start = text.find("\ndef main(")
    if start < 0:
        raise RuntimeError("Could not find def main( in download_universe.py")

    # Keep everything before main; replace main + __main__ guard
    head = text[:start]
    new_text = head + "\n" + FIXED_MAIN + "\n\nif __name__ == '__main__':\n    raise SystemExit(main())\n"
    TARGET.write_text(new_text, encoding="utf-8")
    print(f"Patched {TARGET}")
    print(f"CODE_MARK present in file: {CODE_MARK in TARGET.read_text(encoding='utf-8')}")


def _require_api_key() -> None:
    key = os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY") or ""
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
    os.environ["MASSIVE_API_KEY"] = key
    os.environ["PYTHONUNBUFFERED"] = "1"
    print("API key OK:", key[:4] + "…" + key[-4:])


def main() -> None:
    print("cwd:", REPO)
    print("git HEAD:", subprocess.getoutput("git rev-parse --short HEAD"))
    print("git branch:", subprocess.getoutput("git rev-parse --abbrev-ref HEAD"))

    _require_api_key()
    _patch_download_universe()

    # Prove the mark is importable from disk
    marker_check = subprocess.getoutput(f"grep -n CODE_MARK {TARGET}")
    print("grep CODE_MARK:\n", marker_check)
    if CODE_MARK not in marker_check:
        raise RuntimeError("Patch failed — CODE_MARK not found on disk")

    cache = REPO / "cache"
    if cache.exists():
        for p in cache.rglob("*"):
            if p.is_file() and p.name != ".gitkeep":
                p.unlink(missing_ok=True)
    (cache / "massive_cache").mkdir(parents=True, exist_ok=True)
    print("cache cleared")

    print("\n>>> Running download_universe (watch for CODE_MARK and ABOUT_TO_FETCH_PROFILES) <<<\n")
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "downloader.download_universe", "--config", "config/config.yaml"],
        cwd=str(REPO),
        env=dict(os.environ),
    )
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
