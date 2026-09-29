"""Start Supplychainer on one port: the dashboard and the API together.

Builds the dashboard first when it is missing or older than its sources (this
needs Node.js), then serves both at http://127.0.0.1:8000. Run it from the
repository root with the virtual environment's Python:

    python run.py [--port 8000] [--host 127.0.0.1]

For development with hot reload, run the backend and `npm run dev` separately
(see the README).
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

FRONTEND = Path(__file__).resolve().parent / "frontend"


def needs_build(frontend=FRONTEND):
    """Whether dist/ is missing or older than anything it is built from."""
    built = frontend / "dist" / "index.html"
    if not built.exists():
        return True
    sources = [frontend / "index.html", frontend / "package-lock.json", frontend / "vite.config.js",
               *(frontend / "src").rglob("*")]
    return any(p.is_file() and p.stat().st_mtime > built.stat().st_mtime for p in sources)


def build(frontend=FRONTEND):
    npm = shutil.which("npm")
    if npm is None:
        sys.exit("Building the dashboard needs Node.js 18 or newer (https://nodejs.org).")
    if not (frontend / "node_modules").exists():
        subprocess.run([npm, "ci"], cwd=frontend, check=True)
    subprocess.run([npm, "run", "build"], cwd=frontend, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept connections from other machines")
    args = parser.parse_args()

    if needs_build():
        print("Building the dashboard...", flush=True)
        build()
    if os.getenv("SUPPLYCHAINER_API_KEY"):
        # The key stays on the server: only the development proxy adds it to requests.
        print("Note: SUPPLYCHAINER_API_KEY is set, so the dashboard served here can't plan routes. "
              "Unset it, or use the development setup in the README.")

    import uvicorn
    print(f"Supplychainer: http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port} "
          "(the news model warms up for a few seconds after start-up)")
    uvicorn.run("backend.main:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
