"""Loads .env, on every platform, without a dependency.

The README used to say `set -a; source .env; set +a`, which is bash. On Windows
that is not a command, so there was no documented way to get the gateway URL
into the process and every run fell back to --mock. Importing this module reads
the file directly, which works the same in cmd, PowerShell and any shell.

Real environment variables always win, so CI and `VISION_URL=... python run.py`
keep behaving the way you would expect.
"""
import os

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def load_env(path=ENV_PATH):
    """Parse KEY=VALUE lines. Ignores blanks, comments and `export` prefixes,
    and strips one layer of matching quotes."""
    if not os.path.exists(path):
        return False

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.removeprefix("export ").partition("=")
            key = key.strip()
            value = value.split(" #")[0].strip()
            if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)
    return True


def require(name):
    """Fetch a setting, or explain what to do about it.

    A bare KeyError at import time reads like a bug in the tool rather than a
    missing config file, which is the wrong first thought to give someone.
    """
    value = os.environ.get(name)
    if value:
        return value
    raise SystemExit(
        f"{name} is not set.\n"
        f"  1. copy .env.example to .env\n"
        f"  2. put your gateway URL and model in it\n"
        f"  3. re-run — .env is read automatically, no shell setup needed\n"
        f"Looked for: {ENV_PATH}")


load_env()
