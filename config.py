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


class AuthError(RuntimeError):
    """The gateway rejected the credential.

    Its own class because it is not a per-image failure. It will be true for
    every image in the batch, so the caller should stop rather than repeat the
    same rejected call twenty more times. It lives here rather than in llm.py so
    that run.py can catch it without importing llm — llm needs a gateway URL at
    import time, and --mock has to keep working without one.
    """


AUTH_HELP = """The gateway rejected the credential (HTTP {code}).

That is a config problem, not an image problem — it would fail the same way for
every photo, so the run stopped here.

  1. Check VISION_URL in .env. On this gateway the token is part of the URL
     path, so a truncated or stale path IS an invalid credential.
  2. If you copied .env.example before this repo was cleaned up, that token had
     been public in git history and needs rotating — it may already be dead.
  3. Run `python probe.py`. It reports which request shapes the gateway accepts.
     All four failing the same way confirms the credential; some passing means
     GATEWAY_STYLE is wrong instead.
  4. If your gateway authenticates by header rather than by path, put the token
     in VISION_KEY and use the plain base URL."""


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
