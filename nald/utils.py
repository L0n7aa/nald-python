import os 
import subprocess
import hashlib


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state():
    """(commit, dirty); dirty=True if no commit for the modifications."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=here,
                                             stderr=subprocess.DEVNULL, text=True).strip())
        return commit, dirty
    except Exception:
        return None, None