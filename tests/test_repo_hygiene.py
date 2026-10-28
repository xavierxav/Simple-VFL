from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".cfg",
    ".ini",
    ".ipynb",
}
IGNORED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "outputs",
    "data",
}
FORBIDDEN_PATTERNS = (
    "C:" + "\\Users\\",
    "/home/" + "users/",
    "XD" + "278777",
    "xdes" + "prez",
)


def test_no_personal_paths_or_usernames() -> None:
    """Keep personal machine paths and usernames out of the public repository."""
    offenders: list[str] = []

    for path in REPOSITORY_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if any(part in IGNORED_PARTS for part in path.parts):
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for pattern in FORBIDDEN_PATTERNS:
            if pattern in text:
                offenders.append(f"{path.relative_to(REPOSITORY_ROOT)} contains {pattern!r}")

    assert not offenders, "\n".join(offenders)
