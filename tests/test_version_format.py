# tests/test_version_format.py
import re
from pathlib import Path

def test_version_txt_uses_semver():
    """
    version.txt should contain semantic version: major.minor.changelist
    e.g., 1.0.245
    """
    path = Path("version.txt")
    assert path.exists(), "version.txt must exist"

    content = path.read_text().strip()
    # Match something like 1.0.245
    assert re.fullmatch(r"\d+\.\d+\.\d+", content), (
        f"version.txt should be in major.minor.changelist format, got: {content!r}"
    )
