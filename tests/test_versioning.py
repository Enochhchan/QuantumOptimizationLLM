import re
from pathlib import Path

VERSION_FILE = Path("version.txt")

def test_version_file_exists():
    """
    The version.txt file should exist in the repo root.
    This is important because the CI pipeline reads it
    to name the versioned artifact.
    """
    assert VERSION_FILE.exists(), "version.txt file is missing"


def test_version_format():
    """
    The version should follow the Major.Minor.Changelist format,
    e.g., 1.0.1, 2.3.5, etc.
    """
    version = VERSION_FILE.read_text().strip()
    pattern = r"^\d+\.\d+\.\d+$"
    assert re.match(pattern, version) is not None, (
        f"Version '{version}' does not match Major.Minor.Changelist format"
    )
