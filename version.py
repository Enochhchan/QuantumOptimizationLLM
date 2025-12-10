#!/usr/bin/env python3
"""
Version management: automatically increments build number.
Reads version.txt, increments build number, writes back.
"""
import re
from pathlib import Path

VERSION_FILE = Path("version.txt")

def increment_build_number():
    """Read version.txt, increment build number, write back, return new version"""
    if not VERSION_FILE.exists():
        # Initialize if doesn't exist
        new_version = "1.0.1"
        VERSION_FILE.write_text(f"{new_version}\n")
        return new_version
    
    # Read current version
    current = VERSION_FILE.read_text().strip()
    
    # Parse major.minor.buildNumber
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", current)
    if not match:
        raise ValueError(f"Invalid version format: {current}. Expected major.minor.buildNumber")
    
    major, minor, build = map(int, match.groups())
    new_build = build + 1
    new_version = f"{major}.{minor}.{new_build}"
    
    # Write back
    VERSION_FILE.write_text(f"{new_version}\n")
    return new_version

if __name__ == "__main__":
    new_version = increment_build_number()
    print(new_version)
