#!/usr/bin/env python3
"""
Reconstruct this ultralytics2 repo's requirements.txt from its pyproject.toml.

The verdant yolo11 trainer docker image bakes the ultralytics repo's dependencies in at build time 
which expects a requirements.txt. The ultralytics repo migrated from requirements.txt
to pyproject.toml, so this script regenerates requirements.txt (one requirement per line) in place
from the project's dependency metadata. It operates on the directory the script lives in.

Supports both PEP 621 (`[project].dependencies` / `[project.optional-dependencies]`) and Poetry
(`[tool.poetry.dependencies]`). Prefers PEP 621 when both are present.

Usage:
    verdant_reconstruct_requirements_txt.py [--extra dev --extra export]
"""

import argparse
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        sys.exit(
            "Error: need Python 3.11+ (for tomllib) or the 'tomli' package to read pyproject.toml."
        )


def _poetry_version_to_pep508(version: str) -> str:
    """Best-effort conversion of a Poetry version constraint to a PEP 508 specifier.

    Handles caret (^) and tilde (~) ranges; passes through anything already PEP 440-shaped
    (>=, <, ==, etc.) and the catch-all "*".
    """
    version = version.strip()
    if version in ("", "*"):
        return ""

    if version[0] in "^~":
        op, base = version[0], version[1:]
        parts = [int(p) for p in base.split(".")]
        lower = ".".join(str(p) for p in parts)
        if op == "^":
            # ^1.2.3 -> >=1.2.3,<2.0.0 ; ^0.2.3 -> >=0.2.3,<0.3.0 ; ^0.0.3 -> >=0.0.3,<0.0.4
            upper = list(parts)
            for i, p in enumerate(parts):
                if p != 0:
                    upper = parts[: i + 1]
                    upper[i] += 1
                    upper += [0] * (len(parts) - len(upper))
                    break
            else:
                upper = [0] * (len(parts) - 1) + [parts[-1] + 1]
        else:  # tilde: ~1.2.3 -> >=1.2.3,<1.3.0 ; ~1.2 -> >=1.2,<1.3 ; ~1 -> >=1,<2
            upper = list(parts)
            idx = min(1, len(parts) - 1)
            upper[idx] += 1
            upper = upper[: idx + 1] + [0] * (len(parts) - idx - 1)
        upper_str = ".".join(str(p) for p in upper)
        return f">={lower},<{upper_str}"

    # Already a PEP 440-style specifier (>=, <=, ==, !=, ~=, <, >) or a bare version.
    if version[0].isdigit():
        return f"=={version}"
    return version


def _poetry_requirement(name: str, spec) -> str | None:
    """Convert one Poetry dependency entry to a PEP 508 requirement string."""
    if name.lower() == "python":
        return None  # Interpreter constraint, not a pip requirement.

    if isinstance(spec, str):
        return f"{name}{_poetry_version_to_pep508(spec)}"

    if isinstance(spec, dict):
        # Skip path/git/url deps — they can't be reproduced as a plain pinned requirement here.
        if any(k in spec for k in ("path", "git", "url")):
            print(f"Warning: skipping non-PyPI dependency '{name}'.", file=sys.stderr)
            return None
        extras = spec.get("extras")
        extras_str = f"[{','.join(extras)}]" if extras else ""
        version = _poetry_version_to_pep508(spec.get("version", "*"))
        marker = spec.get("markers")
        marker_str = f" ; {marker}" if marker else ""
        return f"{name}{extras_str}{version}{marker_str}"

    if isinstance(spec, list):
        print(
            f"Warning: skipping '{name}' (multiple-constraint dependency not supported).",
            file=sys.stderr,
        )
        return None

    return None


def extract_requirements(pyproject: dict, extras: list[str]) -> list[str]:
    project = pyproject.get("project")
    if project is not None:
        # PEP 621
        if "dependencies" in project.get("dynamic", []):
            print(
                "Warning: [project].dependencies is declared dynamic; the static list may be "
                "incomplete.",
                file=sys.stderr,
            )
        requirements = list(project.get("dependencies", []))
        optional = project.get("optional-dependencies", {})
        for extra in extras:
            if extra not in optional:
                sys.exit(f"Error: no optional-dependency group '{extra}' in [project].")
            requirements += optional[extra]
        return requirements

    poetry = pyproject.get("tool", {}).get("poetry")
    if poetry is not None:
        requirements = []
        for name, spec in poetry.get("dependencies", {}).items():
            req = _poetry_requirement(name, spec)
            if req:
                requirements.append(req)
        groups = poetry.get("group", {})
        for extra in extras:
            if extra not in groups:
                sys.exit(f"Error: no Poetry dependency group '{extra}'.")
            for name, spec in groups[extra].get("dependencies", {}).items():
                req = _poetry_requirement(name, spec)
                if req:
                    requirements.append(req)
        return requirements

    sys.exit(
        "Error: found neither [project].dependencies (PEP 621) nor [tool.poetry.dependencies]."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--extra",
        dest="extras",
        action="append",
        default=[],
        metavar="NAME",
        help="Include an optional-dependency group / Poetry group (repeatable).",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent

    pyproject_path = repo / "pyproject.toml"
    if not pyproject_path.is_file():
        sys.exit(f"Error: '{pyproject_path}' does not exist.")

    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    requirements = extract_requirements(pyproject, args.extras)

    requirements_path = repo / "requirements.txt"
    requirements_path.write_text("".join(f"{req}\n" for req in requirements))
    print(
        f"Wrote {len(requirements)} requirements to {requirements_path}", file=sys.stderr
    )


if __name__ == "__main__":
    main()
