#!/usr/bin/env python3
"""Install UENR's missing system dependencies globally.

The script is intentionally stdlib-only and idempotent: tools that already
exist are skipped unless --force/--global is used to reinstall them.  It never
handles GitHub credentials; use scripts/github_auth.sh setup afterwards.

Examples:
    python3 install.py --check          # report only, no changes
    python3 install.py                  # install what is missing (asks first)
    python3 install.py --yes            # non-interactive
    python3 install.py --global --yes   # force system-wide install
    python3 install.py --force --yes    # reinstall/upgrade even if present

Supported: Debian/Ubuntu (apt), Fedora/RHEL (dnf/yum), Arch (pacman),
openSUSE (zypper), Alpine (apk), macOS (Homebrew).  Windows is not supported;
use WSL or install Git, GCC, Make and GitHub CLI manually.

Exit codes: 0 success/nothing to do, 1 missing after install, 2 usage,
3 platform/package-manager unsupported, 4 permission problem.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

TOOLS = [
    # (command, apt, dnf, pacman, zypper, apk, brew)
    ("git", "git", "git", "git", "git", "git", "git"),
    ("make", "make", "make", "make", "make", "make", "make"),
    ("gcc", "gcc", "gcc", "gcc", "gcc", "gcc", "gcc"),
    ("curl", "curl", "curl", "curl", "curl", "curl", "curl"),
    ("gh", "gh", "gh", "github-cli", "gh", "github-cli", "gh"),
]

PACKAGE_INDEX = {"apt": 1, "dnf": 2, "pacman": 3, "zypper": 4, "apk": 5, "brew": 6}
EXTRA_PACKAGES = {
    "apt": ["build-essential", "ca-certificates"],
    "dnf": ["gcc", "make", "ca-certificates"],
    "pacman": ["base-devel", "ca-certificates"],
    "zypper": ["gcc", "make", "ca-certificates"],
    "apk": ["build-base", "ca-certificates"],
    "brew": [],
}


class InstallError(RuntimeError):
    pass


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise InstallError(f"command not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise InstallError(f"command failed ({exc.returncode}): {' '.join(cmd)}") from exc


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def have_c_compiler() -> bool:
    return have("gcc") or have("cc") or have("clang")


def detect_manager() -> tuple[str | None, list[str]]:
    if sys.platform == "darwin":
        return ("brew", ["brew"]) if have("brew") else (None, [])
    if sys.platform.startswith("win"):
        return None, []
    for name, cmd in (
        ("apt", ["apt-get"]),
        ("dnf", ["dnf"]),
        ("yum", ["yum"]),
        ("pacman", ["pacman"]),
        ("zypper", ["zypper"]),
        ("apk", ["apk"]),
    ):
        if have(cmd[0]):
            return name, cmd
    return None, []


def is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def sudo_prefix(require_global: bool) -> list[str]:
    if is_root():
        return []
    if have("sudo"):
        return ["sudo"]
    if require_global:
        raise InstallError(
            "global installation was requested but sudo/root is unavailable; "
            "run as root or install without --global"
        )
    return []


def missing_tools() -> list[str]:
    out: list[str] = []
    for row in TOOLS:
        cmd = row[0]
        if cmd == "gcc":
            if not have_c_compiler():
                out.append("gcc")
        elif not have(cmd):
            out.append(cmd)
    return out


def package_names(manager: str, missing: list[str]) -> list[str]:
    idx = PACKAGE_INDEX[manager]
    names = list(EXTRA_PACKAGES.get(manager, []))
    for row in TOOLS:
        if row[0] in missing:
            pkg = row[idx]
            if pkg not in names:
                names.append(pkg)
    return names


def manager_base(manager: str, pm: list[str], sudo: list[str]) -> list[str]:
    if manager == "apt":
        return sudo + pm + ["-y"]
    if manager in ("dnf", "yum"):
        return sudo + pm + ["install", "-y"]
    if manager == "pacman":
        return sudo + pm + ["-S", "--needed", "--noconfirm"]
    if manager == "zypper":
        return sudo + pm + ["install", "-y"]
    if manager == "apk":
        return sudo + pm + ["add"]
    if manager == "brew":
        return pm + ["install"]
    raise InstallError(f"no package manager mapping for {manager}")


def install_packages(manager: str, pm: list[str], packages: list[str],
                     *, require_global: bool, dry_run: bool) -> None:
    # Homebrew must not run under sudo.
    sudo = [] if manager == "brew" else sudo_prefix(require_global)
    if manager == "apt":
        run(sudo + pm + ["update"], dry_run=dry_run)
    run(manager_base(manager, pm, sudo) + packages, dry_run=dry_run)


def _download(url: str, dest: Path) -> None:
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)


def install_gh_binary(*, require_global: bool, dry_run: bool) -> None:
    """Fallback when the distro has no gh package."""
    if sys.platform != "linux":
        raise InstallError("automatic GitHub CLI fallback is Linux-only")
    machine = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64",
            "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    if not arch:
        raise InstallError(f"unsupported architecture for gh: {machine}")
    import json
    api = "https://api.github.com/repos/cli/cli/releases/latest"
    try:
        with urllib.request.urlopen(api, timeout=30) as resp:
            tag = json.load(resp)["tag_name"]
    except Exception as exc:  # network/API failure
        raise InstallError(f"cannot resolve the latest gh release: {exc}") from exc
    ver = tag.lstrip("v")
    url = (f"https://github.com/cli/cli/releases/download/{tag}/"
           f"gh_{ver}_linux_{arch}.tar.gz")
    with tempfile.TemporaryDirectory(prefix="uenr-gh-") as td:
        archive = Path(td) / "gh.tar.gz"
        if dry_run:
            print(f"$ download {url}")
            return
        _download(url, archive)
        with tarfile.open(archive, "r:gz") as tf:
            try:
                tf.extractall(td, filter="data")  # Python 3.12+
            except TypeError:  # pragma: no cover - older Python
                for member in tf.getmembers():
                    target = (Path(td) / member.name).resolve()
                    if not str(target).startswith(str(Path(td).resolve())):
                        raise InstallError("unsafe path in gh archive")
                tf.extractall(td)
        found = list(Path(td).glob("**/bin/gh"))
        if not found:
            raise InstallError("gh binary not found in downloaded archive")
        sudo = sudo_prefix(require_global)
        if require_global or is_root() or have("sudo"):
            prefix = Path("/usr/local/bin")
            run(sudo + ["install", "-d", "-m", "755", str(prefix)])
            run(sudo + ["install", "-m", "755", str(found[0]), str(prefix / "gh")])
        else:
            prefix = Path.home() / ".local" / "bin"
            prefix.mkdir(parents=True, exist_ok=True)
            shutil.copy2(found[0], prefix / "gh")
            os.chmod(prefix / "gh", 0o755)
            print(f"installed {prefix / 'gh'}; ensure {prefix} is on PATH")


def report(tools: list[str]) -> None:
    print("tool status:")
    for row in TOOLS:
        cmd = row[0]
        ok = have_c_compiler() if cmd == "gcc" else have(cmd)
        print(f"  {'OK  ' if ok else 'MISS'} {cmd}")
    print(f"  OK   python3 ({sys.version.split()[0]})")
    if tools:
        print("missing: " + ", ".join(tools))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Install missing UENR build/cloud dependencies globally.")
    ap.add_argument("--check", action="store_true", help="report only")
    ap.add_argument("--yes", "-y", action="store_true", help="do not prompt")
    ap.add_argument("--global", dest="global_install", action="store_true",
                    help="require system-wide installation (forces sudo)")
    ap.add_argument("--force", action="store_true",
                    help="reinstall/upgrade even when a tool is already present")
    ap.add_argument("--dry-run", action="store_true", help="print commands only")
    args = ap.parse_args()

    manager, pm = detect_manager()
    missing = missing_tools()
    report(missing)

    if args.check:
        return 0 if not missing else 1
    if not missing and not args.force:
        print("all dependencies are already installed; nothing to do")
        return 0
    if not manager:
        print("error: no supported package manager found (apt/dnf/pacman/zypper/apk/brew)",
              file=sys.stderr)
        print("install Git, make, GCC, curl and GitHub CLI manually.", file=sys.stderr)
        return 3

    packages = package_names(manager, missing if not args.force else [r[0] for r in TOOLS])
    print(f"plan: {manager} install {' '.join(packages)}")
    if not args.yes and not args.dry_run:
        if not sys.stdin.isatty():
            print("error: non-interactive stdin; re-run with --yes", file=sys.stderr)
            return 2
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("aborted")
            return 1
    try:
        install_packages(manager, pm, packages,
                         require_global=args.global_install or args.force,
                         dry_run=args.dry_run)
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4 if "sudo/root" in str(exc) else 1

    if not have("gh") and not args.dry_run:
        print("gh still missing after package install; trying the official binary")
        try:
            install_gh_binary(require_global=args.global_install or args.force,
                              dry_run=args.dry_run)
        except InstallError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    missing = missing_tools()
    report(missing)
    if missing:
        print("error: still missing: " + ", ".join(missing), file=sys.stderr)
        return 1
    if args.dry_run:
        print("dry run complete; no packages were changed")
    else:
        print("dependencies installed. Next: scripts/github_auth.sh setup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
