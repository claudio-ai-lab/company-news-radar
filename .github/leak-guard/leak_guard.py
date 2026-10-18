#!/usr/bin/env python3
"""Leak guard: keep protected terms out of a git repository.

Generic and dependency-free (standard library only). The script itself contains
no protected terms; they live in a private deny list that is passed at runtime
(a local file, or a masked CI secret written to a temporary file).

Findings report only location and category, never the matched text. Paths that
themselves contain a protected term are printed masked.

Modes
  staged          pre-commit: staged diff (added lines), staged file names,
                  staged binaries, author and committer identity
  msg FILE        commit-msg: the commit message file (comment lines ignored)
  pending         everything the next commit could contain: staged and unstaged
                  changes, untracked files, identity (used by agent hooks)
  tree [REV]      every file of REV (default HEAD): content, names, binary
                  allow list, image metadata
  history         every object reachable from all refs (or --range ARGS...):
                  file contents, names, commit and tag messages,
                  author/committer addresses, ref names
  push [REMOTE]   like history, limited to objects not yet on REMOTE
  text            standard input (e.g. a command line), reported as --path

Deny list format (UTF-8)
  [category]      section header; findings report this name
  term            literal, case-insensitive, flexible whitespace, whole word
  re:pattern      Python regular expression, case-insensitive
  # comment

Allow list format (one rule per line)
  <path-glob> | <exact matched text>    accept this match in matching paths
  binary | <glob>                       binary files allowed by name
  Pseudo paths: ":commit-msg" (commit/tag messages, command text),
  ":author" (author/committer/tagger names), ":ref" (branch and tag names).

Identity: author, committer and tagger addresses must be GitHub noreply addresses.

Exit codes: 0 clean, 1 findings, 2 configuration or runtime error.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
NOREPLY = re.compile(
    r"^(?:\d+\+)?[A-Za-z0-9-]+(?:\[bot\])?@users\.noreply\.github\.com$|^noreply@github\.com$",
    re.IGNORECASE,
)
WORD_LEFT = r"(?<![^\W_])"
WORD_RIGHT = r"(?![^\W_])"
CATEGORY_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
MAX_BYTES = 20 * 1024 * 1024
BINARY_SNIFF = 8000
MSG_PATH = ":commit-msg"
AUTHOR_PATH = ":author"
REF_PATH = ":ref"

CAT_IDENTITY = "identity"
CAT_BINARY = "binary"
CAT_IMAGE_META = "image-metadata"
CAT_SECRET = "secret"
CAT_OVERSIZE = "oversize"


class GuardError(Exception):
    """Configuration or runtime problem (exit code 2)."""


# --------------------------------------------------------------------------- lists


@dataclass(frozen=True)
class Hit:
    path: str
    line: int | None
    category: str
    context: str = ""


class DenyList:
    """Protected terms grouped by category, compiled to one regex per category."""

    def __init__(self, patterns: dict[str, list[str]]) -> None:
        self.categories: dict[str, re.Pattern[str]] = {}
        for category, items in patterns.items():
            if items:
                joined = "|".join(f"(?:{p})" for p in items)
                self.categories[category] = re.compile(joined, re.IGNORECASE)

    @classmethod
    def parse(cls, text: str, source: str = "deny list") -> DenyList:
        patterns: dict[str, list[str]] = {}
        category: str | None = None
        for lineno, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                category = line[1:-1].strip().lower()
                if not CATEGORY_NAME.match(category):
                    raise GuardError(f"{source}:{lineno}: invalid category name")
                patterns.setdefault(category, [])
                continue
            if category is None:
                raise GuardError(f"{source}:{lineno}: entry before the first [category]")
            if line.startswith("re:"):
                pattern = line[3:].strip()
            else:
                pattern = WORD_LEFT + r"\s+".join(map(re.escape, line.split())) + WORD_RIGHT
            try:
                compiled = re.compile(f"(?:{pattern})", re.IGNORECASE)
            except re.error:
                raise GuardError(f"{source}:{lineno}: invalid regular expression") from None
            if compiled.search(""):
                raise GuardError(f"{source}:{lineno}: pattern matches the empty string")
            patterns[category].append(pattern)
        deny = cls(patterns)
        if not deny.categories:
            raise GuardError(f"{source}: no entries")
        return deny

    def find(self, text: str) -> Iterator[tuple[str, str]]:
        """Yield (category, matched text) for every match."""
        for category, regex in self.categories.items():
            for match in regex.finditer(text):
                yield category, match.group(0)

    def mask(self, text: str) -> str:
        for regex in self.categories.values():
            text = regex.sub("***", text)
        return text


@dataclass
class AllowList:
    rules: list[tuple[str, str]] = field(default_factory=list)
    binary_globs: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str, source: str = "allow list") -> AllowList:
        allow = cls()
        for lineno, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            glob, sep, value = line.partition("|")
            glob, value = glob.strip(), value.strip()
            if not sep or not glob or not value:
                raise GuardError(f"{source}:{lineno}: expected '<path-glob> | <text>'")
            if glob.lower() == "binary":
                allow.binary_globs.append(value)
            else:
                allow.rules.append((glob, " ".join(value.split()).casefold()))
        return allow

    def allows(self, path: str, matched: str) -> bool:
        text = " ".join(matched.split()).casefold()
        lowered = path.lower()
        return any(
            text == allowed and fnmatch.fnmatchcase(lowered, glob.lower())
            for glob, allowed in self.rules
        )

    def binary_allowed(self, path: str) -> bool:
        lowered = path.lower()
        name = lowered.rsplit("/", 1)[-1]
        return any(
            fnmatch.fnmatchcase(lowered, g.lower()) or fnmatch.fnmatchcase(name, g.lower())
            for g in self.binary_globs
        )


# --------------------------------------------------------------------------- scanner


def is_binary(data: bytes) -> bool:
    return b"\0" in data[:BINARY_SNIFF]


def decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8", errors="replace")


def image_has_metadata(data: bytes) -> bool:
    """True if a PNG or JPEG carries text/EXIF/XMP/IPTC/comment metadata."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        pos = 8
        while pos + 8 <= len(data):
            length, ctype = struct.unpack(">I4s", data[pos : pos + 8])
            if ctype in (b"tEXt", b"zTXt", b"iTXt", b"eXIf"):
                return True
            if ctype == b"IEND":
                break
            pos += 12 + length
        return False
    if data.startswith(b"\xff\xd8"):
        pos = 2
        while pos + 4 <= len(data):
            if data[pos] != 0xFF:
                break
            marker = data[pos + 1]
            if marker == 0xFF:
                pos += 1
                continue
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                pos += 2
                continue
            if marker == 0xDA:
                break
            if marker in (0xE1, 0xED, 0xFE):
                return True
            (length,) = struct.unpack(">H", data[pos + 2 : pos + 4])
            pos += 2 + length
        return False
    return False


class Scanner:
    def __init__(self, deny: DenyList, allow: AllowList, check_binaries: bool = True) -> None:
        self.deny = deny
        self.allow = allow
        self.check_binaries = check_binaries
        self.hits: list[Hit] = []
        self._seen: set[Hit] = set()

    def child(self) -> Scanner:
        return Scanner(self.deny, self.allow, self.check_binaries)

    def add(self, hit: Hit) -> None:
        if hit not in self._seen:
            self._seen.add(hit)
            self.hits.append(hit)

    def scan_text(self, path: str, text: str, line: int | None, context: str = "") -> None:
        for category, matched in self.deny.find(text):
            if not self.allow.allows(path, matched):
                self.add(Hit(path, line, category, context))

    def scan_lines(self, path: str, lines: Iterable[tuple[int, str]], context: str = "") -> None:
        for lineno, text in lines:
            self.scan_text(path, text, lineno, context)

    def scan_name(self, path: str, pseudo: str | None = None, context: str = "file name") -> None:
        self.scan_text(pseudo or path, path, None, context)

    def scan_blob(self, path: str, data: bytes, context: str = "") -> None:
        if len(data) > MAX_BYTES:
            self.add(Hit(path, None, CAT_OVERSIZE, context))
            return
        if image_has_metadata(data):
            self.add(Hit(path, None, CAT_IMAGE_META, context))
        if is_binary(data):
            if self.check_binaries and not self.allow.binary_allowed(path):
                self.add(Hit(path, None, CAT_BINARY, context))
            for decoded in (data.decode("latin-1"), data.decode("utf-16-le", errors="ignore")):
                self.scan_text(path, decoded, None, (context + " binary content").strip())
            return
        self.scan_lines(path, enumerate(decode_text(data).splitlines(), 1), context)

    def check_identity(self, role: str, name: str, email: str, context: str = "") -> None:
        if not NOREPLY.match(email.strip()):
            self.add(Hit(f"{role} address", None, CAT_IDENTITY, context))
        self.scan_text(AUTHOR_PATH, name, None, (f"{role} name " + context).strip())

    def scan_message(self, text: str, context: str = "") -> None:
        self.scan_lines(MSG_PATH, enumerate(text.splitlines(), 1), context)


# --------------------------------------------------------------------------- git


class Git:
    def __init__(self, repo: Path) -> None:
        self.repo = repo

    def run(self, *args: str, check: bool = True, stdin: bytes | None = None) -> bytes:
        proc = subprocess.run(
            ["git", "-C", str(self.repo), "-c", "core.quotePath=false", *args],
            input=stdin,
            capture_output=True,
        )
        if check and proc.returncode != 0:
            detail = proc.stderr.decode(errors="replace").strip()[:300]
            raise GuardError(f"git {args[0]} failed: {detail}")
        return proc.stdout

    def text(self, *args: str, check: bool = True) -> str:
        return self.run(*args, check=check).decode("utf-8", errors="replace")

    def has_refs(self) -> bool:
        return bool(self.text("for-each-ref", "--count=1").strip())

    def has_head(self) -> bool:
        return self.run("rev-parse", "--verify", "--quiet", "HEAD^{commit}", check=False) != b""

    def read_objects(self, shas: Sequence[str]) -> dict[str, tuple[str, bytes]]:
        """Read many objects with one `git cat-file --batch` call."""
        if not shas:
            return {}
        out = self.run("cat-file", "--batch", stdin=("\n".join(shas) + "\n").encode())
        result: dict[str, tuple[str, bytes]] = {}
        pos = 0
        for sha in shas:
            end = out.index(b"\n", pos)
            parts = out[pos:end].decode().split()
            pos = end + 1
            if len(parts) < 3 or parts[1] == "missing":
                continue
            size = int(parts[2])
            result[sha] = (parts[1], out[pos : pos + size])
            pos += size + 1
        return result

    def ident(self, var: str) -> tuple[str, str]:
        raw = self.text("var", var, check=False).strip()
        match = re.match(r"^(.*?)\s*<([^>]*)>", raw)
        return (match.group(1), match.group(2)) if match else ("", "")


def unquote_c(text: str) -> str:
    """Undo git's C-style quoting of unusual path names."""
    if not (len(text) >= 2 and text.startswith('"') and text.endswith('"')):
        return text
    escapes = {"n": 10, "t": 9, "r": 13, "a": 7, "b": 8, "f": 12, "v": 11}
    raw, out, i = text[1:-1], bytearray(), 0
    while i < len(raw):
        ch = raw[i]
        if ch == "\\" and i + 1 < len(raw):
            octal = raw[i + 1 : i + 4]
            if len(octal) == 3 and all(c in "01234567" for c in octal):
                out.append(int(octal, 8) & 0xFF)
                i += 4
                continue
            out.append(escapes.get(raw[i + 1], ord(raw[i + 1])))
            i += 2
            continue
        out.extend(ch.encode("utf-8"))
        i += 1
    return out.decode("utf-8", errors="replace")


def parse_diff(diff: str) -> Iterator[tuple[str, int, str]]:
    """Yield (path, new line number, text) for every added line of a unified diff."""
    path = ""
    lineno = 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            target = unquote_c(line[4:])
            path = "" if target == "/dev/null" else target[2:] if target.startswith("b/") else target
            continue
        if line.startswith("@@"):
            match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            lineno = int(match.group(1)) if match else 0
            continue
        if line.startswith("+") and path:
            yield path, lineno, line[1:]
            lineno += 1
        elif line.startswith(" "):
            lineno += 1


DIFF_FILTER = ("--no-renames", "--diff-filter=ACMRT")


def diff_text(git: Git, *extra: str) -> str:
    return git.text(
        "diff", "-U0", "--no-color", "--no-ext-diff", *DIFF_FILTER,
        "--src-prefix=a/", "--dst-prefix=b/", *extra,
    )  # fmt: skip


def diff_names(git: Git, *extra: str) -> list[str]:
    return git.text("diff", "--name-only", *DIFF_FILTER, *extra).splitlines()


def diff_binaries(git: Git, *extra: str) -> set[str]:
    out = git.text("diff", "--numstat", *DIFF_FILTER, *extra)
    return {line.split("\t", 2)[2] for line in out.splitlines() if line.startswith("-\t-\t")}


# --------------------------------------------------------------------------- gitleaks


def find_gitleaks(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for candidate in (os.environ.get("GITLEAKS"), shutil.which("gitleaks")):
        if candidate and Path(candidate).exists():
            return candidate
    local = os.environ.get("LOCALAPPDATA")
    if local:
        winget = Path(local) / "Microsoft" / "WinGet"
        candidates = [winget / "Links" / "gitleaks.exe"]
        candidates += sorted(winget.glob("Packages/Gitleaks.Gitleaks_*/gitleaks.exe"))
        for exe in candidates:
            if exe.is_file():
                return str(exe)
    return None


def run_gitleaks(exe: str, repo: Path, args: Sequence[str], config: Path | None) -> list[Hit]:
    """Run gitleaks with its default rules; repo-local config and ignore files are ignored."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        report = tmpdir / "report.json"
        if config is None:
            config = tmpdir / "gitleaks.toml"
            config.write_text("[extend]\nuseDefault = true\n", encoding="utf-8")
        cmd = [
            exe, *args, "--config", str(config), "--gitleaks-ignore-path", str(tmpdir),
            "--ignore-gitleaks-allow", "--redact", "--no-banner", "--log-level", "error",
            "--exit-code", "1", "--report-format", "json", "--report-path", str(report),
        ]  # fmt: skip
        proc = subprocess.run(cmd, cwd=repo, capture_output=True)
        if proc.returncode not in (0, 1):
            raise GuardError(f"gitleaks failed (exit {proc.returncode})")
        content = report.read_text(encoding="utf-8") if report.exists() else ""
        findings = json.loads(content or "[]")
    hits = []
    for f in findings or []:
        context = f"gitleaks {f.get('RuleID', '')}".strip()
        if f.get("Commit"):
            context += f", commit {str(f['Commit'])[:7]}"
        hits.append(Hit(str(f.get("File", "")), f.get("StartLine"), CAT_SECRET, context))
    if proc.returncode == 1 and not hits:
        hits.append(Hit("(gitleaks)", None, CAT_SECRET, "finding without details"))
    return hits


# --------------------------------------------------------------------------- modes


def check_local_identity(scanner: Scanner, git: Git) -> None:
    for role, var in (("author", "GIT_AUTHOR_IDENT"), ("committer", "GIT_COMMITTER_IDENT")):
        scanner.check_identity(role, *git.ident(var))


def scan_diff(scanner: Scanner, git: Git, *extra: str) -> set[str]:
    """Scan names and added lines of one diff; return the binary paths (not yet scanned)."""
    for name in diff_names(git, *extra):
        scanner.scan_name(name)
    binaries = diff_binaries(git, *extra)
    for path, lineno, text in parse_diff(diff_text(git, *extra)):
        if path not in binaries:
            scanner.scan_text(path, text, lineno)
    return binaries


def mode_staged(scanner: Scanner, repo: Path) -> list[list[str]]:
    git = Git(repo)
    check_local_identity(scanner, git)
    base = [] if git.has_head() else [EMPTY_TREE]
    for path in sorted(scan_diff(scanner, git, "--cached", *base)):
        scanner.scan_blob(path, git.run("cat-file", "blob", f":{path}", check=False))
    return [["git", "--pre-commit", "--staged", str(repo)]]


def mode_msg(scanner: Scanner, msg_file: Path) -> None:
    kept = []
    for line in msg_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("# ") and ">8" in line and "-----" in line:
            break  # scissors line of `git commit -v`
        if not line.startswith("#"):
            kept.append(line)
    scanner.scan_message("\n".join(kept))


def mode_pending(scanner: Scanner, repo: Path) -> list[list[str]]:
    git = Git(repo)
    check_local_identity(scanner, git)
    base = "HEAD" if git.has_head() else EMPTY_TREE
    for path in sorted(scan_diff(scanner, git, "--cached", base)):
        scanner.scan_blob(path, git.run("cat-file", "blob", f":{path}", check=False))
    for path in sorted(scan_diff(scanner, git, base)):
        file = repo / path
        scanner.scan_blob(path, file.read_bytes() if file.is_file() else b"")
    commands = [["git", "--pre-commit", "--staged", str(repo)], ["git", "--pre-commit", str(repo)]]
    untracked = git.run("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    for raw in filter(None, untracked):
        path = raw.decode("utf-8", errors="replace")
        scanner.scan_name(path)
        file = repo / path
        if not file.is_file():
            continue
        if file.stat().st_size > MAX_BYTES:
            scanner.add(Hit(path, None, CAT_OVERSIZE))
            continue
        scanner.scan_blob(path, file.read_bytes())
        commands.append(["dir", str(file)])
    return commands


def mode_tree(scanner: Scanner, repo: Path, rev: str) -> None:
    git = Git(repo)
    entries = []
    for record in git.run("ls-tree", "-r", "-z", "--full-tree", rev).split(b"\0"):
        if not record:
            continue
        meta, _, path = record.decode("utf-8", errors="replace").partition("\t")
        _mode, otype, sha = meta.split()
        scanner.scan_name(path)
        if otype == "blob":
            entries.append((sha, path))
    objects = git.read_objects([sha for sha, _ in entries])
    for sha, path in entries:
        scanner.scan_blob(path, objects.get(sha, ("blob", b""))[1])


def mode_history(scanner: Scanner, repo: Path, rev_args: Sequence[str]) -> list[list[str]]:
    git = Git(repo)
    gitleaks_run = [["git", f"--log-opts={' '.join(rev_args)}", str(repo)]]
    if not git.has_refs():
        return []
    # Commit messages and identities
    fmt = "%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%B%x1e"
    for record in git.text("log", f"--format={fmt}", *rev_args).split("\x1e"):
        record = record.lstrip("\n")
        if not record:
            continue
        sha, an, ae, cn, ce, body = ([*record.split("\x1f"), "", "", "", "", ""])[:6]
        context = f"commit {sha[:7]}"
        scanner.check_identity("author", an, ae, context)
        scanner.check_identity("committer", cn, ce, context)
        scanner.scan_message(body, context)
    # Ref names that would be published
    for ref in git.text("for-each-ref", "--format=%(refname)", "refs/heads", "refs/tags").splitlines():
        scanner.scan_name(ref, pseudo=REF_PATH, context="ref name")
    # Every reachable object: blobs, file and directory names, annotated tags
    paths: dict[str, str] = {}
    for line in git.text("rev-list", "--objects", *rev_args).splitlines():
        sha, _, path = line.partition(" ")
        if path and sha not in paths:
            paths[sha] = path
    for path in sorted(set(paths.values())):
        scanner.scan_name(path)
    objects = git.read_objects(list(paths))
    for sha, path in paths.items():
        otype, data = objects.get(sha, ("missing", b""))
        if otype == "blob":
            blob_scan = scanner.child()
            blob_scan.scan_blob(path, data)
            if blob_scan.hits:
                first = git.text("log", "--format=%h", "--reverse", f"--find-object={sha}", *rev_args)
                commit = first.split()[0] if first.split() else "?"
                for hit in blob_scan.hits:
                    scanner.add(replace(hit, context=f"commit {commit}"))
        elif otype == "tag":
            head, _, message = data.decode("utf-8", errors="replace").partition("\n\n")
            tagger = re.search(r"^tagger (.*?)\s*<([^>]*)>", head, re.MULTILINE)
            if tagger:
                scanner.check_identity("tagger", tagger.group(1), tagger.group(2), "annotated tag")
            scanner.scan_message(message, "annotated tag")
    return gitleaks_run


def push_range(git: Git, target: str | None) -> list[str]:
    """Revisions not yet on the target remote (all local branches and tags for a URL)."""
    remotes = git.text("remote").split()
    if target is None:
        upstream = git.text(
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", check=False
        ).strip()
        target = upstream.split("/", 1)[0] if upstream else ("origin" if "origin" in remotes else None)
    if target in remotes:
        return ["--branches", "--tags", "--not", f"--remotes={target}"]
    return ["--branches", "--tags"]


# --------------------------------------------------------------------------- cli


def repo_name_from_origin(git: Git) -> str | None:
    url = git.text("remote", "get-url", "origin", check=False).strip()
    if not url:
        return None
    name = re.split(r"[/:]", url.rstrip("/"))[-1]
    return name.removesuffix(".git")


def load_lists(args: argparse.Namespace, repo: Path) -> tuple[DenyList, AllowList]:
    here = Path(__file__).resolve().parent
    deny_path = Path(args.deny or os.environ.get("LEAK_GUARD_DENY_FILE") or here / "deny.txt")
    if not deny_path.is_file():
        raise GuardError(f"deny list not found: {deny_path}")
    deny = DenyList.parse(deny_path.read_text(encoding="utf-8-sig"), deny_path.name)
    allow_path: Path | None = None
    explicit = args.allow or os.environ.get("LEAK_GUARD_ALLOW_FILE")
    if explicit:
        allow_path = Path(explicit)
        if not allow_path.is_file():
            raise GuardError(f"allow list not found: {allow_path}")
    else:
        name = args.repo_name or repo_name_from_origin(Git(repo))
        candidate = here / "repos" / f"{name}.allow" if name else None
        if candidate and candidate.is_file():
            allow_path = candidate
    allow = AllowList()
    if allow_path:
        allow = AllowList.parse(allow_path.read_text(encoding="utf-8-sig"), allow_path.name)
    allow.binary_globs += [g.strip() for g in (args.binary_allow or "").split(",") if g.strip()]
    return deny, allow


def format_hit(hit: Hit, deny: DenyList) -> str:
    where = deny.mask(hit.path)
    if hit.line is not None:
        where += f":{hit.line}"
    suffix = f" ({deny.mask(hit.context)})" if hit.context else ""
    return f"{where}: {hit.category}{suffix}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leak guard: keep protected terms out of git.")
    parser.add_argument("--repo", default=".", help="repository (default: current directory)")
    parser.add_argument("--repo-name", help="name used to find repos/<name>.allow")
    parser.add_argument("--deny", help="deny list (default: deny.txt next to this script)")
    parser.add_argument("--allow", help="allow list (default: repos/<repo-name>.allow)")
    parser.add_argument("--binary-allow", help="comma-separated globs of allowed binary files")
    parser.add_argument("--gitleaks", help="path to the gitleaks executable")
    parser.add_argument("--gitleaks-config", help="gitleaks config (default: built-in rules)")
    parser.add_argument("--no-gitleaks", action="store_true", help="skip gitleaks")
    parser.add_argument("--no-binary-check", action="store_true", help="skip the binary allow list")
    parser.add_argument("--json", action="store_true", help="print findings as JSON")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("staged")
    sub.add_parser("msg").add_argument("file")
    sub.add_parser("pending")
    sub.add_parser("tree").add_argument("rev", nargs="?", default="HEAD")
    sub.add_parser("history").add_argument(
        "--range", nargs=argparse.REMAINDER, help="git rev-list arguments (default: --all)"
    )
    sub.add_parser("push").add_argument("remote", nargs="?")
    sub.add_parser("text").add_argument("--path", default=MSG_PATH)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(args.repo).resolve()
    deny, allow = load_lists(args, repo)
    scanner = Scanner(deny, allow, check_binaries=not args.no_binary_check)
    gitleaks_config = Path(args.gitleaks_config) if args.gitleaks_config else None
    gitleaks_runs: list[list[str]] = []
    if args.mode == "staged":
        gitleaks_runs = mode_staged(scanner, repo)
    elif args.mode == "msg":
        mode_msg(scanner, Path(args.file))
    elif args.mode == "pending":
        gitleaks_runs = mode_pending(scanner, repo)
    elif args.mode == "tree":
        mode_tree(scanner, repo, args.rev)
    elif args.mode == "history":
        gitleaks_runs = mode_history(scanner, repo, args.range or ["--all"])
    elif args.mode == "push":
        gitleaks_runs = mode_history(scanner, repo, push_range(Git(repo), args.remote))
    elif args.mode == "text":
        scanner.scan_lines(args.path, enumerate(sys.stdin.read().splitlines(), 1))
    if gitleaks_runs and not args.no_gitleaks:
        exe = find_gitleaks(args.gitleaks)
        if exe is None:
            raise GuardError("gitleaks not found (install it, set GITLEAKS, or pass --no-gitleaks)")
        for gl_args in gitleaks_runs:
            for hit in run_gitleaks(exe, repo, gl_args, gitleaks_config):
                scanner.add(hit)
    hits = scanner.hits
    if args.json:
        rows = [
            {"path": deny.mask(h.path), "line": h.line, "category": h.category,
             "context": deny.mask(h.context)}
            for h in hits
        ]  # fmt: skip
        print(json.dumps(rows, ensure_ascii=False))
    else:
        for hit in hits:
            print(format_hit(hit, deny))
        if hits:
            counts = Counter(h.category for h in hits)
            summary = ", ".join(f"{cat} {n}" for cat, n in sorted(counts.items()))
            print(f"leak guard: {len(hits)} finding(s) [{summary}]")
        else:
            print(f"leak guard: clean ({args.mode})")
    return 1 if hits else 0


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    try:
        sys.exit(run())
    except GuardError as exc:
        print(f"leak guard error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
