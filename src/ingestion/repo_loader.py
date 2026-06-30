import os
import shutil
from pathlib import Path
from typing import List
import git

from src.config import config
from src.ingestion.models import CodeFile
from src.ingestion.language_detector import detect_language, is_supported


EXCLUDED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", ".env", "dist", "build", ".next", ".nuxt",
    "coverage", ".pytest_cache", ".mypy_cache", "eggs",
    ".eggs", "htmlcov", ".tox", "docs", "doc",
}

EXCLUDED_FILES = {
    "setup.py", "conftest.py", "__init__.py",
    "manage.py", "wsgi.py", "asgi.py",
}


def _extract_repo_name(github_url: str) -> str:
    url = github_url.rstrip("/")
    return url.split("/")[-1]


def _get_clone_path(repo_name: str) -> Path:
    return Path(config.REPOS_PATH) / repo_name


def clone_repo(github_url: str, force_reclone: bool = False) -> Path:
    repo_name = _extract_repo_name(github_url)
    clone_path = _get_clone_path(repo_name)

    if clone_path.exists():
        if force_reclone:
            print(f"Removing existing clone at {clone_path}...")
            shutil.rmtree(clone_path)
        else:
            print(f"Repo already cloned at {clone_path} — skipping clone.")
            return clone_path

    print(f"Cloning {github_url} into {clone_path}...")
    os.makedirs(clone_path.parent, exist_ok=True)
    git.Repo.clone_from(github_url, clone_path, depth=1)
    print(f"Clone complete.")
    return clone_path


def load_files_from_repo(
    github_url: str,
    force_reclone: bool = False,
    max_file_size_kb: int = 100,
) -> List[CodeFile]:
    repo_name = _extract_repo_name(github_url)
    clone_path = clone_repo(github_url, force_reclone)

    code_files: List[CodeFile] = []
    skipped_count = 0
    max_bytes = max_file_size_kb * 1024

    print(f"Scanning files in {clone_path}...")

    for root, dirs, files in os.walk(clone_path):
        dirs[:] = [
            d for d in dirs
            if d not in EXCLUDED_DIRS and not d.startswith(".")
        ]

        for filename in files:
            file_path = Path(root) / filename
            extension = file_path.suffix.lower()

            if not is_supported(extension):
                skipped_count += 1
                continue

            if filename in EXCLUDED_FILES:
                skipped_count += 1
                continue

            try:
                file_size = file_path.stat().st_size
                if file_size > max_bytes:
                    print(f"  Skipping large file: {filename} ({file_size // 1024}KB)")
                    skipped_count += 1
                    continue
            except OSError:
                skipped_count += 1
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"  Could not read {filename}: {e}")
                skipped_count += 1
                continue

            if not content.strip():
                skipped_count += 1
                continue

            relative_path = str(file_path.relative_to(clone_path))

            code_file = CodeFile(
                content=content,
                file_path=relative_path,
                language=detect_language(extension),
                repo_name=repo_name,
                repo_url=github_url,
                extension=extension,
            )
            code_files.append(code_file)

    print(f"Loaded {len(code_files)} files. Skipped {skipped_count} files.")
    return code_files