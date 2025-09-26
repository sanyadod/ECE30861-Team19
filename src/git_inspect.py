"""
Git repository inspection
"""

import os
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

from dulwich import porcelain
from dulwich.errors import NotGitRepository
from dulwich.repo import Repo

from .logging_utils import get_logger
from .models import ParsedURL

logger = get_logger()


class GitInspector:
    """Git repository inspector using Dulwich."""

    def __init__(self, cache_dir: Optional[str] = None, timeout: int = 30):
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="src_git_cache_")
        self.timeout = timeout
        os.makedirs(self.cache_dir, exist_ok=True)

    def clone_repo(self, repo_url: ParsedURL) -> Optional[str]:
        """
        Clone a repository to cache directory with timeout handling.

        Returns path to cloned repo or None if failed.
        """
        if repo_url.platform != "github":
            return None

        repo_name = (
            f"{repo_url.owner}_{repo_url.repo}"
            if repo_url.owner and repo_url.repo
            else "unknown"
        )
        clone_path = os.path.join(self.cache_dir, repo_name)

        # Check if already cloned
        if os.path.exists(clone_path):
            logger.debug(f"Using cached clone at {clone_path}")
            return clone_path

        try:
            logger.info(f"Cloning {repo_url.url} to {clone_path}")
            
            # Use threading to implement timeout
            result = [None]
            exception = [None]
            
            def clone_worker():
                try:
                    # Use smaller depth for faster cloning
                    porcelain.clone(
                        repo_url.url, clone_path, depth=5
                    )  # Much smaller depth for efficiency
                    result[0] = clone_path
                except Exception as e:
                    exception[0] = e
            
            # Start clone in separate thread
            thread = threading.Thread(target=clone_worker)
            thread.daemon = True
            thread.start()
            
            # Wait for completion or timeout
            thread.join(timeout=self.timeout)
            
            if thread.is_alive():
                logger.warning(f"Timeout cloning {repo_url.url} after {self.timeout}s")
                # Clean up partial clone
                if os.path.exists(clone_path):
                    shutil.rmtree(clone_path, ignore_errors=True)
                return None
            
            if exception[0]:
                raise exception[0]
                
            return result[0]
                
        except Exception as e:
            logger.warning(f"Failed to clone {repo_url.url}: {e}")
            # Clean up partial clone
            if os.path.exists(clone_path):
                shutil.rmtree(clone_path, ignore_errors=True)
            return None

    def analyze_repository(self, repo_path: str) -> Dict[str, Any]:
        """
        Analyze a cloned repository for various quality metrics.

        Returns dict with analysis results.
        """
        try:
            repo = Repo(repo_path)
        except NotGitRepository:
            logger.error(f"Not a git repository: {repo_path}")
            return self._empty_analysis()

        analysis = {
            "commit_analysis": self._analyze_commits(repo),
            "contributor_analysis": self._analyze_contributors(repo),
            "file_analysis": self._analyze_files(repo_path),
            "structure_analysis": self._analyze_structure(repo_path),
            "documentation_analysis": self._analyze_documentation(repo_path),
        }

        return analysis

    def _analyze_commits(self, repo: Repo) -> Dict[str, Any]:
        """Analyze commit history."""
        try:
            commits = list(repo.get_walker(max_entries=100))  # Limit for performance

            if not commits:
                return {
                    "total_commits": 0,
                    "recent_commits": 0,
                    "avg_commit_frequency": 0,
                }

            # Get recent commits (last 90 days)
            now = datetime.now(timezone.utc)
            recent_threshold = now.timestamp() - (90 * 24 * 60 * 60)  # 90 days ago

            recent_commits = []
            commit_dates = []

            for entry in commits:
                commit = entry.commit
                commit_time = commit.commit_time
                commit_dates.append(commit_time)

                if commit_time >= recent_threshold:
                    recent_commits.append(commit)

            # Calculate average commit frequency (commits per day)
            if len(commit_dates) > 1:
                time_span = max(commit_dates) - min(commit_dates)
                days_span = max(1, time_span / (24 * 60 * 60))
                avg_frequency = len(commit_dates) / days_span
            else:
                avg_frequency = 0

            return {
                "total_commits": len(commits),
                "recent_commits": len(recent_commits),
                "avg_commit_frequency": avg_frequency,
                "time_span_days": (
                    (max(commit_dates) - min(commit_dates)) / (24 * 60 * 60)
                    if len(commit_dates) > 1
                    else 0
                ),
            }

        except Exception as e:
            logger.warning(f"Error analyzing commits: {e}")
            return {"total_commits": 0, "recent_commits": 0, "avg_commit_frequency": 0}

    def _analyze_contributors(self, repo: Repo) -> Dict[str, Any]:
        """Analyze contributor diversity."""
        try:
            commits = list(repo.get_walker(max_entries=100))

            authors: Set[str] = set()
            committers: Set[str] = set()

            for entry in commits:
                commit = entry.commit
                authors.add(commit.author.decode("utf-8", errors="ignore"))
                committers.add(commit.committer.decode("utf-8", errors="ignore"))

            # Simple bus factor calculation
            total_commits = len(commits)
            unique_authors = len(authors)

            if total_commits == 0:
                bus_factor_score = 0.0
            elif unique_authors == 1:
                bus_factor_score = 0.3  # Low diversity
            elif unique_authors < 3:
                bus_factor_score = 0.5  # Medium diversity
            else:
                bus_factor_score = 0.8  # High diversity

            return {
                "unique_authors": unique_authors,
                "unique_committers": len(committers),
                "bus_factor_score": bus_factor_score,
                "authors": list(authors)[:10],  # Limit for privacy
            }

        except Exception as e:
            logger.warning(f"Error analyzing contributors: {e}")
            return {
                "unique_authors": 0,
                "unique_committers": 0,
                "bus_factor_score": 0.0,
            }

    def _analyze_files(self, repo_path: str) -> Dict[str, Any]:
        """Analyze file structure and types."""
        try:
            repo_root = Path(repo_path)

            python_files = list(repo_root.glob("**/*.py"))
            test_files = [
                f
                for f in python_files
                if "test" in f.name.lower()
                or f.parent.name.lower() in ["test", "tests"]
            ]

            # Count lines of code (simple version)
            total_lines = 0
            for py_file in python_files:
                try:
                    with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                        total_lines += len(f.readlines())
                except Exception:
                    continue

            return {
                "total_files": len(list(repo_root.glob("**/*"))),
                "python_files": len(python_files),
                "test_files": len(test_files),
                "total_lines_of_code": total_lines,
                "test_coverage_estimate": (
                    len(test_files) / max(1, len(python_files)) if python_files else 0
                ),
            }

        except Exception as e:
            logger.warning(f"Error analyzing files: {e}")
            return {
                "total_files": 0,
                "python_files": 0,
                "test_files": 0,
                "total_lines_of_code": 0,
            }

    def _analyze_structure(self, repo_path: str) -> Dict[str, Any]:
        """Analyze repository structure quality."""
        try:
            repo_root = Path(repo_path)

            # Check for standard files
            has_readme = any(repo_root.glob("README.*"))
            has_license = any(repo_root.glob("LICENSE*")) or any(
                repo_root.glob("LICENCE*")
            )
            has_requirements = (repo_root / "requirements.txt").exists() or (
                repo_root / "pyproject.toml"
            ).exists()
            has_setup = (repo_root / "setup.py").exists() or (
                repo_root / "setup.cfg"
            ).exists()
            has_ci = (repo_root / ".github").exists() or (
                repo_root / ".gitlab-ci.yml"
            ).exists()

            structure_score = (
                sum([has_readme, has_license, has_requirements, has_setup, has_ci])
                / 5.0
            )

            return {
                "has_readme": has_readme,
                "has_license": has_license,
                "has_requirements": has_requirements,
                "has_setup": has_setup,
                "has_ci": has_ci,
                "structure_score": structure_score,
            }

        except Exception as e:
            logger.warning(f"Error analyzing structure: {e}")
            return {"structure_score": 0.0}

    def _analyze_documentation(self, repo_path: str) -> Dict[str, Any]:
        """Analyze documentation quality."""
        try:
            repo_root = Path(repo_path)

            # Find README content
            readme_content = ""
            for readme_file in repo_root.glob("README.*"):
                try:
                    with open(readme_file, "r", encoding="utf-8", errors="ignore") as f:
                        readme_content = f.read()
                    break
                except Exception:
                    continue

            # Basic documentation analysis
            has_usage = "usage" in readme_content.lower()
            has_installation = "install" in readme_content.lower()
            has_examples = "example" in readme_content.lower()

            doc_score = (
                sum([bool(readme_content), has_usage, has_installation, has_examples])
                / 4.0
            )

            return {
                "readme_length": len(readme_content),
                "has_usage_section": has_usage,
                "has_installation_section": has_installation,
                "has_examples": has_examples,
                "documentation_score": doc_score,
                "readme_content": readme_content[:2000],  # Limit size
            }

        except Exception as e:
            logger.warning(f"Error analyzing documentation: {e}")
            return {"documentation_score": 0.0, "readme_content": ""}

    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            "commit_analysis": {
                "total_commits": 0,
                "recent_commits": 0,
                "avg_commit_frequency": 0,
            },
            "contributor_analysis": {"unique_authors": 0, "bus_factor_score": 0.0},
            "file_analysis": {"total_files": 0, "python_files": 0, "test_files": 0},
            "structure_analysis": {"structure_score": 0.0},
            "documentation_analysis": {
                "documentation_score": 0.0,
                "readme_content": "",
            },
        }

    def cleanup(self):
        """Clean up cache directory."""
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup cache directory: {e}")
