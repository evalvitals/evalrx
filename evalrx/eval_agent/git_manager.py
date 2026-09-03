"""Git-native experiment version management for the eval_agent loop.

Every successful diagnosis run is committed on a branch ``eval/{run_id}``;
failed or unresolved runs are reset with ``git reset --hard``.  This enables
``git log`` as a diagnosis journal and easy rollback to a prior state.

Ported from ``researchclaw/experiment/git_manager.py`` with adaptations:
- Branch naming: ``eval/{run_id}`` (was ``experiment/{tag}``)
- Commit format: ``eval({run_id}) cycle {N}: {desc}\\n\\nHypotheses: {json}``
- History grep: ``eval(`` (was ``experiment(``)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentGitManager:
    """Git-native versioning for eval_agent diagnosis runs.

    Every successful run is a commit on ``eval/{run_id}``.  Failed / unresolved
    runs are discarded with ``git reset --hard HEAD``.  Git log becomes the
    experiment journal; ``git checkout eval/{run_id}`` replays any prior run.
    """

    def __init__(self, repo_dir: Path) -> None:
        self.repo_dir: Path = repo_dir
        self._active_branch: str | None = None
        self._original_branch: str | None = self._detect_current_branch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_git_repo(self) -> bool:
        """Return True if *repo_dir* is inside a git repository."""
        result = self._run_git(["rev-parse", "--is-inside-work-tree"])
        return result is not None and result.returncode == 0

    def get_current_branch(self) -> str:
        """Return the active branch name, or '' if detection fails."""
        return self._detect_current_branch() or ""

    def create_experiment_branch(self, run_id: str) -> str:
        """Create and check out branch ``eval/{run_id}``.

        Returns the branch name on success, or '' on failure.
        """
        branch = f"eval/{run_id}"
        result = self._run_git(["checkout", "-b", branch])
        if result is None or result.returncode != 0:
            self._log_git_failure("create_experiment_branch", result)
            return ""
        self._active_branch = branch
        return branch

    def commit_experiment(
        self,
        run_id: str,
        cycle: int,
        hypothesis_statuses: dict[str, str],
        description: str = "",
    ) -> str:
        """Stage all changes and commit.

        Returns the commit hash on success, or '' on failure.
        """
        add_result = self._run_git(["add", "-A"])
        if add_result is None or add_result.returncode != 0:
            self._log_git_failure("git add", add_result)
            return ""

        message = self._format_commit_message(
            run_id=run_id,
            cycle=cycle,
            hypothesis_statuses=hypothesis_statuses,
            description=description,
        )
        commit_result = self._run_git(["commit", "-m", message])
        if commit_result is None or commit_result.returncode != 0:
            self._log_git_failure("git commit", commit_result)
            return ""

        hash_result = self._run_git(["rev-parse", "HEAD"])
        if hash_result is None or hash_result.returncode != 0:
            self._log_git_failure("git rev-parse HEAD", hash_result)
            return ""
        return self._clean_output(hash_result.stdout)

    def discard_experiment(self, run_id: str, reason: str) -> bool:
        """Reset working tree to HEAD (discard uncommitted changes).

        Returns True on success.
        """
        logger.info("Discarding eval run %s: %s", run_id, reason)
        result = self._run_git(["reset", "--hard", "HEAD"])
        if result is None or result.returncode != 0:
            self._log_git_failure("discard_experiment", result)
            return False
        return True

    def get_experiment_history(self) -> list[dict[str, str]]:
        """Return a list of prior eval commits parsed from git log."""
        result = self._run_git(
            ["log", "--oneline", "--fixed-strings", "--grep", "eval("]
        )
        if result is None or result.returncode != 0:
            self._log_git_failure("git log", result)
            return []

        history: list[dict[str, str]] = []
        for line in result.stdout.splitlines():
            parsed = self._parse_experiment_log_line(line)
            if parsed is not None:
                history.append(parsed)
        return history

    def return_to_original_branch(self) -> bool:
        """Switch back to the branch that was active at init time."""
        if not self._original_branch:
            return False
        result = self._run_git(["checkout", self._original_branch])
        if result is None or result.returncode != 0:
            self._log_git_failure("return_to_original_branch", result)
            return False
        self._active_branch = self._original_branch
        return True

    def get_experiment_diff(self) -> str:
        """Return ``git diff --stat`` of uncommitted changes (for logging)."""
        result = self._run_git(["diff", "--stat"])
        if result is None or result.returncode != 0:
            return ""
        return result.stdout.strip()

    def clean_untracked(self) -> bool:
        """Remove untracked files and directories (``git clean -fd``)."""
        result = self._run_git(["clean", "-fd"])
        return result is not None and result.returncode == 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_git(self, args: list[str]) -> subprocess.CompletedProcess[str] | None:
        try:
            logger.debug("git %s", " ".join(args))
            return subprocess.run(
                ["git", *args],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Git operation failed (%s): %s", " ".join(args), exc)
            return None

    @staticmethod
    def _format_commit_message(
        *,
        run_id: str,
        cycle: int,
        hypothesis_statuses: dict[str, str],
        description: str,
    ) -> str:
        statuses_json = json.dumps(hypothesis_statuses, sort_keys=True)
        desc = description or "diagnosis complete"
        return (
            f"eval({run_id}) cycle {cycle}: {desc}\n\n"
            f"Hypotheses: {statuses_json}"
        )

    @staticmethod
    def _clean_output(output: str) -> str:
        return output.strip()

    @staticmethod
    def _parse_experiment_log_line(line: str) -> dict[str, str] | None:
        pattern = re.compile(r"^([0-9a-fA-F]+)\s+eval\(([^)]+)\)\s+cycle\s+(\d+):\s*(.*)$")
        match = pattern.match(line.strip())
        if match is None:
            return None
        commit_hash, run_id, cycle, message = match.groups()
        return {"hash": commit_hash, "run_id": run_id, "cycle": cycle, "message": message}

    @staticmethod
    def _log_git_failure(
        operation: str, result: subprocess.CompletedProcess[str] | None
    ) -> None:
        if result is None:
            logger.warning("Git operation failed for %s", operation)
            return
        stderr = result.stderr.strip()
        if stderr:
            logger.warning("Git operation failed for %s: %s", operation, stderr)
        else:
            logger.warning(
                "Git operation failed for %s with code %s", operation, result.returncode
            )

    def _detect_current_branch(self) -> str | None:
        result = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        if result is None or result.returncode != 0:
            return None
        name = result.stdout.strip()
        return name if name else None
