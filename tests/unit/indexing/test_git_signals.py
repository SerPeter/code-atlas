"""Unit tests for git history mining (indexing/git_signals.py) — throwaway git repos, no mocking."""

from __future__ import annotations

from pathlib import Path

from git import Actor, Repo

from code_atlas.indexing.git_signals import mine_git_signals


def _commit(repo: Repo, files: dict[str, str], *, author: str) -> None:
    """Write/update *files* (name -> content) in the repo's working tree and commit them."""
    for name, content in files.items():
        (Path(repo.working_dir) / name).write_text(content, encoding="utf-8")
    repo.index.add(list(files.keys()))
    actor = Actor(author, f"{author}@example.com")
    repo.index.commit(f"update {', '.join(sorted(files))}", author=actor, committer=actor)


class TestMineGitSignals:
    def test_counts_commits_and_distinct_authors_per_file(self, tmp_path):
        repo = Repo.init(tmp_path)
        _commit(repo, {"a.py": "1"}, author="alice")
        _commit(repo, {"a.py": "2", "b.py": "1"}, author="bob")
        _commit(repo, {"b.py": "2"}, author="alice")

        result = mine_git_signals(tmp_path)

        signals = {s.file_path: s for s in result.file_signals}
        assert result.commits_scanned == 3
        assert signals["a.py"].commit_count == 2
        assert signals["a.py"].author_count == 2  # alice, bob
        assert signals["b.py"].commit_count == 2
        assert signals["b.py"].author_count == 2
        assert signals["a.py"].days_since_last_commit >= 0

    def test_single_author_file_has_author_count_one(self, tmp_path):
        repo = Repo.init(tmp_path)
        _commit(repo, {"solo.py": "1"}, author="alice")
        _commit(repo, {"solo.py": "2"}, author="alice")

        result = mine_git_signals(tmp_path)

        signal = next(s for s in result.file_signals if s.file_path == "solo.py")
        assert signal.commit_count == 2
        assert signal.author_count == 1

    def test_co_change_pairs_dropped_below_threshold(self, tmp_path):
        repo = Repo.init(tmp_path)
        _commit(repo, {"a.py": "1", "b.py": "1"}, author="alice")
        _commit(repo, {"a.py": "2"}, author="alice")

        result = mine_git_signals(tmp_path, co_change_threshold=2)

        assert result.co_change_pairs == ()

    def test_co_change_pairs_included_at_threshold(self, tmp_path):
        repo = Repo.init(tmp_path)
        _commit(repo, {"a.py": "1", "b.py": "1"}, author="alice")
        _commit(repo, {"a.py": "2"}, author="alice")

        result = mine_git_signals(tmp_path, co_change_threshold=1)

        assert len(result.co_change_pairs) == 1
        pair = result.co_change_pairs[0]
        assert {pair.file_a, pair.file_b} == {"a.py", "b.py"}
        assert pair.count == 1

    def test_co_change_only_counts_shared_commits(self, tmp_path):
        """A pair touched together in 2 of 3 commits gets count=2, not 3."""
        repo = Repo.init(tmp_path)
        _commit(repo, {"a.py": "1", "b.py": "1"}, author="alice")
        _commit(repo, {"a.py": "2", "b.py": "2"}, author="alice")
        _commit(repo, {"a.py": "3"}, author="alice")

        result = mine_git_signals(tmp_path, co_change_threshold=1)

        pair = next(p for p in result.co_change_pairs if {p.file_a, p.file_b} == {"a.py", "b.py"})
        assert pair.count == 2

    def test_empty_repo_returns_no_signals(self, tmp_path):
        """A freshly `git init`'d repo with zero commits has an unborn HEAD —
        must not raise, just report nothing mined."""
        Repo.init(tmp_path)

        result = mine_git_signals(tmp_path)

        assert result.commits_scanned == 0
        assert result.file_signals == ()
        assert result.co_change_pairs == ()
