#!/usr/bin/env python3
"""
Aqwel-Aion - Git Integration
=============================

Git repository operations via GitPython: status, commit history, branches,
diffs, file history. Optional dependency; install with: pip install gitpython.
"""

import os
from typing import List, Dict, Optional, Tuple

try:
    from git import Repo, GitCommandError  # pyright: ignore [reportMissingImports]
    from git.objects.commit import Commit  # pyright: ignore [reportMissingImports]
    from git.refs.remote import RemoteReference  # pyright: ignore [reportMissingImports]
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    class Repo:
        pass
    class GitCommandError:
        pass
    class Commit:
        pass
    class RemoteReference:
        pass

class GitManager:
    """Git repository operations: status, history, branches, diff, file history."""

    def __init__(self, repo_path: str = None):
        """Open repo at repo_path (default: current directory)."""
        self.repo_path = repo_path or os.getcwd()
        self.repo = None
        self._initialize_repo()

    def _initialize_repo(self):
        if not GIT_AVAILABLE:
            raise ValueError("GitPython is not available. Install with: pip install gitpython")
        
        try:
            self.repo = Repo(self.repo_path)
        except Exception as e:
            self.repo = None
            raise ValueError(f"Not a Git repository: {self.repo_path}")
    
    def get_status(self) -> Dict[str, any]:
        """Return current branch, working_dir_clean, staged_files, untracked_files, repo_path."""
        if not self.repo:
            return {"error": "Not a Git repository"}
        try:
            current_branch = self.repo.active_branch.name
            working_dir_clean = not self.repo.is_dirty()
            staged_files = [item.a_path for item in self.repo.index.diff("HEAD")]
            untracked_files = self.repo.untracked_files
            return {
                "current_branch": current_branch,
                "working_dir_clean": working_dir_clean,
                "staged_files": staged_files,
                "untracked_files": untracked_files,
                "repo_path": self.repo_path
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_commit_history(self, max_commits: int = 10) -> List[Dict[str, any]]:
        """Return list of dicts: hash, author, date, message, files_changed (up to max_commits)."""
        if not self.repo:
            return []
        
        try:
            commits = []
            for commit in self.repo.iter_commits('HEAD', max_count=max_commits):
                commits.append({
                    "hash": commit.hexsha[:8],
                    "author": commit.author.name,
                    "date": commit.committed_datetime.isoformat(),
                    "message": commit.message.strip(),
                    "files_changed": len(commit.stats.files)
                })
            return commits
        except Exception as e:
            return [{"error": str(e)}]
    
    def get_branches(self) -> List[Dict[str, any]]:
        """Return list of dicts: name, is_active, last_commit, last_commit_date."""
        if not self.repo:
            return []
        
        try:
            branches = []
            for branch in self.repo.branches:
                branches.append({
                    "name": branch.name,
                    "is_active": branch.name == self.repo.active_branch.name,
                    "last_commit": branch.commit.hexsha[:8],
                    "last_commit_date": branch.commit.committed_datetime.isoformat()
                })
            return branches
        except Exception as e:
            return [{"error": str(e)}]
    
    def get_diff(self, commit_hash: str = None) -> str:
        """Return diff for commit_hash (vs parent) or working directory if commit_hash is None."""
        if not self.repo:
            return "Not a Git repository"
        try:
            if commit_hash:
                commit = self.repo.commit(commit_hash)
                parent = commit.parents[0] if commit.parents else None
                if parent:
                    return self.repo.git.diff(parent.hexsha, commit.hexsha)
                return "Initial commit - no diff available"
            return self.repo.git.diff()
        except Exception as e:
            return f"Error getting diff: {str(e)}"
    
    def get_file_history(self, file_path: str, max_commits: int = 5) -> List[Dict[str, any]]:
        """Return list of commits touching file_path: hash, date, message, author."""
        if not self.repo:
            return []
        
        try:
            commits = []
            for commit in self.repo.iter_commits('HEAD', paths=file_path, max_count=max_commits):
                commits.append({
                    "hash": commit.hexsha[:8],
                    "date": commit.committed_datetime.isoformat(),
                    "message": commit.message.strip(),
                    "author": commit.author.name
                })
            return commits
        except Exception as e:
            return [{"error": str(e)}]

def get_git_status(repo_path: str = None) -> Dict[str, any]:
    """Get Git status for a repository."""
    if not GIT_AVAILABLE:
        return {"error": "GitPython is not available. Install with: pip install gitpython"}
    
    try:
        git_mgr = GitManager(repo_path)
        return git_mgr.get_status()
    except Exception as e:
        return {"error": str(e)}

def get_recent_commits(repo_path: str = None, max_commits: int = 10) -> List[Dict[str, any]]:
    """Return recent commit history for the repository."""
    if not GIT_AVAILABLE:
        return [{"error": "GitPython is not available. Install with: pip install gitpython"}]
    
    try:
        git_mgr = GitManager(repo_path)
        return git_mgr.get_commit_history(max_commits)
    except Exception as e:
        return [{"error": str(e)}]

def list_branches(repo_path: str = None) -> List[Dict[str, any]]:
    """Return all branches in the repository."""
    if not GIT_AVAILABLE:
        return [{"error": "GitPython is not available. Install with: pip install gitpython"}]
    
    try:
        git_mgr = GitManager(repo_path)
        return git_mgr.get_branches()
    except Exception as e:
        return [{"error": str(e)}]
