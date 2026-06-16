from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class GitHubError(Exception):
    pass


@dataclass(frozen=True)
class Issue:
    number: int
    title: str
    body: str
    author: str
    html_url: str


class GitHubClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        github = config.get("github") or {}
        token_env = str(github.get("token_env") or "GITHUB_TOKEN")
        self.token = github_token(github)
        self.api_url = str(github.get("api_url") or "https://api.github.com").rstrip("/")
        self.owner = str(github.get("owner") or "")
        self.repo = str(github.get("repo") or "")
        self.timeout = int(github.get("request_timeout_seconds") or 90)
        if not self.token:
            raise GitHubError("Missing GitHub token in ${}".format(token_env))

    def request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        data = None
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer {}".format(self.token),
            "User-Agent": "codex-rag-runner",
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.api_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                text = response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            raise GitHubError("{} {} failed: {}".format(method, path, exc)) from exc
        return json.loads(text) if text.strip() else None

    def list_queued_issues(self) -> List[Issue]:
        github = self.config.get("github") or {}
        label = str(github.get("queued_label") or "codex:queued")
        query = urllib.parse.urlencode({"state": "open", "labels": label, "per_page": int((self.config.get("runner") or {}).get("max_issues_per_poll") or 1)})
        rows = self.request("GET", "/repos/{}/{}/issues?{}".format(self.owner, self.repo, query))
        issues = []
        allowed = set(github.get("allowed_authors") or [])
        for row in rows or []:
            if "pull_request" in row:
                continue
            author = str((row.get("user") or {}).get("login") or "")
            if allowed and author not in allowed:
                continue
            issues.append(Issue(number=int(row["number"]), title=str(row.get("title") or ""), body=str(row.get("body") or ""), author=author, html_url=str(row.get("html_url") or "")))
        return issues

    def add_comment(self, issue_number: int, body: str) -> None:
        self.request("POST", "/repos/{}/{}/issues/{}/comments".format(self.owner, self.repo, issue_number), {"body": body})

    def add_labels(self, issue_number: int, labels: List[str]) -> None:
        clean = [label for label in labels if label]
        if clean:
            self.request("POST", "/repos/{}/{}/issues/{}/labels".format(self.owner, self.repo, issue_number), {"labels": clean})

    def remove_label(self, issue_number: int, label: str) -> None:
        if not label:
            return
        encoded = urllib.parse.quote(label, safe="")
        try:
            self.request("DELETE", "/repos/{}/{}/issues/{}/labels/{}".format(self.owner, self.repo, issue_number, encoded))
        except GitHubError:
            return


def github_token(github: Dict[str, Any]) -> str:
    configured = str(github.get("token") or github.get("github_token") or "").strip()
    if configured:
        return configured
    token_env = str(github.get("token_env") or "GITHUB_TOKEN").strip()
    if looks_like_github_token(token_env):
        return token_env
    return os.environ.get(token_env, "")


def looks_like_github_token(value: str) -> bool:
    text = str(value or "").strip()
    return text.startswith(("github_pat_", "ghp_", "gho_", "ghu_", "ghs_", "ghr_"))
