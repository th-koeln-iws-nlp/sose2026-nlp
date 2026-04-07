"""
generate_quiz.py — Generates personalized quiz questions from student code and posts them
to the Course Hub API for every team member individually.

Called from llm-review.yml after the LLM code review step.

Inputs (env vars):
  GEMINI_API_KEY       — Gemini API key
  COURSE_HUB_URL       — Base URL of the deployed Course Hub (no trailing slash)
  COURSE_HUB_API_TOKEN — Bearer token for the Course Hub API
  GITHUB_TOKEN         — Auto-provided by Actions (for collaborators API + issue comments)

Inputs (files written by earlier steps):
  /tmp/student_code.txt  — concatenated student .py files
  /tmp/feedback.md       — LLM code review output

CLI args:
  --repo-owner     GitHub org/user owning the student repo
  --repo-name      Student repo name (e.g. "team01")
  --issue-number   Issue number that triggered the workflow
  --session-number Assignment session number (integer, e.g. 3)
"""
import argparse
import json
import os
import sys

import requests
from google import genai

ADMIN_GITHUB_USERNAMES = {"richardsiegth"}
COURSE_HUB_URL = os.environ["COURSE_HUB_URL"].rstrip("/")
COURSE_HUB_API_TOKEN = os.environ["COURSE_HUB_API_TOKEN"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]

_hub_headers = {"Authorization": f"Bearer {COURSE_HUB_API_TOKEN}"}
_gh_headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}


# ---------------------------------------------------------------------------
# Fetch from Course Hub API
# ---------------------------------------------------------------------------

def fetch_prompt(name: str) -> str:
    resp = requests.get(f"{COURSE_HUB_URL}/api/content/prompt/{name}", headers=_hub_headers)
    if resp.status_code == 404:
        print(f"WARNING: prompt '{name}' not found in Course Hub — using empty string")
        return ""
    resp.raise_for_status()
    return resp.json()["content"]


def fetch_lecture_content(session_number: int) -> str:
    resp = requests.get(f"{COURSE_HUB_URL}/api/content/lecture/{session_number}", headers=_hub_headers)
    if resp.status_code == 404:
        print(f"WARNING: no lecture content for session {session_number} — proceeding without it")
        return ""
    resp.raise_for_status()
    return resp.json()["content"]


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------

def get_team_members(repo_owner: str, repo_name: str) -> list[str]:
    resp = requests.get(
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/collaborators",
        headers=_gh_headers,
    )
    resp.raise_for_status()
    return [c["login"] for c in resp.json() if c["login"] not in ADMIN_GITHUB_USERNAMES]


def post_issue_comment(repo_owner: str, repo_name: str, issue_number: int, body: str) -> None:
    requests.post(
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}/comments",
        json={"body": body},
        headers=_gh_headers,
    ).raise_for_status()


# ---------------------------------------------------------------------------
# Read local files
# ---------------------------------------------------------------------------

def read_file(path: str, default: str = "") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"WARNING: {path} not found — using default")
        return default


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(repo_owner: str, repo_name: str, issue_number: int, session_number: int) -> None:
    # 1. Gather inputs
    student_code = read_file("/tmp/student_code.txt", "# (no code found)")
    llm_review   = read_file("/tmp/feedback.md", "No review available.")

    quiz_system_prompt = fetch_prompt("quiz_system_prompt")
    lecture_content    = fetch_lecture_content(session_number)

    if not quiz_system_prompt:
        print("ERROR: quiz_system_prompt is empty — aborting quiz generation")
        sys.exit(1)

    # 2. Assemble prompt
    user_message = f"""\
## Lecture Content (Session {session_number})
{lecture_content if lecture_content else "(No lecture content available for this session.)"}

---
## Student Code
```python
{student_code[:8000]}
```

---
## Automated Code Review
{llm_review[:3000]}
"""

    # 3. Generate questions with Gemini (once for the whole team)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_message,
        config=genai.types.GenerateContentConfig(
            system_instruction=quiz_system_prompt,
            temperature=0.4,
            response_mime_type="application/json",
        ),
    )

    raw = response.text.strip()
    # Strip markdown fences if model ignores the instruction
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    base_questions = json.loads(raw)
    print(f"Generated {len(base_questions)} questions")

    # 4. Upload for each team member individually
    team_members = get_team_members(repo_owner, repo_name)
    print(f"Team members: {team_members}")

    for member in team_members:
        questions = [
            {
                **q,
                "session_number": session_number,
                "username": member,
                "question_type": "mcq",
            }
            for q in base_questions
        ]
        resp = requests.post(
            f"{COURSE_HUB_URL}/api/quiz",
            json={"questions": questions, "mode": "replace"},
            headers=_hub_headers,
        )
        resp.raise_for_status()
        print(f"Uploaded {len(questions)} questions for {member}")

    # 5. Post issue comment with direct quiz links
    quiz_url = f"{COURSE_HUB_URL}/Quiz?session={session_number}"
    link_lines = [f"- **{m}**: [Open Session {session_number} quiz]({quiz_url})" for m in team_members]

    comment = (
        f"### 📝 Quiz ready — Session {session_number}\n\n"
        f"Your personalized quiz has been generated based on your code and the review above.\n\n"
        + "\n".join(link_lines)
        + "\n\nLog in with your email and password. The link takes you directly to your quiz."
    )
    post_issue_comment(repo_owner, repo_name, issue_number, comment)
    print("Posted quiz comment on issue")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-owner",     required=True)
    parser.add_argument("--repo-name",      required=True)
    parser.add_argument("--issue-number",   required=True, type=int)
    parser.add_argument("--session-number", required=True, type=int)
    args = parser.parse_args()
    main(args.repo_owner, args.repo_name, args.issue_number, args.session_number)
