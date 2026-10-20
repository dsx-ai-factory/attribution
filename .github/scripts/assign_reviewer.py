#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep one primary reviewer on each ready pull request.

Run by .github/workflows/auto-assign-pr.yml. See
.github/reviewer-assignment.md for the process this implements.

Pull request events: once the PR is ready, pick one primary reviewer from
the reviewer team and drop the team's CODEOWNERS request. A reviewer a person
requested wins over a random pick. When someone removes the primary before
they review, pick a replacement. Each primary is recorded in a bot comment,
and the bot never picks an earlier primary again.

Schedule: post one comment per review round when the primary misses the
deadline. For a PR whose author is outside the reviewer team, also post once
when the primary approves.

Slack: the script writes the messages it wants sent to the ``messages`` step
output. The workflow sends each one with the shared dsx-github-actions
slack-notify action. The primary hears about a new pick, a re-request, or an
overdue round. On-call hears about PRs whose author is outside the reviewer
team, and about PRs with no eligible reviewer.

Environment:
    GITHUB_TOKEN: Workflow token. Reads the PR, requests reviewers, comments.
    TEAM_TOKEN: Token that can read the reviewer team and remove its request.
    SLACK_USER_IDS: Optional JSON object that maps GitHub logins to Slack user IDs.
    ONCALL_SLACK: Optional JSON object with the on-call Slack ``channel`` ID and
        user ``group`` ID.
    GITHUB_REPOSITORY, GITHUB_EVENT_NAME, GITHUB_EVENT_PATH, GITHUB_OUTPUT: Set by Actions.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib import error, request
from zoneinfo import ZoneInfo

API_URL = "https://api.github.com"
BOT_LOGIN = "github-actions[bot]"
REVIEWER_TEAM = "attribution"
# Never picked as primary: Ajay Hotchandani and Felix Abecassis.
EXCLUDED_REVIEWERS = {"ajaymh", "flx42"}
PRIMARY_MARKER = re.compile(r"<!-- primary-reviewer: ([A-Za-z0-9-]+) -->")
OVERDUE_MARKER = "<!-- review-overdue: {} -->"
APPROVED_MARKER = "<!-- approved-notice: {} -->"
# Read from priority/P0, priority/P1, or priority/P2 labels. The PR title is
# a Conventional Commit, so it cannot carry a [P1] tag.
REVIEW_DAYS = {"P0": 1, "P1": 3, "P2": 4}
REVIEW_TIMEZONE = ZoneInfo("America/Los_Angeles")
# A plain PR comment is not a review. A submitted review of any kind is.
COMPLETED_REVIEW_STATES = {"approved", "changes_requested", "commented"}


def github(token: str, path: str, method: str = "GET", payload: dict[str, Any] | None = None) -> Any:
    """Call the GitHub REST API.

    Args:
        token: API token.
        path: API path, including any query string.
        method: HTTP method.
        payload: Optional JSON body.

    Returns:
        Decoded JSON response, or None for an empty body.
    """
    data = None if payload is None else json.dumps(payload).encode()
    req = request.Request(
        f"{API_URL}{path}",
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with request.urlopen(req, timeout=60) as response:
            body = response.read()
    except error.HTTPError as exc:
        raise RuntimeError(f"{method} {path} returned HTTP {exc.code}: {exc.read().decode(errors='replace')}") from None
    return json.loads(body) if body else None


def github_pages(token: str, path: str) -> list[dict[str, Any]]:
    """Read every page of a GitHub list endpoint.

    Args:
        token: API token.
        path: API path, optionally with a query string.

    Returns:
        All items from all pages.
    """
    items: list[dict[str, Any]] = []
    separator = "&" if "?" in path else "?"
    page = 1
    while True:
        batch = github(token, f"{path}{separator}per_page=100&page={page}")
        items.extend(batch)
        if len(batch) < 100:
            return items
        page += 1


def event_time(event: dict[str, Any]) -> datetime:
    """Return when a timeline event happened. Reviews use submitted_at.

    Args:
        event: Issue timeline event.

    Returns:
        Timezone-aware event time.
    """
    stamp = event.get("created_at")
    if stamp is None:
        stamp = event["submitted_at"]
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def login(user: dict[str, Any] | None) -> str | None:
    """Return a user's login, or None for a missing user.

    Args:
        user: User object from the API, possibly None.

    Returns:
        The login, or None.
    """
    return None if user is None else user.get("login")


def by_person(event: dict[str, Any]) -> bool:
    """Return True for a timeline event a person made, not the bot or CODEOWNERS.

    Args:
        event: Issue timeline event.

    Returns:
        True if a person is the actor.
    """
    return login(event.get("actor")) not in {None, BOT_LOGIN}


def outside_team(pr: dict[str, Any], team: list[dict[str, Any]]) -> bool:
    """Return True when a person outside the reviewer team opened the PR.

    Args:
        pr: Pull request object.
        team: Reviewer team members.

    Returns:
        True for a non-bot author who is not a team member.
    """
    author = pr["user"]
    return author.get("type") != "Bot" and author["login"] not in {member["login"] for member in team}


def primary_history(timeline: list[dict[str, Any]]) -> list[tuple[str, datetime]]:
    """Return every primary the bot recorded, oldest first.

    Args:
        timeline: Issue timeline events, oldest first.

    Returns:
        (login, recorded at) pairs.
    """
    history = []
    for event in timeline:
        if event["event"] != "commented" or login(event.get("user")) != BOT_LOGIN:
            continue
        match = PRIMARY_MARKER.search(event.get("body") or "")
        if match is not None:
            history.append((match[1], event_time(event)))
    return history


def primary_missing(timeline: list[dict[str, Any]], primary: str) -> bool:
    """Return True unless the primary is still requested or has reviewed.

    A submitted review ends the request without removing the reviewer, so it
    does not count as a removal. A primary with no request at all counts as
    missing, which covers a run that recorded the primary and then failed.

    Args:
        timeline: Issue timeline events, oldest first.
        primary: Primary reviewer login.

    Returns:
        True if the primary needs replacing.
    """
    last = None
    for event in timeline:
        requested = event["event"] in {"review_requested", "review_request_removed"}
        if requested is True and login(event.get("requested_reviewer")) == primary:
            last = event["event"]
        elif event["event"] == "reviewed" and login(event.get("user")) == primary:
            last = "reviewed"
    return last not in {"review_requested", "reviewed"}


def next_primary(
    pr: dict[str, Any], timeline: list[dict[str, Any]], team: list[dict[str, Any]], rng: random.Random
) -> str | None:
    """Return the reviewer to record as the new primary.

    Args:
        pr: Pull request object.
        timeline: Issue timeline events, oldest first.
        team: Reviewer team members.
        rng: Random source for the pick.

    Returns:
        A login, or None while the current primary is still in place.

    Raises:
        LookupError: No eligible reviewer is left.
    """
    author = pr["user"]["login"]
    history = primary_history(timeline)
    if len(history) > 0 and primary_missing(timeline, history[-1][0]) is False:
        return None
    since = history[-1][1] if len(history) > 0 else None
    requested = {user["login"] for user in pr["requested_reviewers"]}
    eligible = {member["login"] for member in team if member["type"] == "User"} - EXCLUDED_REVIEWERS - {author}
    for event in timeline:
        picked = login(event.get("requested_reviewer"))
        if (
            event["event"] == "review_requested"
            and by_person(event) is True
            and picked in requested & eligible
            and (since is None or event_time(event) > since)
        ):
            return picked
    pool = sorted(eligible - requested - {primary for primary, _ in history})
    if len(pool) == 0:
        raise LookupError("No eligible primary reviewer")
    return rng.choice(pool)


def review_deadline(start: datetime, days: int) -> datetime:
    """Add weekdays at the same Pacific wall-clock time. Holidays count as weekdays.

    Args:
        start: When the review round started.
        days: Weekdays allowed.

    Returns:
        The deadline in Pacific time.
    """
    due = start.astimezone(REVIEW_TIMEZONE)
    while days > 0:
        due += timedelta(days=1)
        if due.weekday() < 5:
            days -= 1
    return due


def review_round(pr: dict[str, Any], timeline: list[dict[str, Any]]) -> tuple[str, datetime, datetime] | None:
    """Return the current primary, review round start, and deadline.

    A round starts when the PR is opened ready, marked ready, or a person
    re-requests review from the primary. Replacing the primary keeps the round.

    Args:
        pr: Pull request object.
        timeline: Issue timeline events, oldest first.

    Returns:
        (primary, round start, deadline), or None without an active primary.
    """
    history = primary_history(timeline)
    if len(history) == 0 or primary_missing(timeline, history[-1][0]) is True:
        return None
    primary = history[-1][0]
    start = round_start(pr, timeline)
    return primary, start, review_deadline(start, review_days(pr))


def round_start(pr: dict[str, Any], timeline: list[dict[str, Any]]) -> datetime:
    """Return when the current review round started.

    A re-request counts only when it names whoever was primary at that moment,
    so a later handover keeps the round. A person's own pick is requested
    before the bot records it, so it does not count.

    Args:
        pr: Pull request object.
        timeline: Issue timeline events, oldest first.

    Returns:
        The latest ready time, or the latest re-request of the then-current primary.
    """
    start = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
    primary = None
    for event in timeline:
        if event["event"] == "ready_for_review":
            start = event_time(event)
        elif event["event"] == "commented" and login(event.get("user")) == BOT_LOGIN:
            match = PRIMARY_MARKER.search(event.get("body") or "")
            if match is not None:
                primary = match[1]
        elif (
            event["event"] == "review_requested"
            and by_person(event) is True
            and login(event.get("requested_reviewer")) == primary
        ):
            start = event_time(event)
    return start


def review_days(pr: dict[str, Any]) -> int:
    """Return the review allowance from the PR's priority label.

    Args:
        pr: Pull request object.

    Returns:
        Weekdays allowed. The most urgent label wins, and no label means P1.
    """
    labels = {label["name"] for label in pr["labels"]}
    for priority, days in REVIEW_DAYS.items():
        if f"priority/{priority}" in labels:
            return days
    return REVIEW_DAYS["P1"]


def queue_dm(messages: list[dict[str, str]], user: str, text: str) -> None:
    """Queue a Slack direct message for a GitHub user.

    Args:
        messages: Queue the workflow sends after this script exits.
        user: Recipient's GitHub login.
        text: Plain message text.
    """
    slack_id = json.loads(os.environ.get("SLACK_USER_IDS") or "{}").get(user)
    if slack_id is None:
        print(f"::warning::No Slack user ID for {user} in REVIEWER_SLACK_IDS; skipped the DM")
        return
    messages.append({"channel": slack_id, "text": slack_escape(text)})


def queue_oncall(messages: list[dict[str, str]], text: str) -> None:
    """Queue a Slack post that tags on-call in the team channel.

    Args:
        messages: Queue the workflow sends after this script exits.
        text: Plain message text.
    """
    oncall = json.loads(os.environ.get("ONCALL_SLACK") or "{}")
    if "channel" not in oncall or "group" not in oncall:
        print("::warning::REVIEWER_ONCALL_SLACK is not set; skipped the on-call post")
        return
    messages.append({"channel": oncall["channel"], "text": f"<!subteam^{oncall['group']}> {slack_escape(text)}"})


def slack_escape(text: str) -> str:
    """Escape Slack's control characters.

    PR titles are untrusted, so a title must not render as a link or an @here
    mention in a bot message.

    Args:
        text: Plain message text.

    Returns:
        Text that Slack shows as written.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def overdue_comment(pr: dict[str, Any], timeline: list[dict[str, Any]], now: datetime) -> str | None:
    """Return the primary's overdue comment, or None if nothing is due.

    Args:
        pr: Pull request object.
        timeline: Issue timeline events, oldest first.
        now: Current time.

    Returns:
        Comment body, or None.
    """
    current = None if pr["draft"] is True else review_round(pr, timeline)
    if current is None:
        return None
    primary, start, due = current
    if now <= due:
        return None
    for event in timeline:
        if (
            event["event"] == "reviewed"
            and login(event.get("user")) == primary
            and event.get("state") in COMPLETED_REVIEW_STATES
            and event_time(event) >= start
        ):
            return None
    marker = OVERDUE_MARKER.format(start.isoformat())
    for event in timeline:
        # A person quoting the reminder copies the marker. Only the bot's counts.
        if (
            event["event"] == "commented"
            and login(event.get("user")) == BOT_LOGIN
            and marker in (event.get("body") or "")
        ):
            return None
    return (
        f"{marker}\n@{primary} review overdue. It was due {due:%Y-%m-%d %H:%M %Z}. "
        "Please review, or agree a new date with the author, or hand the review over."
    )


def approval_comment(pr: dict[str, Any], timeline: list[dict[str, Any]]) -> str | None:
    """Return the next-step comment for a primary's approval, or None.

    Args:
        pr: Pull request object.
        timeline: Issue timeline events, oldest first.

    Returns:
        Comment body, or None if there is no new approval to announce.
    """
    current = review_round(pr, timeline)
    if current is None:
        return None
    primary = current[0]
    # A later changes-requested review or a dismissal replaces an approval.
    # A plain comment review does not.
    verdicts = [
        event
        for event in timeline
        if event["event"] == "reviewed"
        and login(event.get("user")) == primary
        and event.get("state") in {"approved", "changes_requested", "dismissed"}
    ]
    if len(verdicts) == 0 or verdicts[-1]["state"] != "approved":
        return None
    marker = APPROVED_MARKER.format(event_time(verdicts[-1]).isoformat())
    for event in timeline:
        if (
            event["event"] == "commented"
            and login(event.get("user")) == BOT_LOGIN
            and marker in (event.get("body") or "")
        ):
            return None
    return f"{marker}\nApproved by @{primary}. A maintainer will merge this or explain the next step."


def assign(token: str, team_token: str, repo: str, event: dict[str, Any]) -> list[dict[str, str]]:
    """Record a primary reviewer if needed and drop the team request.

    Args:
        token: Workflow token.
        team_token: Token that can read the reviewer team.
        repo: owner/name.
        event: Pull request event payload.

    Returns:
        Slack messages for the workflow to send.
    """
    messages: list[dict[str, str]] = []
    number = event["pull_request"]["number"]
    pr = github(token, f"/repos/{repo}/pulls/{number}")
    if pr["state"] != "open":
        return messages
    team = github_pages(team_token, f"/orgs/{repo.split('/')[0]}/teams/{REVIEWER_TEAM}/members")
    # Drafts included: on-call owns the PR before automated reviews finish.
    if event["action"] == "opened" and outside_team(pr, team) is True:
        queue_oncall(
            messages,
            f"New pull request from {pr['user']['login']}: {pr['title']}\n{pr['html_url']}\n"
            "Acknowledge it and make sure it has a reviewer.",
        )
    if pr["draft"] is True:
        return messages
    timeline = github_pages(token, f"/repos/{repo}/issues/{number}/timeline")
    try:
        primary = next_primary(pr, timeline, team, random.Random())
    except LookupError:
        github(
            token,
            f"/repos/{repo}/issues/{number}/comments",
            "POST",
            {"body": "No eligible primary reviewer is left. On-call will arrange coverage."},
        )
        queue_oncall(
            messages, f"No eligible primary reviewer is left: {pr['title']}\n{pr['html_url']}\nArrange coverage."
        )
        return messages
    if primary is not None:
        # Record first. If the request then fails, a re-run sees a primary with
        # no request and picks someone else instead of adding a second primary.
        github(
            token,
            f"/repos/{repo}/issues/{number}/comments",
            "POST",
            {"body": f"<!-- primary-reviewer: {primary} -->\nPrimary reviewer: @{primary}"},
        )
        if primary not in {user["login"] for user in pr["requested_reviewers"]}:
            github(token, f"/repos/{repo}/pulls/{number}/requested_reviewers", "POST", {"reviewers": [primary]})
        # A replacement keeps the current round.
        due = review_deadline(round_start(pr, timeline), review_days(pr))
        queue_dm(
            messages,
            primary,
            f"You are the primary reviewer: {pr['title']}\n{pr['html_url']}\nReview due: {due:%Y-%m-%d %H:%M %Z}.",
        )
    elif event["action"] == "review_requested" and login(event.get("sender")) != BOT_LOGIN:
        current = review_round(pr, timeline)
        # Only a request after the pick starts a new round. A person's own
        # pick was requested before the bot recorded it.
        if (
            current is not None
            and login(event.get("requested_reviewer")) == current[0]
            and current[1] > primary_history(timeline)[-1][1]
        ):
            queue_dm(
                messages,
                current[0],
                f"Review re-requested: {pr['title']}\n{pr['html_url']}\nReview due: {current[2]:%Y-%m-%d %H:%M %Z}.",
            )
    if REVIEWER_TEAM in {team["slug"] for team in pr["requested_teams"]}:
        # The workflow token cannot see org teams, so it cannot edit a team request.
        github(
            team_token,
            f"/repos/{repo}/pulls/{number}/requested_reviewers",
            "DELETE",
            {"reviewers": [], "team_reviewers": [REVIEWER_TEAM]},
        )
    return messages


def remind(token: str, team_token: str, repo: str) -> tuple[bool, list[dict[str, str]]]:
    """Post overdue and approval comments on every open pull request.

    Args:
        token: Workflow token.
        team_token: Token that can read the reviewer team.
        repo: owner/name.

    Returns:
        Whether any pull request failed, and Slack messages for the workflow to send.
    """
    now = datetime.now(REVIEW_TIMEZONE)
    failed = False
    messages: list[dict[str, str]] = []
    team = github_pages(team_token, f"/orgs/{repo.split('/')[0]}/teams/{REVIEWER_TEAM}/members")
    for pr in github_pages(token, f"/repos/{repo}/pulls?state=open"):
        try:
            timeline = github_pages(token, f"/repos/{repo}/issues/{pr['number']}/timeline")
            outside = outside_team(pr, team)
            body = overdue_comment(pr, timeline, now)
            if body is not None:
                # Each comment is its own once-only record, so it goes first.
                github(token, f"/repos/{repo}/issues/{pr['number']}/comments", "POST", {"body": body})
                primary, _, due = review_round(pr, timeline)
                text = f"Review overdue: {pr['title']}\n{pr['html_url']}\nDue: {due:%Y-%m-%d %H:%M %Z}."
                queue_dm(messages, primary, f"{text} Please review it or hand it over.")
                if outside is True:
                    queue_oncall(messages, f"{text} Primary reviewer: {primary}. Make sure someone reviews it.")
            body = None if outside is False else approval_comment(pr, timeline)
            if body is not None:
                github(token, f"/repos/{repo}/issues/{pr['number']}/comments", "POST", {"body": body})
                queue_oncall(
                    messages,
                    f"Approved and waiting to merge: {pr['title']}\n{pr['html_url']}\n"
                    "Merge it, or tell the author the next step.",
                )
        except (RuntimeError, OSError, KeyError, ValueError) as exc:
            print(f"PR #{pr['number']} reminder failed: {exc}", file=sys.stderr)
            failed = True
    return failed, messages


def main() -> int:
    """Handle one pull request event or the hourly schedule.

    Returns:
        Process exit status.
    """
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    if os.environ["GITHUB_EVENT_NAME"] == "schedule":
        failed, messages = remind(token, os.environ["TEAM_TOKEN"], repo)
    else:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
        failed, messages = False, assign(token, os.environ["TEAM_TOKEN"], repo, event)
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"messages={json.dumps(messages)}\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
