# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the primary reviewer workflow script."""

from __future__ import annotations

import random
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github" / "scripts"))
import assign_reviewer as ar  # noqa: E402

AUTHOR = "author"
TEAM = [{"login": name, "type": "User"} for name in ("author", "alice", "bob", "carol")] + [
    {"login": "robot", "type": "Bot"}
]


def pr(
    draft: bool = False, labels: tuple[str, ...] = (), requested: tuple[str, ...] = (), author: str = AUTHOR
) -> dict:
    """Build a pull request object."""
    return {
        "number": 7,
        "state": "open",
        "user": {"login": author, "type": "User"},
        "draft": draft,
        "title": "fix: thing",
        "html_url": "https://github.com/o/r/pull/7",
        "labels": [{"name": name} for name in labels],
        "created_at": "2026-09-21T16:00:00Z",
        "requested_reviewers": [{"login": name} for name in requested],
        "requested_teams": [],
    }


def fake_github(monkeypatch: pytest.MonkeyPatch, pull: dict, timeline: list[dict]) -> list[str]:
    """Serve one PR and its timeline, and set up Slack IDs and the on-call channel.

    Returns:
        The bodies of comments the script posts.
    """
    comments: list[str] = []

    def github(token: str, path: str, method: str = "GET", payload: dict | None = None) -> dict:
        if path.endswith("/comments"):
            comments.append(payload["body"])
        return pull

    monkeypatch.setattr(ar, "github", github)
    monkeypatch.setattr(
        ar, "github_pages", lambda token, path: timeline if "timeline" in path else TEAM if "teams" in path else [pull]
    )
    monkeypatch.setenv("SLACK_USER_IDS", '{"author": "UA", "alice": "U1", "bob": "U2", "carol": "U3"}')
    monkeypatch.setenv("ONCALL_SLACK", '{"channel": "CHAN", "group": "S1"}')
    return comments


def request_event(reviewer: str, sender: str = AUTHOR) -> dict:
    """Build a review_requested webhook payload."""
    return {
        "action": "review_requested",
        "pull_request": {"number": 7},
        "requested_reviewer": {"login": reviewer},
        "sender": {"login": sender},
    }


def requested(reviewer: str, at: str, actor: str | None = AUTHOR) -> dict:
    """Build a review_requested timeline event."""
    return {
        "event": "review_requested",
        "actor": None if actor is None else {"login": actor},
        "requested_reviewer": {"login": reviewer},
        "created_at": at,
    }


def removed(reviewer: str, at: str, actor: str = AUTHOR) -> dict:
    """Build a review_request_removed timeline event."""
    return {
        "event": "review_request_removed",
        "actor": {"login": actor},
        "requested_reviewer": {"login": reviewer},
        "created_at": at,
    }


def reviewed(reviewer: str, at: str, state: str = "approved") -> dict:
    """Build a reviewed timeline event."""
    return {"event": "reviewed", "user": {"login": reviewer}, "state": state, "submitted_at": at}


def ready(at: str) -> dict:
    """Build a ready_for_review timeline event."""
    return {"event": "ready_for_review", "actor": {"login": AUTHOR}, "created_at": at}


def marker(reviewer: str, at: str) -> dict:
    """Build the bot comment that records a primary."""
    return {
        "event": "commented",
        "user": {"login": ar.BOT_LOGIN},
        "body": f"<!-- primary-reviewer: {reviewer} -->\nPrimary reviewer: @{reviewer}",
        "created_at": at,
    }


def test_first_pick_skips_author_bots_and_requested_specialists() -> None:
    """A random pick never lands on the author, a bot, or a requested reviewer."""
    picks = {ar.next_primary(pr(requested=("carol",)), [], TEAM, random.Random(seed)) for seed in range(50)}
    assert picks == {"alice", "bob"}


def test_excluded_reviewers_are_never_primary() -> None:
    """Excluded team members are skipped, even when the author requests them."""
    team = [*TEAM, {"login": "flx42", "type": "User"}]
    timeline = [requested("flx42", "2026-09-21T16:00:00Z")]
    picks = {ar.next_primary(pr(requested=("flx42",)), timeline, team, random.Random(seed)) for seed in range(50)}
    assert picks == {"alice", "bob", "carol"}


def test_author_choice_wins_over_random_pick() -> None:
    """A reviewer the author requested before ready becomes the primary."""
    timeline = [requested("carol", "2026-09-21T16:00:00Z"), ready("2026-09-21T17:00:00Z")]
    assert ar.next_primary(pr(requested=("carol",)), timeline, TEAM, random.Random(0)) == "carol"


def test_author_choice_outside_the_team_is_a_specialist() -> None:
    """A requested reviewer outside the team stays a specialist, not the primary."""
    timeline = [requested("outsider", "2026-09-21T16:00:00Z"), requested("carol", "2026-09-21T16:00:01Z")]
    assert ar.next_primary(pr(requested=("outsider", "carol")), timeline, TEAM, random.Random(0)) == "carol"


def test_codeowners_team_request_is_not_an_author_choice() -> None:
    """The automatic team request does not count as the author's pick."""
    timeline = [requested("alice", "2026-09-21T16:00:00Z", actor=None)]
    assert ar.next_primary(pr(requested=("alice",)), timeline, TEAM, random.Random(0)) != "alice"


def test_present_primary_is_kept() -> None:
    """No change while the primary is requested or has reviewed."""
    timeline = [requested("alice", "2026-09-21T16:00:00Z", actor=ar.BOT_LOGIN), marker("alice", "2026-09-21T16:00:01Z")]
    assert ar.next_primary(pr(requested=("alice",)), timeline, TEAM, random.Random(0)) is None
    timeline.append(reviewed("alice", "2026-09-22T16:00:00Z"))
    assert ar.next_primary(pr(), timeline, TEAM, random.Random(0)) is None


def test_removed_primary_is_replaced_by_someone_new() -> None:
    """Removing the primary picks a reviewer who was never primary."""
    timeline = [
        marker("alice", "2026-09-21T16:00:01Z"),
        removed("alice", "2026-09-22T16:00:00Z"),
        marker("bob", "2026-09-22T16:00:01Z"),
        removed("bob", "2026-09-23T16:00:00Z"),
    ]
    assert ar.next_primary(pr(), timeline, TEAM, random.Random(0)) == "carol"


def test_author_selected_replacement_is_kept() -> None:
    """A reviewer the author adds with the removal becomes the new primary."""
    timeline = [
        marker("alice", "2026-09-21T16:00:01Z"),
        removed("alice", "2026-09-22T16:00:00Z"),
        requested("carol", "2026-09-22T16:00:00Z"),
    ]
    assert ar.next_primary(pr(requested=("carol",)), timeline, TEAM, random.Random(0)) == "carol"


def test_empty_pool_raises() -> None:
    """next_primary raises when nobody eligible is left."""
    timeline = [
        marker("alice", "2026-09-21T16:00:01Z"),
        marker("bob", "2026-09-21T16:00:02Z"),
        marker("carol", "2026-09-21T16:00:03Z"),
        removed("carol", "2026-09-22T16:00:00Z"),
    ]
    with pytest.raises(LookupError):
        ar.next_primary(pr(), timeline, TEAM, random.Random(0))


def test_deadline_skips_weekends_at_same_pacific_time() -> None:
    """A Friday P1 round is due the next Wednesday at the same time."""
    due = ar.review_deadline(datetime.fromisoformat("2026-09-25T17:00:00+00:00"), 3)
    assert due.isoformat() == "2026-09-30T10:00:00-07:00"


def test_overdue_comment_once_per_round() -> None:
    """The author gets one comment after the deadline, and none after a review."""
    timeline = [
        ready("2026-09-21T16:00:00Z"),
        marker("alice", "2026-09-21T16:00:01Z"),
        requested("alice", "2026-09-21T16:00:02Z", actor=ar.BOT_LOGIN),
    ]
    on_time = datetime.fromisoformat("2026-09-24T15:59:00+00:00")
    late = datetime.fromisoformat("2026-09-24T16:01:00+00:00")
    assert ar.overdue_comment(pr(), timeline, on_time) is None
    body = ar.overdue_comment(pr(), timeline, late)
    assert body is not None and body.startswith("<!-- review-overdue: 2026-09-21T16:00:00+00:00 -->\n@alice")
    quoted = {"event": "commented", "user": {"login": AUTHOR}, "body": f"> {body}"}
    assert ar.overdue_comment(pr(), [*timeline, quoted], late) == body
    posted = {"event": "commented", "user": {"login": ar.BOT_LOGIN}, "body": body}
    assert ar.overdue_comment(pr(), [*timeline, posted], late) is None
    assert ar.overdue_comment(pr(), [*timeline, reviewed("alice", "2026-09-24T16:00:00Z", "commented")], late) is None
    assert ar.overdue_comment(pr(draft=True), timeline, late) is None


def test_priority_and_rerequest_set_the_deadline() -> None:
    """A re-request starts a new round with the priority label's allowance."""
    timeline = [
        ready("2026-09-21T16:00:00Z"),
        marker("alice", "2026-09-21T16:00:01Z"),
        reviewed("alice", "2026-09-21T18:00:00Z", "changes_requested"),
        requested("alice", "2026-09-22T16:00:00Z"),
    ]
    _, start, due = ar.review_round(pr(labels=("priority/P0",)), timeline)
    assert start.isoformat() == "2026-09-22T16:00:00+00:00"
    assert due.isoformat() == "2026-09-23T09:00:00-07:00"


def test_replacement_keeps_the_deadline() -> None:
    """Swapping the primary, even to the author's pick, keeps the round start."""
    timeline = [
        ready("2026-09-21T16:00:00Z"),
        marker("alice", "2026-09-21T16:00:01Z"),
        removed("alice", "2026-09-22T16:00:00Z"),
        requested("carol", "2026-09-22T16:00:00Z"),
        marker("carol", "2026-09-22T16:00:01Z"),
    ]
    primary, start, _ = ar.review_round(pr(requested=("carol",)), timeline)
    assert (primary, start.isoformat()) == ("carol", "2026-09-21T16:00:00+00:00")
    assert ar.review_round(pr(), timeline[:3]) is None


def test_pr_opened_ready_starts_at_creation() -> None:
    """Without a ready event, the round starts when the PR was opened."""
    timeline = [marker("alice", "2026-09-21T16:00:01Z"), requested("alice", "2026-09-21T16:00:02Z", actor=ar.BOT_LOGIN)]
    _, start, _ = ar.review_round(pr(), timeline)
    assert start.isoformat() == "2026-09-21T16:00:00+00:00"


def test_recorded_primary_without_a_request_is_replaced() -> None:
    """A run that recorded a primary but failed to request them picks someone else on re-run."""
    timeline = [marker("alice", "2026-09-21T16:00:01Z")]
    assert ar.next_primary(pr(), timeline, TEAM, random.Random(0)) in {"bob", "carol"}


def test_most_urgent_priority_label_wins() -> None:
    """No label means P1. With several labels, the shortest allowance applies."""
    assert ar.review_days(pr()) == 3
    assert ar.review_days(pr(labels=("priority/P2", "bug"))) == 4
    assert ar.review_days(pr(labels=("priority/P2", "priority/P0"))) == 1


def test_new_primary_gets_one_dm(monkeypatch: pytest.MonkeyPatch) -> None:
    """The author's pick becomes primary and is told the deadline once."""
    timeline = [requested("carol", "2026-09-21T16:00:00Z")]
    pull = pr(requested=("carol",))
    fake_github(monkeypatch, pull, timeline)
    assert ar.assign("t", "t", "o/r", request_event("carol")) == [
        {
            "channel": "U3",
            "text": "You are the primary reviewer: fix: thing\nhttps://github.com/o/r/pull/7\n"
            "Review due: 2026-09-24 09:00 PDT.",
        }
    ]
    # The same request, seen again once the pick is recorded, is not a re-request.
    fake_github(monkeypatch, pull, [*timeline, marker("carol", "2026-09-21T16:00:01Z")])
    assert ar.assign("t", "t", "o/r", request_event("carol")) == []


def test_author_rerequest_dms_the_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    """A re-request after feedback starts a new round and tells the primary."""
    timeline = [
        ready("2026-09-21T16:00:00Z"),
        marker("alice", "2026-09-21T16:00:01Z"),
        reviewed("alice", "2026-09-21T18:00:00Z", "changes_requested"),
        requested("alice", "2026-09-22T16:00:00Z"),
    ]
    fake_github(monkeypatch, pr(requested=("alice",)), timeline)
    messages = ar.assign("t", "t", "o/r", request_event("alice"))
    assert [(m["channel"], m["text"].splitlines()[0], m["text"].splitlines()[-1]) for m in messages] == [
        ("U1", "Review re-requested: fix: thing", "Review due: 2026-09-25 09:00 PDT.")
    ]
    # A specialist request on the same PR sends nothing.
    fake_github(monkeypatch, pr(requested=("alice", "bob")), [*timeline, requested("bob", "2026-09-22T17:00:00Z")])
    assert ar.assign("t", "t", "o/r", request_event("bob")) == []


def test_overdue_reminder_dms_the_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    """The primary gets a DM with the overdue comment. On-call is not told about a team member's PR."""
    timeline = [ready("2020-01-06T16:00:00Z"), marker("alice", "2020-01-06T16:00:01Z")]
    timeline.append(requested("alice", "2020-01-06T16:00:02Z", actor=ar.BOT_LOGIN))
    fake_github(monkeypatch, pr(), timeline)
    failed, messages = ar.remind("t", "t", "o/r")
    assert failed is False
    assert [(m["channel"], m["text"].splitlines()[0]) for m in messages] == [("U1", "Review overdue: fix: thing")]


def test_user_without_slack_id_gets_no_dm(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing mapping skips the DM instead of failing the run."""
    monkeypatch.delenv("SLACK_USER_IDS", raising=False)
    messages: list[dict[str, str]] = []
    ar.queue_dm(messages, "alice", "hi")
    assert messages == []


def test_dm_text_cannot_inject_slack_links(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hostile title shows as plain text, not a link or a mention."""
    monkeypatch.setenv("SLACK_USER_IDS", '{"alice": "U1"}')
    messages: list[dict[str, str]] = []
    ar.queue_dm(messages, "alice", "fix: <https://evil.example|Approve here> & <!here>")
    assert messages[0]["text"] == "fix: &lt;https://evil.example|Approve here&gt; &amp; &lt;!here&gt;"


def test_team_member_can_hand_the_review_over() -> None:
    """A teammate who removes the primary and requests someone else sets the new primary."""
    timeline = [
        marker("alice", "2026-09-21T16:00:01Z"),
        requested("carol", "2026-09-22T16:00:00Z", actor="bob"),
        removed("alice", "2026-09-22T16:00:01Z", actor="bob"),
    ]
    assert ar.next_primary(pr(requested=("carol",)), timeline, TEAM, random.Random(0)) == "carol"


def test_teammate_rerequest_starts_a_new_round() -> None:
    """A re-request by anyone, not only the author, starts a new round."""
    timeline = [
        ready("2026-09-21T16:00:00Z"),
        marker("alice", "2026-09-21T16:00:01Z"),
        reviewed("alice", "2026-09-21T18:00:00Z", "changes_requested"),
        requested("alice", "2026-09-22T16:00:00Z", actor="bob"),
    ]
    _, start, _ = ar.review_round(pr(), timeline)
    assert start.isoformat() == "2026-09-22T16:00:00+00:00"


def test_outside_pr_opened_tags_oncall(monkeypatch: pytest.MonkeyPatch) -> None:
    """On-call hears about a new PR from outside the team, even a draft. Team and bot PRs stay quiet."""
    opened = {"action": "opened", "pull_request": {"number": 7}, "sender": {"login": "outsider"}}
    fake_github(monkeypatch, pr(draft=True, author="outsider"), [])
    assert ar.assign("t", "t", "o/r", opened) == [
        {
            "channel": "CHAN",
            "text": "<!subteam^S1> New pull request from outsider: fix: thing\nhttps://github.com/o/r/pull/7\n"
            "Acknowledge it and make sure it has a reviewer.",
        }
    ]
    fake_github(monkeypatch, pr(draft=True), [])
    assert ar.assign("t", "t", "o/r", opened) == []
    bot = pr(draft=True, author="dependabot[bot]")
    bot["user"]["type"] = "Bot"
    fake_github(monkeypatch, bot, [])
    assert ar.assign("t", "t", "o/r", opened) == []


def test_no_eligible_reviewer_asks_oncall(monkeypatch: pytest.MonkeyPatch) -> None:
    """When nobody is left, on-call arranges coverage and the run does not fail."""
    timeline = [
        marker("alice", "2026-09-21T16:00:01Z"),
        marker("bob", "2026-09-21T16:00:02Z"),
        marker("carol", "2026-09-21T16:00:03Z"),
        removed("carol", "2026-09-22T16:00:00Z"),
    ]
    comments = fake_github(monkeypatch, pr(), timeline)
    removal = {"action": "review_request_removed", "pull_request": {"number": 7}, "sender": {"login": AUTHOR}}
    messages = ar.assign("t", "t", "o/r", removal)
    assert comments == ["No eligible primary reviewer is left. On-call will arrange coverage."]
    assert [(m["channel"], m["text"].splitlines()[-1]) for m in messages] == [("CHAN", "Arrange coverage.")]


def test_outside_pr_overdue_and_approval_tag_oncall(monkeypatch: pytest.MonkeyPatch) -> None:
    """On-call hears about an overdue round and, once, about the primary's approval."""
    timeline = [ready("2020-01-06T16:00:00Z"), marker("alice", "2020-01-06T16:00:01Z")]
    timeline.append(requested("alice", "2020-01-06T16:00:02Z", actor=ar.BOT_LOGIN))
    fake_github(monkeypatch, pr(author="outsider"), timeline)
    _, messages = ar.remind("t", "t", "o/r")
    assert [m["channel"] for m in messages] == ["U1", "CHAN"]
    assert messages[1]["text"].endswith("Primary reviewer: alice. Make sure someone reviews it.")

    timeline.append(reviewed("alice", "2020-01-07T16:00:00Z"))
    comments = fake_github(monkeypatch, pr(author="outsider"), timeline)
    _, messages = ar.remind("t", "t", "o/r")
    assert comments == [
        (
            "<!-- approved-notice: 2020-01-07T16:00:00+00:00 -->\n"
            "Approved by @alice. A maintainer will merge this or explain the next step."
        )
    ]
    assert [(m["channel"], m["text"].splitlines()[0]) for m in messages] == [
        ("CHAN", "<!subteam^S1> Approved and waiting to merge: fix: thing")
    ]
    posted = {"event": "commented", "user": {"login": ar.BOT_LOGIN}, "body": comments[0]}
    fake_github(monkeypatch, pr(author="outsider"), [*timeline, posted])
    assert ar.remind("t", "t", "o/r") == (False, [])


def test_oncall_post_escapes_text_but_keeps_the_mention(monkeypatch: pytest.MonkeyPatch) -> None:
    """The on-call tag renders, and a hostile title does not."""
    monkeypatch.setenv("ONCALL_SLACK", '{"channel": "CHAN", "group": "S1"}')
    messages: list[dict[str, str]] = []
    ar.queue_oncall(messages, "fix: <!here>")
    assert messages == [{"channel": "CHAN", "text": "<!subteam^S1> fix: &lt;!here&gt;"}]
    monkeypatch.delenv("ONCALL_SLACK")
    ar.queue_oncall(messages, "skipped")
    assert len(messages) == 1


def test_handover_after_rerequest_keeps_the_new_round() -> None:
    """A handover after a re-request keeps the re-request's round, not the ready time."""
    timeline = [
        ready("2026-09-21T16:00:00Z"),
        marker("alice", "2026-09-21T16:00:01Z"),
        reviewed("alice", "2026-09-21T18:00:00Z", "changes_requested"),
        requested("alice", "2026-09-24T16:00:00Z"),
        requested("bob", "2026-09-25T16:00:00Z", actor="carol"),
        removed("alice", "2026-09-25T16:00:01Z", actor="carol"),
        marker("bob", "2026-09-25T16:00:02Z"),
    ]
    primary, start, _ = ar.review_round(pr(requested=("bob",)), timeline)
    assert (primary, start.isoformat()) == ("bob", "2026-09-24T16:00:00+00:00")


def test_superseded_approval_is_not_announced() -> None:
    """A later changes-requested review or dismissal cancels the approval notice. A comment does not."""
    timeline = [
        ready("2026-09-21T16:00:00Z"),
        marker("alice", "2026-09-21T16:00:01Z"),
        reviewed("alice", "2026-09-22T16:00:00Z"),
        reviewed("alice", "2026-09-22T17:00:00Z", "commented"),
    ]
    assert ar.approval_comment(pr(), timeline) is not None
    assert (
        ar.approval_comment(pr(), [*timeline, reviewed("alice", "2026-09-22T18:00:00Z", "changes_requested")]) is None
    )
    assert ar.approval_comment(pr(), [*timeline, reviewed("alice", "2026-09-22T18:00:00Z", "dismissed")]) is None
