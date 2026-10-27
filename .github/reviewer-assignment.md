# Pull request reviewer assignment

Each pull request (PR) has one primary reviewer and no standing backups. The
author is the assignee. Automation picks the primary once the PR is ready for
review. The team, not the author, makes sure every PR has a reviewer.

## Roles

**PR author**

- Open the PR as a draft with a priority label (`priority/P0`, `priority/P1`, or
  `priority/P2`). For changes over 500 lines, write an RFC first.
- Keep the PR in draft until automated reviews finish. Fix each finding or explain
  why no change is needed, and resolve every automated thread.
- Mark the PR ready for review. The human review deadline starts then.
- After addressing feedback, re-request review from the primary.
- Merge after required approvals and checks pass, or wait for a maintainer to
  merge.

**Primary reviewer**

- Review the full change and submit approval or actionable feedback by the deadline.
- Aim for a complete first pass. Later rounds may still surface new issues.
- Keep requests within the PR's stated purpose. Do not ask for extra features.
- If the deadline is not achievable, agree a new one with the author.
- If you will be away, hand the review over: remove yourself as a reviewer, and
  automation picks a replacement.
- Request specialists as extra reviewers when you need help.
- Own the review until the PR is merged, closed, or handed over.

**On-call engineer**

- Acknowledge each PR the bot announces in the team's Slack channel, and make
  sure it has a reviewer, even while automated reviews are still running.
- Arrange coverage when no eligible reviewer is left or a review is overdue.
- After approval, merge the PR or tell the author the next step.

Any team member can re-request a review or hand it over on the author's behalf.

## Review schedule

| Priority | Review due |
|----------|------------|
| P0 | 1 weekday |
| P1 | 3 weekdays |
| P2 | 4 weekdays |

A PR without a priority label uses P1. If a PR has several priority labels, the
most urgent one applies. Deadlines count weekdays at the same Pacific wall-clock
time. Company holidays count as weekdays.

A round starts in three cases. The first is when the author opens a PR that is
not a draft. The second is when the author marks the PR ready. The third is when
anyone re-requests review from the primary. Each round gets the same allowance.
Replacing the primary keeps the current deadline.

A round is complete when the primary submits a review: approve, request changes,
or comment. A plain PR comment does not count.

## What the automation does

`.github/workflows/auto-assign-pr.yml` runs `.github/scripts/assign_reviewer.py`.

- When a PR opens, the author becomes the assignee. If the author is not on
  the `attribution` team, the bot posts the PR to the team's Slack channel and
  tags on-call. Drafts are included.
- When a PR is ready, the bot picks one primary from the `attribution` team and
  requests their review. It skips the author, anyone already requested, and
  the members in `EXCLUDED_REVIEWERS` in the script. If someone already
  requested an eligible team member, that person becomes the primary instead.
  The bot records the primary in a PR comment and removes the team's CODEOWNERS
  request, so only the primary is notified.
- When anyone removes the primary before they review, the bot picks a
  replacement. It never re-picks an earlier primary on its own. If someone
  requested another team member since the primary was picked, that person
  becomes the primary. Removing a specialist changes nothing.
- If no eligible reviewer is left, the bot comments and asks on-call to arrange
  coverage.
- Every hour, the bot comments once per overdue round and tags the primary.
  Drafts get no overdue comments. If the author is not on the team, the bot
  also tells on-call.
- Every hour, for a PR whose author is not on the team, the bot comments once
  when the primary approves and asks on-call to merge or explain the next step.
- The bot sends Slack messages through the shared `slack-notify` action from
  `dsx-ai-factory/dsx-github-actions`. The primary gets a direct message when
  picked, when review is re-requested, and when a round is overdue. Each
  message includes the deadline.

## Setup

- `PR_ASSIGNER_TOKEN` secret: looks up the team and removes the team request.
  It needs read access to the organization's teams and write access to pull
  requests in this repository (`Pull requests: write` for a fine-grained token).
- `SLACK_BOT_TOKEN` secret: Slack bot token (`xoxb-...`) with the `chat:write`
  scope. Invite the bot to the on-call channel.
- `REVIEWER_SLACK_IDS` repository variable: a JSON object that maps each GitHub
  login to a Slack member ID, for example `{"octocat": "U0123ABCD"}`. The bot
  skips the message, with a warning in the run log, for a login that is not in
  the map.
- `REVIEWER_ONCALL_SLACK` repository variable: a JSON object with the on-call
  channel ID and user group ID, for example
  `{"channel": "C0123ABCD", "group": "S0123ABCD"}`. Without it, the bot skips
  on-call messages with a warning in the run log.

## Recovery

- The primary is whoever the latest `<!-- primary-reviewer: ... -->` bot comment
  names. Do not delete these comments.
- Failed assignment: read the log, fix the cause, and re-run the job. A re-run
  reads the live PR, so it never adds a second primary.
- Failed Slack message: use "Re-run failed jobs". This re-sends only the failed
  messages and does not pick a new primary.
- To stop Slack messages, delete the `REVIEWER_SLACK_IDS` and
  `REVIEWER_ONCALL_SLACK` variables. Assignment and comments continue.
- To stop automation, disable the workflow in the Actions tab. On-call then
  requests a primary by hand for each new PR.
