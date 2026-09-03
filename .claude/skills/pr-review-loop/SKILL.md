---
name: pr-review-loop
description: Drive a pull request through repeated AI code review — re-trigger CodeRabbit and GitHub Copilot, wait for their comments, reply to every comment, fix the ones that are valid, and repeat until both reviewers are quiet or 5 rounds are spent. Use this whenever the user asks to re-trigger, re-run, or re-request review from CodeRabbit and/or Copilot, to loop or iterate on AI/bot review comments, to "get the bots off my PR", to address CodeRabbit or Copilot feedback, or to clean up a PR until the automated reviewers stop reporting findings — even if they do not name the bots or the word "loop".
---

# AI reviewer loop

Iterates CodeRabbit and Copilot review on one pull request until neither has
anything actionable left, capped at **5 rounds**. Each round is: trigger both
reviewers → wait → triage every new comment → fix or rebut → validate → push.

Fixes must follow this repo's rules: `CLAUDE.md`, `CONTRIBUTING.md`
(pre-review checklist, commit subject `module: changes`, Jira refs), and
`docs/rust_instructions.md`. A reviewer suggestion that conflicts with those
documents is **not** valid — rebut it, do not implement it.

## Setup (once)

1. Resolve `OWNER`/`REPO` from `git remote get-url origin`, and the PR number
   from the user's argument or from the current branch
   (`list_pull_requests` filtered by `head`). Stop and ask if the branch has
   no open PR.
2. Read the PR body and diff (`pull_request_read` with `get` and `get_diff`)
   so you can judge comments on their merits rather than pattern-matching.
3. Set `ROUND=1`.

Prefer the GitHub MCP tools (`mcp__github__*`). Where they fall short, use
`gh api`; the recipes below give both.

## Each round

### 1. Mark the boundary, then trigger both reviewers

Record `SINCE` (current UTC time, `date -u +%Y-%m-%dT%H:%M:%SZ`) and the head
SHA **before** triggering. Everything newer than `SINCE` is this round's
feedback; everything older was already handled.

**CodeRabbit** — post a PR comment with `add_issue_comment`
(`gh pr comment $PR --body ...`):

- `@coderabbitai full review` on round 1, and on any later round where the head
  SHA has not moved — CodeRabbit skips a plain `review` when there are no new
  commits to look at.
- `@coderabbitai review` on later rounds after you have pushed fixes, for an
  incremental pass over the new commits.

**Copilot** — `request_copilot_review`, or
`gh api -X POST "repos/$OWNER/$REPO/pulls/$PR/requested_reviewers" -f "reviewers[]=copilot-pull-request-reviewer[bot]"`.

If either call fails (422/403, app not installed, Copilot review not enabled
for the repo), that reviewer is unavailable: say so once and keep going with
the other. If both are unavailable, stop and report — do not spin.

### 2. Wait for the reviews to land

Both bots take minutes, not seconds. Never block on a foreground `sleep`; poll
in the background with `Monitor` and keep the deadline bounded:

```sh
END=$(( $(date +%s) + 900 )); cr=0; cp=0
while :; do
  a=$( { gh api "repos/$OWNER/$REPO/pulls/$PR/reviews" --paginate \
           --jq ".[] | select(.submitted_at > \"$SINCE\") | .user.login";
         gh api "repos/$OWNER/$REPO/issues/$PR/comments" --paginate \
           --jq ".[] | select(.created_at > \"$SINCE\") | .user.login"; } 2>/dev/null | sort -u )
  [ $cr -eq 0 ] && grep -q '^coderabbitai\[bot\]$' <<<"$a" && { echo "coderabbit responded"; cr=1; }
  [ $cp -eq 0 ] && grep -qi '^copilot' <<<"$a" && { echo "copilot responded"; cp=1; }
  [ $cr -eq 1 ] && [ $cp -eq 1 ] && { echo "both responded"; break; }
  [ "$(date +%s)" -ge $END ] && { echo "timeout coderabbit=$cr copilot=$cp"; break; }
  sleep 30
done
```

Without `gh`, call `subscribe_pr_activity` for the PR and schedule a re-check
with `send_later` (~5 minutes) instead of polling. On timeout, proceed with
whatever arrived and note which reviewer never answered.

### 3. Triage every new comment

Collect this round's feedback: `pull_request_read` with `get_review_comments`
(review threads, with their GraphQL thread IDs), `get_reviews` (Copilot's
review body), and `get_comments` (CodeRabbit's summary comment).

Ignore non-actionable noise: CodeRabbit's summary/walkthrough, its collapsed
"nitpick"/"outside diff range" blocks it marked non-blocking, Copilot's
"reviewed N files" preamble, and anything you already addressed in an earlier
round. Also skip comments authored by you.

For each remaining comment, decide and then act — every comment gets a visible
outcome, so a reviewer can see what happened:

- **Valid** → fix the code, then reply on the thread
  (`add_reply_to_pull_request_comment`) saying what you changed, and resolve
  the thread (`resolve_review_thread`) once the fix is pushed.
- **Valid but out of this PR's scope** → reply saying so and why; leave the
  thread open. Do not widen the PR (`CONTRIBUTING.md`: one logical change).
- **Wrong, or against repo convention** → reply with the concrete reason,
  citing the guideline or the code that makes it wrong. Addressing
  `@coderabbitai` in the reply gets you an answer you can argue with; Copilot
  does not converse, so one clear rebuttal is enough. Do not resolve a thread
  just to silence it.

Never disable, skip, or weaken a test to satisfy a comment.

### 4. Validate before pushing

Run what CI runs — warnings are errors:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -Dwarnings
cargo test --workspace
```

Regenerate `api/openapi.json` with `cargo openapi` if the REST API changed;
never hand-edit it. A push that turns CI red costs a whole round, so push only
once these come back clean.

### 5. Commit and push

Commit per `CONTRIBUTING.md`: subject `module: changes`, a body explaining
*why*, and `Fixes:`/`Refs: VECTOR-<n>` where the PR already carries one.

- Branch is yours and the PR is a patch series → fold each fix into the commit
  it belongs to (`git commit --fixup` + `git rebase --autosquash`) so every
  patch stays individually correct, then force-push with `--force-with-lease`.
- Otherwise, or if anyone else may have the branch checked out → add a plain
  follow-up commit and push normally. Never rewrite history on a branch that
  is not yours.

Push with `git push -u origin <branch>`.

### 6. Decide whether to loop

Stop when **both** reviewers have gone quiet:

- CodeRabbit reports `Actionable comments posted: 0`, and
- Copilot's review adds no new comments,

with no unaddressed threads left. Otherwise `ROUND=$((ROUND+1))` and go back to
step 1 — up to `ROUND=5`.

Nothing pushed this round and the same comments coming back means you are not
converging: stop early, and report what is disputed rather than burning rounds.

## Final report

Tell the user, in a few lines: rounds used, how many comments came from each
reviewer, what you fixed, what you pushed back on and why, and anything still
open (including a reviewer that never responded). If CI is still red or the PR
has a merge conflict, say so plainly — the loop is not done until the PR is
green.

Keep posted comments short and factual. End each comment you post on GitHub
with the attribution footer your environment requires, so reviewers can tell
which replies were written by Claude.
