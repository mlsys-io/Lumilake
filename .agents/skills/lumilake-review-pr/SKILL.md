---
name: lumilake-review-pr
description: >-
  Review a Lumilake PR by number, owner/repo#N, or URL; follow up after fixes;
  or self-check a local branch before opening a PR. Use for PR reviews,
  re-checking an author's new commits, and branch checks. Optionally post
  findings as a pending GitHub review when explicitly asked.
---

# Lumilake PR review

Review critically for correctness, coding style, and concrete maintenance costs.
Verify findings; zero findings is valid. Keep the review read-only unless the
user explicitly requests an action described below.

## Modes

- **Fresh PR:** a number, `owner/repo#N`, or PR URL selects that PR and repository.
- **Branch:** a branch name, or no argument, selects the named/current branch.
  Resolve its PR with `gh pr view <branch>`; if none exists, self-check against
  `main`. Distinguish a missing PR from authentication or network failure.
- **Follow-up:** when asked to re-check a reviewed PR, use the follow-up section.
- **Posting:** only an explicit request enables the pending-review section.

## 1. Resolve and check out

Record `git status --short`, the original branch (or detached SHA), and the
resolved repository. Pass `--repo <owner/repo>` on PR commands; do not discard
repository identity from a URL or `owner/repo#N`.

For a PR, require a clean worktree before checkout. Fetch metadata with
`gh pr view <N> --repo <owner/repo> --json number,title,body,baseRefName,baseRefOid,headRefName,headRefOid,url,state,isDraft,commits`.
Use `gh pr checkout <N> --repo <owner/repo>`, fetch the base, and record the
reviewed base/head SHAs. Honor a supplied historical head instead of reviewing
today's head. Never overwrite local changes or reset a diverged branch.

For local self-check, stay on the current branch, including when it is dirty.
Use `base=$(git merge-base main HEAD)`, `git log "$base"..HEAD`, and
`git diff "$base"` to cover committed changes plus staged and unstaged changes.
Read relevant untracked files from status separately. To inspect another local
branch, require a clean tree before switching; then apply the same procedure.
If there is nothing to review, report that and stop.

## 2. Understand intent

Read the PR body, or infer intent from commit messages and the diff when absent;
state any inference. A local check skips PR-body checks and checks commit
subjects against [CONTRIBUTING.md](../../../CONTRIBUTING.md).

Read [AGENTS.md](../../../AGENTS.md) and the routed docs relevant to the change,
including [ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the package map,
runtime flow and storage model, and [CODE_STYLE.md](../../../docs/CODE_STYLE.md)
for Python, logging, deploy and test rules. Check the title type and
`[BREAKING]` prefix, DCO sign-off, a refreshed `uv.lock` after a dependency
change, and logical commits against CONTRIBUTING.md; use
[CODEOWNERS](../../../.github/CODEOWNERS) for reviewer suggestions.

For PRs, compare the body with
[PULL_REQUEST_TEMPLATE.md](../../../.github/PULL_REQUEST_TEMPLATE.md): Purpose,
Changes, Design, Test Plan, Test Result, and the pre-submission checklist. Boxes
are ticked only for checks actually run; test claims must match the diff and
available evidence. A fix outside the stated purpose is a scope finding even
when correct.

## 3. Read the changes

Read the entire base-to-head PR diff (`gh pr diff <N> --repo <owner/repo>`), or
the pinned historical equivalent `git diff <merge-base> <reviewed-head>`.
Review every hunk, including inherited and merged-in commits. Read each changed
document in full against the base: check repeated sections/examples, dropped
behavior, and whether prose still introduces its example. Inspect merge
resolutions for omissions or duplication. For large changes, use a todo list by
logical area; finish every area before reporting. Read full source files where
surrounding context affects behavior.

**Correctness:** trace logic, defaults, boundary inputs, async exception paths,
races, schema/API contracts, updated callers, and dead or unreachable code.
Read installed dependency source or primary documentation for external APIs,
FlowMesh included; check each execution path preserves supported options and
result semantics.
- A runtime graph change must keep the FlowMesh task spec it emits valid for
  the FlowMesh version the server targets: result paths, dependency edges and
  task options read by FlowMesh.
- Job lifecycle: cancelling a job cancels the underlying runtime request, and
  job records, runtime artifacts and intermediate outputs go through the archive
  layer, never a local result directory.
- Inputs and data retrieval route through lumid-data-app with the effective
  token contract in ARCHITECTURE.md's storage model.
- Optional dependencies must remain optional at import time, including
  annotations and fallback names when an import fails.

Inspect whether tests exercise the reachable failure and meaningful outputs
rather than mirror the implementation. Rank rare races by real likelihood and
impact; keep fixes proportional.

**Organization and style:** apply CODE_STYLE.md and the established module pattern.
- Respect the package map: `lumilake_server` ships only in the server image;
  `lumilake` (SDK), `lumilake_cli`, `lumilake_deploy` and `lumilake_hook` are
  separate distributions. Flag an import a distribution's dependencies do not
  cover, and logic placed in the wrong package.
- Cluster management goes through the Lumilake CLI, SDK and deploy helpers;
  one-shot container operations use the Docker SDK wrapper in
  `lumilake_deploy`. Flag raw HTTP or Docker calls where a helper exists.
- Flag untyped dicts, dynamic `setattr`/`getattr`, `hasattr` control flow, and
  `.get()` on keys known to exist where a typed model or explicit field fits;
  Pydantic v2 models for structured data crossing a boundary.
- Flag `print()`, `except` blocks that translate undocumented cases or hide
  errors behind `None`/`False`, defaults duplicated outside the env or profile
  layer, `del arg` for unused parameters, unneeded keyword-only `*`, inline or
  `importlib` imports without a circular-import or heavy-dependency reason, and
  `from __future__ import annotations` without need.
- Flag a `# type: ignore` without a specific code, or one where an ordinary
  typing fix exists; a missing-dependency type error is fixed with the
  dependency or stub. Check ruff's enabled rules before accepting `# noqa`.
- Flag unused parameters/types, dead imports, premature abstractions, unclear
  names, extension points for services outside the repository, and
  compatibility layers (aliases, re-exports, stubs, dual contracts) for
  unreleased behavior.
- Flag comments that restate code, narrate the task, reference the current PR,
  or leave source-history breadcrumbs ("previously", "replacement for").
- Nest sibling branches that repeat a condition; check numbered sequences for
  gaps or ordering errors. Compare sibling declarations for inconsistent flags
  or settings. Deduplicate repeated blocks into helpers, and require a mutation
  to justify a defensive copy.
- Sweep renamed/split concepts through source, tests, docs, annotations, logs,
  examples and `.env.example`. Read each occurrence and verify both leftover and
  incorrectly changed uses; a search count alone does not prove consistency.

**Others:** inspect security at trust boundaries (tokens, hooks, plugin
loading, routes), hot-path regressions, coverage, doc fit, and scope. Docs
change in the same PR as behavior, per AGENTS.md's routing:
- [API.md](../../../docs/API.md) for routes and responses,
  [SDK.md](../../../docs/SDK.md) and [CLI.md](../../../docs/CLI.md) for client
  and command behavior, [WORKFLOWS.md](../../../docs/WORKFLOWS.md) and
  [OPS.md](../../../docs/OPS.md) for workflow formats and operations,
  [ENV.md](../../../docs/ENV.md) with `.env.example` for deploy-time variables,
  and [PLUGINS.md](../../../docs/PLUGINS.md) for hooks.
- Runnable examples in `examples/templates/` and `examples/plugins/` must still
  match the behavior they show.
- Describe current behavior without changelog voice ("now", "no longer",
  "rather than"), and without hardcoded file-path examples that will drift.

## 4. Verify every finding

Treat every candidate finding as a hypothesis, not a fact. Before it earns a
place in the report, confirm it against the actual code — a grep hit or a diff
hunk is a lead, not proof.

- **Read the real source, not just the diff.** Open the changed file and its
  surrounding context. Many "bugs" evaporate once you see the lines the diff
  didn't show (an early return above, a guard below, a default set elsewhere).
- **Trace the call sites — read them, don't just grep.** For a signature change,
  renamed symbol, changed return type, or any contract change, locate callers,
  then read each one and confirm it is genuinely broken — not already updated
  in this PR, not guarded, not unreachable. A grep count is not a finding.
- **Double-check every condition for inversion.** Walk the true and false
  branches explicitly and confirm the polarity is actually wrong. Inverted
  comparisons, flipped guards, swapped early returns, De Morgan rewrites:
  re-read the condition and state the correct logic; if correct, drop it.
- **Confirm the failing path is reachable.** An edge case sitting behind a guard
  the diff also adds is not a bug.

Drop findings that fail verification. Survivors cite a reachable trigger,
`path:line`, impact, and the smallest fix. Remaining uncertainty is an explicit
question, not a defect. Pure taste is not a finding; an established convention
or a concrete organization, consistency, or doc-fit cost is.

Do not run the suite unless asked. When asked, use CONTRIBUTING.md's commands
(`uv run pytest tests/`, `uv run pre-commit run --all-files`). A red test is a
finding even if unrelated to the diff (CODE_STYLE.md).

## 5. Report

Use this order:

    **Verdict:** Ready to merge / Needs changes / Major rework required

    **Executive summary:** 1–3 sentences: purpose and reason for verdict.

    ### Correctness
    1. <verified issue, trigger, impact, `path:line`>.
       *Recommended action:* <one concrete fix>.
    ### Coding style
    2. ...
    ### Others
    3. ...

Number findings continuously across sections and omit empty sections. If none
survive, say so under the summary. Ready means no required action (any remaining
suggestion has an explicitly optional Recommended action); Needs changes means
at least one actionable issue; Major rework means fundamental design/contract
failure or a diff that misses its purpose. Include verification limits.

## 6. Leave the workspace usable

State the current branch and how to return if checkout moved it. When invoked
by another skill, restore the original branch/detached SHA before returning.
Leave files and the index untouched.

## Follow-up review

Fetch and fast-forward the PR branch before reading; stop on divergence. Honor
historical targets. Record the last reviewed SHA from the prior review or ask
with the ask-question tool if available when it cannot be established reliably.
Fetch all review threads and replies with paginated GraphQL `reviewThreads` and
nested `comments`; include resolved/outdated threads and original anchors.

Check each reply's claim against source at the reviewed head, not reply text or
resolution status. Report **addressed / partly addressed / not addressed** per
thread with evidence. Inspect prior defects even outside the new diff; review
only commits after the last reviewed SHA for new issues, including merge effects.
Re-check the PR body for stale claims. Put thread dispositions before new
findings, which use step 5's numbering and verdict rules. Finish with step 6.

## Posting a pending review

Only when explicitly asked to post, sample the reviewer's recent repository
comments and match their register. Write 2–5 sentences: defect, concrete
consequence, `path:line`, then one ask ("Suggest …" / "Could we …?"). Use "Nit:"
for minor points; omit first-person hedging and chatty openers. Public posts
must not cite private repositories, forks, or their SHAs; describe the approach
inline. Before any authorized body edit, fetch it fresh and apply a minimal edit.

Resolve the authenticated user and list reviews via
`gh api --paginate repos/{owner}/{repo}/pulls/{n}/reviews`; reuse only that user's
`PENDING` review (one per user per PR). Pin `commit_id` to the reviewed head and
confirm it still matches the PR head before writing; re-review if it changed.
Every inline anchor must lie in a diff hunk at that commit.

- Create with `gh api -X POST repos/{owner}/{repo}/pulls/{n}/reviews --input review.json`:
  `commit_id`, optional `body`, `comments: [{path, line, side: "RIGHT", body}]`;
  add `start_line`/`start_side` for multi-line comments. Omit `event` to keep
  `PENDING`; never submit the review. The user submits it.
- Existing pending review: retain its `node_id` (`PRR_...`). REST can return null
  lines for pending comments; read anchors and IDs through GraphQL
  `node(id:){... on PullRequestReview{comments{nodes{id databaseId path line}}}}`,
  paginating comments as needed.
- Add via GraphQL `addPullRequestReviewThread(input:{pullRequestReviewId, path,
  line, side: RIGHT, startLine?, startSide?, body})`.
- Edit via `updatePullRequestReviewComment(input:{pullRequestReviewCommentId:
  "PRRC_...", body})`; REST PATCH on pending comments returns 404.

## Guardrails

No code edits, staging, commits, pushes, or posts without an explicit request.
Never submit `APPROVE`, `REQUEST_CHANGES`, or `COMMENT` reviews. Preserve a dirty
worktree; checkout requires a clean one. Report verified findings without padding.
