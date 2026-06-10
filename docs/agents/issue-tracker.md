# Issue tracker: GitHub (canonical) + Local Markdown (drafts)

This repo uses **both**. GitHub Issues is the source of truth; local
markdown under `.scratch/` is for drafting, offline work, and anything not
yet ready to publish.

## GitHub (canonical) — use the `gh` CLI

- **Create**: `gh issue create --title "..." --body "..."` (heredoc for multi-line bodies)
- **Read**: `gh issue view <number> --comments`
- **List**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`
- **Comment**: `gh issue comment <number> --body "..."`
- **Label**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

`gh` infers the repo from `git remote -v` when run inside the clone
(`witold-andelie/quant-alpha-foundation`).

## Local markdown (drafts) — `.scratch/`

- One feature per directory: `.scratch/<feature-slug>/`
- PRD draft: `.scratch/<feature-slug>/PRD.md`
- Issue drafts: `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`
- Triage state as a `Status:` line near the top (see `triage-labels.md`)
- Comments append under a `## Comments` heading

## Resolution rules

- **"publish to the issue tracker"** → create a **GitHub** issue (default).
  Draft in `.scratch/` first only if the user asks for a draft or is offline.
- **"fetch the relevant ticket"** → `gh issue view <number> --comments`,
  unless the user passes a `.scratch/` path, then read that file.
- A `.scratch/` draft that gets published should reference the resulting
  GitHub issue number so the two stay linked.
