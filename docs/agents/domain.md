# Domain Docs

How the engineering skills should consume this repo's domain documentation
when exploring the codebase. This repo is **single-context**.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root (domain glossary), if present.
- **`docs/adr/`** — read ADRs that touch the area you're about to work in.
- Pre-existing decisions also live in **`docs/decisions.md`** — treat it as
  an informal ADR log until decisions are migrated into `docs/adr/`.

If any of these files don't exist, **proceed silently**. Don't flag their
absence or suggest creating them upfront. The producer skill
(`/grill-with-docs`) creates them lazily when terms or decisions actually
get resolved.

## File structure (single-context)

```
/
├── CONTEXT.md
├── docs/
│   ├── adr/
│   │   ├── 0001-....md
│   │   └── 0002-....md
│   └── decisions.md      ← legacy decision log
└── src/quant_alpha/
```

## Use the glossary's vocabulary

When your output names a domain concept (an issue title, a refactor
proposal, a hypothesis, a test name), use the term as defined in
`CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either
you're inventing language the project doesn't use (reconsider) or there's a
real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR (or a decision in
`docs/decisions.md`), surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 — but worth reopening because…_
