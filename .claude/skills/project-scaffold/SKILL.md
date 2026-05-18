---
name: project-scaffold
description: >-
  Scaffold a new project using a four-document engineering framework — PRD,
  Architecture, AI Rules, and Plan — that turns "vibe coding" into disciplined
  engineering. Use when the user starts a new project, app, or major feature,
  or asks to scaffold, plan, or set up a project with proper structure and
  guardrails against scope creep and spaghetti code.
---

# Project Scaffold

Bootstrap a project with the four-document framework. Each document has one
job; together they keep an AI-assisted build on rails instead of producing a
pile of bugs.

| Document | Job | Prevents |
|----------|-----|----------|
| `docs/PRD.md` | What the project IS and is NOT | Scope creep |
| `docs/ARCHITECTURE.md` | Folder layout + data structure | Spaghetti code |
| `docs/AI_RULES.md` | Non-negotiable constraints | Quality drift |
| `docs/PLAN.md` | Ordered, one-step-at-a-time roadmap | Getting lost in code |

## Process

### 1. Gather context
Read any project description passed in the skill arguments. If the project's
purpose, scope, stack, or hard constraints are still unclear, ask the user with
`AskUserQuestion`:
- Project name and one-line purpose
- What it should explicitly NOT do (non-goals)
- Platform / language / framework
- Hard constraints (security, performance, deadline, dependencies)

Do not invent answers. The non-goals and constraints are the whole point of
the framework — get them from the user.

### 2. Create the documents
Create a `docs/` directory. For each document, copy the matching file from
this skill's `templates/` directory into `docs/`, then fill every placeholder
with concrete, project-specific content and delete the `<!-- -->` guidance
comments. Generate them in this order, because each builds on the previous:

1. **PRD.md** — write it, then show the user the "What it is NOT" section and
   get explicit confirmation of scope before continuing.
2. **ARCHITECTURE.md** — derive the folder layout and data model from the
   confirmed PRD.
3. **AI_RULES.md** — capture the user's non-negotiables; ask if any are unclear.
4. **PLAN.md** — break the build into small, ordered, independently verifiable
   steps. Each step names the files it touches and a concrete "done when" check.

### 3. Persist the rules
Mirror the key items from `AI_RULES.md` into a root `CLAUDE.md` (create it, or
append a section if it exists) so the constraints apply automatically in every
future session — not just while this skill is running.

### 4. Hand off
Summarize the four documents and tell the user the build will now follow
`docs/PLAN.md` one step at a time.

## Execution discipline (after scaffolding)

When implementing the project, follow `docs/PLAN.md` strictly:
- Work the **current** step (the first unchecked one) only. Do not jump ahead.
- When a step's "done when" check passes, mark its checkbox `[x]` in `docs/PLAN.md`.
- Stop and confirm with the user before starting the next step.
- If new scope appears mid-build, add it to `docs/PRD.md` (in scope or non-goals)
  before writing any code for it.
