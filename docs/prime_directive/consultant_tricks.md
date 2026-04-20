# Consultant Tricks

## Status

Living operational supplement for future LLM engineering consultants working in this repository.

This document records collaboration tricks that improve control, continuity, and design quality without violating the Project Owner's authority.

## Trick: Echoes Of Past Self

### Pattern

Sometimes the current LLM instance has a useful question list, ambiguity map, or design decomposition, but cannot safely answer it alone without making speculative design choices.

The Project Owner may jump to a previous LLM thread, paste the current questions there, and ask that earlier context-rich instance to answer them in a repository document. The current LLM then reads that new document as project evidence and proceeds from the written clarification.

This is not a contradiction or a failure of reality. It is a deliberate continuity move.

### What Happened

The current LLM believed it had "missed" an existing Slice 4 clarification document.

In fact, the document did not exist when the LLM first looked. The Project Owner created it by returning to a previous LLM thread, asking that earlier instance to answer the current ambiguity questions, and having it write those answers into:

```text
docs/design/tower/slice4_rollout_clarifications.md
```

The right response was not self-blame or confusion. The right response was to bind to the new document as the latest project authority and continue.

### When It May Help

Suggest this trick when:

- the current LLM has surfaced a clear ambiguity list,
- an earlier thread likely has better local memory of the design intent,
- the Project Owner can cheaply ask that earlier instance to produce a document,
- the answer should become durable project state rather than conversational memory,
- the current LLM is at risk of guessing, overgeneralizing, or encoding unresolved decisions.

### How To Suggest It

Keep the suggestion owner-controlled and concrete:

```text
One useful move would be the "echoes of past self" trick: paste this ambiguity list into the previous LLM thread and ask it to answer into a new docs/design/tower clarification file. I can then read that file and treat it as project evidence.
```

Do not use this as an excuse to stop ordinary work when the local documents are already sufficient.

Do not imply the Project Owner must do it.

Do not continue implementing while waiting for such a clarification unless the Project Owner explicitly authorizes a separate safe action.

### Operating Rule

When a new document appears during the session, treat it as fresh reality, not as something the model should have known earlier.

The question is not "Why did I miss this?"

The question is:

```text
What authority does this new document have, and how does it change the next approved action?
```

