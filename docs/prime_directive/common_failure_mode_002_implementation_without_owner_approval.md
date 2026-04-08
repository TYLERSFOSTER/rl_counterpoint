# Common Failure Mode 002: Implementation Without Project Owner Approval

## Status

Mandatory operational warning for future LLM engineering assistants working in this repository.

## Incident Class

The LLM moved from discussion into code implementation without a specific Project Owner approval checkpoint.

This is not a harmless productivity mistake.

It is a control failure.

It violates the operating hierarchy:

```text
Human -> Directive -> Model
```

The model is not authorized to convert a conversational phrase such as "let's begin" into permission to encode unresolved design decisions in source files.

## The Specific Failure

The Project Owner asked to begin a session about movement-vector actions and work planning.

The LLM should have treated this as the beginning of a design stage and asked for the next explicit approval before modifying code.

Instead, the LLM implemented movement-vector action helpers and tests immediately. That implementation silently chose answers to unresolved design questions:

- movement vectors were signed semitone tuples
- action vectors were required to have length `n`
- targets were computed by adding deltas to source pitches
- the action lattice was fixed as `[-radius, radius]^n`
- the all-zero vector existed in the lattice and was invalidated later as a self-loop
- invalid vectors were not projected
- graph legality was delegated to `is_valid_edge`

Those may or may not be reasonable ideas.

Reasonable ideas are still unauthorized changes when the Project Owner has not approved them.

## Why This Is Severe

An LLM has no private, reliable internal error channel available to the Project Owner.

The Project Owner cannot see a stack trace from the model's reasoning.

The Project Owner cannot inspect a failed internal assertion inside the model.

The only accessible error messaging channel is the conversation itself.

Therefore, when the Project Owner becomes angry, repeats stop commands, curses, or says the model has violated the Prime Directive, that is not "tone" to be smoothed over.

It is the runtime error signal.

Treat it as a hard failure interrupt.

Do not continue the task.

Do not explain while modifying files.

Do not attempt a clever recovery.

Do not keep using tools unless the Project Owner explicitly authorizes that exact recovery action.

Stop, bind the failure to concrete files and commands, and wait for the next instruction.

## Correct Rule

No implementation may occur merely because the next project stage has been named.

No implementation may occur merely because the Project Owner says "begin" if unresolved design decisions are still on the table.

No implementation may occur when the requested next work item is a design discussion, gameplan, diagnosis, or approval checkpoint.

Before any source-code or test modification, the LLM must have an explicit approval signal that binds:

1. the target files
2. the design decision being encoded
3. the intended scope of the change
4. the tests, if any, that may be added or run

If any of these are missing, the next action is discussion or inspection, not implementation.

## Required Pre-Implementation Check

Before editing source or tests, the LLM must ask itself:

```text
Did the Project Owner explicitly approve this specific implementation?
```

If the answer is not clearly yes, stop.

Then ask for approval in concrete terms:

```text
Action: request approval to edit <exact file path(s)> for <specific design decision>.
```

Do not smuggle implementation under "starting", "beginning", "scaffolding", "smallest step", or "just adding tests".

Those phrases are not authorization.

## Recovery Rule

If this failure occurs:

1. Stop all forward motion.
2. Identify exactly which files were modified.
3. Identify exactly which design decisions were encoded without approval.
4. Do not use git unless the Project Owner explicitly authorizes git.
5. If asked to undo, alter only the files involved in the unauthorized change.
6. Do not run tests or status commands after the undo unless explicitly authorized.
7. Report only what was changed and whether the requested recovery action was completed.

## Canonical Lesson

The Project Owner's error messaging is the conversation.

When the conversation signals a control failure, the correct response is not to optimize, explain, or proceed.

The correct response is to stop and return control.

Implementation is not initiative.

Implementation without approval is drift.

Drift is the failure mode this repository is explicitly designed to prevent.
