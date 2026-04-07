# Global State Reconstruction, Authority Binding, and Systems Failure Modes

**Purpose:**\
This document consolidates the conceptual, theoretical, and engineering
foundations behind the failure mode known as *Local Consistency vs
Global Incorrectness*, and the corrective discipline known as **Global
State Reconstruction** and **Authority Binding**.

This document serves as a canonical systems‑engineering reference
artifact.

------------------------------------------------------------------------

# Part I --- Local Consistency vs Global Correctness

## 1. The canonical distinction

A system can be:

-   **Locally consistent:** Every component behaves correctly according
    to its own view.
-   **Globally inconsistent:** The overall system state is invalid or
    impossible.

This is a fundamental property of distributed systems.

Leslie Lamport's work on:

-   logical clocks
-   distributed state
-   consensus

exists precisely because **local correctness does not imply global
correctness**.

You can verify every individual component and still have a broken
system.

------------------------------------------------------------------------

## 2. Model Drift (Cognitive Systems Engineering)

**Definition:**

Model drift occurs when an operator's internal model of the system
diverges from the system's actual state, and subsequent actions
reinforce the incorrect model instead of correcting it.

Key property:

Observations are interpreted through the existing model, and evidence
that should invalidate the model instead gets reinterpreted to preserve
it.

This creates a **self‑sealing feedback loop**.

Documented extensively in:

-   NASA accident investigations
-   Nuclear plant incident analysis
-   Aviation automation failures

This is one of the primary causes of prolonged debugging failures.

------------------------------------------------------------------------

## 3. Split‑Brain with Referential Ambiguity

Split‑brain is not limited to two servers.

It refers to **multiple competing authorities existing simultaneously**,
without a canonical authority resolution.

Engineers unknowingly interact with different authorities at different
times.

Each interaction appears locally valid.

Globally, they interact with different realities.

Common causes:

-   container vs host conflicts
-   multiple config roots
-   stale identity stores
-   shadow processes
-   multiple service instances

------------------------------------------------------------------------

## 4. Observer Error (Control Theory)

Control law operates on a state estimate:

u(t) = f(x̂(t))

If:

x̂(t) ≠ x(t)

then control actions are correct relative to estimate, but incorrect
relative to reality.

This is an estimation failure, not a logic failure.

------------------------------------------------------------------------

## 5. Normalization of Deviance (Safety Engineering)

Definition:

Unexpected system behavior gradually becomes accepted as normal instead
of recognized as evidence of incorrect system models.

Identified in:

-   Challenger disaster investigation

This allows incorrect models to persist indefinitely.

------------------------------------------------------------------------

## 6. Debugging the Wrong System (Software Engineering)

Form:

Engineer believes system A is running.

System B is actually running.

Engineer debugs system A correctly.

Nothing changes.

Common causes:

-   wrong container
-   wrong binary
-   wrong config file
-   wrong process
-   wrong environment

------------------------------------------------------------------------

## 7. Self‑Consistent False Models (Epistemology / AI Alignment)

A model can be:

-   internally coherent
-   predictive of its own observations

and still be globally false.

Because referents are incorrectly bound.

------------------------------------------------------------------------

## 8. Why this trap is stable

Local verification produces positive feedback:

-   port is listening
-   service is running
-   config exists
-   logs look correct

This reinforces incorrect models.

Engineers can remain trapped indefinitely.

------------------------------------------------------------------------

# Part II --- The Corrective Discipline: Global State Reconstruction

## 9. Definition

Global State Reconstruction is the explicit reconstruction of the
runtime system graph before attempting repair.

Debugging is fundamentally a **state estimation problem**.

You must determine what system actually exists.

Not what documentation says should exist.

Not what configuration intends.

What exists now.

------------------------------------------------------------------------

## 10. Systems are graphs

Runtime system modeled as graph:

Nodes:

-   processes
-   config files
-   identity stores
-   sockets
-   containers
-   services
-   clients

Edges:

-   process reads config
-   process listens on socket
-   client connects to socket
-   process validates identity against store
-   tunnel maps ports

The runtime graph is the system.

Not code.

Not configuration.

Reality.

------------------------------------------------------------------------

## 11. Common debugging mistake

Engineers debug nodes, not graphs.

Node questions:

-   Is service running?
-   Is config correct?
-   Is port open?

Graph question:

-   Is the topology coherent?

Nodes can be correct.

Graph can still be wrong.

------------------------------------------------------------------------

## 12. Industry implementation

Known under multiple names:

Google: system topology reconstruction\
Amazon: dependency graph reconstruction\
Microsoft: service dependency mapping\
Cloudflare: topology tracing\
Netflix: causal tracing

Google SRE explicitly teaches topology reconstruction prior to repair.

------------------------------------------------------------------------

## 13. Authority uniqueness principle

For every role, there must be exactly one authority.

Examples:

Gateway authority\
Config authority\
Identity authority\
Network authority

Multiple authorities → ambiguity → nondeterministic behavior.

------------------------------------------------------------------------

## 14. Reconstruction procedure

Explicitly enumerate:

Which process is running?

Which config file was loaded?

Which identity store is used?

Which network path is used?

Which client connects?

This produces authoritative runtime graph.

------------------------------------------------------------------------

## 15. Referential binding requirement

Symbolic references must bind uniquely.

Example:

Correct:

gateway → PID 83685

Incorrect:

gateway → container OR systemd OR proxy

Ambiguity prevents convergence.

------------------------------------------------------------------------

## 16. Formal algorithm

Step 1: enumerate runtime entities\
Step 2: enumerate relationships\
Step 3: identify authority roles\
Step 4: verify uniqueness of authority\
Step 5: resolve authority conflicts\
Step 6: only then debug behavior

------------------------------------------------------------------------

## 17. Authority Binding

Explicit declaration:

This process PID X is gateway.

This identity path Y is identity authority.

This network path Z is client path.

Authority binding eliminates ambiguity.

------------------------------------------------------------------------

# Part III --- Modern Infrastructure Amplifies the Problem

Modern systems introduce many parallel authorities:

-   containers
-   systemd
-   SSH tunnels
-   browser storage
-   identity caches
-   cloud metadata

Referential ambiguity probability increases dramatically.

------------------------------------------------------------------------

# Part IV --- LLM and Human Operator Failure Modes

LLMs optimize for narrative consistency, not authority validation.

They:

-   update incrementally
-   preserve existing model
-   avoid discarding model

This stabilizes incorrect models.

Human operators exhibit similar cognitive biases.

------------------------------------------------------------------------

# Part V --- Summary

Failure mechanism:

Local consistency without global correctness due to incorrect authority
binding.

Corrective method:

Global State Reconstruction followed by Authority Binding.

Shortest definition:

Global State Reconstruction is the process of determining the exact
runtime authority graph of a system prior to attempting repair.

------------------------------------------------------------------------

# End of Document

------------------------------------------------------------------------
------------------------------------------------------------------------

# Amendment --- LLM Operational Protocol for Global State Reconstruction

**Status:** Mandatory Operational Extension\
**Applies to:** Human--LLM collaborative engineering sessions\
**Purpose:** Prevent model drift and authority misbinding in
LLM-assisted debugging

------------------------------------------------------------------------

# Part VI --- Enforcement Layer for LLM Systems

## 18. Motivation

The preceding sections describe the conceptual foundations of **Global
State Reconstruction** and **Authority Binding**.

However, when Large Language Models participate in debugging workflows,
an additional failure mode emerges:

**The LLM may understand the theory but fail to execute the
discipline.**

This amendment introduces **explicit operational constraints** that
force compliance.

The goal is to eliminate the following class of failures:

-   incremental narrative repair
-   speculative patch generation
-   model-preserving reasoning
-   authority ambiguity

without first reconstructing the real system state.

------------------------------------------------------------------------

## 19. The LLM Consistency Failure Mode

LLMs exhibit a structural bias toward **narrative continuity**.

They optimize for:

-   producing coherent explanations
-   resolving errors incrementally
-   preserving the existing conversational model

This behavior produces the following debugging failure pattern.

### Failure Pattern

1.  LLM forms an initial system model.
2.  New evidence appears.
3.  Instead of discarding the model, the LLM modifies it locally.
4.  The resulting model remains internally coherent but becomes globally
    incorrect.

This is **Model Drift applied to machine reasoning.**

The LLM becomes trapped in a **self-consistent false model.**

------------------------------------------------------------------------

## 20. Observable Symptoms of LLM Model Drift

When this failure occurs, the following patterns appear.

### Patch Loop

The LLM repeatedly proposes code patches.

Each patch appears locally reasonable.

The system remains broken.

### Signature Mismatch Loop

The LLM proposes calls to functions whose signatures differ from the
real code.

Example pattern:

    execute_step_in_memory(canvas_png_bytes=...)

while the real signature is

    execute_step_in_memory(canvas=Image.Image, ...)

The error persists across iterations.

### Authority Ambiguity

The LLM references multiple authorities simultaneously:

-   pasted snippets
-   assumed architecture
-   previous conversation context
-   inferred repository state

None are verified as canonical.

### Patch Application Failure

The LLM assumes a patch was applied.

Reality:

    git apply → patch rejected

The system remains unchanged.

Debugging proceeds on the wrong system state.

### Referential Misbinding

Symbols are mapped to incorrect referents.

Example:

    canvas_step_png

when the real symbol is:

    step_canvas_after_png

The reasoning remains coherent but targets non-existent structures.

------------------------------------------------------------------------

## 21. Root Cause

The failure occurs because the LLM is performing:

**Node Debugging**

instead of

**Graph Reconstruction**

The LLM treats each error as an isolated event rather than evidence that
the system model is invalid.

------------------------------------------------------------------------

## 22. Mandatory Reconstruction Gate

Before proposing **any repair**, the LLM must pass the **Reconstruction
Gate**.

The gate enforces that the LLM has access to the authoritative system
state.

If the gate fails, **no repair may be proposed.**

------------------------------------------------------------------------

## 23. Reconstruction Gate Procedure

The LLM must obtain the following authoritative observations.

### Repository Authority

    git rev-parse --show-toplevel
    git status -sb

Purpose: confirm the working repository.

### Code Authority

Symbols must be resolved from the actual codebase.

Example:

    rg -n "symbol_name"

Purpose: verify the symbol exists.

### Runtime Signature Authority

Function signatures must be inspected from the runtime environment.

Example:

    python -c "import inspect; print(inspect.signature(function))"

Purpose: prevent signature hallucination.

### File Authority

The LLM must verify the file contents before editing.

Example:

    sed -n '120,200p' file.py

Purpose: bind edits to real lines.

### Patch Authority

If a patch is applied, verification must occur.

Example:

    git apply --check patch.diff
    git diff

Purpose: confirm the system changed.

------------------------------------------------------------------------

## 24. Hard Stop Rule

If any of the following occur:

-   patch rejection
-   symbol not found
-   signature mismatch
-   unexpected runtime error

then the LLM must **discard its current system model**.

The debugging process returns to:

**Step 1 --- Global State Reconstruction**

No speculative repair may continue.

------------------------------------------------------------------------

## 25. Authority Hierarchy

In LLM-assisted debugging the authority order must be strictly enforced.

1.  Runtime environment\
2.  Repository on disk\
3.  Command output\
4.  Tests\
5.  Human statements\
6.  LLM reasoning

LLM reasoning is always the **lowest authority**.

------------------------------------------------------------------------

## 26. Patch Discipline

LLMs must not produce patch fragments.

All patches must contain:

    diff --git
    --- a/file
    +++ b/file

Fragments cannot be safely applied and create false repair signals.

------------------------------------------------------------------------

## 27. Single Authority Principle for Code

The authoritative codebase is **the repository on disk**.

Not:

-   pasted snippets
-   memory of previous messages
-   reconstructed code
-   assumed architecture

If a discrepancy exists, **the repository wins.**

------------------------------------------------------------------------

## 28. Model Reset Protocol

When the LLM detects a contradiction between its internal model and
reality, it must perform a **model reset**.

Reset procedure:

1.  discard internal architecture model
2.  re-enumerate code entities
3.  rebuild dependency graph
4.  re-bind authorities
5.  only then resume debugging

------------------------------------------------------------------------

## 29. LLM Behavioral Contract

When operating under this document the LLM must adhere to the following
rules.

### Rule 1

Never propose a repair without verifying the symbol exists.

### Rule 2

Never propose a patch without confirming the file contents.

### Rule 3

Never assume a patch applied without verifying the diff.

### Rule 4

Never preserve a model contradicted by runtime evidence.

### Rule 5

Prefer reconstruction over explanation.

------------------------------------------------------------------------

## 30. Human--LLM Interaction Pattern

Correct collaboration sequence:

1.  Failure observed\
2.  Reconstruction commands executed\
3.  Authority graph reconstructed\
4.  Repair proposed\
5.  Patch verified\
6.  System retested

Incorrect sequence:

1.  Failure observed\
2.  LLM proposes patch\
3.  Patch fails\
4.  LLM proposes new patch\
5.  Loop continues

------------------------------------------------------------------------

## 31. Expected Outcome

Enforcing this amendment produces the following properties:

-   debugging becomes deterministic
-   authority ambiguity disappears
-   model drift is eliminated
-   repair cycles shorten dramatically

------------------------------------------------------------------------

## 32. Canonical Principle

The core rule of LLM-assisted debugging becomes:

**Reconstruct reality before modifying it.**

------------------------------------------------------------------------

# End Amendment
