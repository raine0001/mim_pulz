# METHOD

## System Overview

The project uses structure-aware retrieval with deterministic routing and conservative reranking.

Pipeline:

1. Structural profiling
2. Routing policy selection
3. Candidate retrieval (internal + optional ORACC memory)
4. Candidate reranking and gates
5. Evidence trace output

## Structural Profiling Dimensions

- `fragmentation`: complete / partial / fragmentary
- `formula_density`: low / medium / high
- `numeric_density`: low / high
- `template_type`: slot-structured / narrative / hybrid
- `length_bucket`: short / medium / long
- `domain_intent`: letter / legal / economic / ritual / unknown

## Routing Policies

- `P0 INTERNAL`: internal retrieval only
- `P1 HYBRID`: mixed internal + external candidate pool
- `P2 FALLBACK`: external assistance when internal confidence is weak
- `P3 STRONG_RERANK`: larger rerank emphasis when ambiguity is high

## Border/Section Logic

Border-aware routing can apply section constraints (`closing` vs `body`) to reduce mismatched exemplars.

## Reranker

- Linear pairwise reranker
- Trained on hard cases
- Objective aligned with combined metric gain
- Canonical model: seed43
- Probe model: seed44

Conservative switching rules:

- do not switch unless reranker margin is positive
- retain numeric/slot fidelity guards
- keep deterministic tie-break behavior

## Evidence Trace

Submission artifacts include:

- selected exemplar id
- policy and route labels
- source origin (internal / ORACC)
- structural labels
- scores and routing telemetry

