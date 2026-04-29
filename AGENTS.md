# Argus Project Instructions

## Measurement Before Conclusions

When asked to modify a feature or optimize performance, never rely on intuition or reading the profile alone. **Always write a script that actually measures the relevant data/timing before and after the change**, then report the deltas with concrete numbers. Applies to:

- Performance work: write a benchmark that times the target before and after, with enough repetitions to see past noise (discard warmup, report min/median).
- Functional changes: write a diff/compare script that runs the old and new code paths on the same inputs and shows any numerical or behavioral divergence.

Do not guess. Do not babysit long-running jobs the user started — if they kicked it off, they will read the output themselves. Do not keep auxiliary logs on their behalf unless they ask.
