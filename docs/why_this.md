# Why Lens (and why Clay/Apollo-style tools don’t solve this)

## The actual failure mode
Agencies lose deals because outreach is generic.
Generic happens because research is expensive and not operationally scalable.

## Why Clay/Apollo still produce “generic outreach syndrome”
1) They optimize for volume mechanics (lists, enrichment, sequencing), not proof-backed specificity.
2) They don’t enforce an evidence trail (a pitch can be “confident” without citations).
3) They don’t verify solution-fit against *your* internal capabilities (case studies / services).
4) No quality gate: nothing forces a “do not send” outcome when signals are weak.
5) They don’t loop: failure doesn’t automatically trigger deeper research with a reason.

## Lens thesis
A pitch is only good if it:
- cites measurable failures on the prospect’s live presence
- maps those failures to comparable proof you already have
- exposes confidence transparently
- refuses to ship when evidence is weak

## Design decisions (non-negotiables)
- 3-tier signal taxonomy → separates hard facts from soft inference
- typed agent contracts → no string soup between agents
- evidence trail attached to every signal
- validator score gate with loop-back on failure

## What Lens outputs (not “a message”, a decision artifact)
- signals (tiered) + sources
- matched case study + similarity score
- structured pitch (JSON)
- pass/fail + confidence breakdown