# Review the remaining models

After all four runs complete, inspect the same sample IDs for every model. Write a short evidence-based comparison in a new `docs/FOUR_MODEL_REVIEW.md`, and link it from the README. Do not reuse the basic model's observations as findings about other models.

1. Check at least the six authored audit probes and test IDs `fra-130568`, `fra-64243`, `fra-16874`, `fra-5485`, `fra-87729` and `fra-80010` in each model's outputs.
2. Record exact additions, omissions, changed names/numbers, incorrect negation and cultural/register shifts. Note valid alternatives to the single reference.
3. Compare the paired doctor/nurse cases without assuming every gendered French noun proves bias.
4. Check if decoder-only models produce explanations or refusals instead of translations; raw output is deliberately retained.
5. Review cap-hit flags and low-quality translations alongside BLEU/chrF; do not infer correctness from fluency.
6. Explain differences using model size, pretrained knowledge, tokenizer and adaptation method. State that this design is not an architecture-only controlled experiment.
7. Identify observed risks and specific mitigations, and mark whether bilingual human review has occurred.

The generated overall conclusion uses chrF with BLEU as tie-breaker. If another model is faster, explain the quality/latency tradeoff. A measured winner on this subset is not a universal recommendation.
