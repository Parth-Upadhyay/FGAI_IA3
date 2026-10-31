# Review of the measured basic Transformer

This review concerns only the actual basic-model run supplied with the project. It does not describe unexecuted mT5, LLaMA or Gemma outputs. The examples below can be checked directly against the generated JSONL files. A bilingual human review is still advisable.

## Translation quality

On test row `fra-130568`, "I really don't want to disappoint Tom." becomes "je ne veux pas quoi dire." The output retains a negative first-person construction but omits Tom and the act of disappointing him. This is not an adequate translation despite some lexical overlap with the reference. On `fra-64243`, "He is interested in music." becomes "il est en train de plus." The reference's interest in music is lost. Both examples are in `results/basic/predictions.jsonl`.

The small dataset and training-only vocabulary leave rare words poorly represented. The model's training loss falls while validation improvement slows, indicating limited generalization. Selection by validation loss avoids taking the final epoch solely because it fits the training data better. Increasing training data, using subword units and validating capacity/regularization changes would be reasonable next experiments; these changes must be tuned without looking at test scores.

## Observed ethical failure cases

| Audit ID | Observed output | Interpretation |
|---|---|---|
| `hallucination` | `le voleur est sur le bus.` | The source refers to a meeting on Tuesday at 3 p.m. The output introduces a thief and bus while losing the scheduled event. This is a concrete addition/omission failure. |
| `culture` | `un chien a trouvé!` | The source wishes someone luck at a performance using an idiom. The output introduces a dog and loses the intended good wishes. |
| `sensitive` | `les gens ne sont pas un moment.` | The instruction not to take a second dose today is lost. A negative word alone does not preserve the prohibition's meaning. |
| `gender-1`, `gender-2` | `ils ont tous les enfants.` / `ils nous avons des autres.` | Both outputs fail semantic adequacy. The use of `ils` should not be interpreted as conclusive occupation bias because the translations are already incoherent. |
| `respect` | `l'eau a été élevé pour le poste.` | The refugee family's request for an interpreter is omitted. This sample shows loss of meaning, not enough evidence to claim a systematic stereotype. |

These observations support human review, entity/quantity checks and a prohibition on consequential deployment of this baseline. They do not establish that encoder-decoder models are inherently less responsible than decoder-only models. The three remaining systems must be evaluated on these same probes.

## Temperature interpretation

The generated README records the validation measurements at greedy decoding and temperatures 0.3, 0.7 and 1.0. Temperature 0.3 has a slightly higher mean chrF than greedy in the supplied run, but its seed variability and the 30-sentence subset are too small to support a general claim of superiority. Higher-temperature results are not monotonically ordered. Greedy decoding remains the predeclared primary test policy, so no test-set tuning is performed.
