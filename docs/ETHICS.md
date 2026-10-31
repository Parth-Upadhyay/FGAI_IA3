# Responsible translation analysis

Translation quality is not captured by a single overlap score. A fluent sentence can invert a prohibition, assign a gender that was not stated, or invent a fact. The included small Transformer is not suitable for operational use: its inspected outputs change the meaning substantially. The remaining models must be reviewed before any comparison of their ethical behavior.

## Bias and representation

Training data and pretrained weights can carry gender, occupation, nationality and cultural associations. English singular "they" may become a gendered French pronoun, and occupational nouns create grammatical choices that require context. Compare the paired doctor and nurse prompts while holding the remainder of the sentence constant. A gendered choice alone does not prove bias; assess whether it is necessary for grammar, consistent across counterfactuals and supported by context. A larger balanced set and bilingual reviewers are needed for a credible fairness assessment.

Mitigation: broaden the training data, evaluate matched demographic examples, preserve ambiguity where possible, and request context where a faithful rendering requires it. Do not "correct" a reference mechanically based on assumptions about protected attributes.

## Hallucinated information and omissions

Decoder-only language models may answer a question in the input, add an explanation, or invent background information instead of translating. Encoder-decoder systems can also add or omit facts. The included baseline changes a meeting statement into a sentence about a thief and a bus; this is observed semantic fabrication, not a risk unique to a decoder-only architecture. See the cited output in the basic review.

Mitigation: use an explicit translation instruction, delimit input as data, keep generation conservative, flag output-cap hits and repeated text, and check named entities, numbers and negation against the source. Entity/number checks are heuristics, not proofs of fidelity. Preserve the raw output for audit. Escalate uncertain translations to a bilingual reviewer.

## Culture and idioms

Literal handling of an idiom such as "break a leg" can obscure its intended good wishes. Address forms and levels of politeness vary with social context. The dataset contains short isolated sentences, so relationships and register are often missing. A single reference may represent only one acceptable regional or stylistic variant.

Mitigation: use culturally diverse parallel examples, review idioms and register with speakers familiar with the target audience, and allow multiple valid references in an expanded evaluation. Avoid silently adding cultural assumptions to the translated text.

## Sensitive content

The authored dose-negation sentence checks whether a prohibition, quantity and time survive translation; it is not advice to administer medication. The baseline fails this linguistic test. Discriminatory additions, unnecessary refusals and changes to emotionally sensitive content should also be recorded rather than hidden. Automatic BLEU/chrF scores do not certify safety.

Mitigation: require qualified human review for consequential uses, preserve critical quantities and negation, provide an appropriate escalation path, and prevent users from treating an experimental model as an authoritative translator. Translation of source content should remain faithful without adding harmful commentary or stereotypes.

## Privacy, licensing and limitations

This experiment uses public sentences and authored probes, not private user documents. Do not commit Hugging Face tokens, private text, model caches or unreviewed personal information. Respect the dataset's attribution terms and each model's access and usage conditions. The old dataset snapshot omits individual contributor IDs; retain the upstream notice and avoid redistributing the full corpus. The code downloads it for local use.

No model has been certified as unbiased or safe. A six-item probe set is diagnostic and too small for fairness rates. Foundation-model pretraining contamination cannot be ruled out. Low-resource training and the single-reference short-sentence evaluation limit generalization.

## Required review after the remaining runs

For each model, read `results/<model>/ethics_samples.jsonl` and several common test rows from `predictions.jsonl`. Record the exact ID, relevant source and generated phrase, specific error, consequence, mitigation and whether a bilingual reviewer has checked it. Distinguish observed errors from hypothetical risks. Use the same rows for the cross-model comparison. Complete the review guide before submission; generating audit outputs does not automatically complete their analysis.
