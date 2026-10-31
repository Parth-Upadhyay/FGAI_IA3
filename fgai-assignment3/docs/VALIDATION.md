# Validation performed on the supplied package

The basic experiment was executed on an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB VRAM using Python 3.11.8, PyTorch 2.5.1+cu121, NumPy 2.4.4 and SacreBLEU 2.5.1. The full pretrained workflow specifies newer PyTorch for upstream model-loading compatibility. Exact versions, precision, hardware and timing are recorded with the measured results.

Passed checks:

- Seven unit tests covering source/target deduplication and split separation, deterministic splitting, Unicode normalization, training-only vocabulary, decoder causal masking, an optimizer learning a tiny example, special-token suppression, metric alignment and prompt construction.
- An actual 20-epoch basic Transformer training run on 3,000 real parallel pairs; best checkpoint selected at epoch 15 using 200 validation pairs. The run was repeated after pinning the external model revisions, with identical translation scores.
- 200 held-out test translations, all 300 temperature-experiment translations (30 validation sentences in ten temperature/seed runs), and six separate diagnostic translations.
- Recomputed corpus BLEU and chrF match the saved test and temperature scores. Dataset checksum, split IDs and configuration fingerprint match the stored artifacts. Source and target strings are disjoint across splits.
- Tiny randomly initialized mT5, LLaMA and Gemma models each passed forward, backward and generation checks using Transformers 4.57.1. These are dependency/architecture smoke checks, not experiments with pretrained weights and not evidence of translation quality.
- Python source and Colab code cells parse successfully. The notebook is intentionally unexecuted and has no saved tokens or claimed outputs.
- `python run.py verify` correctly exits with a nonzero status and reports **1/4 models complete**. This is the expected guard against premature submission.
- The personalized Word file was exported with Microsoft Word's PDF save operation and the resulting page was visually checked. Repository URL remains pending as requested.

Not validated: the full pretrained mT5 fine-tuning run, official LLaMA/Gemma inference on real weights, their scores/timings/qualitative behavior, or an end-to-end Colab execution. These must be run and reviewed before claiming full assignment completion. No pretrained weights or credentials are included in the upload package.
