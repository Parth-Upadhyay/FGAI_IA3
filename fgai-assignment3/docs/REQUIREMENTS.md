# Assignment requirement mapping

This checklist maps the two-page assignment to actual files. The PDF allows smaller models and a manageable subset; it does not prescribe a dataset, minimum score, epoch count or language pair. Its README requirement includes temperature analysis even though the main question is about machine translation. The submission document therefore names the project as an autoregressive translation system, retaining the requested terminology.

| PDF requirement | Implementation or deliverable | Status in supplied package |
|---|---|---|
| Public parallel-text dataset and source URL | `mt/data.py`, README, `data/manifest.json`, dataset notice | Obtained; 2019-04-07 English-French snapshot |
| Describe, preprocess and split data | README; seeded disjoint splits, deduplication and checksum | Executed |
| Implement/train basic encoder-decoder Transformer | `mt/basic.py`, `mt/training.py` | Executed; measured outputs included |
| Fine-tune mT5 | `mt/training.py` | Implemented; final experiment pending |
| Fine-tune or prompt LLaMA and Gemma | `mt/experiment.py` | Prompting implemented; licensed model access and execution pending |
| Same test set and comparable generation | `run.py`, pinned config, IDs and run fingerprint | Implemented; full four-model execution pending |
| BLEU, chrF, training time, inference time, parameters | `results/<model>/metrics.json`, comparison CSV | Measured for basic; others pending |
| Compare architecture and computation with confounders | README generated report and methodology | Framework provided; empirical four-model conclusion pending |
| Examine all four outputs and discuss ethics | `docs/ETHICS.md`, probes, review guide | Basic review complete; three reviews pending |
| Source files/notebook, supporting files, results, plots, samples | Python modules, Colab notebook, results directory | Included for available runs |
| README: objective, dataset, models, steps, dependencies, results, comparison, temperature, best model | README and `mt/reporting.py` | All sections present; winner gated on four complete runs |
| Correct, accessible GitHub repository | `docs/UPLOAD_AND_SUBMIT.md` | User will create/upload repository |
| Named Word document and required personal/project/dataset/GitHub details | `submission/Parth_Upadhyay_230953328_FGAI_Assignment3.docx` | Name and registration supplied; repository URL pending |
| Convert Word to PDF | Matching PDF plus rebuild/export instructions | Draft PDF; regenerate after adding URL |
| Upload final PDF to LMS by deadline | Upload instructions | User action; no deadline is stated in the supplied PDF |

The requested deliverable is a folder to upload to Git. Creating a public repository, granting evaluator access and submitting to LMS are later user actions. The user explicitly chose to receive run instructions for models requiring access. Pending work is documented rather than presented as complete.
