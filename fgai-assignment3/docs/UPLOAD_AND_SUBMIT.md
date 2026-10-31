# Upload and finish the assignment

The supplied package contains working source code and one measured baseline. You chose to receive run instructions for the other models. Complete the remaining experiments and review before the final LMS submission.

## Run and inspect

1. Install dependencies and enable a GPU in your chosen environment. Accept the official LLaMA and Gemma model terms and log in to Hugging Face locally.
2. Run `python run.py prepare`, `python run.py preflight` and `python run.py run --device cuda`. For consistent timing, rerun the basic model on the same machine as the other three. The Colab notebook automates these commands.
3. Run `python run.py report` and `python run.py verify`. A missing model causes verification to fail deliberately. Generated scores are recomputed from predictions, and mismatched test sets/configurations are rejected.
4. Complete `docs/FOUR_MODEL_REVIEW.md` using the review guide and link it from README. Check the generated README comparison, temperature discussion and best-model conclusion. Review claims yourself rather than treating the generated conclusion as sufficient qualitative analysis.
5. If using Colab, download the generated ZIP after the run and replace the local results with it before uploading to GitHub.

## Create the repository

Suggested name: `fgai-assignment3-machine-translation`. If private, grant the evaluator access. Upload the **contents** of the clean project folder. A ZIP uploaded as a single repository file is inconvenient for assessment: extract it first so README and code are visible directly.

For command-line Git, first copy/extract the project into its own directory outside any existing repository. This project was prepared inside another workspace, so do not initialize or push that parent workspace by mistake. In the new standalone project directory:

```text
git init
git add .
git status
git commit -m "Add ICT4403 machine translation assignment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Use your actual URL. Check `git status` before committing: no tokens, `.env`, datasets, checkpoints, virtual environments or unrelated workspace files should be present. Keep results, plots and small generated samples. `.gitignore` excludes local student submission documents from the public repository; the upload ZIP includes them for your convenience. You can keep those documents private and submit only the PDF to LMS.

## Add the repository URL and export the PDF

The Word file is named `Parth_Upadhyay_230953328_FGAI_Assignment3.docx`, with the matching PDF name. Both supplied copies have an explicit repository-link placeholder. They are not ready for LMS while it remains.

1. Edit `submission/details.json` and set `github_url` to the actual accessible repository URL.
2. Run `python scripts/build_submission.py` to regenerate the named Word file. The script refuses to create a final file with a blank URL unless `--allow-pending` is explicitly selected.
3. Open the Word file and verify your name, registration number, assignment title, system title, dataset source and repository URL. Click the links to check them.
4. In Microsoft Word, use **File > Save As > PDF**, as the assignment specifies. Replace the previous PDF and check its contents.
5. Upload the final PDF to the correct LMS assignment section before your institution's deadline. No submission deadline is printed in the supplied PDF, so check LMS.

## Rebuild a clean upload archive

From the project directory, run `python scripts/package_upload.py`. It writes `fgai-assignment3-upload.zip` beside the project, with source, small results, plots, documentation and submission documents. The script excludes raw/processed corpus files, checkpoints, caches and credentials. Never commit Hugging Face tokens.
