"""Generate tables and plots exclusively from completed, matching experiments."""
import csv
import statistics
from .common import ROOT, read_json, read_rows, write_json, fingerprint

NAMES = ("basic", "mt5", "llama", "gemma")


def completed():
    results = []
    for name in NAMES:
        folder = ROOT / "results" / name
        if (folder / "status.json").exists() and read_json(folder / "status.json")["state"] == "complete":
            results.append(read_json(folder / "metrics.json"))
    if len({r["run_hash"] for r in results}) > 1 or len({r["test_ids"] for r in results}) > 1:
        raise ValueError("Cannot compare experiments from different configurations or test sets")
    return results


def table(rows):
    text = "| Model | BLEU | chrF | Training / tuning (s) | Inference (s) | s / sentence | Parameters |\n"
    text += "|---|---:|---:|---:|---:|---:|---:|\n"
    lookup = {r["model"]: r for r in rows}
    for name in NAMES:
        if name not in lookup:
            text += f"| {name} | pending | pending | pending | pending | pending | pending |\n"
            continue
        r = lookup[name]
        train = f'{r["training_seconds"]:.2f}' if r["training_seconds"] is not None else "N/A (prompted)"
        text += f'| {name} | {r["bleu"]:.2f} | {r["chrf"]:.2f} | {train} | {r["inference_seconds"]:.2f} | {r["seconds_per_sentence"]:.4f} | {r["parameters"]:,} |\n'
    return text


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = completed()
    folder = ROOT / "results"
    plots = folder / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fields = ["model", "bleu", "chrf", "training_seconds", "inference_seconds", "seconds_per_sentence", "parameters", "test_size", "hardware", "precision"]
    with (folder / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    body = "## Measured results\n\n"
    body += f"{len(rows)} of 4 models have completed training/adaptation, test evaluation, temperature experiments and audit generation. "
    body += "Pending entries are unmeasured, not zero scores.\n\n" + table(rows)
    if rows:
        count = rows[0]["test_size"]
        body += f"\nThe shared held-out test set contains {count} sentence pairs. BLEU and chrF are corpus-level SacreBLEU scores on a 0-100 scale. "
        body += "Raw predictions, timing, token-limit flags, model revisions and metric signatures accompany each model.\n\n"
        body += "| Model | Hardware | Precision | Output-cap hits |\n|---|---|---|---:|\n"
        for r in rows:
            body += f'| {r["model"]} | {r["hardware"]} | {r["precision"]} | {r["hit_token_limit_count"]} |\n'
        if len({(r["hardware"], r["device"]) for r in rows}) > 1:
            body += "\n**Hardware differs across these runs. The latency figures are not a controlled comparison. Rerun on the same hardware to compare model speed.**\n"
        body += "\n"
        body += "![Measured translation scores](results/plots/quality.png)\n\n![Measured inference latency](results/plots/latency.png)\n\n"
        for metric, title, filename in (("bleu", "BLEU", "quality"), ("seconds_per_sentence", "Seconds per sentence", "latency")):
            fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
            names = [r["model"] for r in rows]
            if filename == "quality":
                x = list(range(len(rows)))
                ax.bar([v-.18 for v in x], [r["bleu"] for r in rows], width=.36, label="BLEU", color="#2364AA")
                ax.bar([v+.18 for v in x], [r["chrf"] for r in rows], width=.36, label="chrF", color="#48A9A6")
                ax.set_xticks(x, names)
                ax.legend()
                ax.set_ylabel("Corpus score (0-100)")
            else:
                ax.bar(names, [r[metric] for r in rows], color="#2364AA")
                ax.set_ylabel(title)
            ax.set_title(f"Measured models only ({len(rows)}/4 complete)")
            ax.spines[["top", "right"]].set_visible(False)
            fig.savefig(plots / f"{filename}.png", dpi=180)
            plt.close(fig)
    body += "## Sampling temperature analysis\n\n"
    body += "The primary test comparison uses greedy decoding (sampling disabled). The temperature experiment uses the same validation subset for all models, with top-p=0.9, top-k disabled, and three seeds per nonzero temperature. Greedy decoding runs once. Mean and sample standard deviation below describe seed variability, not confidence intervals. No temperature is selected using the test set.\n\n"
    body += "| Model | Temperature | Runs | Validation BLEU mean ± SD | Validation chrF mean ± SD |\n|---|---:|---:|---:|---:|\n"
    summaries = []
    for r in rows:
        records = read_json(folder / r["model"] / "temperature_metrics.json")
        for t in sorted({v["temperature"] for v in records}):
            group = [v for v in records if v["temperature"] == t]
            bs, cs = [v["bleu"] for v in group], [v["chrf"] for v in group]
            bsd, csd = statistics.stdev(bs) if len(bs)>1 else 0., statistics.stdev(cs) if len(cs)>1 else 0.
            summaries.append({"model": r["model"], "temperature": t, "bleu": statistics.mean(bs), "chrf": statistics.mean(cs), "bleu_sd": bsd, "chrf_sd": csd})
            body += f'| {r["model"]} | {t:g} | {len(group)} | {statistics.mean(bs):.2f} ± {bsd:.2f} | {statistics.mean(cs):.2f} ± {csd:.2f} |\n'
    if summaries:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        for r in rows:
            group = [v for v in summaries if v["model"] == r["model"]]
            ax.errorbar([v["temperature"] for v in group], [v["chrf"] for v in group],
                        yerr=[v["chrf_sd"] for v in group], marker="o", capsize=3, label=r["model"])
            best = max(group, key=lambda v: v["chrf"])
            greedy = next(v for v in group if v["temperature"] == 0)
            body += f'\nFor {r["model"]}, the highest observed mean validation chrF is {best["chrf"]:.2f} at temperature {best["temperature"]:g}; the greedy baseline is {greedy["chrf"]:.2f}. This small validation experiment does not establish a general optimum.\n'
        ax.set(xlabel="Temperature (0 denotes greedy)", ylabel="Validation chrF", title="Temperature sensitivity with seed variability")
        ax.legend()
        fig.savefig(plots / "temperature.png", dpi=180)
        plt.close(fig)
        body += "\n![Temperature experiment](results/plots/temperature.png)\n"
    body += "\n## Model comparison and conclusion\n\n"
    if len(rows) == 4:
        winner = max(rows, key=lambda r: (r["chrf"], r["bleu"]))
        bleu_winner = max(rows, key=lambda r: r["bleu"])
        fastest = min(rows, key=lambda r: r["seconds_per_sentence"])
        body += f'Using chrF as the predeclared primary metric and BLEU as a tie-breaker, **{winner["model"]}** is the best-performing model on this test set (chrF {winner["chrf"]:.2f}, BLEU {winner["bleu"]:.2f}). '
        body += f'The highest BLEU belongs to {bleu_winner["model"]}; {fastest["model"]} has the lowest measured per-sentence latency. '
        body += "These rankings apply to these checkpoints, adaptation budgets, dataset subset and hardware; they do not establish architectural superiority.\n"
    else:
        body += "**The four-model comparison is incomplete. No overall best-performing model can yet be identified.** Run the remaining models and regenerate this section before submitting.\n"
    body += "\nThe basic Transformer starts with random weights and learns a word/punctuation vocabulary only from training sentences. mT5 starts from multilingual pretraining and receives supervised task fine-tuning. LLaMA and Gemma use instruction-tuned checkpoints with three training examples in their prompts. Differences in parameter counts, tokenization, pretraining, prompt length and adaptation prevent a controlled architecture-only comparison. The same token cap is not the same character budget across tokenizers; inspect cap-hit counts. Inference timing uses batch size one, synchronization and a warm-up, and includes tokenization and decoding but excludes model loading, downloads and scoring. Training timing includes validation and best-checkpoint writes, but excludes loading and tokenization.\n"
    (folder / "RESULTS.md").write_text(body.replace("(results/plots/", "(plots/"), encoding="utf-8")
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
        if start in text and end in text:
            text = text.split(start)[0] + start + "\n" + body + "\n" + end + text.split(end, 1)[1]
            readme.write_text(text, encoding="utf-8")
    write_json(folder / "completion.json", {"completed_models": [r["model"] for r in rows], "all_four_complete": len(rows) == 4})
    print(f"Reports generated: {len(rows)}/4 models complete", flush=True)


def verify(config):
    from .experiment import score, AUDIT_CASES
    manifest = read_json(ROOT / "data/manifest.json")
    run_hash = fingerprint({"config": config, "manifest": manifest})
    rows = completed()
    if len(rows) != 4:
        raise SystemExit(f"NOT submission-ready: {len(rows)}/4 models complete")
    validation = read_rows(ROOT / "data/processed/validation.jsonl")
    validation_refs = {r["id"]: r["target"] for r in validation}
    for metric in rows:
        if metric["run_hash"] != run_hash:
            raise ValueError("Results do not match current config/data")
        folder = ROOT / "results" / metric["model"]
        predictions = read_rows(folder / "predictions.jsonl")
        if [r["id"] for r in predictions] != manifest["split_ids"]["test"]:
            raise ValueError("Test IDs/order differ")
        actual = score([r["prediction"] for r in predictions], [r["target"] for r in predictions])
        for key in ("bleu", "chrf"):
            if abs(actual[key] - metric[key]) > 1e-8:
                raise ValueError("Stored scores do not reproduce")
        temp = read_rows(folder / "temperature_predictions.jsonl")
        temp_metrics = read_json(folder / "temperature_metrics.json")
        expected_groups = {(0.0, config["seed"])} | {(t, s) for t in config["temperature_values"] for s in config["temperature_seeds"]}
        if {(r["temperature"], r["seed"]) for r in temp_metrics} != expected_groups or len(temp_metrics) != len(expected_groups):
            raise ValueError("Temperature runs are incomplete or duplicated")
        for record in temp_metrics:
            selected = [r for r in temp if r["temperature"] == record["temperature"] and r["seed"] == record["seed"]]
            expected = manifest["split_ids"]["validation"][:config["temperature_validation_size"]]
            if [r["id"] for r in selected] != expected:
                raise ValueError("Temperature validation IDs differ")
            actual = score([r["prediction"] for r in selected], [validation_refs[r["id"]] for r in selected])
            for key in ("bleu", "chrf"):
                if abs(actual[key]-record[key]) > 1e-8:
                    raise ValueError("Temperature scores do not reproduce")
        if [r["id"] for r in read_rows(folder / "ethics_samples.jsonl")] != [r["id"] for r in AUDIT_CASES]:
            raise ValueError("Ethics probes are incomplete")
    print("All four model results match the dataset/config and test metrics reproduce. Review ethics and submission details separately.")
