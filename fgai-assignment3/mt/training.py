import math
import time
import torch
from torch.utils.data import DataLoader
from .basic import BasicTransformer, Vocabulary, PAD, collate
from .common import ROOT, synchronize, write_json


def train_basic(train, validation, config, device, run_hash):
    c = config["basic"]
    source_vocab = Vocabulary.fit([r["source"] for r in train], c["vocab_size"])
    target_vocab = Vocabulary.fit([r["target"] for r in train], c["vocab_size"])
    def encode(rows):
        return [(source_vocab.encode(r["source"], config["max_source_tokens"]),
                 target_vocab.encode(r["target"], config["max_new_tokens"] + 1)) for r in rows]
    train_loader = DataLoader(encode(train), batch_size=c["batch_size"], shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(encode(validation), batch_size=c["batch_size"], collate_fn=collate)
    model = BasicTransformer(len(source_vocab.tokens), len(target_vocab.tokens), c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")
    checkpoint = ROOT / "checkpoints/basic.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    history, best, best_epoch = [], float("inf"), None
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, c["epochs"] + 1):
        totals = {}
        for mode, loader in (("train", train_loader), ("validation", validation_loader)):
            model.train(mode == "train")
            loss_sum, count = 0.0, 0
            with torch.set_grad_enabled(mode == "train"):
                for src, tgt in loader:
                    src, tgt = src.to(device), tgt.to(device)
                    labels = tgt[:, 1:]
                    logits = model(src, tgt[:, :-1])
                    n = int(labels.ne(PAD).sum())
                    loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite basic Transformer loss")
                    if mode == "train":
                        optimizer.zero_grad(set_to_none=True)
                        (loss/n).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    loss_sum += loss.item()
                    count += n
            totals[mode + "_loss"] = loss_sum/count
        history.append({"epoch": epoch, **totals})
        print(f"basic epoch {epoch}: {totals}", flush=True)
        if totals["validation_loss"] < best:
            best, best_epoch = totals["validation_loss"], epoch
            torch.save({"state": model.state_dict(), "source_vocab": source_vocab.tokens,
                        "target_vocab": target_vocab.tokens, "config": c, "run_hash": run_hash}, checkpoint)
    synchronize(device)
    seconds = time.perf_counter() - start
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state"])
    metadata = {"adaptation": "trained from scratch", "training_seconds": seconds,
                "best_epoch": best_epoch, "history": history, "run_hash": run_hash}
    write_json(ROOT / "checkpoints/basic_training.json", metadata)
    return model.eval(), source_vocab, target_vocab, metadata


def load_basic(device, run_hash):
    from .common import read_json
    saved = torch.load(ROOT / "checkpoints/basic.pt", map_location=device, weights_only=True)
    if saved["run_hash"] != run_hash:
        raise ValueError("Checkpoint belongs to a different configuration/dataset; retrain")
    model = BasicTransformer(len(saved["source_vocab"]), len(saved["target_vocab"]), saved["config"]).to(device)
    model.load_state_dict(saved["state"])
    return model.eval(), Vocabulary(saved["source_vocab"]), Vocabulary(saved["target_vocab"]), read_json(ROOT / "checkpoints/basic_training.json")


def train_mt5(train, validation, config, device, run_hash):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Adafactor, DataCollatorForSeq2Seq
    from huggingface_hub import HfApi
    c = config["mt5"]
    revision = HfApi().model_info(c["model_id"], revision=c["revision"]).sha
    tokenizer = AutoTokenizer.from_pretrained(c["model_id"], revision=revision, use_fast=False)
    # FP32 + Adafactor avoids FP16 overflow in T5 and Adam's two full-sized states.
    model = AutoModelForSeq2SeqLM.from_pretrained(c["model_id"], revision=revision).to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    def encode(rows):
        examples = []
        for r in rows:
            item = tokenizer("translate English to French: " + r["source"], truncation=False)
            labels = tokenizer(text_target=r["target"], truncation=False)["input_ids"]
            if len(item["input_ids"]) > config["max_source_tokens"] or len(labels) > config["max_new_tokens"]:
                raise ValueError("mT5 input/target exceeds configured limit")
            item["labels"] = labels
            examples.append(item)
        return examples
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    train_loader = DataLoader(encode(train), batch_size=c["batch_size"], shuffle=True, collate_fn=collator)
    validation_loader = DataLoader(encode(validation), batch_size=c["batch_size"], collate_fn=collator)
    optimizer = Adafactor(model.parameters(), lr=c["learning_rate"], relative_step=False, scale_parameter=False, warmup_init=False)
    folder = ROOT / "checkpoints/mt5"
    history, best, best_epoch = [], float("inf"), None
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, c["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, count = 0., 0
        accumulation = c["gradient_accumulation"]
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite mT5 loss")
            # Correctly scale the final incomplete accumulation group.
            group_start = (step // accumulation) * accumulation
            group_size = min(accumulation, len(train_loader)-group_start)
            (loss/group_size).backward()
            if (step+1) % accumulation == 0 or step+1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            n = int(batch["labels"].ne(-100).sum())
            loss_sum += loss.item()*n
            count += n
        model.eval()
        val_sum, val_count = 0., 0
        with torch.no_grad():
            for batch in validation_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                n = int(batch["labels"].ne(-100).sum())
                val_sum += model(**batch).loss.item()*n
                val_count += n
        val_loss = val_sum/val_count
        if not math.isfinite(val_loss):
            raise RuntimeError("Non-finite mT5 validation loss")
        history.append({"epoch": epoch, "train_loss": loss_sum/count, "validation_loss": val_loss})
        print(f"mt5 epoch {epoch}: {history[-1]}", flush=True)
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            model.save_pretrained(folder)
            tokenizer.save_pretrained(folder)
    synchronize(device)
    seconds = time.perf_counter()-start
    del optimizer, model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    model = AutoModelForSeq2SeqLM.from_pretrained(folder).to(device).eval()
    model.config.use_cache = True
    metadata = {"adaptation": "full-parameter fine-tuning with Adafactor", "training_seconds": seconds,
                "best_epoch": best_epoch, "history": history, "revision": revision, "run_hash": run_hash}
    write_json(folder / "training.json", metadata)
    return model, tokenizer, metadata
