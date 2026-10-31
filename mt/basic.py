"""A basic encoder-decoder Transformer trained entirely from scratch."""
import math
import re
from collections import Counter
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

PAD, BOS, EOS, UNK = 0, 1, 2, 3


class Vocabulary:
    def __init__(self, tokens):
        self.tokens = tokens
        self.ids = {token: i for i, token in enumerate(tokens)}

    @staticmethod
    def tokenize(text):
        return re.findall(r"\w+|[^\w\s]", text.casefold(), flags=re.UNICODE)

    @classmethod
    def fit(cls, texts, size):
        counts = Counter(token for text in texts for token in cls.tokenize(text))
        words = sorted(counts, key=lambda token: (-counts[token], token))[:size-4]
        return cls(["<pad>", "<bos>", "<eos>", "<unk>"] + words)

    def encode(self, text, limit):
        ids = [self.ids.get(t, UNK) for t in self.tokenize(text)]
        if len(ids) + 2 > limit:
            raise ValueError("Input exceeds token limit; increase it rather than silently truncating")
        return [BOS] + ids + [EOS]

    def decode(self, ids):
        tokens = []
        for i in ids:
            if i == EOS:
                break
            if i not in (PAD, BOS):
                tokens.append(self.tokens[i])
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return re.sub(r"\s*'\s*", "'", text).strip()


class BasicTransformer(nn.Module):
    def __init__(self, source_size, target_size, config):
        super().__init__()
        d = config["d_model"]
        self.scale = math.sqrt(d)
        self.source_embedding = nn.Embedding(source_size, d, padding_idx=PAD)
        self.target_embedding = nn.Embedding(target_size, d, padding_idx=PAD)
        position = torch.arange(2048).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, d, 2) * (-math.log(10000) / d))
        pe = torch.zeros(2048, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * frequency), torch.cos(position * frequency)
        self.register_buffer("position", pe.unsqueeze(0))
        self.dropout = nn.Dropout(config["dropout"])
        self.transformer = nn.Transformer(d_model=d, nhead=config["nhead"],
            num_encoder_layers=config["layers"], num_decoder_layers=config["layers"],
            dim_feedforward=config["dim_feedforward"], dropout=config["dropout"], batch_first=True)
        self.output = nn.Linear(d, target_size)

    def embedded(self, ids, layer):
        return self.dropout(layer(ids) * self.scale + self.position[:, :ids.size(1)])

    def encode(self, source):
        return self.transformer.encoder(self.embedded(source, self.source_embedding), src_key_padding_mask=source.eq(PAD))

    def decode(self, target, memory, source):
        mask = torch.triu(torch.ones(target.size(1), target.size(1), device=target.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer.decoder(self.embedded(target, self.target_embedding), memory,
            tgt_mask=mask, tgt_key_padding_mask=target.eq(PAD), memory_key_padding_mask=source.eq(PAD))
        return self.output(hidden)

    def forward(self, source, target):
        return self.decode(target, self.encode(source), source)


def collate(pairs):
    source, target = zip(*pairs)
    return (pad_sequence([torch.tensor(s) for s in source], batch_first=True, padding_value=PAD),
            pad_sequence([torch.tensor(t) for t in target], batch_first=True, padding_value=PAD))


def choose_token(logits, temperature, top_p):
    logits = logits.clone()
    logits[:, [PAD, BOS]] = -float("inf")
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid temperature or top_p")
    scores, indices = torch.sort(logits / temperature, descending=True)
    cumulative = torch.softmax(scores, -1).cumsum(-1)
    remove = cumulative > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    scores[remove] = -float("inf")
    sampled = torch.multinomial(torch.softmax(scores, -1), 1)
    return indices.gather(-1, sampled)


@torch.inference_mode()
def translate(model, source_vocab, target_vocab, text, config, device, temperature=0):
    model.eval()
    source = torch.tensor([source_vocab.encode(text, config["max_source_tokens"])], device=device)
    target = torch.tensor([[BOS]], device=device)
    memory = model.encode(source)
    for _ in range(config["max_new_tokens"]):
        nxt = choose_token(model.decode(target, memory, source)[:, -1], temperature, config["top_p"])
        target = torch.cat((target, nxt), dim=1)
        if nxt.item() == EOS:
            break
    ids = target[0, 1:].tolist()
    return target_vocab.decode(ids), len(ids), ids[-1] != EOS
