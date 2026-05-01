#!/usr/bin/env python3
"""
Unified DiMSUM runner for VS Code/local Python and Google Colab.

Supports:
  - BERT/RoBERTa/DeBERTa backbones through Hugging Face AutoModel/AutoTokenizer
  - linear multitask heads: MWE + supersense
  - optional CRF decoding for the MWE head
  - train/dev split from training data only
  - prediction file writing in DiMSUM format
  - optional official dimsumeval.py call

Example local/VS Code:
  python dimsum_unified.py --data_dir ./dimsum-data/data --eval_file ./dimsum-data/eval/dimsumeval.py \
    --model_name roberta-base --architecture linear --epochs 3 --batch_size 16

Example Colab:
  !python dimsum_unified.py --data_dir /content/drive/MyDrive/DiMSUM/data \
    --eval_file /content/drive/MyDrive/DiMSUM/eval/dimsumeval.py \
    --model_name microsoft/deberta-v3-small --architecture mtl_crf --epochs 3
"""

from __future__ import annotations

import argparse
from html import parser
import json
import os
import random
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, logging as transformers_logging

try:
    from torchcrf import CRF
except Exception:  # pragma: no cover
    CRF = None

try:
    from sklearn.metrics import f1_score
except Exception:  # pragma: no cover
    f1_score = None

Sentence = List[Tuple[str, str, Optional[str]]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def running_in_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ


def maybe_mount_drive() -> None:
    if not running_in_colab():
        return
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
    except Exception as exc:
        print(f"Google Drive mount skipped: {exc}")


def parse_dimsum_file(file_path: Path) -> List[Sentence]:
    """Parse DiMSUM CoNLL-style file into sentences of (word, mwe_tag, supersense)."""
    sentences: List[Sentence] = []
    current: Sentence = []

    with file_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue

            cols = line.split("\t")
            if len(cols) < 5:
                continue

            word = cols[1]
            mwe_tag = cols[4] if cols[4] else "O"
            sup_tag = cols[7].strip() if len(cols) > 7 and cols[7].strip() else None
            current.append((word, mwe_tag, sup_tag))

    if current:
        sentences.append(current)
    return sentences


def build_vocabs(train_data: Sequence[Sentence]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build label vocabularies from training data only. This avoids test-label leakage."""
    mwe_vocab = {"O"}
    sup_vocab = {"O"}
    for sentence in train_data:
        for _, mwe, sup in sentence:
            mwe_vocab.add(mwe if mwe else "O")
            if sup:
                sup_vocab.add(sup)
    return ({tag: i for i, tag in enumerate(sorted(mwe_vocab))},
            {tag: i for i, tag in enumerate(sorted(sup_vocab))})


def invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    return {idx: tag for tag, idx in vocab.items()}


class DiMSUMDataset(Dataset):
    def __init__(self, data: Sequence[Sentence], tokenizer, max_len: int,
                 mwe2id: Dict[str, int], sup2id: Dict[str, int]):
        self.data = list(data)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mwe2id = mwe2id
        self.sup2id = sup2id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        sentence = self.data[index]
        words = [x[0] for x in sentence]
        mwe_tags = [x[1] for x in sentence]
        sup_tags = [x[2] for x in sentence]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids()

        mwe_label_ids: List[int] = []
        sup_label_ids: List[int] = []
        first_subword_mask: List[int] = []

        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                mwe_label_ids.append(self.mwe2id["O"])
                sup_label_ids.append(-100)
                first_subword_mask.append(0)
            elif word_idx != prev_word_idx:
                mwe_label_ids.append(self.mwe2id.get(mwe_tags[word_idx], self.mwe2id["O"]))
                sup = sup_tags[word_idx]
                sup_label_ids.append(self.sup2id.get(sup, self.sup2id["O"]) if sup else self.sup2id["O"])
                first_subword_mask.append(1)
            else:
                mwe_label_ids.append(self.mwe2id["O"])
                sup_label_ids.append(-100)
                first_subword_mask.append(0)
            prev_word_idx = word_idx

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        return (
            item["input_ids"],
            item["attention_mask"],
            torch.tensor(first_subword_mask, dtype=torch.bool),
            torch.tensor(mwe_label_ids, dtype=torch.long),
            torch.tensor(sup_label_ids, dtype=torch.long),
        )


class LinearMultitaskTagger(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_mwe_tags: int,
        num_sup_tags: int,
        dropout: float = 0.1,
        mwe_loss_weight: float = 1.0,
        sup_loss_weight: float = 1.0,
    ):
        super().__init__()
        transformers_logging.set_verbosity_error()
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True).float()
        transformers_logging.set_verbosity_warning()
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.mwe_head = nn.Linear(hidden, num_mwe_tags)
        self.sup_head = nn.Linear(hidden, num_sup_tags)
        self.num_mwe_tags = num_mwe_tags
        self.num_sup_tags = num_sup_tags
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.mwe_loss_weight = mwe_loss_weight
        self.sup_loss_weight = sup_loss_weight

    def forward(self, input_ids, attention_mask, first_subword_mask=None, mwe_tags=None, sup_tags=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        out = self.dropout(out)
        out = out.to(self.mwe_head.weight.dtype)
        mwe_logits = self.mwe_head(out)
        sup_logits = self.sup_head(out)
        mwe_preds = torch.argmax(mwe_logits, dim=-1).tolist()
        sup_preds = torch.argmax(sup_logits, dim=-1)

        if mwe_tags is not None and sup_tags is not None:
            # Only score first subwords for MWE; ignore special/padding/trailing subwords.
            masked_mwe_tags = mwe_tags.masked_fill(~first_subword_mask, -100)
            mwe_loss = self.loss_fn(mwe_logits.view(-1, self.num_mwe_tags), masked_mwe_tags.view(-1))
            sup_loss = self.loss_fn(sup_logits.view(-1, self.num_sup_tags), sup_tags.view(-1))
            loss = (
                self.mwe_loss_weight * mwe_loss
                + self.sup_loss_weight * sup_loss
            )
            return loss, mwe_preds, sup_preds
        return mwe_preds, sup_preds


class CRFMultitaskTagger(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_mwe_tags: int,
        num_sup_tags: int,
        dropout: float = 0.1,
        mwe_loss_weight: float = 1.0,
        sup_loss_weight: float = 1.0,
    ):
        super().__init__()
        if CRF is None:
            raise RuntimeError("pytorch-crf is required for --architecture mtl_crf. Install with: pip install pytorch-crf")
        transformers_logging.set_verbosity_error()
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True).float()
        transformers_logging.set_verbosity_warning()
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.mwe_head = nn.Linear(hidden, num_mwe_tags)
        self.sup_head = nn.Linear(hidden, num_sup_tags)
        self.crf = CRF(num_mwe_tags, batch_first=True)
        self.num_mwe_tags = num_mwe_tags
        self.num_sup_tags = num_sup_tags
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.mwe_loss_weight = mwe_loss_weight
        self.sup_loss_weight = sup_loss_weight

    @staticmethod
    def _pack_crf_inputs(
        mwe_logits: torch.Tensor,
        first_subword_mask: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        mwe_tags: Optional[torch.Tensor] = None,
    ):
        if first_subword_mask is None:
            return mwe_logits, mwe_tags, attention_mask.bool()

        batch_size, _, num_tags = mwe_logits.shape
        packed_positions = []
        max_len = 0

        for i in range(batch_size):
            positions = torch.where(first_subword_mask[i])[0]
            packed_positions.append(positions)
            max_len = max(max_len, int(positions.numel()))

        packed_logits = mwe_logits.new_zeros((batch_size, max_len, num_tags))
        packed_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=mwe_logits.device)
        packed_tags = None
        if mwe_tags is not None:
            packed_tags = mwe_tags.new_zeros((batch_size, max_len))

        for i, positions in enumerate(packed_positions):
            length = int(positions.numel())
            packed_logits[i, :length] = mwe_logits[i, positions]
            packed_mask[i, :length] = True
            if packed_tags is not None:
                packed_tags[i, :length] = mwe_tags[i, positions]

        return packed_logits, packed_tags, packed_mask

    def forward(self, input_ids, attention_mask, first_subword_mask=None, mwe_tags=None, sup_tags=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        out = self.dropout(out)
        out = out.to(self.mwe_head.weight.dtype)
        mwe_logits = self.mwe_head(out)
        sup_logits = self.sup_head(out)

        # torchcrf expects each sequence mask to be contiguous. Pack first-subword
        # emissions into word-level CRF sequences before scoring/decoding.
        crf_logits, crf_tags, crf_mask = self._pack_crf_inputs(
            mwe_logits,
            first_subword_mask,
            attention_mask,
            mwe_tags,
        )

        mwe_preds = self.crf.decode(crf_logits, mask=crf_mask)
        sup_preds = torch.argmax(sup_logits, dim=-1)

        if mwe_tags is not None and sup_tags is not None:
            mwe_loss = -self.crf(crf_logits, crf_tags, mask=crf_mask, reduction="mean")
            sup_loss = self.loss_fn(sup_logits.view(-1, self.num_sup_tags), sup_tags.view(-1))

            loss = (
                self.mwe_loss_weight * mwe_loss
                + self.sup_loss_weight * sup_loss
            )
            return loss, mwe_preds, sup_preds
        return mwe_preds, sup_preds


def make_model(
    architecture: str,
    model_name: str,
    num_mwe_tags: int,
    num_sup_tags: int,
    dropout: float,
    mwe_loss_weight: float,
    sup_loss_weight: float,
):
    if architecture == "linear":
        return LinearMultitaskTagger(
            model_name,
            num_mwe_tags,
            num_sup_tags,
            dropout,
            mwe_loss_weight,
            sup_loss_weight,
        )
    if architecture == "mtl_crf":
        return CRFMultitaskTagger(
            model_name,
            num_mwe_tags,
            num_sup_tags,
            dropout,
            mwe_loss_weight,
            sup_loss_weight,
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def split_train_dev(data: Sequence[Sentence], dev_split: float, seed: int):
    items = list(data)
    random.Random(seed).shuffle(items)
    split = int(len(items) * (1.0 - dev_split))
    return items[:split], items[split:]


def train_one(
    model,
    loader,
    device,
    epochs: int,
    lr: float,
    grad_clip: float,
):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        total = 0.0

        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")

        for batch in bar:
            batch = [x.to(device) for x in batch]

            optimizer.zero_grad(set_to_none=True)

            loss, _, _ = model(*batch[:3], mwe_tags=batch[3], sup_tags=batch[4])

            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total += float(loss.item())
            bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total / max(len(loader), 1)

        row = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }

        loss_history.append(row)

        print(
            f"epoch {epoch + 1}: "
            f"avg_train_loss={avg_train_loss:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

    return model, loss_history


def macro_f1(y_true, y_pred) -> float:
    if not y_true:
        return 0.0
    if f1_score is None:
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        return correct / len(y_true)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def decode_mwe_predictions(architecture: str, raw_preds, valid_indices: torch.Tensor, id2mwe: Dict[int, str]) -> List[str]:
    if architecture == "mtl_crf":
        return [id2mwe[raw_preds[j]] for j in range(len(valid_indices))]
    return [id2mwe[raw_preds[int(idx)]] for idx in valid_indices]


def evaluate_dev(model, loader, device, architecture: str, id2mwe: Dict[int, str], id2sup: Dict[int, str]):
    model.eval()
    total_loss = 0.0
    flat_mwe_t, flat_mwe_p = [], []
    flat_sup_t, flat_sup_p = [], []
    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device) for x in batch]
            input_ids, attention_mask, first_mask, mwe_tags, sup_tags = batch
            loss, mwe_preds, sup_preds = model(input_ids, attention_mask, first_mask, mwe_tags, sup_tags)
            total_loss += float(loss.item())
            for i in range(input_ids.size(0)):
                valid_indices = torch.where(first_mask[i])[0]
                mwe_p = decode_mwe_predictions(architecture, mwe_preds[i], valid_indices, id2mwe)
                mwe_t = [id2mwe[int(mwe_tags[i, idx])] for idx in valid_indices]
                sup_p_raw = [id2sup[int(sup_preds[i, idx])] for idx in valid_indices]
                sup_t_raw = [id2sup[int(sup_tags[i, idx])] if int(sup_tags[i, idx]) != -100 else "O" for idx in valid_indices]
                flat_mwe_t.extend(mwe_t)
                flat_mwe_p.extend(mwe_p)
                flat_sup_t.extend(sup_t_raw)
                flat_sup_p.extend(sup_p_raw)
    return {
        "dev_loss": total_loss / max(len(loader), 1),
        "mwe_macro_f1": macro_f1(flat_mwe_t, flat_mwe_p),
        "sup_macro_f1": macro_f1(flat_sup_t, flat_sup_p),
    }


def normalize_mwe_tag(tag: Optional[str]) -> str:
    """
    Collapse DiMSUM's gappy lowercase MWE tags into simple BIO.

    The official DiMSUM evaluator supports lowercase b/i/o only for valid
    discontinuous MWEs. This baseline does not explicitly model gappy MWEs,
    so writing lowercase sequences can create invalid outputs such as "bio".
    For stable evaluation, predicted MWEs are normalized to standard B/I/O.
    """
    if tag in {"B", "b"}:
        return "B"
    if tag in {"I", "i"}:
        return "I"
    return "O"


def clean_mwe_tags(tags: Sequence[str]) -> List[str]:
    """
    DiMSUM-aware cleanup for the official 6-tag scheme.

    Keeps: B, I, O, b, i, o

    Valid examples:
      B I
      B I I
      B o o I
      B b i I

    Invalid examples repaired:
      B O        -> O O
      B o O      -> O O O
      B o o i I  -> B o o o I
      I O        -> O O
      o I        -> O O
    """
    valid = {"B", "I", "O", "b", "i", "o"}
    tags = [t if t in valid else "O" for t in tags]
    n = len(tags)
    out = ["O"] * n

    i = 0
    while i < n:
        t = tags[i]

        if t == "O":
            i += 1
            continue

        # These cannot start an outer DiMSUM MWE chunk.
        if t in {"I", "i", "o"}:
            i += 1
            continue

        # Uppercase B starts the outer MWE chunk.
        # Lowercase b at the start is repaired as uppercase B.
        if t in {"B", "b"}:
            start = i
            out[start] = "B"

            j = i + 1
            found_upper_I = False
            lower_b_active = False

            while j < n:
                tj = tags[j]

                if tj == "O":
                    break

                if tj == "B":
                    # New chunk begins. Stop current chunk here.
                    break

                if tj == "I":
                    out[j] = "I"
                    found_upper_I = True
                    lower_b_active = False
                    j += 1
                    continue

                if tj == "o":
                    out[j] = "o"
                    lower_b_active = False
                    j += 1
                    continue

                if tj == "b":
                    # Lowercase b starts a nested/weak expression inside a larger MWE.
                    # It only stays b if followed by at least one i.
                    if j + 1 < n and tags[j + 1] == "i":
                        out[j] = "b"
                        lower_b_active = True
                    else:
                        out[j] = "o"
                        lower_b_active = False
                    j += 1
                    continue

                if tj == "i":
                    if lower_b_active:
                        out[j] = "i"
                    else:
                        # Bare lowercase i is invalid after o/I/B.
                        # Treat it as a gap marker inside the outer MWE.
                        out[j] = "o"
                    j += 1
                    continue

                break

            if not found_upper_I:
                # Singleton B or B...o without a final I is not a valid MWE.
                for k in range(start, j):
                    out[k] = "O"
            else:
                # Valid DiMSUM chunks must end with an uppercase I.
                last_upper_I = max(k for k in range(start + 1, j) if out[k] == "I")
                for k in range(last_upper_I + 1, j):
                    out[k] = "O"

            i = j
            continue

        i += 1

    return out


def write_prediction_file(
    test_file: Path,
    pred_file: Path,
    mwe_preds: List[List[str]],
    sup_preds: List[List[Optional[str]]],
):
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned_mwe = [clean_mwe_tags(x) for x in mwe_preds]

    with test_file.open("r", encoding="utf-8") as f_in, pred_file.open("w", encoding="utf-8") as f_out:
        sent_idx, word_idx = 0, 0
        strong_head = "0"  # head for B ... I
        weak_head = "0"    # head for b ... i

        for raw_line in f_in:
            line = raw_line.rstrip("\n")

            if not line.strip():
                f_out.write("\n")
                sent_idx += 1
                word_idx = 0
                strong_head = "0"
                weak_head = "0"
                continue

            cols = line.split("\t")
            while len(cols) < 8:
                cols.append("")

            if sent_idx < len(cleaned_mwe) and word_idx < len(cleaned_mwe[sent_idx]):
                mwe = cleaned_mwe[sent_idx][word_idx]
                sup = sup_preds[sent_idx][word_idx]

                if mwe == "B":
                    strong_head = cols[0]
                    weak_head = "0"
                    cols[4] = "B"
                    cols[5] = "0"
                    cols[6] = ""
                    cols[7] = sup if sup and sup != "O" else ""

                elif mwe == "I":
                    cols[4] = "I"
                    cols[5] = strong_head if strong_head != "0" else "0"
                    cols[6] = ""
                    cols[7] = ""

                elif mwe == "o":
                    # Gap token inside a discontinuous MWE.
                    # It is not linked to the MWE group, but it may have its own supersense.
                    cols[4] = "o"
                    cols[5] = "0"
                    cols[6] = ""
                    cols[7] = sup if sup and sup != "O" else ""

                elif mwe == "b":
                    weak_head = cols[0]
                    cols[4] = "b"
                    cols[5] = "0"
                    cols[6] = ""
                    cols[7] = sup if sup and sup != "O" else ""

                elif mwe == "i":
                    cols[4] = "i"
                    cols[5] = weak_head if weak_head != "0" else "0"
                    cols[6] = ""
                    cols[7] = ""

                else:
                    strong_head = "0"
                    weak_head = "0"
                    cols[4] = "O"
                    cols[5] = "0"
                    cols[6] = ""
                    cols[7] = sup if sup and sup != "O" else ""

            f_out.write("\t".join(cols) + "\n")
            word_idx += 1


def predict_and_write(model, loader, device, architecture: str, id2mwe, id2sup, test_file: Path, pred_file: Path):
    model.eval()
    all_mwe_preds: List[List[str]] = []
    all_sup_preds: List[List[Optional[str]]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict"):
            batch = [x.to(device) for x in batch]
            input_ids, attention_mask, first_mask, _, _ = batch
            mwe_preds, sup_preds = model(input_ids, attention_mask, first_mask)
            for i in range(input_ids.size(0)):
                valid_indices = torch.where(first_mask[i])[0]
                all_mwe_preds.append(decode_mwe_predictions(architecture, mwe_preds[i], valid_indices, id2mwe))
                raw_sup = [id2sup[int(sup_preds[i, idx])] for idx in valid_indices]
                all_sup_preds.append([x if x != "O" else None for x in raw_sup])
    write_prediction_file(test_file, pred_file, all_mwe_preds, all_sup_preds)


def run_official_eval(eval_file: Optional[Path], gold_file: Path, pred_file: Path) -> str:
    if not eval_file or not eval_file.exists():
        return "Official evaluator not found; skipped."

    cmd = ["python", str(eval_file), "-C", str(gold_file), str(pred_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    if result.returncode != 0:
        raise RuntimeError(
            f"Official evaluator failed with exit code {result.returncode}.\n"
            f"Command: {' '.join(cmd)}\n\n{output}"
        )

    return output


def _ratio_decimal_to_percent(value: str) -> Optional[float]:
    try:
        return float(value) * 100.0
    except Exception:
        return None


def parse_official_scores(text: str) -> Dict[str, float]:
    """
    Parse official dimsumeval.py summary lines.

    Expected lines look like:
      MWEs: P=145/537=0.2700 R=145/1115=0.1300 F=17.55%
      Supersenses: P=1498/3851=0.3890 R=1498/4745=0.3157 F=34.85%
      Combined: Acc=11493/16500=0.6965 P=1643/4388=0.3744 R=1643/5860=0.2804 F=32.06%

    Values are returned as percentages for easy table generation.
    """
    scores: Dict[str, float] = {}

    task_map = {
        "MWEs": "mwe",
        "Supersenses": "sup",
        "Combined": "combined",
    }

    for raw in text.splitlines():
        line = raw.strip()
        for label, prefix in task_map.items():
            if not line.startswith(label + ":"):
                continue

            acc = re.search(r"Acc=[^=\s]+=[ ]*([0-9.]+)", line)
            p = re.search(r"\bP=[^=\s]+=[ ]*([0-9.]+)", line)
            r = re.search(r"\bR=[^=\s]+=[ ]*([0-9.]+)", line)
            f = re.search(r"\bF=([0-9.]+)%", line)

            if acc:
                val = _ratio_decimal_to_percent(acc.group(1))
                if val is not None:
                    scores[f"official_{prefix}_acc"] = val
            if p:
                val = _ratio_decimal_to_percent(p.group(1))
                if val is not None:
                    scores[f"official_{prefix}_precision"] = val
            if r:
                val = _ratio_decimal_to_percent(r.group(1))
                if val is not None:
                    scores[f"official_{prefix}_recall"] = val
            if f:
                scores[f"official_{prefix}_f1"] = float(f.group(1))

    return scores


@dataclass
class RunResult:
    architecture: str
    model_name: str
    lr: float
    epochs: int
    batch_size: int
    dev_loss: float
    mwe_macro_f1: float
    sup_macro_f1: float
    pred_file: str
    model_file: str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("./dimsum-data/data"))
    parser.add_argument("--train_file", type=Path, default=None)
    parser.add_argument("--test_file", type=Path, default=None)
    parser.add_argument("--eval_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("./runs"))
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--architecture", choices=["linear", "mtl_crf"], default="linear")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--dev_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--mount_drive", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quick_cpu", action="store_true", help="Small, CPU-friendly run for smoke testing.")
    parser.add_argument("--mwe_loss_weight", type=float, default=1.0)
    parser.add_argument("--sup_loss_weight", type=float, default=1.0)
    args = parser.parse_args()

    if args.mount_drive:
        maybe_mount_drive()

    if args.quick_cpu:
        args.cpu = True
        args.model_name = "distilbert-base-uncased"
        args.epochs = min(args.epochs, 1)
        args.batch_size = min(args.batch_size, 4)
        args.max_len = min(args.max_len, 96)

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}")

    train_file = args.train_file or args.data_dir / "dimsum16.train"
    test_file = args.test_file or args.data_dir / "dimsum16.test"
    if args.eval_file is None:
        candidates = [
            args.data_dir / "scripts" / "dimsumeval.py",
            args.data_dir.parent / "scripts" / "dimsumeval.py",
            args.data_dir.parent / "eval" / "dimsumeval.py",
        ]
        args.eval_file = next((c for c in candidates if c.exists()), None)

    print(f"train_file={train_file}")
    print(f"test_file={test_file}")
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError("Could not find train/test files. Set --data_dir or pass --train_file and --test_file.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_data = parse_dimsum_file(train_file)
    test_data = parse_dimsum_file(test_file)
    mwe2id, sup2id = build_vocabs(train_data)
    id2mwe, id2sup = invert_vocab(mwe2id), invert_vocab(sup2id)
    print(f"train_sentences={len(train_data)} test_sentences={len(test_data)}")
    print(f"mwe_labels={len(mwe2id)} sup_labels={len(sup2id)}")

    train_split, dev_split = split_train_dev(train_data, args.dev_split, args.seed)
    train_loader = DataLoader(DiMSUMDataset(train_split, tokenizer, args.max_len, mwe2id, sup2id), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(DiMSUMDataset(dev_split, tokenizer, args.max_len, mwe2id, sup2id), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(DiMSUMDataset(test_data, tokenizer, args.max_len, mwe2id, sup2id), batch_size=args.batch_size, shuffle=False)

    safe_model_name = args.model_name.replace("/", "__")
    run_name = f"{args.architecture}_{safe_model_name}_lr{args.lr}_ep{args.epochs}_bs{args.batch_size}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_file = run_dir / "predictions.pred"
    model_file = run_dir / "model.pt"
    label_file = run_dir / "labels.json"

    with label_file.open("w", encoding="utf-8") as f:
        json.dump({"mwe2id": mwe2id, "sup2id": sup2id}, f, indent=2)

    start = time.time()
    model = make_model(
        args.architecture,
        args.model_name,
        len(mwe2id),
        len(sup2id),
        args.dropout,
        args.mwe_loss_weight,
        args.sup_loss_weight,
    )
    model, loss_history = train_one(
        model,
        train_loader,
        device,
        args.epochs,
        args.lr,
        args.grad_clip,
    )
    dev_metrics = evaluate_dev(model, dev_loader, device, args.architecture, id2mwe, id2sup)
    print("dev_metrics=", dev_metrics)

    torch.save(model.state_dict(), model_file)
    predict_and_write(model, test_loader, device, args.architecture, id2mwe, id2sup, test_file, pred_file)
    eval_text = run_official_eval(args.eval_file, test_file, pred_file)
    with (run_dir / "official_eval.txt").open("w", encoding="utf-8") as f:
        f.write(eval_text)
    print(eval_text)

    result = RunResult(
        architecture=args.architecture,
        model_name=args.model_name,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dev_loss=dev_metrics["dev_loss"],
        mwe_macro_f1=dev_metrics["mwe_macro_f1"],
        sup_macro_f1=dev_metrics["sup_macro_f1"],
        pred_file=str(pred_file),
        model_file=str(model_file),
    )
    summary = {
        **asdict(result),
        **parse_official_scores(eval_text),
        "seconds": round(time.time() - start, 2),
        "loss_history": loss_history,
        "dropout": args.dropout,
        "grad_clip": args.grad_clip,
        "mwe_loss_weight": args.mwe_loss_weight,
        "sup_loss_weight": args.sup_loss_weight,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    loss_csv = run_dir / "loss_history.csv"
    
    with loss_csv.open("w", encoding="utf-8") as f:
        f.write("epoch,train_loss,lr\n")
        for row in loss_history:
            f.write(f"{row['epoch']},{row['train_loss']},{row['lr']}\n")


if __name__ == "__main__":
    main()
