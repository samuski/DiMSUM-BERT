#!/usr/bin/env python3
"""
DiMSUM report generator.

Normal use:
  python dimsum_report.py

It auto-detects the latest run under ./runs and writes:
  runs/<latest>/report/

Useful manual modes:
  python dimsum_report.py --run_dir runs/<run_name>
  python dimsum_report.py --gold dimsum-data/dimsum16.test --pred runs/<run>/predictions.pred --out runs/<run>/report
  python dimsum_report.py --all_runs
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Token = Dict[str, object]

# Colorblind-safe choices:
# - Okabe-Ito for categorical bar charts
# - cividis for heatmaps
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]
HEATMAP_CMAP = "cividis"


def read_dimsum(path: Path) -> List[List[Token]]:
    """Read DiMSUM CoNLL-style data.

    Important columns:
      0 token id, 1 word, 2 lemma, 3 POS, 4 MWE tag, 5 MWE parent, 7 supersense
    """
    sents: List[List[Token]] = []
    sent: List[Token] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if sent:
                    sents.append(sent)
                    sent = []
                continue

            cols = line.split("\t")
            while len(cols) < 8:
                cols.append("")

            sent.append({
                "id": cols[0],
                "word": cols[1],
                "lemma": cols[2],
                "pos": cols[3],
                "mwe": cols[4] if cols[4] else "O",
                "mwe_parent": cols[5] if cols[5] else "0",
                "sup": cols[7] if cols[7] else "O",
                "raw_cols": cols,
            })

    if sent:
        sents.append(sent)
    return sents


def align_gold_pred(gold_sents: List[List[Token]], pred_sents: List[List[Token]]) -> List[Dict[str, object]]:
    if len(gold_sents) != len(pred_sents):
        raise ValueError(f"Sentence count mismatch: gold={len(gold_sents)} pred={len(pred_sents)}")

    rows: List[Dict[str, object]] = []
    for sent_id, (g_sent, p_sent) in enumerate(zip(gold_sents, pred_sents), start=1):
        if len(g_sent) != len(p_sent):
            raise ValueError(f"Token count mismatch in sentence {sent_id}: gold={len(g_sent)} pred={len(p_sent)}")

        sent_text = " ".join(str(t["word"]) for t in g_sent)
        for tok_id, (g, p) in enumerate(zip(g_sent, p_sent), start=1):
            gold_mwe = str(g["mwe"] or "O")
            pred_mwe = str(p["mwe"] or "O")
            gold_sup = str(g["sup"] or "O")
            pred_sup = str(p["sup"] or "O")

            mwe_correct = gold_mwe == pred_mwe
            sup_correct = gold_sup == pred_sup

            error_parts: List[str] = []
            if not mwe_correct:
                if gold_mwe != "O" and pred_mwe == "O":
                    error_parts.append("mwe_missed")
                elif gold_mwe == "O" and pred_mwe != "O":
                    error_parts.append("mwe_false_positive")
                else:
                    error_parts.append("mwe_boundary_or_tag")
            if not sup_correct:
                if gold_sup != "O" and pred_sup == "O":
                    error_parts.append("sup_missed")
                elif gold_sup == "O" and pred_sup != "O":
                    error_parts.append("sup_false_positive")
                else:
                    error_parts.append("sup_wrong_label")

            rows.append({
                "sent_id": sent_id,
                "tok_id": tok_id,
                "sentence": sent_text,
                "word": g["word"],
                "lemma": g["lemma"],
                "pos": g["pos"],
                "gold_mwe": gold_mwe,
                "pred_mwe": pred_mwe,
                "gold_sup": gold_sup,
                "pred_sup": pred_sup,
                "mwe_correct": mwe_correct,
                "sup_correct": sup_correct,
                "combined_correct": mwe_correct and sup_correct,
                "error_type": "+".join(error_parts) if error_parts else "correct",
            })
    return rows


def safe_pct(numer: float, denom: float) -> float:
    return 100.0 * numer / denom if denom else 0.0


def precision_recall_f1(tp: int, pred_pos: int, gold_pos: int) -> Tuple[float, float, float]:
    p = tp / pred_pos if pred_pos else 0.0
    r = tp / gold_pos if gold_pos else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def compute_basic_metrics(rows: List[Dict[str, object]]) -> Dict[str, object]:
    total = len(rows)

    mwe_correct = sum(bool(r["mwe_correct"]) for r in rows)
    sup_correct = sum(bool(r["sup_correct"]) for r in rows)
    combined_correct = sum(bool(r["combined_correct"]) for r in rows)

    gold_mwe_pos = sum(str(r["gold_mwe"]) != "O" for r in rows)
    pred_mwe_pos = sum(str(r["pred_mwe"]) != "O" for r in rows)
    tp_mwe_presence = sum(str(r["gold_mwe"]) != "O" and str(r["pred_mwe"]) != "O" for r in rows)

    gold_sup_pos = sum(str(r["gold_sup"]) != "O" for r in rows)
    pred_sup_pos = sum(str(r["pred_sup"]) != "O" for r in rows)
    tp_sup_exact = sum(str(r["gold_sup"]) != "O" and str(r["gold_sup"]) == str(r["pred_sup"]) for r in rows)

    mwe_p, mwe_r, mwe_f = precision_recall_f1(tp_mwe_presence, pred_mwe_pos, gold_mwe_pos)
    sup_p, sup_r, sup_f = precision_recall_f1(tp_sup_exact, pred_sup_pos, gold_sup_pos)

    return {
        "tokens": total,
        "token_mwe_accuracy": safe_pct(mwe_correct, total),
        "token_sup_accuracy": safe_pct(sup_correct, total),
        "token_combined_accuracy": safe_pct(combined_correct, total),
        "token_mwe_presence_precision": 100 * mwe_p,
        "token_mwe_presence_recall": 100 * mwe_r,
        "token_mwe_presence_f1": 100 * mwe_f,
        "token_sup_exact_precision": 100 * sup_p,
        "token_sup_exact_recall": 100 * sup_r,
        "token_sup_exact_f1": 100 * sup_f,
        "gold_mwe_tokens": gold_mwe_pos,
        "pred_mwe_tokens": pred_mwe_pos,
        "gold_sup_tokens": gold_sup_pos,
        "pred_sup_tokens": pred_sup_pos,
    }


def parse_official_eval_text(text: str) -> Dict[str, float]:
    """Parse official P/R/F and combined accuracy from dimsumeval.py output.

    This version is deliberately lenient. It handles:
      MWEs: P=145/537=0.2700 R=145/1115=0.1300 F=17.55%
      Supersenses: P=1498/3851=0.3890 R=1498/4745=0.3157 F=34.85%
      Combined: Acc=11493/16500=0.6965 P=1643/4388=0.3744 R=1643/5860=0.2804 F=32.06%

    Returned values are percentages, e.g. 27.00 instead of 0.2700.
    """
    out: Dict[str, float] = {}

    def as_percent_from_decimal(value: str) -> float:
        return 100.0 * float(value)

    # Normalize whitespace, but also parse line-by-line first so a greedy regex cannot
    # accidentally jump across the giant confusion-matrix dump.
    clean_lines = [" ".join(line.strip().split()) for line in text.replace("\r", "\n").split("\n")]

    def find_summary_line(prefix: str) -> Optional[str]:
        for line in clean_lines:
            if line.startswith(prefix + ":"):
                return line
        # Fallback: search the flattened text.
        flat = " ".join(clean_lines)
        m = re.search(rf"({re.escape(prefix)}:\s+.*?)(?=\s+(?:MWEs|Supersenses|Combined):|$)", flat)
        return m.group(1) if m else None

    def extract_ratio_decimal(line: str, label: str) -> Optional[float]:
        # Example: P=145/537=0.2700
        m = re.search(rf"\b{label}=([0-9]+)/([0-9]+)=([0-9.]+)", line)
        if m:
            return as_percent_from_decimal(m.group(3))

        # More lenient fallback: P=0.2700 or P=27.00%
        m = re.search(rf"\b{label}=([0-9.]+)%", line)
        if m:
            return float(m.group(1))
        m = re.search(rf"\b{label}=([0-9.]+)", line)
        if m:
            val = float(m.group(1))
            return val if val > 1.0 else 100.0 * val
        return None

    def extract_f(line: str) -> Optional[float]:
        # Example: F=17.55%
        m = re.search(r"\bF=([0-9.]+)%", line)
        if m:
            return float(m.group(1))
        m = re.search(r"\bF=([0-9.]+)", line)
        if m:
            val = float(m.group(1))
            return val if val > 1.0 else 100.0 * val
        return None

    mwe_line = find_summary_line("MWEs")
    if mwe_line:
        p_val = extract_ratio_decimal(mwe_line, "P")
        r_val = extract_ratio_decimal(mwe_line, "R")
        f_val = extract_f(mwe_line)
        if p_val is not None:
            out["official_mwe_precision"] = p_val
        if r_val is not None:
            out["official_mwe_recall"] = r_val
        if f_val is not None:
            out["official_mwe_f1"] = f_val

    sup_line = find_summary_line("Supersenses")
    if sup_line:
        p_val = extract_ratio_decimal(sup_line, "P")
        r_val = extract_ratio_decimal(sup_line, "R")
        f_val = extract_f(sup_line)
        if p_val is not None:
            out["official_sup_precision"] = p_val
        if r_val is not None:
            out["official_sup_recall"] = r_val
        if f_val is not None:
            out["official_sup_f1"] = f_val

    combined_line = find_summary_line("Combined")
    if combined_line:
        acc_val = extract_ratio_decimal(combined_line, "Acc")
        p_val = extract_ratio_decimal(combined_line, "P")
        r_val = extract_ratio_decimal(combined_line, "R")
        f_val = extract_f(combined_line)
        if acc_val is not None:
            out["official_combined_accuracy"] = acc_val
        if p_val is not None:
            out["official_combined_precision"] = p_val
        if r_val is not None:
            out["official_combined_recall"] = r_val
        if f_val is not None:
            out["official_combined_f1"] = f_val

    return out


def load_summary(summary_path: Optional[Path], eval_log: Optional[Path]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    if summary_path and summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary.update(json.load(f))

    if eval_log and eval_log.exists():
        summary.update(parse_official_eval_text(eval_log.read_text(encoding="utf-8", errors="replace")))

    for key in list(summary.keys()):
        if isinstance(summary[key], str) and str(summary[key]).endswith("%"):
            try:
                summary[key] = float(str(summary[key]).rstrip("%"))
            except Exception:
                pass
    return summary


def official_score_rows(summary: Dict[str, object], metrics: Dict[str, object]) -> List[Dict[str, object]]:
    """Build score rows.

    Priority:
      1. Official evaluator P/R/F when parsed from official_eval.txt.
      2. Token-level fallback P/R/F so cells are never blank.
      3. Accuracy: official combined accuracy when available; otherwise token-level sanity check.
    """
    rows = []

    specs = [
        (
            "MWE",
            "official_mwe",
            "token_mwe_presence_precision",
            "token_mwe_presence_recall",
            "token_mwe_presence_f1",
            "token_mwe_accuracy",
        ),
        (
            "Supersense",
            "official_sup",
            "token_sup_exact_precision",
            "token_sup_exact_recall",
            "token_sup_exact_f1",
            "token_sup_accuracy",
        ),
        (
            "Combined",
            "official_combined",
            "token_combined_accuracy",
            "token_combined_accuracy",
            "token_combined_accuracy",
            "token_combined_accuracy",
        ),
    ]

    for task, prefix, fallback_p, fallback_r, fallback_f, fallback_acc in specs:
        has_official_prf = (
            f"{prefix}_precision" in summary
            or f"{prefix}_recall" in summary
            or f"{prefix}_f1" in summary
        )
        rows.append({
            "task": task,
            "precision": summary.get(f"{prefix}_precision", metrics.get(fallback_p, "")),
            "recall": summary.get(f"{prefix}_recall", metrics.get(fallback_r, "")),
            "f1": summary.get(f"{prefix}_f1", metrics.get(fallback_f, "")),
            "accuracy": summary.get(f"{prefix}_accuracy", metrics.get(fallback_acc, "")),
            "score_source": "official" if has_official_prf else "token-level fallback",
            "accuracy_note": "official" if f"{prefix}_accuracy" in summary else "token-level sanity check",
        })

    return rows


def per_label_metrics(rows: List[Dict[str, object]], gold_key: str, pred_key: str, exclude_o: bool = True) -> List[Dict[str, object]]:
    labels = sorted(set(str(r[gold_key]) for r in rows) | set(str(r[pred_key]) for r in rows))
    if exclude_o:
        labels = [label for label in labels if label != "O"]

    out: List[Dict[str, object]] = []
    for label in labels:
        tp = sum(str(r[gold_key]) == label and str(r[pred_key]) == label for r in rows)
        pred_pos = sum(str(r[pred_key]) == label for r in rows)
        gold_pos = sum(str(r[gold_key]) == label for r in rows)
        p, r_, f = precision_recall_f1(tp, pred_pos, gold_pos)
        out.append({
            "label": label,
            "gold_count": gold_pos,
            "pred_count": pred_pos,
            "tp": tp,
            "precision": 100 * p,
            "recall": 100 * r_,
            "f1": 100 * f,
        })
    out.sort(key=lambda x: (float(x["f1"]), -int(x["gold_count"]), str(x["label"])))
    return out


def confusion_rows(rows: List[Dict[str, object]], gold_key: str, pred_key: str) -> List[Dict[str, object]]:
    c = Counter((str(r[gold_key]), str(r[pred_key])) for r in rows)
    return [{"gold": g, "pred": p, "count": n, "correct": g == p} for (g, p), n in c.most_common()]


def select_top_supersense_labels(rows: List[Dict[str, object]], max_labels: int, include_o: bool = False) -> List[str]:
    counts: Counter[str] = Counter()
    for r in rows:
        g = str(r["gold_sup"])
        p = str(r["pred_sup"])
        if include_o or g != "O":
            counts[g] += 1
        if include_o or p != "O":
            counts[p] += 1
    return sorted([label for label, _ in counts.most_common(max_labels)])


def build_confusion_matrix(
    rows: List[Dict[str, object]],
    gold_key: str,
    pred_key: str,
    labels: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[List[int]]]:
    if labels is None:
        labels = sorted(set(str(r[gold_key]) for r in rows) | set(str(r[pred_key]) for r in rows))
    labels = [str(label) for label in labels]
    idx = {label: i for i, label in enumerate(labels)}
    mat = [[0 for _ in labels] for _ in labels]

    for r in rows:
        g = str(r[gold_key])
        p = str(r[pred_key])
        if g in idx and p in idx:
            mat[idx[g]][idx[p]] += 1
    return labels, mat


def normalize_rows(mat: List[List[int]]) -> List[List[float]]:
    out: List[List[float]] = []
    for row in mat:
        total = sum(row)
        out.append([100.0 * x / total if total else 0.0 for x in row])
    return out


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_confusion_matrix_csv(path: Path, labels: List[str], mat: List[List[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred"] + labels)
        for label, row in zip(labels, mat):
            writer.writerow([label] + row)


def md_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def fmt_num(value: object, digits: int = 2) -> str:
    if value == "" or value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def make_markdown_table(headers: List[str], rows: Iterable[Iterable[object]], aligns: Optional[List[str]] = None) -> List[str]:
    aligns = list(aligns or ["left"] * len(headers))
    if len(aligns) < len(headers):
        aligns.extend(["left"] * (len(headers) - len(aligns)))
    elif len(aligns) > len(headers):
        aligns = aligns[:len(headers)]

    clean_headers = [md_escape(h) for h in headers]
    clean_rows = [[md_escape(c) for c in row] for row in rows]

    # Pad/truncate rows defensively so markdown generation never crashes.
    fixed_rows = []
    for row in clean_rows:
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))
        elif len(row) > len(headers):
            row = row[:len(headers)]
        fixed_rows.append(row)
    clean_rows = fixed_rows

    widths = [len(h) for h in clean_headers]
    for row in clean_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_cell(cell: str, width: int, align: str) -> str:
        if align == "right":
            return cell.rjust(width)
        if align == "center":
            return cell.center(width)
        return cell.ljust(width)

    header_line = "| " + " | ".join(fmt_cell(h, widths[i], aligns[i]) for i, h in enumerate(clean_headers)) + " |"
    sep = []
    for width, align in zip(widths, aligns):
        if align == "right":
            sep.append("-" * max(3, width) + ":")
        elif align == "center":
            sep.append(":" + "-" * max(3, width) + ":")
        else:
            sep.append(":" + "-" * max(3, width))
    sep_line = "| " + " | ".join(sep) + " |"
    data_lines = [
        "| " + " | ".join(fmt_cell(cell, widths[i], aligns[i]) for i, cell in enumerate(row)) + " |"
        for row in clean_rows
    ]
    return [header_line, sep_line] + data_lines


def write_scores_files(out_dir: Path, score_rows: List[Dict[str, object]]) -> None:
    fields = ["task", "precision", "recall", "f1", "accuracy", "score_source", "accuracy_note"]
    write_csv(out_dir / "scores.csv", score_rows, fields)

    table_rows = [
        [
            r["task"],
            fmt_num(r["precision"]),
            fmt_num(r["recall"]),
            fmt_num(r["f1"]),
            fmt_num(r["accuracy"]),
            r.get("score_source", ""),
            r.get("accuracy_note", ""),
        ]
        for r in score_rows
    ]
    lines = ["# Score Summary", ""]
    lines.extend(make_markdown_table(
        ["task", "precision (%)", "recall (%)", "F1 (%)", "accuracy (%)", "score source", "accuracy source"],
        table_rows,
        ["left", "right", "right", "right", "right", "left", "left"],
    ))
    (out_dir / "scores.md").write_text("\n".join(lines), encoding="utf-8")


def write_sentence_errors(rows: List[Dict[str, object]], out_path: Path, max_sentences: int) -> None:
    by_sent: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_sent[int(r["sent_id"])].append(r)

    ranked = []
    for sid, toks in by_sent.items():
        n_err = sum(t["error_type"] != "correct" for t in toks)
        n_mwe_err = sum("mwe" in str(t["error_type"]) for t in toks)
        n_sup_err = sum("sup" in str(t["error_type"]) for t in toks)
        ranked.append((n_err, n_mwe_err, n_sup_err, sid, toks))
    ranked.sort(reverse=True)

    lines = ["# Sentence-Level Error Samples", ""]
    headers = ["tok", "word", "POS", "gold MWE", "pred MWE", "gold supersense", "pred supersense", "status", "error"]
    aligns = ["right", "left", "left", "left", "left", "left", "left", "left", "left"]

    for n_err, n_mwe_err, n_sup_err, sid, toks in ranked[:max_sentences]:
        lines.append(f"## Sentence {sid}: {n_err} total errors, {n_mwe_err} MWE errors, {n_sup_err} supersense errors")
        lines.append("")
        lines.append(str(toks[0]["sentence"]))
        lines.append("")
        table_rows = []
        for t in toks:
            table_rows.append([
                t["tok_id"], t["word"], t["pos"], t["gold_mwe"], t["pred_mwe"],
                t["gold_sup"], t["pred_sup"], "OK" if t["error_type"] == "correct" else "ERR", t["error_type"],
            ])
        lines.extend(make_markdown_table(headers, table_rows, aligns))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_report(
    out_dir: Path,
    metrics: Dict[str, object],
    summary: Dict[str, object],
    score_rows: List[Dict[str, object]],
    rows: List[Dict[str, object]],
) -> None:
    error_counts = Counter(str(r["error_type"]) for r in rows)
    top_sup_conf = [r for r in confusion_rows(rows, "gold_sup", "pred_sup") if not r["correct"]][:10]
    top_mwe_conf = [r for r in confusion_rows(rows, "gold_mwe", "pred_mwe") if not r["correct"]][:10]

    lines = ["# DiMSUM Run Report", ""]

    if summary:
        lines.append("## Run configuration")
        lines.append("")
        config_rows = []
        for key in ["architecture", "model_name", "epochs", "batch_size", "lr", "seconds", "dev_loss", "mwe_macro_f1", "sup_macro_f1"]:
            if key in summary:
                config_rows.append([key, format_config_value(key, summary.get(key, ""))])
        if config_rows:
            lines.extend(make_markdown_table(["field", "value"], config_rows, ["left", "left"]))
            lines.append("")

    lines.append("## Scores")
    lines.append("")
    lines.append("Precision/recall/F1 use official evaluator values when available. Accuracy is official for the combined task when available; otherwise it is a token-level sanity check.")
    lines.append("")
    lines.extend(make_markdown_table(
        ["task", "precision (%)", "recall (%)", "F1 (%)", "accuracy (%)", "score source", "accuracy source"],
        [[r["task"], fmt_num(r["precision"]), fmt_num(r["recall"]), fmt_num(r["f1"]), fmt_num(r["accuracy"]), r.get("score_source", ""), r["accuracy_note"]] for r in score_rows],
        ["left", "right", "right", "right", "right", "left", "left"],
    ))
    lines.append("")

    lines.append("## Token-level sanity checks")
    lines.append("")
    lines.extend(make_markdown_table(
        ["metric", "value"],
        [[k, fmt_num(v) if isinstance(v, float) else v] for k, v in metrics.items()],
        ["left", "right"],
    ))
    lines.append("")

    lines.append("## Error breakdown")
    lines.append("")
    lines.extend(make_markdown_table(
        ["error type", "count"],
        [[err, n] for err, n in error_counts.most_common()],
        ["left", "right"],
    ))
    lines.append("")

    lines.append("## Top supersense confusions")
    lines.append("")
    lines.extend(make_markdown_table(
        ["gold", "predicted", "count"],
        [[r["gold"], r["pred"], r["count"]] for r in top_sup_conf],
        ["left", "left", "right"],
    ))
    lines.append("")

    lines.append("## Top MWE tag confusions")
    lines.append("")
    lines.extend(make_markdown_table(
        ["gold", "predicted", "count"],
        [[r["gold"], r["pred"], r["count"]] for r in top_mwe_conf],
        ["left", "left", "right"],
    ))
    lines.append("")

    lines.append("## Generated analysis files")
    lines.append("")
    lines.append("- `scores.csv` / `scores.md`: precision, recall, F1, and accuracy summary")
    lines.append("- `supersense_confusion_matrix.png`: colorblind-safe supersense confusion matrix")
    lines.append("- `supersense_confusion_matrix_row_normalized.png`: row-normalized supersense confusion matrix")
    lines.append("- `supersense_per_label_metrics.csv`: per-label precision, recall, and F1")
    lines.append("- `sentence_errors.md`: qualitative sentence-level examples")
    lines.append("- `token_errors.csv`: token-level data for spreadsheet filtering")

    loss_lines = write_loss_history_report(out_dir, summary)
    if loss_lines:
        lines.append("")
        lines.extend(loss_lines)

    (out_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


def plot_heatmap(
    out_path: Path,
    labels: List[str],
    mat: List[List[float]],
    title: str,
    colorbar_label: str,
    annotate: bool = True,
    value_fmt: str = ".0f",
) -> None:
    plt = try_import_matplotlib()
    if plt is None:
        return

    # Use Matplotlib utilities directly here so annotation color follows the
    # actual rendered cell color. This fixes the common readability problem
    # where a simple value threshold chooses white text on yellow cells or black
    # text on dark blue cells.
    import matplotlib.colors as mcolors  # type: ignore
    import matplotlib.patheffects as pe  # type: ignore

    n = len(labels)
    fig_w = max(8, min(18, n * 0.52 + 4))
    fig_h = max(7, min(18, n * 0.48 + 3))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Normalize explicitly so text color can use the same mapping as imshow.
    max_val = max((max(row) if row else 0 for row in mat), default=0)
    norm = mcolors.Normalize(vmin=0, vmax=max_val if max_val > 0 else 1)
    cmap = plt.get_cmap(HEATMAP_CMAP)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=15)
    ax.set_yticklabels(labels, fontsize=15)

    # Keep labels generic because this function is also used for MWE matrices.
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    if annotate and n <= 30:
        for i, row in enumerate(mat):
            for j, val in enumerate(row):
                if val == 0:
                    continue

                rgba = cmap(norm(val))
                r, g, b = rgba[:3]
                # Perceived luminance. Bright cividis/yellow cells get black text;
                # dark blue/purple cells get white text.
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                text_color = "black" if luminance > 0.55 else "white"
                stroke_color = "white" if text_color == "black" else "black"

                ax.text(
                    j,
                    i,
                    format(val, value_fmt),
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=text_color,
                    path_effects=[pe.withStroke(linewidth=1.25, foreground=stroke_color)],
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_reports(out_dir: Path, rows: List[Dict[str, object]], score_rows: List[Dict[str, object]], max_matrix_labels: int) -> None:
    plt = try_import_matplotlib()
    if plt is None:
        (out_dir / "PLOTS_SKIPPED.txt").write_text(
            "matplotlib is not installed. Add matplotlib to requirements-dimsum.txt to generate PNG plots.\n",
            encoding="utf-8",
        )
        return

    # Score bar chart.
    tasks = [str(r["task"]) for r in score_rows]
    metric_keys = ["precision", "recall", "f1", "accuracy"]
    metric_labels = ["Precision", "Recall", "F1", "Accuracy"]
    x = list(range(len(tasks)))
    width = 0.18

    fig, ax = plt.subplots(figsize=(8, 5))
    for k_idx, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        vals = []
        for r in score_rows:
            try:
                vals.append(float(r[key]))
            except Exception:
                vals.append(0.0)
        offsets = [i + (k_idx - 1.5) * width for i in x]
        ax.bar(offsets, vals, width=width, label=label, color=OKABE_ITO[k_idx])
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("DiMSUM Scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "scores_colorblind_safe.png", dpi=180)
    plt.close(fig)

    # Error breakdown.
    err_counts = Counter(str(r["error_type"]) for r in rows if r["error_type"] != "correct")
    if err_counts:
        labels, values = zip(*err_counts.most_common())
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.7), 4.8))
        ax.bar(labels, values, color=OKABE_ITO[0])
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.set_ylabel("Token count")
        ax.set_title("Token Error Breakdown")
        fig.tight_layout()
        fig.savefig(out_dir / "error_breakdown_colorblind_safe.png", dpi=180)
        plt.close(fig)

    # Top supersense confusions.
    sup_conf = [r for r in confusion_rows(rows, "gold_sup", "pred_sup") if not r["correct"]][:20]
    if sup_conf:
        labels = [f"{r['gold']} -> {r['pred']}" for r in reversed(sup_conf)]
        values = [int(r["count"]) for r in reversed(sup_conf)]
        fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.32)))
        ax.barh(labels, values, color=OKABE_ITO[1])
        ax.set_xlabel("Count")
        ax.set_title("Top Supersense Confusions")
        fig.tight_layout()
        fig.savefig(out_dir / "supersense_top_confusions_colorblind_safe.png", dpi=180)
        plt.close(fig)

    # Supersense confusion matrix, excluding O and limiting to most frequent labels.
    sup_labels = select_top_supersense_labels(rows, max_matrix_labels, include_o=False)
    if sup_labels:
        labels, mat = build_confusion_matrix(rows, "gold_sup", "pred_sup", labels=sup_labels)
        write_confusion_matrix_csv(out_dir / "supersense_confusion_matrix.csv", labels, mat)

        plot_heatmap(
            out_dir / "supersense_confusion_matrix.png",
            labels,
            mat,
            title=f"Supersense Confusion Matrix: Top {len(labels)} Non-O Labels",
            colorbar_label="Token count",
            annotate=len(labels) <= 30,
            value_fmt=".0f",
        )

        plot_heatmap(
            out_dir / "supersense_confusion_matrix_row_normalized.png",
            labels,
            normalize_rows(mat),
            title=f"Supersense Confusion Matrix: Row-Normalized Top {len(labels)} Non-O Labels",
            colorbar_label="% of gold label",
            annotate=len(labels) <= 25,
            value_fmt=".1f",
        )

    # MWE confusion matrix.
    mwe_labels = sorted(set(str(r["gold_mwe"]) for r in rows) | set(str(r["pred_mwe"]) for r in rows))
    if mwe_labels:
        labels, mat = build_confusion_matrix(rows, "gold_mwe", "pred_mwe", labels=mwe_labels)
        write_confusion_matrix_csv(out_dir / "mwe_tag_confusion_matrix.csv", labels, mat)
        plot_heatmap(
            out_dir / "mwe_tag_confusion_matrix.png",
            labels,
            mat,
            title="MWE Tag Confusion Matrix",
            colorbar_label="Token count",
            annotate=len(labels) <= 30,
            value_fmt=".0f",
        )


def maybe_run_official_eval(eval_file: Optional[Path], gold: Path, pred: Path, eval_log: Optional[Path], force: bool = False) -> Optional[Path]:
    if eval_file is None:
        default_eval = Path("dimsum-data/scripts/dimsumeval.py")
        if default_eval.exists():
            eval_file = default_eval
        else:
            return eval_log

    if eval_log is None:
        eval_log = pred.parent / "official_eval.txt"

    # If an old official_eval.txt exists but only contains an exception/partial output,
    # rerun it so precision/recall/F1 can be parsed instead of leaving score cells blank.
    if eval_log.exists() and not force:
        existing = eval_log.read_text(encoding="utf-8", errors="replace")
        parsed = parse_official_eval_text(existing)
        if parsed:
            return eval_log
        print(f"Existing evaluator log did not contain parseable official scores; rerunning: {eval_log}")

    cmd = ["python", str(eval_file), "-C", str(gold), str(pred)]
    print("Running official evaluator:", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    eval_log.parent.mkdir(parents=True, exist_ok=True)
    eval_log.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Official evaluator failed with exit code {proc.returncode}. Output saved to {eval_log}")
    return eval_log

def find_latest_run(runs_dir: Path) -> Path:
    candidates: List[Tuple[float, Path]] = []
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        pred = child / "predictions.pred"
        summary = child / "summary.json"
        eval_log = child / "official_eval.txt"
        marker = pred if pred.exists() else summary if summary.exists() else eval_log if eval_log.exists() else None
        if marker is not None:
            candidates.append((marker.stat().st_mtime, child))

    if not candidates:
        raise FileNotFoundError(f"No run directories with predictions.pred, summary.json, or official_eval.txt found under {runs_dir}")
    candidates.sort(reverse=True)
    return candidates[0][1]


def aggregate_runs(runs_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        eval_log = summary_path.parent / "official_eval.txt"
        try:
            data = load_summary(summary_path, eval_log)
        except Exception:
            continue

        rows.append({
            "run": summary_path.parent.name,
            "architecture": data.get("architecture", ""),
            "model_name": data.get("model_name", ""),
            "epochs": data.get("epochs", ""),
            "batch_size": data.get("batch_size", ""),
            "lr": data.get("lr", ""),
            "seconds": data.get("seconds", ""),
            "dev_loss": data.get("dev_loss", ""),
            "official_mwe_precision": data.get("official_mwe_precision", ""),
            "official_mwe_recall": data.get("official_mwe_recall", ""),
            "official_mwe_f1": data.get("official_mwe_f1", ""),
            "official_sup_precision": data.get("official_sup_precision", ""),
            "official_sup_recall": data.get("official_sup_recall", ""),
            "official_sup_f1": data.get("official_sup_f1", ""),
            "official_combined_accuracy": data.get("official_combined_accuracy", ""),
            "official_combined_precision": data.get("official_combined_precision", ""),
            "official_combined_recall": data.get("official_combined_recall", ""),
            "official_combined_f1": data.get("official_combined_f1", ""),
            "mwe_macro_f1": data.get("mwe_macro_f1", ""),
            "sup_macro_f1": data.get("sup_macro_f1", ""),
        })

    if not rows:
        print(f"No summary.json files found under {runs_dir}")
        return

    fieldnames = list(rows[0].keys())
    write_csv(out_dir / "runs_summary.csv", rows, fieldnames)

    plt = try_import_matplotlib()
    if plt is None:
        return

    labels = [str(r["run"]) for r in rows]
    keys = ["official_mwe_f1", "official_sup_f1", "official_combined_f1"]
    pretty = ["MWE F1", "Supersense F1", "Combined F1"]
    x = list(range(len(labels)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.2), 5))
    for idx, (offset, key, label) in enumerate(zip([-width, 0, width], keys, pretty)):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except Exception:
                vals.append(0.0)
        ax.bar([i + offset for i in x], vals, width=width, label=label, color=OKABE_ITO[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("F1 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Model Comparison Across Runs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "runs_f1_comparison_colorblind_safe.png", dpi=180)
    plt.close(fig)


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.all_runs:
        args.runs_dir = args.runs_dir or Path("runs")
        if args.out == Path("report"):
            args.out = Path("reports/all_runs")
        return args

    if args.run_dir is None and args.pred is None:
        args.run_dir = find_latest_run(args.runs_dir or Path("runs"))
        print(f"Auto-selected latest run: {args.run_dir}")

    if args.run_dir is not None:
        args.pred = args.pred or args.run_dir / "predictions.pred"
        args.summary = args.summary or args.run_dir / "summary.json"
        args.eval_log = args.eval_log or args.run_dir / "official_eval.txt"
        if args.out == Path("report"):
            args.out = args.run_dir / "report"

    args.gold = args.gold or Path("dimsum-data/dimsum16.test")
    return args

def write_loss_history_report(out_dir: Path, summary: dict):
    history = summary.get("loss_history", [])

    if not history:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "loss_history.csv"
    png_path = out_dir / "loss_curve.png"

    # Save CSV
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("epoch,train_loss,lr\n")
        for row in history:
            f.write(
                f"{row.get('epoch','')},"
                f"{row.get('train_loss','')},"
                f"{row.get('lr','')}\n"
            )

    # Save plot
    try:
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in history]
        train_losses = [row["train_loss"] for row in history]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_losses, marker="o", label="Train loss")

        ax.set_title("Training Loss by Epoch", fontsize=16)
        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel("Loss", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(png_path, dpi=220)
        plt.close(fig)

    except Exception as exc:
        print(f"Could not generate loss curve: {exc}")

    # Markdown section
    lines = []
    lines.append("## Loss history")
    lines.append("")
    lines.append("| epoch | train loss | learning rate |")
    lines.append("| ----: | ---------: | ------------: |")

    for row in history:
        epoch = row.get("epoch", "")
        train_loss = row.get("train_loss", "")
        lr = row.get("lr", "")

        train_loss_str = f"{train_loss:.4f}" if isinstance(train_loss, (int, float)) else str(train_loss)
        lr_str = f"{lr:.2e}" if isinstance(lr, (int, float)) else str(lr)

        lines.append(f"| {epoch} | {train_loss_str} | {lr_str} |")

    lines.append("")
    lines.append("![Training loss curve](loss_curve.png)")
    lines.append("")

    return lines

def format_config_value(key, value):
    if value is None:
        return ""

    # Learning rates are tiny, so decimal formatting makes them look like 0.00.
    if key in {"lr", "learning_rate"} and isinstance(value, (int, float)):
        return f"{value:.2e}"

    # These are readable as normal decimals.
    if key in {
        "seconds",
        "dev_loss",
        "mwe_macro_f1",
        "sup_macro_f1",
        "dropout",
        "grad_clip",
        "mwe_loss_weight",
        "sup_loss_weight",
    } and isinstance(value, (int, float)):
        return f"{value:.2f}"

    return str(value)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=None, help="Gold DiMSUM file, usually dimsum16.test")
    parser.add_argument("--pred", type=Path, default=None, help="Prediction file from dimsum_unified.py")
    parser.add_argument("--summary", type=Path, default=None, help="Optional run summary.json")
    parser.add_argument("--eval_log", type=Path, default=None, help="Optional saved official evaluator output")
    parser.add_argument("--eval_file", type=Path, default=Path("dimsum-data/scripts/dimsumeval.py"), help="Optional dimsumeval.py; auto-runs if official_eval.txt is missing")
    parser.add_argument("--out", type=Path, default=Path("report"))
    parser.add_argument("--run_dir", type=Path, default=None, help="Run dir containing predictions.pred")
    parser.add_argument("--runs_dir", type=Path, default=Path("runs"), help="Directory containing multiple run folders")
    parser.add_argument("--all_runs", action="store_true", help="Aggregate all runs instead of one detailed report")
    parser.add_argument("--max_sentences", type=int, default=25)
    parser.add_argument("--max_matrix_labels", type=int, default=30, help="Max non-O supersense labels shown in matrix PNG")
    parser.add_argument("--force_eval", action="store_true", help="Rerun official evaluator even if official_eval.txt already exists")
    args = resolve_args(parser.parse_args())

    args.out.mkdir(parents=True, exist_ok=True)

    if args.all_runs:
        aggregate_runs(args.runs_dir, args.out)
        print(f"Wrote aggregate run report to {args.out}")
        return

    if not args.gold or not args.pred:
        raise SystemExit("Need --gold and --pred, or use --run_dir, or run from a project with ./runs/<run>/predictions.pred.")

    args.eval_log = maybe_run_official_eval(args.eval_file, args.gold, args.pred, args.eval_log, force=args.force_eval)

    gold = read_dimsum(args.gold)
    pred = read_dimsum(args.pred)
    rows = align_gold_pred(gold, pred)
    metrics = compute_basic_metrics(rows)
    summary = load_summary(args.summary, args.eval_log)
    score_rows = official_score_rows(summary, metrics)

    token_fields = [
        "sent_id", "tok_id", "word", "lemma", "pos",
        "gold_mwe", "pred_mwe", "gold_sup", "pred_sup",
        "mwe_correct", "sup_correct", "combined_correct", "error_type", "sentence",
    ]

    write_csv(args.out / "token_errors.csv", rows, token_fields)
    write_csv(args.out / "supersense_confusions.csv", confusion_rows(rows, "gold_sup", "pred_sup"), ["gold", "pred", "count", "correct"])
    write_csv(args.out / "mwe_tag_confusions.csv", confusion_rows(rows, "gold_mwe", "pred_mwe"), ["gold", "pred", "count", "correct"])
    write_csv(args.out / "supersense_per_label_metrics.csv", per_label_metrics(rows, "gold_sup", "pred_sup", exclude_o=True), ["label", "gold_count", "pred_count", "tp", "precision", "recall", "f1"])
    write_csv(args.out / "mwe_per_label_metrics.csv", per_label_metrics(rows, "gold_mwe", "pred_mwe", exclude_o=False), ["label", "gold_count", "pred_count", "tp", "precision", "recall", "f1"])

    write_sentence_errors(rows, args.out / "sentence_errors.md", args.max_sentences)
    write_scores_files(args.out, score_rows)
    write_markdown_report(args.out, metrics, summary, score_rows, rows)
    plot_reports(args.out, rows, score_rows, args.max_matrix_labels)

    print(f"Wrote report to: {args.out}")
    print("Key files:")
    for name in [
        "summary_report.md",
        "scores.csv",
        "scores.md",
        "scores_colorblind_safe.png",
        "supersense_confusion_matrix.png",
        "supersense_confusion_matrix_row_normalized.png",
        "supersense_per_label_metrics.csv",
        "sentence_errors.md",
        "token_errors.csv",
    ]:
        print(f"  {args.out / name}")


if __name__ == "__main__":
    main()
