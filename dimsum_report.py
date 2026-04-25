#!/usr/bin/env python3
"""
Generate human-readable DiMSUM analysis files from a gold file and prediction file.

Outputs:
  - summary_report.md
  - token_errors.csv
  - sentence_errors.md
  - supersense_confusions.csv
  - mwe_tag_confusions.csv
  - optional PNG plots if matplotlib is installed
  - optional runs_summary.csv / runs_f1_comparison.png when --runs_dir is used

Example:
  python dimsum_report.py \
    --gold dimsum-data/dimsum16.test \
    --pred runs/mtl_crf_roberta-base_lr2e-05_ep3_bs16/predictions.pred \
    --summary runs/mtl_crf_roberta-base_lr2e-05_ep3_bs16/summary.json \
    --eval_log runs/mtl_crf_roberta-base_lr2e-05_ep3_bs16/official_eval.txt \
    --out runs/mtl_crf_roberta-base_lr2e-05_ep3_bs16/report

To compare multiple model runs:
  python dimsum_report.py --runs_dir runs --out reports/all_runs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

Token = Dict[str, object]


def read_dimsum(path: Path) -> List[List[Token]]:
    """Read a DiMSUM CoNLL-style file.

    Expected useful columns:
      0 = token id
      1 = word
      2 = lemma
      3 = POS
      4 = MWE tag
      5 = MWE parent/head
      7 = supersense
    """
    sentences: List[List[Token]] = []
    sent: List[Token] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if sent:
                    sentences.append(sent)
                    sent = []
                continue

            cols = line.split("\t")
            while len(cols) < 8:
                cols.append("")

            tok: Token = {
                "id": cols[0],
                "word": cols[1],
                "lemma": cols[2],
                "pos": cols[3],
                "mwe": cols[4] if cols[4] else "O",
                "mwe_parent": cols[5] if cols[5] else "0",
                "sup": cols[7] if cols[7] else "O",
                "raw_cols": cols,
            }
            sent.append(tok)

    if sent:
        sentences.append(sent)
    return sentences


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

            gold_is_mwe = gold_mwe != "O"
            pred_is_mwe = pred_mwe != "O"
            gold_has_sup = gold_sup != "O"
            pred_has_sup = pred_sup != "O"

            mwe_correct = gold_mwe == pred_mwe
            sup_correct = gold_sup == pred_sup
            combined_correct = mwe_correct and sup_correct

            error_parts = []
            if not mwe_correct:
                if gold_is_mwe and not pred_is_mwe:
                    error_parts.append("mwe_missed")
                elif not gold_is_mwe and pred_is_mwe:
                    error_parts.append("mwe_false_positive")
                else:
                    error_parts.append("mwe_boundary_or_tag")
            if not sup_correct:
                if gold_has_sup and not pred_has_sup:
                    error_parts.append("sup_missed")
                elif not gold_has_sup and pred_has_sup:
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
                "combined_correct": combined_correct,
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
    tp_mwe_pos = sum(str(r["gold_mwe"]) != "O" and str(r["pred_mwe"]) != "O" for r in rows)

    gold_sup_pos = sum(str(r["gold_sup"]) != "O" for r in rows)
    pred_sup_pos = sum(str(r["pred_sup"]) != "O" for r in rows)
    tp_sup_exact = sum(str(r["gold_sup"]) != "O" and str(r["gold_sup"]) == str(r["pred_sup"]) for r in rows)

    mwe_p, mwe_r, mwe_f = precision_recall_f1(tp_mwe_pos, pred_mwe_pos, gold_mwe_pos)
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


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def md_escape(value: object) -> str:
    """Escape Markdown table-sensitive characters and normalize empty cells."""
    text = "" if value is None else str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("|", "\\|")
    text = text.replace("\n", "<br>")
    return text


def visible_len(text: str) -> int:
    """Approximate visible width for Markdown table padding.

    To keep raw Markdown aligned in VS Code, avoid emoji in padded tables
    because emoji display width varies by editor/font.
    """
    return len(text)


def make_markdown_table(
    headers: List[str],
    rows: Iterable[Iterable[object]],
    aligns: Optional[List[str]] = None,
) -> List[str]:
    """Build a padded Markdown table that looks aligned in raw VS Code view.

    aligns values: "left", "right", or "center".
    """
    aligns = aligns or ["left"] * len(headers)
    clean_headers = [md_escape(h) for h in headers]
    clean_rows = [[md_escape(cell) for cell in row] for row in rows]

    widths = [visible_len(h) for h in clean_headers]
    for row in clean_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], visible_len(cell))

    def fmt_cell(cell: str, width: int, align: str) -> str:
        if align == "right":
            return cell.rjust(width)
        if align == "center":
            left = (width - visible_len(cell)) // 2
            right = width - visible_len(cell) - left
            return " " * left + cell + " " * right
        return cell.ljust(width)

    def fmt_row(row: List[str]) -> str:
        cells = [fmt_cell(cell, widths[i], aligns[i]) for i, cell in enumerate(row)]
        return "| " + " | ".join(cells) + " |"

    sep_cells = []
    for width, align in zip(widths, aligns):
        dashes = "-" * max(3, width)
        if align == "right":
            sep_cells.append(dashes + ":")
        elif align == "center":
            sep_cells.append(":" + dashes + ":")
        else:
            sep_cells.append(dashes)

    return [fmt_row(clean_headers), "| " + " | ".join(sep_cells) + " |"] + [fmt_row(row) for row in clean_rows]


def confusion_rows(rows: List[Dict[str, object]], gold_key: str, pred_key: str) -> List[Dict[str, object]]:
    c = Counter((str(r[gold_key]), str(r[pred_key])) for r in rows)
    out = []
    for (g, p), n in c.most_common():
        out.append({"gold": g, "pred": p, "count": n, "correct": g == p})
    return out


def load_summary(summary_path: Optional[Path], eval_log: Optional[Path]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    if summary_path and summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary.update(json.load(f))

    text = ""
    if eval_log and eval_log.exists():
        text = eval_log.read_text(encoding="utf-8", errors="replace")

    # Parse official summary lines if present.
    patterns = {
        "official_mwe_f1": r"MWEs:.*F=([0-9.]+)%",
        "official_sup_f1": r"Supersenses:.*F=([0-9.]+)%",
        "official_combined_f1": r"Combined:.*F=([0-9.]+)%",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            summary[key] = float(m.group(1))
        elif key in summary and isinstance(summary[key], str) and summary[key].endswith("%"):
            summary[key] = float(summary[key].rstrip("%"))
    return summary


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
    lines.append(
        "Note: tables are padded for raw VS Code readability. Markdown preview will also render them normally."
    )
    lines.append("")

    for n_err, n_mwe_err, n_sup_err, sid, toks in ranked[:max_sentences]:
        sentence = str(toks[0]["sentence"])
        lines.append(f"## Sentence {sid}: {n_err} total errors, {n_mwe_err} MWE errors, {n_sup_err} supersense errors")
        lines.append("")
        lines.append(sentence)
        lines.append("")

        table_rows = []
        for t in toks:
            status = "OK" if t["error_type"] == "correct" else "ERR"
            table_rows.append([
                t["tok_id"],
                t["word"],
                t["pos"],
                t["gold_mwe"],
                t["pred_mwe"],
                t["gold_sup"],
                t["pred_sup"],
                status,
                t["error_type"],
            ])

        lines.extend(make_markdown_table(
            ["tok", "word", "POS", "gold MWE", "pred MWE", "gold supersense", "pred supersense", "status", "error"],
            table_rows,
            aligns=["right", "left", "left", "left", "left", "left", "left", "left", "left"],
        ))
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_report(out_dir: Path, metrics: Dict[str, object], summary: Dict[str, object], rows: List[Dict[str, object]]) -> None:
    error_counts = Counter(str(r["error_type"]) for r in rows)
    top_sup_conf = [r for r in confusion_rows(rows, "gold_sup", "pred_sup") if not r["correct"]][:10]
    top_mwe_conf = [r for r in confusion_rows(rows, "gold_mwe", "pred_mwe") if not r["correct"]][:10]

    lines = ["# DiMSUM Run Report", ""]

    if summary:
        lines.append("## Official / run-level scores")
        lines.append("")
        for key in [
            "architecture", "model_name", "epochs", "batch_size", "lr", "seconds",
            "official_mwe_f1", "official_sup_f1", "official_combined_f1",
            "mwe_macro_f1", "sup_macro_f1", "dev_loss",
        ]:
            if key in summary:
                val = summary[key]
                suffix = "%" if key.startswith("official_") and isinstance(val, (int, float)) else ""
                lines.append(f"- **{key}**: {val}{suffix}")
        lines.append("")

    lines.append("## Token-level sanity checks")
    lines.append("")
    for key, val in metrics.items():
        if isinstance(val, float):
            lines.append(f"- **{key}**: {val:.2f}")
        else:
            lines.append(f"- **{key}**: {val}")
    lines.append("")

    lines.append("## Error breakdown")
    lines.append("")
    lines.extend(make_markdown_table(
        ["error type", "count"],
        [[err, n] for err, n in error_counts.most_common()],
        aligns=["left", "right"],
    ))
    lines.append("")

    lines.append("## Top supersense confusions")
    lines.append("")
    lines.extend(make_markdown_table(
        ["gold", "predicted", "count"],
        [[r["gold"], r["pred"], r["count"]] for r in top_sup_conf],
        aligns=["left", "left", "right"],
    ))
    lines.append("")

    lines.append("## Top MWE tag confusions")
    lines.append("")
    lines.extend(make_markdown_table(
        ["gold", "predicted", "count"],
        [[r["gold"], r["pred"], r["count"]] for r in top_mwe_conf],
        aligns=["left", "left", "right"],
    ))
    lines.append("")

    lines.append("## How to use this for analysis")
    lines.append("")
    lines.append("- Use `summary_report.md` for the paper/table summary.")
    lines.append("- Use `sentence_errors.md` for qualitative examples.")
    lines.append("- Use `supersense_confusions.csv` to find common semantic confusions.")
    lines.append("- Use `token_errors.csv` to filter exact bad examples in Excel/Sheets.")

    (out_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


def plot_reports(out_dir: Path, rows: List[Dict[str, object]], summary: Dict[str, object]) -> None:
    plt = try_import_matplotlib()
    if plt is None:
        (out_dir / "PLOTS_SKIPPED.txt").write_text(
            "matplotlib is not installed. Add matplotlib to requirements-dimsum.txt to generate PNG plots.\n",
            encoding="utf-8",
        )
        return

    # Official F1 bar chart
    f1_items = []
    for label, key in [("MWE", "official_mwe_f1"), ("Supersense", "official_sup_f1"), ("Combined", "official_combined_f1")]:
        if key in summary:
            try:
                f1_items.append((label, float(summary[key])))
            except Exception:
                pass
    if f1_items:
        labels, values = zip(*f1_items)
        plt.figure(figsize=(7, 4))
        plt.bar(labels, values)
        plt.ylabel("F1 (%)")
        plt.title("Official DiMSUM Scores")
        plt.ylim(0, max(100, max(values) + 5))
        plt.tight_layout()
        plt.savefig(out_dir / "official_scores.png", dpi=160)
        plt.close()

    # Error breakdown
    err_counts = Counter(str(r["error_type"]) for r in rows if r["error_type"] != "correct")
    if err_counts:
        labels, values = zip(*err_counts.most_common())
        plt.figure(figsize=(max(7, len(labels) * 0.7), 4))
        plt.bar(labels, values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Token count")
        plt.title("Token Error Breakdown")
        plt.tight_layout()
        plt.savefig(out_dir / "error_breakdown.png", dpi=160)
        plt.close()

    # Top supersense confusions, excluding correct.
    sup_conf = [r for r in confusion_rows(rows, "gold_sup", "pred_sup") if not r["correct"]]
    sup_conf = sup_conf[:20]
    if sup_conf:
        labels = [f"{r['gold']} → {r['pred']}" for r in reversed(sup_conf)]
        values = [int(r["count"]) for r in reversed(sup_conf)]
        plt.figure(figsize=(9, max(5, len(labels) * 0.3)))
        plt.barh(labels, values)
        plt.xlabel("Count")
        plt.title("Top Supersense Confusions")
        plt.tight_layout()
        plt.savefig(out_dir / "supersense_top_confusions.png", dpi=160)
        plt.close()

    # MWE tag confusion matrix for compact labels.
    mwe_labels = sorted(set(str(r["gold_mwe"]) for r in rows) | set(str(r["pred_mwe"]) for r in rows))
    if len(mwe_labels) <= 20:
        idx = {lab: i for i, lab in enumerate(mwe_labels)}
        mat = [[0 for _ in mwe_labels] for _ in mwe_labels]
        for r in rows:
            mat[idx[str(r["gold_mwe"])]][idx[str(r["pred_mwe"])]] += 1
        plt.figure(figsize=(7, 6))
        plt.imshow(mat)
        plt.xticks(range(len(mwe_labels)), mwe_labels, rotation=45, ha="right")
        plt.yticks(range(len(mwe_labels)), mwe_labels)
        plt.xlabel("Predicted MWE tag")
        plt.ylabel("Gold MWE tag")
        plt.title("MWE Tag Confusion Matrix")
        plt.colorbar(label="Count")
        plt.tight_layout()
        plt.savefig(out_dir / "mwe_tag_confusion.png", dpi=160)
        plt.close()



def aggregate_runs(runs_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        summary_path = run_dir / "summary.json"
        eval_log = run_dir / "official_eval.txt"
        data: Dict[str, object] = {}

        if summary_path.exists():
            try:
                data.update(json.loads(summary_path.read_text(encoding="utf-8")))
            except Exception:
                pass

        # Pull official scores from official_eval.txt even if summary.json does not contain them.
        data.update(load_summary(summary_path if summary_path.exists() else None, eval_log if eval_log.exists() else None))

        if not data and not (run_dir / "predictions.pred").exists():
            continue

        row = {
            "run": run_dir.name,
            "architecture": data.get("architecture", ""),
            "model_name": data.get("model_name", ""),
            "epochs": data.get("epochs", ""),
            "batch_size": data.get("batch_size", ""),
            "lr": data.get("lr", ""),
            "seconds": data.get("seconds", ""),
            "dev_loss": data.get("dev_loss", ""),
            "mwe_macro_f1": data.get("mwe_macro_f1", ""),
            "sup_macro_f1": data.get("sup_macro_f1", ""),
            "official_mwe_f1": data.get("official_mwe_f1", ""),
            "official_sup_f1": data.get("official_sup_f1", ""),
            "official_combined_f1": data.get("official_combined_f1", ""),
        }
        for k in ["official_mwe_f1", "official_sup_f1", "official_combined_f1"]:
            if isinstance(row[k], str) and row[k].endswith("%"):
                row[k] = row[k].rstrip("%")
        rows.append(row)

    if not rows:
        print(f"No run folders found under {runs_dir}")
        return

    fieldnames = list(rows[0].keys())
    write_csv(out_dir / "runs_summary.csv", rows, fieldnames)

    plt = try_import_matplotlib()
    if plt is None:
        return

    labels = [str(r["run"]) for r in rows]
    keys = ["official_mwe_f1", "official_sup_f1", "official_combined_f1"]
    x = list(range(len(labels)))
    width = 0.25

    plt.figure(figsize=(max(9, len(labels) * 1.2), 5))
    for offset, key in zip([-width, 0, width], keys):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except Exception:
                vals.append(0.0)
        plt.bar([i + offset for i in x], vals, width=width, label=key)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("F1 (%)")
    plt.title("Model Comparison Across Runs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "runs_f1_comparison.png", dpi=160)
    plt.close()


def find_latest_run(runs_dir: Path = Path("runs")) -> Path:
    """Find the newest run folder that looks like a dimsum_unified.py output."""
    if not runs_dir.exists():
        raise SystemExit(f"Could not find runs directory: {runs_dir}")

    candidates = []
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        pred = p / "predictions.pred"
        summary = p / "summary.json"
        model = p / "model.pt"
        if pred.exists() or summary.exists() or model.exists():
            mtimes = [x.stat().st_mtime for x in [pred, summary, model] if x.exists()]
            candidates.append((max(mtimes) if mtimes else p.stat().st_mtime, p))

    if not candidates:
        raise SystemExit(f"No run folders with predictions.pred/summary.json/model.pt found under {runs_dir}")

    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[0][1]


def resolve_run_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Auto-fill gold/pred/summary/eval_log/out from --run_dir or latest ./runs folder."""
    if getattr(args, "all_runs", False):
        args.runs_dir = args.runs_dir or Path("runs")
        args.out = args.out if args.out != Path("report") else Path("reports/all_runs")
        return args

    run_dir = args.run_dir
    if run_dir is None and args.pred is None and args.gold is None:
        run_dir = find_latest_run(args.runs_dir or Path("runs"))
        print(f"Auto-selected latest run: {run_dir}")

    if run_dir is not None:
        if not run_dir.exists():
            raise SystemExit(f"Run directory does not exist: {run_dir}")
        args.pred = args.pred or (run_dir / "predictions.pred")
        args.summary = args.summary or (run_dir / "summary.json")
        args.eval_log = args.eval_log or (run_dir / "official_eval.txt")
        args.out = args.out if args.out != Path("report") else (run_dir / "report")

    args.gold = args.gold or Path("dimsum-data/dimsum16.test")
    return args


def maybe_run_official_eval(args: argparse.Namespace) -> None:
    """If official_eval.txt is missing and --eval_file is provided, generate it."""
    if not args.eval_file or not args.gold or not args.pred:
        return

    eval_log = args.eval_log or (args.pred.parent / "official_eval.txt")
    if eval_log.exists() and not args.force_eval:
        return

    cmd = ["python", str(args.eval_file), "-C", str(args.gold), str(args.pred)]
    print("Running official evaluator:", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    eval_log.write_text(result.stdout + result.stderr, encoding="utf-8")
    args.eval_log = eval_log

    if result.returncode != 0:
        raise SystemExit(f"Official evaluator failed. Saved output to {eval_log}")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=None, help="Gold DiMSUM file. Defaults to dimsum-data/dimsum16.test")
    parser.add_argument("--pred", type=Path, default=None, help="Prediction file produced by dimsum_unified.py")
    parser.add_argument("--summary", type=Path, default=None, help="Optional run summary.json")
    parser.add_argument("--eval_log", type=Path, default=None, help="Optional saved official evaluator output")
    parser.add_argument("--out", type=Path, default=Path("report"), help="Output report directory")
    parser.add_argument("--max_sentences", type=int, default=25)

    # Automation-friendly options
    parser.add_argument("--run_dir", type=Path, default=None, help="A single run folder, e.g. runs/mtl_crf_roberta-base_lr2e-05_ep3_bs16")
    parser.add_argument("--runs_dir", type=Path, default=Path("runs"), help="Runs root directory. Default: runs")
    parser.add_argument("--all_runs", action="store_true", help="Aggregate all run folders under --runs_dir")
    parser.add_argument("--eval_file", type=Path, default=None, help="Optional evaluator path; used to create official_eval.txt if missing")
    parser.add_argument("--force_eval", action="store_true", help="Rerun official evaluator even if official_eval.txt exists")
    args = parser.parse_args()

    args = resolve_run_paths(args)

    if args.all_runs:
        args.out.mkdir(parents=True, exist_ok=True)
        aggregate_runs(args.runs_dir, args.out)
        print(f"Wrote aggregate run report to {args.out}")
        return

    if not args.gold or not args.pred:
        raise SystemExit(
            "Need --gold and --pred, or use --run_dir, or run with no arguments after a run exists under ./runs."
        )

    if not args.gold.exists():
        raise SystemExit(f"Gold file not found: {args.gold}")
    if not args.pred.exists():
        raise SystemExit(f"Prediction file not found: {args.pred}")

    maybe_run_official_eval(args)

    args.out.mkdir(parents=True, exist_ok=True)

    gold = read_dimsum(args.gold)
    pred = read_dimsum(args.pred)
    rows = align_gold_pred(gold, pred)
    metrics = compute_basic_metrics(rows)
    summary = load_summary(args.summary, args.eval_log)

    token_fields = [
        "sent_id", "tok_id", "word", "lemma", "pos",
        "gold_mwe", "pred_mwe", "gold_sup", "pred_sup",
        "mwe_correct", "sup_correct", "combined_correct", "error_type", "sentence",
    ]
    write_csv(args.out / "token_errors.csv", rows, token_fields)
    write_csv(args.out / "supersense_confusions.csv", confusion_rows(rows, "gold_sup", "pred_sup"), ["gold", "pred", "count", "correct"])
    write_csv(args.out / "mwe_tag_confusions.csv", confusion_rows(rows, "gold_mwe", "pred_mwe"), ["gold", "pred", "count", "correct"])
    write_sentence_errors(rows, args.out / "sentence_errors.md", args.max_sentences)
    write_markdown_report(args.out, metrics, summary, rows)
    plot_reports(args.out, rows, summary)

    print(f"Wrote report to: {args.out}")
    print("Key files:")
    print(f"  {args.out / 'summary_report.md'}")
    print(f"  {args.out / 'sentence_errors.md'}")
    print(f"  {args.out / 'token_errors.csv'}")
    print(f"  {args.out / 'supersense_confusions.csv'}")
    print(f"  {args.out / 'mwe_tag_confusions.csv'}")


if __name__ == "__main__":
    main()
