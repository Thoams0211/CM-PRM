#!/usr/bin/env python3
"""统计生成答案与原始答案的一致性（不使用 argparse）。

直接修改下方 CONFIG 后运行：
    python analyse.py
"""

from __future__ import annotations

import json
import math
import os
import re
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

# Configuration Parameters
CONFIG = {
    "pred_path": "buffer/generator.json", 
    "gt_path": "eval_data/MATH/test.jsonl",  
    "id_key": "id",
    "pred_answer_key": "answer",
    "gt_answer_key": "answer",
    "output_compare_path": "buffer/answer_compare.json", 
    "abs_tol": 1e-9,
    "rel_tol": 1e-9,
}


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass

    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _strip_wrappers(s: str) -> str:
    s = s.strip()
    s = s.replace("\u2212", "-")
    s = s.replace("，", ",")
    s = s.strip(" ")

    # Remove markdown/latex wrappers
    s = s.replace("$", "")
    s = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\left", "", s)
    s = re.sub(r"\\right", "", s)

    # Extract ##answer: ... ## / #answer#: ...
    m = re.search(r"##answer\s*:\s*(.*?)\s*##", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        s = m.group(1).strip()
    m = re.search(r"#answer#\s*:\s*(.*)$", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        s = m.group(1).strip()

    # If x=... / y=... take the right side of the equal sign
    m = re.match(r"^[a-zA-Z]\w*\s*=\s*(.+)$", s)
    if m:
        s = m.group(1).strip()

    # Remove trailing period
    s = s.rstrip("。.")
    return s.strip()


def _latex_frac_to_plain(s: str) -> str:
    # \frac{a}{b} -> (a)/(b)
    pattern = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    while True:
        m = pattern.search(s)
        if not m:
            break
        a, b = m.group(1), m.group(2)
        s = s[: m.start()] + f"({a})/({b})" + s[m.end() :]
    return s


def _to_number_simple(raw: str) -> Optional[float]:
    s = _strip_wrappers(str(raw))
    s = _latex_frac_to_plain(s)
    s = s.replace(" ", "")

    if not s:
        return None

    # Percentage
    if s.endswith("%"):
        core = s[:-1]
        v = _to_number_simple(core)
        return None if v is None else v / 100.0

    # Pure numbers (including scientific notation)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?", s):
        try:
            return float(s)
        except Exception:
            return None

    # Fraction a/b
    if re.fullmatch(r"[+-]?\d+\/[+-]?\d+", s):
        try:
            return float(Fraction(s))
        except Exception:
            return None

    # Parentheses fraction (a)/(b)
    m = re.fullmatch(r"\(([+-]?\d+)\)\/\(([+-]?\d+)\)", s)
    if m:
        try:
            return float(Fraction(int(m.group(1)), int(m.group(2))))
        except Exception:
            return None

    return None


def _to_number_sympy(raw: str) -> Optional[float]:
    """Optional use sympy to parse more complex expressions."""
    try:
        import sympy as sp  # type: ignore
    except Exception:
        return None

    s = _strip_wrappers(str(raw))
    s = _latex_frac_to_plain(s)
    s = s.replace("^", "**")

    if re.search(r"[a-zA-Z]", s):
        # Allow common constant function names
        allowed = {"pi", "e", "sqrt"}
        tokens = set(re.findall(r"[a-zA-Z_]+", s))
        if any(t not in allowed for t in tokens):
            return None

    try:
        expr = sp.sympify(s)
        if expr.free_symbols:
            return None
        val = float(sp.N(expr, 50))
        if math.isfinite(val):
            return val
    except Exception:
        return None
    return None


def parse_numeric_value(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    v = _to_number_simple(s)
    if v is not None:
        return v

    v = _to_number_sympy(s)
    if v is not None:
        return v

    return None


def answers_equal(a: Any, b: Any, abs_tol: float, rel_tol: float) -> Tuple[bool, str]:
    va = parse_numeric_value(a)
    vb = parse_numeric_value(b)

    if va is not None and vb is not None:
        ok = math.isclose(va, vb, rel_tol=rel_tol, abs_tol=abs_tol)
        return ok, f"numeric compare: {va} vs {vb}"

    # fallback: Normalize string exact comparison
    sa = _strip_wrappers(str(a))
    sb = _strip_wrappers(str(b))
    ok = sa == sb
    return ok, f"string fallback: '{sa}' vs '{sb}'"


def to_map(rows: List[Dict[str, Any]], id_key: str) -> Dict[Any, Dict[str, Any]]:
    out: Dict[Any, Dict[str, Any]] = {}
    for r in rows:
        if id_key in r:
            out[r[id_key]] = r
    return out


def main() -> None:
    pred_rows = load_json_or_jsonl(CONFIG["pred_path"])
    gt_rows = load_json_or_jsonl(CONFIG["gt_path"])

    pred_map = to_map(pred_rows, CONFIG["id_key"])
    gt_map = to_map(gt_rows, CONFIG["id_key"])

    pred_ids = set(pred_map.keys())
    gt_ids = set(gt_map.keys())
    common_ids = sorted(pred_ids & gt_ids)

    only_pred = sorted(pred_ids - gt_ids)
    only_gt = sorted(gt_ids - pred_ids)

    comparable = 0
    matched = 0
    pred_missing_answer = 0
    gt_missing_answer = 0

    compare_rows: List[Dict[str, Any]] = []

    for sid in common_ids:
        p = pred_map[sid]
        g = gt_map[sid]
        pa = p.get(CONFIG["pred_answer_key"])
        ga = g.get(CONFIG["gt_answer_key"])

        row: Dict[str, Any] = {
            "id": sid,
            "pred_answer": pa,
            "gt_answer": ga,
            "status": "",
            "match": False,
            "reason": "",
        }

        if pa is None or str(pa).strip() == "":
            pred_missing_answer += 1
            row["status"] = "pred_missing_answer"
            row["reason"] = "prediction answer missing"
            compare_rows.append(row)
            continue

        if ga is None or str(ga).strip() == "":
            gt_missing_answer += 1
            row["status"] = "gt_missing_answer"
            row["reason"] = "ground-truth answer missing"
            compare_rows.append(row)
            continue

        comparable += 1
        ok, reason = answers_equal(pa, ga, CONFIG["abs_tol"], CONFIG["rel_tol"])
        row["status"] = "compared"
        row["match"] = ok
        row["reason"] = reason
        if ok:
            matched += 1
        compare_rows.append(row)

    # ids only in pred/gt also write to result file
    for sid in only_pred:
        p = pred_map[sid]
        compare_rows.append(
            {
                "id": sid,
                "pred_answer": p.get(CONFIG["pred_answer_key"]),
                "gt_answer": None,
                "status": "only_in_pred",
                "match": False,
                "reason": "id only exists in prediction file",
            }
        )

    for sid in only_gt:
        g = gt_map[sid]
        compare_rows.append(
            {
                "id": sid,
                "pred_answer": None,
                "gt_answer": g.get(CONFIG["gt_answer_key"]),
                "status": "only_in_gt",
                "match": False,
                "reason": "id only exists in ground-truth file",
            }
        )

    acc = (matched / comparable) if comparable else 0.0

    output = {
        "summary": {
            "pred_path": CONFIG["pred_path"],
            "gt_path": CONFIG["gt_path"],
            "pred_samples": len(pred_rows),
            "gt_samples": len(gt_rows),
            "id_overlap": len(common_ids),
            "only_in_pred": len(only_pred),
            "only_in_gt": len(only_gt),
            "pred_missing_answer": pred_missing_answer,
            "gt_missing_answer": gt_missing_answer,
            "comparable_samples": comparable,
            "matched_samples": matched,
            "accuracy": acc,
            "abs_tol": CONFIG["abs_tol"],
            "rel_tol": CONFIG["rel_tol"],
        },
        "records": compare_rows,
    }

    output_path = CONFIG["output_compare_path"]
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("Answer Consistency Report")
    print("=" * 72)
    print(f"pred_path                : {CONFIG['pred_path']}")
    print(f"gt_path                  : {CONFIG['gt_path']}")
    print(f"pred samples             : {len(pred_rows)}")
    print(f"gt samples               : {len(gt_rows)}")
    print(f"id overlap               : {len(common_ids)}")
    print(f"only in pred             : {len(only_pred)}")
    print(f"only in gt               : {len(only_gt)}")
    print("-" * 72)
    print(f"pred missing answer      : {pred_missing_answer}")
    print(f"gt missing answer        : {gt_missing_answer}")
    print(f"comparable samples       : {comparable}")
    print(f"matched samples          : {matched}")
    print(f"accuracy                 : {acc:.4%}")
    print(f"saved compare json       : {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
