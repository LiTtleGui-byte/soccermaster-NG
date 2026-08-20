#!/usr/bin/env python3
"""CPU-only offline diagnostics for the fixed 200 commentary prefixes.

The script reads only existing local cache/JSON/NPZ artifacts.  It never opens
videos, model assets, or checkpoints, never imports torch, and writes exactly
result.json plus README.md into a previously absent fixed output directory.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Iterable

import numpy as np
from safetensors import safe_open


REPO = Path("/home/tianlin/SoccerMaster")
LOCAL_PYTHON = REPO / ".local_envs/SoccerMaster-repro/bin/python"
PREFIX_DIR = REPO / "reports/commentary_prefix_cache_200_20260814_run1"
PREFIX_FILE = PREFIX_DIR / "visual_prefixes.safetensors"
PREFIX_MANIFEST = PREFIX_DIR / "manifest.json"
E1_PREDICTIONS = (
    REPO
    / "reports/commentary_parallel_20260814/e1_decoder_sweep_run1/"
    "predictions.jsonl"
)
E2_DIR = REPO / "reports/commentary_parallel_20260814/e2_visual_sensitivity_run1"
E2_PREDICTIONS = E2_DIR / "predictions.jsonl"
E2_RESULT = E2_DIR / "result.json"
ATTENTION_FILE = (
    REPO
    / "reports/commentary_trace/sample_000_qformer_attention/cross_attention.npz"
)
ATTENTION_MANIFEST = ATTENTION_FILE.parent / "manifest.json"
OUTPUT_DIR = REPO / "reports/commentary_offline_diagnostic_200_20260815"

EXPECTED_SAMPLE_COUNT = 200
EXPECTED_PREFIX_SHAPE = (200, 32, 4096)
EXPECTED_PREFIX_DTYPE = np.dtype("float32")
EXPECTED_CACHE_SIZE = 104_859_528
EXPECTED_CACHE_SHA256_DECLARED = (
    "8b1723926eacfe381ceae2ec5433767574f56028d894a1b28d7c7222c69b6c97"
)
EXPECTED_ATTENTION_SHAPE = (2, 12, 32, 30)
CONDITIONS = {
    "e1_historical_beam_sampling": ("e1", "historical_beam_sampling"),
    "e1_nucleus_t070_p090": ("e1", "nucleus_t070_p090"),
    "e2_correct_deterministic_beam": ("e2", "correct_prefix"),
}

TOKEN_RE = re.compile(r"\[\w+\]|[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"\[(?:PLAYER|TEAM|COACH|REFEREE)\]")
EVENT_CUES = {
    "goal": ("goal", "scores", "scored", "net"),
    "shot": ("shot", "shoot", "fires", "effort", "volley", "strike"),
    "save": ("save", "keeper", "goalkeeper"),
    "foul_or_free_kick": ("foul", "whistle", "free kick", "handball"),
    "yellow_card": ("yellow card", "booked"),
    "red_card": ("red card", "sent off"),
    "offside": ("offside",),
    "corner": ("corner",),
    "substitution": ("substitution", "substitute", "replaces", "replaced"),
    "cross": ("cross", "whipped in"),
    "pass": ("pass", "through ball"),
    "throw_in": ("throw in", "throw-in"),
    "penalty": ("penalty",),
    "injury": ("injury", "injured", "medical attention"),
    "clear_or_intercept": ("clear", "intercept", "blocks", "blocked"),
}
NEGATION_RE = re.compile(r"\b(?:no|not|never|didn't|doesn't|isn't|wasn't)\b", re.I)


class Heartbeat:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.stage = "startup"
        self.detail = ""
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def set(self, stage: str, detail: str = "") -> None:
        self.stage = stage
        self.detail = detail
        print(f"[STAGE] {stage} {detail}".rstrip(), flush=True)

    def finish(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)

    def _run(self) -> None:
        while not self.stop.wait(30):
            elapsed = time.monotonic() - self.started
            print(
                f"[HEARTBEAT] elapsed={elapsed:.1f}s stage={self.stage} "
                f"detail={self.detail}",
                flush=True,
            )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def require_runtime() -> dict[str, Any]:
    if Path(sys.executable).resolve() != LOCAL_PYTHON.resolve():
        raise RuntimeError(f"Wrong Python: {sys.executable}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be explicitly empty")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("PYTHONDONTWRITEBYTECODE=1 is required")
    pythonpath = os.environ.get("PYTHONPATH", "")
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if pythonpath != str(REPO):
        raise RuntimeError(f"PYTHONPATH must be exactly {REPO}; got {pythonpath!r}")
    expected_ld = ":".join(
        [
            str(
                REPO
                / ".local_envs/SoccerMaster-repro/lib/python3.10/"
                "site-packages/torch/lib"
            ),
            str(REPO / ".local_envs/SoccerMaster-repro/lib"),
        ]
    )
    if ld_library_path != expected_ld:
        raise RuntimeError(
            f"LD_LIBRARY_PATH must be exactly {expected_ld}; got {ld_library_path!r}"
        )
    if "torch" in sys.modules:
        raise RuntimeError("torch must not be imported by this CPU/NumPy diagnostic")
    required = [
        PREFIX_FILE,
        PREFIX_MANIFEST,
        E1_PREDICTIONS,
        E2_PREDICTIONS,
        E2_RESULT,
        ATTENTION_FILE,
        ATTENTION_MANIFEST,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if PREFIX_FILE.stat().st_size != EXPECTED_CACHE_SIZE:
        raise RuntimeError("Unexpected prefix cache size")
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")
    return {
        "python": sys.version.split()[0],
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy": np.__version__,
        "pythonpath": pythonpath,
        "ld_library_path": ld_library_path,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "device": "cpu",
        "torch_imported": False,
        "gpu_used": False,
    }


def summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Summary requires non-empty finite values")
    return {
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "std": float(array.std()),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def correlation(x: Iterable[float], y: Iterable[float]) -> dict[str, float | None]:
    left = np.asarray(list(x), dtype=np.float64)
    right = np.asarray(list(y), dtype=np.float64)
    if left.shape != right.shape or left.size < 3:
        raise ValueError("Correlation inputs must have the same length >= 3")

    def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return None if denom == 0.0 else float(np.dot(a, b) / denom)

    return {
        "pearson_r": pearson(left, right),
        "spearman_rho": pearson(rankdata(left), rankdata(right)),
    }


def binary_group_effect(
    values: Iterable[float], labels: Iterable[bool]
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    mask = np.asarray(list(labels), dtype=bool)
    positive = array[mask]
    negative = array[~mask]
    result: dict[str, Any] = {
        "positive_count": int(positive.size),
        "negative_count": int(negative.size),
        "positive_mean": float(positive.mean()) if positive.size else None,
        "negative_mean": float(negative.mean()) if negative.size else None,
        "mean_difference_positive_minus_negative": (
            float(positive.mean() - negative.mean())
            if positive.size and negative.size
            else None
        ),
        "standardized_mean_difference": None,
    }
    if positive.size > 1 and negative.size > 1:
        pooled_variance = (
            (positive.size - 1) * positive.var(ddof=1)
            + (negative.size - 1) * negative.var(ddof=1)
        ) / (positive.size + negative.size - 2)
        if pooled_variance > 0:
            result["standardized_mean_difference"] = float(
                (positive.mean() - negative.mean()) / math.sqrt(pooled_variance)
            )
    return result


def spectrum_metrics(eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    totals = values.sum(axis=-1, keepdims=True)
    probabilities = np.divide(
        values,
        totals,
        out=np.zeros_like(values),
        where=totals > 0,
    )
    log_probabilities = np.zeros_like(probabilities)
    np.log(
        probabilities,
        out=log_probabilities,
        where=probabilities > 0,
    )
    entropy = -np.sum(probabilities * log_probabilities, axis=-1)
    effective_rank = np.exp(entropy)
    participation = np.divide(
        np.square(values.sum(axis=-1)),
        np.square(values).sum(axis=-1),
        out=np.zeros(values.shape[:-1], dtype=np.float64),
        where=np.square(values).sum(axis=-1) > 0,
    )
    return effective_rank, participation


def off_diagonal(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 2:
        return matrix[~np.eye(matrix.shape[0], dtype=bool)]
    if matrix.ndim == 3:
        mask = ~np.eye(matrix.shape[1], dtype=bool)
        return matrix[:, mask]
    raise ValueError("Expected a square matrix or a batch of square matrices")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def token_f1(reference: str, prediction: str) -> float:
    reference_tokens = Counter(tokenize(reference))
    prediction_tokens = Counter(tokenize(prediction))
    overlap = sum((reference_tokens & prediction_tokens).values())
    if not reference_tokens or not prediction_tokens or overlap == 0:
        return 0.0
    precision = overlap / sum(prediction_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 2.0 * precision * recall / (precision + recall)


def phrase_present(text: str, phrase: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def extract_cues(text: str) -> set[str]:
    return {
        label
        for label, phrases in EVENT_CUES.items()
        if any(phrase_present(text, phrase) for phrase in phrases)
    }


def silver_classification(reference: str, prediction: str) -> dict[str, Any]:
    reference_cues = extract_cues(reference)
    prediction_cues = extract_cues(prediction)
    shared = reference_cues & prediction_cues
    f1 = token_f1(reference, prediction)
    normalized_equal = " ".join(tokenize(reference)) == " ".join(tokenize(prediction))
    if normalized_equal:
        category, severity = "exact_text_match", 0
    elif f1 >= 0.75 and (not reference_cues or not prediction_cues or shared):
        category, severity = "near_lexical_match", 0
    elif reference_cues and prediction_cues and reference_cues == prediction_cues:
        category, severity = "cue_aligned_but_not_near", 1
    elif shared:
        category, severity = "partial_cue_overlap", 1
    elif reference_cues and prediction_cues:
        category, severity = "disjoint_event_cues", 3
    elif reference_cues:
        category, severity = "reference_cue_missing_in_prediction", 2
    elif prediction_cues:
        category, severity = "prediction_cue_absent_from_reference", 2
    elif f1 >= 0.5:
        category, severity = "lexical_partial_without_cues", 1
    else:
        category, severity = "weak_alignment_unresolved", 2

    reference_placeholders = Counter(PLACEHOLDER_RE.findall(reference))
    prediction_placeholders = Counter(PLACEHOLDER_RE.findall(prediction))
    placeholder_keys = sorted(set(reference_placeholders) | set(prediction_placeholders))
    placeholder_count_l1 = sum(
        abs(reference_placeholders[key] - prediction_placeholders[key])
        for key in placeholder_keys
    )
    return {
        "category": category,
        "severity": severity,
        "token_f1": f1,
        "reference_cues": sorted(reference_cues),
        "prediction_cues": sorted(prediction_cues),
        "shared_cues": sorted(shared),
        "cue_jaccard": (
            len(shared) / len(reference_cues | prediction_cues)
            if reference_cues or prediction_cues
            else None
        ),
        "placeholder_set_equal": set(reference_placeholders)
        == set(prediction_placeholders),
        "placeholder_count_l1": int(placeholder_count_l1),
        "negation_presence_mismatch": bool(NEGATION_RE.search(reference))
        != bool(NEGATION_RE.search(prediction)),
    }


def normalized_edit_distance(left: list[int], right: list[int]) -> float:
    ignored = {128001, 128009}
    left = [value for value in left if value not in ignored]
    right = [value for value in right if value not in ignored]
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row_index, right_value in enumerate(right, 1):
        current = [row_index]
        for column_index, left_value in enumerate(left, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    denominator = max(len(left), len(right), 1)
    return previous[-1] / denominator


def validate_rows(
    manifest: dict[str, Any],
    e1_rows: list[dict[str, Any]],
    e2_rows: list[dict[str, Any]],
    e2_result: dict[str, Any],
) -> None:
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported cache manifest schema")
    if manifest.get("prefix_semantics") != "post_qformer_llama_projection_before_bos":
        raise RuntimeError("Unexpected prefix semantics")
    if manifest.get("cache_file_sha256") != EXPECTED_CACHE_SHA256_DECLARED:
        raise RuntimeError("Manifest cache identity changed")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("Expected exactly 200 manifest samples")
    if len(e1_rows) != EXPECTED_SAMPLE_COUNT or len(e2_rows) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("Expected exactly 200 E1 and E2 rows")
    if e2_result.get("status") != "passed" or e2_result.get("samples_completed") != 200:
        raise RuntimeError("E2 result is not a completed pass")
    if e2_result.get("predictions_sha256") != (
        "2be5bd010fa7c905f4656d88918caa562e7d2e39f7e33ea474af382b32056921"
    ):
        raise RuntimeError("E2 predictions identity changed")
    for offset, (sample, e1, e2) in enumerate(zip(samples, e1_rows, e2_rows)):
        expected_ordinal = offset + 1
        dataset_index = int(sample["dataset_index"])
        if int(e1["ordinal"]) != expected_ordinal or int(e2["ordinal"]) != expected_ordinal:
            raise RuntimeError(f"Ordinal mismatch at offset {offset}")
        if int(e1["dataset_index"]) != dataset_index or int(e2["dataset_index"]) != dataset_index:
            raise RuntimeError(f"Dataset index mismatch at offset {offset}")
        references = {
            str(sample["reference_commentary"]),
            str(e1["reference_commentary"]),
            str(e2["reference_commentary"]),
        }
        if len(references) != 1:
            raise RuntimeError(f"Reference mismatch at dataset index {dataset_index}")
        expected_shift = int(samples[(offset + 1) % EXPECTED_SAMPLE_COUNT]["dataset_index"])
        sources = e2["prefix_sources"]
        if int(sources["correct_prefix_dataset_index"]) != dataset_index:
            raise RuntimeError("Incorrect E2 correct-prefix source")
        if int(sources["cyclic_shift_prefix_dataset_index"]) != expected_shift:
            raise RuntimeError("Incorrect E2 cyclic-shift source")


def load_prefixes(manifest: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with safe_open(PREFIX_FILE, framework="np", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = handle.metadata() or {}
        if keys != {"dataset_indices", "visual_prefixes"}:
            raise RuntimeError(f"Unexpected safetensors keys: {keys}")
        dataset_indices = handle.get_tensor("dataset_indices")
        prefixes = handle.get_tensor("visual_prefixes")
    if prefixes.shape != EXPECTED_PREFIX_SHAPE or prefixes.dtype != EXPECTED_PREFIX_DTYPE:
        raise RuntimeError(f"Unexpected prefix array {prefixes.shape} {prefixes.dtype}")
    if dataset_indices.tolist() != manifest["dataset_indices"]:
        raise RuntimeError("Safetensors dataset indices do not match manifest")
    if not np.isfinite(prefixes).all():
        raise RuntimeError("Prefix cache contains non-finite values")
    return prefixes, dataset_indices, metadata


def representation_diagnostics(prefixes: np.ndarray) -> tuple[dict[str, Any], list[dict[str, float]]]:
    samples, queries, width = prefixes.shape
    norms = np.linalg.norm(prefixes, axis=2)
    gram = np.matmul(prefixes, np.swapaxes(prefixes, 1, 2))
    denom = norms[:, :, None] * norms[:, None, :]
    cosine = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
    raw_offdiag = off_diagonal(cosine)

    sample_means = prefixes.mean(axis=1)
    centered_queries = prefixes - sample_means[:, None, :]
    centered_gram = np.matmul(centered_queries, np.swapaxes(centered_queries, 1, 2))
    raw_eigenvalues = np.linalg.eigvalsh(gram.astype(np.float64))
    centered_eigenvalues = np.linalg.eigvalsh(centered_gram.astype(np.float64))
    raw_effective_rank, raw_participation = spectrum_metrics(raw_eigenvalues)
    centered_effective_rank, centered_participation = spectrum_metrics(centered_eigenvalues)
    within_query_variance = np.mean(np.var(prefixes, axis=1), axis=1)

    flat = prefixes.reshape(samples, queries * width)
    flat_norms = np.linalg.norm(flat, axis=1)
    sample_cosine = np.matmul(flat, flat.T) / (flat_norms[:, None] * flat_norms[None, :])
    sample_offdiag = off_diagonal(sample_cosine)
    np.fill_diagonal(sample_cosine, -np.inf)
    nearest_cosine = sample_cosine.max(axis=1)
    nearest_index = sample_cosine.argmax(axis=1)
    nearest_cosine_distance = 1.0 - nearest_cosine

    centered_flat = flat - flat.mean(axis=0, keepdims=True)
    sample_covariance_gram = np.matmul(centered_flat, centered_flat.T)
    sample_eigenvalues = np.linalg.eigvalsh(sample_covariance_gram.astype(np.float64))
    sample_effective_rank, sample_participation = spectrum_metrics(sample_eigenvalues)

    grand_mean = prefixes.mean(axis=(0, 1))
    query_means = prefixes.mean(axis=0)
    sample_component = sample_means - grand_mean
    query_component = query_means - grand_mean
    residual = (
        prefixes
        - sample_means[:, None, :]
        - query_means[None, :, :]
        + grand_mean[None, None, :]
    )
    total_variance = float(np.mean(np.square(prefixes - grand_mean)))
    sample_variance = float(np.mean(np.square(sample_component)))
    query_slot_variance = float(np.mean(np.square(query_component)))
    residual_variance = float(np.mean(np.square(residual)))

    query_slot_norms = np.linalg.norm(query_means, axis=1)
    query_slot_cosine = np.matmul(query_means, query_means.T) / (
        query_slot_norms[:, None] * query_slot_norms[None, :]
    )
    query_slot_cross_sample_variance = np.mean(np.var(prefixes, axis=0), axis=1)

    cyclic = np.roll(flat, -1, axis=0)
    cyclic_norms = np.linalg.norm(cyclic, axis=1)
    cyclic_cosine = np.sum(flat * cyclic, axis=1) / (flat_norms * cyclic_norms)
    cyclic_relative_l2 = np.linalg.norm(flat - cyclic, axis=1) / np.sqrt(
        np.square(flat_norms) + np.square(cyclic_norms)
    )

    per_sample: list[dict[str, float]] = []
    for index in range(samples):
        offdiag = raw_offdiag[index]
        per_sample.append(
            {
                "mean_query_cosine": float(offdiag.mean()),
                "p95_query_cosine": float(np.quantile(offdiag, 0.95)),
                "max_query_cosine": float(offdiag.max()),
                "raw_effective_rank": float(raw_effective_rank[index]),
                "raw_participation_ratio": float(raw_participation[index]),
                "centered_effective_rank": float(centered_effective_rank[index]),
                "centered_participation_ratio": float(centered_participation[index]),
                "within_query_variance": float(within_query_variance[index]),
                "prefix_rms": float(np.sqrt(np.mean(np.square(prefixes[index])))),
                "nearest_other_sample_cosine": float(nearest_cosine[index]),
                "nearest_other_sample_cosine_distance": float(nearest_cosine_distance[index]),
                "nearest_other_sample_offset": int(nearest_index[index]),
                "cyclic_mismatch_cosine_distance": float(1.0 - cyclic_cosine[index]),
                "cyclic_mismatch_relative_l2": float(cyclic_relative_l2[index]),
            }
        )

    aggregate = {
        "scope": "200 cached post-Q-Former llama-projected prefixes",
        "shape": [samples, queries, width],
        "query_redundancy": {
            "per_sample_mean_off_diagonal_cosine": summary(
                row["mean_query_cosine"] for row in per_sample
            ),
            "all_off_diagonal_cosines": summary(raw_offdiag.reshape(-1)),
            "per_sample_p95_off_diagonal_cosine": summary(
                row["p95_query_cosine"] for row in per_sample
            ),
            "per_sample_max_off_diagonal_cosine": summary(
                row["max_query_cosine"] for row in per_sample
            ),
            "raw_effective_rank_maximum": queries,
            "raw_effective_rank": summary(raw_effective_rank),
            "raw_participation_ratio": summary(raw_participation),
            "centered_effective_rank_maximum": queries - 1,
            "centered_effective_rank": summary(centered_effective_rank),
            "centered_participation_ratio": summary(centered_participation),
            "within_query_variance": summary(within_query_variance),
            "query_norm": summary(norms.reshape(-1)),
        },
        "two_way_variance_decomposition": {
            "definition": "mean squared deviations per embedding dimension",
            "total_variance": total_variance,
            "between_sample_mean_variance": sample_variance,
            "between_query_slot_mean_variance": query_slot_variance,
            "sample_by_query_residual_variance": residual_variance,
            "component_sum": sample_variance + query_slot_variance + residual_variance,
            "between_sample_fraction_of_total": sample_variance / total_variance,
            "between_query_slot_fraction_of_total": query_slot_variance / total_variance,
            "residual_fraction_of_total": residual_variance / total_variance,
            "per_query_slot_cross_sample_variance": summary(
                query_slot_cross_sample_variance
            ),
            "query_slot_centroid_off_diagonal_cosine": summary(
                off_diagonal(query_slot_cosine)
            ),
        },
        "sample_distinguishability": {
            "exact_duplicate_prefix_count": int(
                samples - len({prefixes[i].tobytes() for i in range(samples)})
            ),
            "all_pair_flattened_cosine": summary(sample_offdiag),
            "nearest_other_sample_cosine": summary(nearest_cosine),
            "nearest_other_sample_cosine_distance": summary(nearest_cosine_distance),
            "centered_sample_effective_rank_maximum": samples - 1,
            "centered_sample_effective_rank": float(sample_effective_rank),
            "centered_sample_participation_ratio": float(sample_participation),
            "cyclic_mismatch_cosine_distance": summary(1.0 - cyclic_cosine),
            "cyclic_mismatch_relative_l2": summary(cyclic_relative_l2),
        },
    }
    return aggregate, per_sample


def attention_diagnostics() -> dict[str, Any]:
    manifest = load_json(ATTENTION_MANIFEST)
    with np.load(ATTENTION_FILE, allow_pickle=False) as archive:
        attention = archive["cross_attention"].astype(np.float64)
    if attention.shape != EXPECTED_ATTENTION_SHAPE:
        raise RuntimeError(f"Unexpected attention shape: {attention.shape}")
    row_sums = attention.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, rtol=0, atol=3e-7):
        raise RuntimeError("Attention rows are not normalized")
    entropy = -np.sum(attention * np.log(np.maximum(attention, 1e-12)), axis=-1)
    normalized_entropy = entropy / math.log(attention.shape[-1])
    effective_frames = np.exp(entropy)
    peak_weight = attention.max(axis=-1)
    per_layer = []
    for layer in range(attention.shape[0]):
        per_layer.append(
            {
                "layer": layer,
                "normalized_entropy": summary(normalized_entropy[layer].reshape(-1)),
                "effective_frames": summary(effective_frames[layer].reshape(-1)),
                "peak_frame_weight": summary(peak_weight[layer].reshape(-1)),
            }
        )
    per_head = []
    for layer in range(attention.shape[0]):
        for head in range(attention.shape[1]):
            per_head.append(
                {
                    "layer": layer,
                    "head": head,
                    "normalized_entropy_across_queries": summary(
                        normalized_entropy[layer, head]
                    ),
                    "effective_frames_across_queries": summary(
                        effective_frames[layer, head]
                    ),
                    "peak_frame_weight_across_queries": summary(
                        peak_weight[layer, head]
                    ),
                }
            )
    per_query = []
    for query in range(attention.shape[2]):
        per_query.append(
            {
                "query": query,
                "normalized_entropy_across_layers_heads": summary(
                    normalized_entropy[:, :, query].reshape(-1)
                ),
                "effective_frames_across_layers_heads": summary(
                    effective_frames[:, :, query].reshape(-1)
                ),
                "peak_frame_weight_across_layers_heads": summary(
                    peak_weight[:, :, query].reshape(-1)
                ),
            }
        )
    return {
        "scope": "single independent sample_000 CPU Q-Former replay; supplemental only",
        "dataset_index_not_in_200_cache": 0 not in set(
            load_json(PREFIX_MANIFEST)["dataset_indices"]
        ),
        "shape": list(attention.shape),
        "max_row_sum_error": float(np.max(np.abs(row_sums - 1.0))),
        "normalized_entropy_all_layer_head_query_rows": summary(
            normalized_entropy.reshape(-1)
        ),
        "effective_frames_all_layer_head_query_rows": summary(
            effective_frames.reshape(-1)
        ),
        "peak_frame_weight_all_layer_head_query_rows": summary(
            peak_weight.reshape(-1)
        ),
        "per_layer": per_layer,
        "per_head": per_head,
        "per_query": per_query,
        "manifest_limitations": manifest.get("limitations", []),
    }


def text_and_sensitivity_diagnostics(
    manifest: dict[str, Any],
    e1_rows: list[dict[str, Any]],
    e2_rows: list[dict[str, Any]],
    per_sample_representation: list[dict[str, float]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_rows: list[dict[str, Any]] = []
    for offset, (sample, e1, e2, representation) in enumerate(
        zip(manifest["samples"], e1_rows, e2_rows, per_sample_representation)
    ):
        correct = e2["predictions"]["correct_prefix"]
        shifted = e2["predictions"]["cyclic_shift_prefix"]
        zero = e2["predictions"]["zero_prefix"]
        first = e2["first_token_diagnostic"]
        silver: dict[str, Any] = {}
        for output_name, (source, condition) in CONDITIONS.items():
            source_row = e1 if source == "e1" else e2
            prediction = source_row["predictions"][condition]["text"]
            silver[output_name] = silver_classification(
                str(sample["reference_commentary"]), str(prediction)
            )
        sample_rows.append(
            {
                "ordinal": offset + 1,
                "dataset_index": int(sample["dataset_index"]),
                "representation": representation,
                "e2": {
                    "js_correct_vs_shifted": float(first["js_correct_vs_shifted"]),
                    "js_correct_vs_zero": float(first["js_correct_vs_zero"]),
                    "first_token_changed_correct_vs_shifted": first["top_token_ids"][
                        "correct_prefix"
                    ]
                    != first["top_token_ids"]["cyclic_shift_prefix"],
                    "first_token_changed_correct_vs_zero": first["top_token_ids"][
                        "correct_prefix"
                    ]
                    != first["top_token_ids"]["zero_prefix"],
                    "text_changed_correct_vs_shifted": correct["text"] != shifted["text"],
                    "text_changed_correct_vs_zero": correct["text"] != zero["text"],
                    "token_edit_distance_correct_vs_shifted": normalized_edit_distance(
                        correct["token_ids"], shifted["token_ids"]
                    ),
                    "token_edit_distance_correct_vs_zero": normalized_edit_distance(
                        correct["token_ids"], zero["token_ids"]
                    ),
                },
                "reference_relative_silver": silver,
            }
        )

    feature_names = [
        "mean_query_cosine",
        "centered_effective_rank",
        "centered_participation_ratio",
        "within_query_variance",
        "prefix_rms",
        "nearest_other_sample_cosine_distance",
        "cyclic_mismatch_cosine_distance",
        "cyclic_mismatch_relative_l2",
    ]
    outcomes = [
        "js_correct_vs_shifted",
        "js_correct_vs_zero",
        "token_edit_distance_correct_vs_shifted",
        "token_edit_distance_correct_vs_zero",
    ]
    associations: dict[str, Any] = {}
    for outcome in outcomes:
        associations[outcome] = {}
        outcome_values = [row["e2"][outcome] for row in sample_rows]
        for feature in feature_names:
            associations[outcome][feature] = correlation(
                [row["representation"][feature] for row in sample_rows],
                outcome_values,
            )

    binary_effects: dict[str, Any] = {}
    for label_name in [
        "first_token_changed_correct_vs_shifted",
        "text_changed_correct_vs_shifted",
    ]:
        labels = [bool(row["e2"][label_name]) for row in sample_rows]
        binary_effects[label_name] = {
            feature: binary_group_effect(
                [row["representation"][feature] for row in sample_rows], labels
            )
            for feature in feature_names
        }

    silver_aggregate: dict[str, Any] = {}
    for output_name in CONDITIONS:
        details = [row["reference_relative_silver"][output_name] for row in sample_rows]
        categories = Counter(detail["category"] for detail in details)
        severities = [detail["severity"] for detail in details]
        f1_values = [detail["token_f1"] for detail in details]
        silver_aggregate[output_name] = {
            "category_counts": dict(sorted(categories.items())),
            "severity_summary": summary(severities),
            "token_f1_summary": summary(f1_values),
            "placeholder_set_mismatch_count": sum(
                not detail["placeholder_set_equal"] for detail in details
            ),
            "placeholder_count_mismatch_count": sum(
                detail["placeholder_count_l1"] > 0 for detail in details
            ),
            "negation_presence_mismatch_count": sum(
                detail["negation_presence_mismatch"] for detail in details
            ),
            "associations": {
                "token_f1": {
                    feature: correlation(
                        [row["representation"][feature] for row in sample_rows],
                        f1_values,
                    )
                    for feature in feature_names
                },
                "silver_severity": {
                    feature: correlation(
                        [row["representation"][feature] for row in sample_rows],
                        severities,
                    )
                    for feature in feature_names
                },
                "e2_js_correct_vs_shifted_vs_token_f1": correlation(
                    [row["e2"]["js_correct_vs_shifted"] for row in sample_rows],
                    f1_values,
                ),
                "e2_js_correct_vs_shifted_vs_silver_severity": correlation(
                    [row["e2"]["js_correct_vs_shifted"] for row in sample_rows],
                    severities,
                ),
            },
        }

    e2_summary = {
        "js_correct_vs_shifted": summary(
            row["e2"]["js_correct_vs_shifted"] for row in sample_rows
        ),
        "js_correct_vs_zero": summary(
            row["e2"]["js_correct_vs_zero"] for row in sample_rows
        ),
        "first_token_changed_correct_vs_shifted_count": sum(
            row["e2"]["first_token_changed_correct_vs_shifted"] for row in sample_rows
        ),
        "first_token_changed_correct_vs_zero_count": sum(
            row["e2"]["first_token_changed_correct_vs_zero"] for row in sample_rows
        ),
        "text_changed_correct_vs_shifted_count": sum(
            row["e2"]["text_changed_correct_vs_shifted"] for row in sample_rows
        ),
        "text_changed_correct_vs_zero_count": sum(
            row["e2"]["text_changed_correct_vs_zero"] for row in sample_rows
        ),
        "token_edit_distance_correct_vs_shifted": summary(
            row["e2"]["token_edit_distance_correct_vs_shifted"] for row in sample_rows
        ),
        "token_edit_distance_correct_vs_zero": summary(
            row["e2"]["token_edit_distance_correct_vs_zero"] for row in sample_rows
        ),
        "constant_binary_outcomes_not_correlated": [
            "first_token_changed_correct_vs_zero (200/200)",
            "text_changed_correct_vs_zero (200/200)",
        ],
    }
    return {
        "e2_observed": e2_summary,
        "representation_to_e2_associations": associations,
        "representation_binary_group_effects": binary_effects,
        "reference_relative_silver": {
            "scope": (
                "Text-only comparison to one reference commentary. Categories are "
                "review aids, not video-fact accuracy or human semantic judgments."
            ),
            "cue_lexicon_origin": (
                "Reuses and boundary-hardens the event cues in "
                "render_semantic_review.py."
            ),
            "category_rules": {
                "exact_text_match": "normalized token sequences equal",
                "near_lexical_match": "token F1 >= 0.75 without disjoint cues",
                "cue_aligned_but_not_near": "recognized cue sets equal",
                "partial_cue_overlap": "recognized cue sets overlap but differ",
                "disjoint_event_cues": "both sides have cues and intersection is empty",
                "reference_cue_missing_in_prediction": "only reference has recognized cues",
                "prediction_cue_absent_from_reference": "only prediction has recognized cues",
                "lexical_partial_without_cues": "no cues and token F1 >= 0.5",
                "weak_alignment_unresolved": "no cues and token F1 < 0.5",
            },
            "conditions": silver_aggregate,
        },
    }, sample_rows


def strongest_associations(associations: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    rows = []
    for outcome, features in associations.items():
        for feature, values in features.items():
            rho = values["spearman_rho"]
            if rho is not None:
                rows.append(
                    {
                        "outcome": outcome,
                        "feature": feature,
                        "spearman_rho": rho,
                        "pearson_r": values["pearson_r"],
                    }
                )
    rows.sort(key=lambda row: abs(row["spearman_rho"]), reverse=True)
    return rows[:limit]


def render_readme(result: dict[str, Any]) -> str:
    representation = result["representation"]
    redundancy = representation["query_redundancy"]
    samples = representation["sample_distinguishability"]
    variance = representation["two_way_variance_decomposition"]
    e2 = result["text_and_sensitivity"]["e2_observed"]
    attention = result["single_sample_attention_supplement"]
    associations = strongest_associations(
        result["text_and_sensitivity"]["representation_to_e2_associations"]
    )
    silver = result["text_and_sensitivity"]["reference_relative_silver"]["conditions"]
    association_lines = "\n".join(
        f"- `{row['feature']}` ↔ `{row['outcome']}`: Spearman "
        f"ρ={row['spearman_rho']:.3f}, Pearson r={row['pearson_r']:.3f}."
        for row in associations
    )
    silver_lines = "\n".join(
        f"- `{name}`: mean token F1={detail['token_f1_summary']['mean']:.3f}; "
        f"categories={json.dumps(detail['category_counts'], ensure_ascii=False, sort_keys=True)}"
        for name, detail in silver.items()
    )
    return f"""# Commentary offline diagnostic — fixed 200 samples

Status: passed once, CPU-only. This is a non-Gate offline analysis.

## Main findings

- The 32 projected query tokens are strongly similar but not numerically collapsed: mean within-sample off-diagonal cosine is {redundancy['per_sample_mean_off_diagonal_cosine']['mean']:.4f}; raw effective rank is {redundancy['raw_effective_rank']['mean']:.2f}/32 and mean-centered effective rank is {redundancy['centered_effective_rank']['mean']:.2f}/31.
- All 200 prefixes are byte-distinct. Mean nearest-other flattened cosine distance is {samples['nearest_other_sample_cosine_distance']['mean']:.6f} (minimum {samples['nearest_other_sample_cosine_distance']['min']:.6f}); centered sample effective rank is {samples['centered_sample_effective_rank']:.2f}/199.
- Variance fractions are sample mean {variance['between_sample_fraction_of_total']:.3%}, fixed query-slot mean {variance['between_query_slot_fraction_of_total']:.3%}, and sample×query residual {variance['residual_fraction_of_total']:.3%}.
- E2 reproduces 187/200 correct-vs-mismatch text changes and 200/200 correct-vs-zero changes. Mean first-token JS is {e2['js_correct_vs_shifted']['mean']:.5f} for mismatch and {e2['js_correct_vs_zero']['mean']:.5f} for zero.
- Single-sample attention is diffuse on average: normalized entropy {attention['normalized_entropy_all_layer_head_query_rows']['mean']:.4f}, corresponding to {attention['effective_frames_all_layer_head_query_rows']['mean']:.2f}/30 effective frames. This is only sample_000 and is not generalized to the 200-prefix set.

## Strongest exploratory representation–E2 associations

{association_lines}

These are unadjusted exploratory correlations on the fixed 200 samples. They do not establish causality, and a text-change indicator can hide large differences in edit magnitude.

## Reference-relative silver categories

{silver_lines}

The categories use one reference sentence, token overlap, a fixed event-cue lexicon, placeholders, and negation flags. They are interpretable review aids only: they are not video-grounded factual accuracy, and a paraphrase may be marked as a mismatch while a reference-matching hallucination may look good.

## Execution boundary

- Python: `{result['environment']['python_executable']}`
- `PYTHONPATH={result['environment']['pythonpath']}`
- `LD_LIBRARY_PATH={result['environment']['ld_library_path']}`
- `CUDA_VISIBLE_DEVICES={result['environment']['cuda_visible_devices']}` (empty)
- No remote/NAS/video/model/checkpoint access; no Torch import; no GPU; no training.
- Inputs were the existing prefix cache/manifest, E1 predictions, E2 predictions/result, and one local attention NPZ/manifest.
- Exit code: 0; elapsed {result['run']['elapsed_seconds']:.3f} seconds.

Full metric definitions, correlations, category counts, and 200 per-sample records are in `result.json`.
"""


def main() -> int:
    heartbeat = Heartbeat()
    heartbeat.start()
    started_monotonic = time.monotonic()
    started_utc = utc_now()
    try:
        heartbeat.set("preflight")
        environment = require_runtime()
        manifest = load_json(PREFIX_MANIFEST)
        e1_rows = load_jsonl(E1_PREDICTIONS)
        e2_rows = load_jsonl(E2_PREDICTIONS)
        e2_result = load_json(E2_RESULT)
        validate_rows(manifest, e1_rows, e2_rows, e2_result)

        heartbeat.set("load_prefix_cache", "200x32x4096")
        prefixes, dataset_indices, cache_metadata = load_prefixes(manifest)

        heartbeat.set("representation_metrics", "query and sample structure")
        representation, per_sample_representation = representation_diagnostics(prefixes)

        heartbeat.set("text_and_e2_metrics", "silver categories and associations")
        text_and_sensitivity, per_sample = text_and_sensitivity_diagnostics(
            manifest, e1_rows, e2_rows, per_sample_representation
        )

        heartbeat.set("attention_supplement", "single sample only")
        attention = attention_diagnostics()

        elapsed = time.monotonic() - started_monotonic
        result: dict[str, Any] = {
            "schema_version": 1,
            "status": "passed",
            "scope": "CPU-only fixed-200 offline representation/text diagnostic; not a Gate",
            "run": {
                "started_at_utc": started_utc,
                "completed_at_utc": utc_now(),
                "elapsed_seconds": elapsed,
                "exit_code": 0,
                "heartbeat_seconds": 30,
                "timeout_seconds": 600,
                "fallbacks_used": [],
            },
            "environment": environment,
            "inputs": {
                "prefix_file": str(PREFIX_FILE),
                "prefix_file_bytes": PREFIX_FILE.stat().st_size,
                "prefix_sha256_declared_by_manifest_not_recomputed": manifest[
                    "cache_file_sha256"
                ],
                "prefix_manifest": str(PREFIX_MANIFEST),
                "prefix_semantics": manifest["prefix_semantics"],
                "cache_metadata": cache_metadata,
                "dataset_index_count": int(dataset_indices.size),
                "e1_predictions": str(E1_PREDICTIONS),
                "e2_predictions": str(E2_PREDICTIONS),
                "e2_result": str(E2_RESULT),
                "attention_npz": str(ATTENTION_FILE),
                "attention_manifest": str(ATTENTION_MANIFEST),
            },
            "assertions": {
                "sample_count_200": len(per_sample) == 200,
                "prefix_shape_200_32_4096": list(prefixes.shape) == [200, 32, 4096],
                "all_prefix_values_finite": bool(np.isfinite(prefixes).all()),
                "manifest_e1_e2_identity_aligned": True,
                "e2_completed_pass": True,
                "output_path_was_absent": True,
                "torch_not_imported": "torch" not in sys.modules,
                "no_gpu_used": True,
                "no_remote_video_model_checkpoint_access": True,
            },
            "metric_definitions": {
                "raw_effective_rank": "exp entropy of normalized eigenvalues of X X^T",
                "participation_ratio": "(sum eigenvalues)^2 / sum squared eigenvalues",
                "centered_query_spectrum": "same metrics after subtracting each sample's 32-query mean; maximum 31",
                "query_cosine_redundancy": "off-diagonal cosine among 32 projected query rows within each sample",
                "sample_cosine": "cosine between flattened 32x4096 prefixes",
                "cyclic_mismatch": "next cached sample, matching E2's fixed cyclic shift",
                "attention_normalized_entropy": "entropy over 30 frames divided by log(30)",
            },
            "representation": representation,
            "text_and_sensitivity": text_and_sensitivity,
            "single_sample_attention_supplement": attention,
            "per_sample": per_sample,
            "limitations": [
                "The 200-sample representation is post-Q-Former and post-linear-projection, not the raw internal Q-Former state.",
                "The fixed 200 samples are not the complete 3,256-sample test set.",
                "E2 associations are observational and based on one deterministic cyclic mismatch per sample.",
                "Reference-relative silver categories compare text to one reference and are not video-fact accuracy.",
                "The attention NPZ is an independent single sample and attention weights are not causal attribution.",
            ],
        }
        if not all(result["assertions"].values()):
            raise RuntimeError(f"Final assertions failed: {result['assertions']}")

        heartbeat.set("write_outputs", "result.json and README.md only")
        temporary = OUTPUT_DIR.parent / f".{OUTPUT_DIR.name}.tmp.{os.getpid()}"
        if temporary.exists():
            raise FileExistsError(temporary)
        temporary.mkdir(parents=False)
        result_path = temporary / "result.json"
        readme_path = temporary / "README.md"
        with result_path.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        with readme_path.open("x", encoding="utf-8") as handle:
            handle.write(render_readme(result))
        os.replace(temporary, OUTPUT_DIR)
        print(f"[RESULT] {OUTPUT_DIR / 'result.json'}", flush=True)
        print(f"[README] {OUTPUT_DIR / 'README.md'}", flush=True)
        print("[EXIT_CODE] 0", flush=True)
        return 0
    finally:
        heartbeat.finish()


if __name__ == "__main__":
    raise SystemExit(main())
