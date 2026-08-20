#!/usr/bin/env python3
"""Local, CPU-only web server for video-grounding calibration review."""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET = ROOT / "reports/commentary_fact_calibration_24_20260818"
TEACHER_DIR = ROOT / "reports/commentary_multi_event_luna_5_20260818"
ADAPTIVE_DIR = ROOT / "reports/commentary_adaptive_luna_pilot_5_20260818_run4"
FACT_FIELDS = ("primary_event", "action", "result", "actor_role", "target_role", "review_confidence", "evidence_note")
CANDIDATE_FIELDS = ("overall_support", "core_event", "action", "result", "actor_role", "target_role", "unsupported_claims", "confidence", "note")
REVIEW_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


class ReviewApplication:
    def __init__(self, packet_dir: Path) -> None:
        self.packet_dir = packet_dir.resolve()
        self.items_path = self.packet_dir / "items.json"
        self.reviews_path = self.packet_dir / "reviews.json"
        self.index_path = self.packet_dir / "index.html"
        self.teacher_drafts_path = TEACHER_DIR / "teacher_drafts.json"
        self.teacher_page_path = TEACHER_DIR / "teacher_review.html"
        self.teacher_reviews_path = TEACHER_DIR / "teacher_reviews.json"
        self.adaptive_results_path = ADAPTIVE_DIR / "adaptive_results.json"
        self.adaptive_page_path = ADAPTIVE_DIR / "adaptive_review.html"
        self.adaptive_reviews_path = ADAPTIVE_DIR / "adaptive_reviews.json"
        self.packet = json.loads(self.items_path.read_text(encoding="utf-8"))
        self.items = {item["annotation_id"]: item for item in self.packet["items"]}
        if len(self.items) != self.packet["sample_count"]:
            raise RuntimeError("annotation IDs are not unique")
        if not self.index_path.is_file():
            raise FileNotFoundError(self.index_path)
        if not self.reviews_path.exists():
            atomic_write_json(
                self.reviews_path,
                {
                    "schema_version": 1,
                    "packet": self.packet_dir.name,
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                    "reviews": {},
                },
            )
        self.teacher_drafts = json.loads(self.teacher_drafts_path.read_text(encoding="utf-8"))
        self.teacher_items = {item["annotation_id"]: item for item in self.teacher_drafts["items"]}
        if not self.teacher_page_path.is_file():
            raise FileNotFoundError(self.teacher_page_path)
        if not self.teacher_reviews_path.exists():
            atomic_write_json(
                self.teacher_reviews_path,
                {
                    "schema_version": 1,
                    "draft_file": str(self.teacher_drafts_path),
                    "warning": "Human review of Luna 2fps teacher drafts; not gold labels until corrected and approved.",
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                    "reviews": {},
                },
            )
        self.adaptive_results = json.loads(self.adaptive_results_path.read_text(encoding="utf-8"))
        self.adaptive_items = {item["annotation_id"]: item for item in self.adaptive_results["items"]}
        if not self.adaptive_page_path.is_file():
            raise FileNotFoundError(self.adaptive_page_path)
        if not self.adaptive_reviews_path.exists():
            atomic_write_json(
                self.adaptive_reviews_path,
                {
                    "schema_version": 1,
                    "result_file": str(self.adaptive_results_path),
                    "warning": "Human review of adaptive Luna pilot; not gold labels until approved.",
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                    "reviews": {},
                },
            )

    def load_reviews(self) -> dict:
        return json.loads(self.reviews_path.read_text(encoding="utf-8"))

    def load_teacher_reviews(self) -> dict:
        return json.loads(self.teacher_reviews_path.read_text(encoding="utf-8"))

    def load_adaptive_reviews(self) -> dict:
        return json.loads(self.adaptive_reviews_path.read_text(encoding="utf-8"))

    def save_adaptive_review(self, annotation_id: str, incoming: dict) -> dict:
        if annotation_id not in self.adaptive_items:
            raise KeyError(annotation_id)
        if not isinstance(incoming, dict):
            raise ValueError("review must be a JSON object")
        verdict = incoming.get("verdict", "")
        if verdict not in {"", "usable", "needs_revision", "unusable"}:
            raise ValueError("invalid verdict")
        correction = incoming.get("correction", "")
        if not isinstance(correction, str) or len(correction) > 20_000:
            raise ValueError("invalid correction")
        complete = bool(incoming.get("review_complete"))
        if complete and not verdict:
            raise ValueError("提交前请选择可用、需修改或不可用")
        if complete and verdict == "needs_revision" and not correction.strip():
            raise ValueError("选择需修改时，请简要写明修改意见")
        review = {
            "verdict": verdict,
            "correction": correction,
            "review_complete": complete,
            "updated_at": utc_now(),
        }
        with REVIEW_LOCK:
            document = self.load_adaptive_reviews()
            document["reviews"][annotation_id] = review
            document["updated_at"] = utc_now()
            atomic_write_json(self.adaptive_reviews_path, document)
        return review

    def save_teacher_review(self, annotation_id: str, incoming: dict) -> dict:
        if annotation_id not in self.teacher_items:
            raise KeyError(annotation_id)
        if not isinstance(incoming, dict):
            raise ValueError("review must be a JSON object")
        verdict = incoming.get("verdict", "")
        if verdict not in {"", "usable", "needs_revision", "unusable"}:
            raise ValueError("invalid verdict")
        correction = incoming.get("correction", "")
        if not isinstance(correction, str) or len(correction) > 20_000:
            raise ValueError("invalid correction")
        complete = bool(incoming.get("review_complete"))
        if complete and not verdict:
            raise ValueError("提交前请选择可用、需修改或不可用")
        if complete and verdict == "needs_revision" and not correction.strip():
            raise ValueError("选择需修改时，请简要写明修改意见")
        review = {
            "verdict": verdict,
            "correction": correction,
            "review_complete": complete,
            "updated_at": utc_now(),
        }
        with REVIEW_LOCK:
            document = self.load_teacher_reviews()
            document["reviews"][annotation_id] = review
            document["updated_at"] = utc_now()
            atomic_write_json(self.teacher_reviews_path, document)
        return review

    def public_items(self) -> dict:
        return {
            "purpose": self.packet["purpose"],
            "sample_count": self.packet["sample_count"],
            "items": [
                {
                    "annotation_id": x["annotation_id"],
                    "review_position": x["review_position"],
                    "dataset_index": x["dataset_index"],
                }
                for x in self.packet["items"]
            ],
        }

    def reveal(self, annotation_id: str) -> dict:
        item = self.items[annotation_id]
        review = self.load_reviews()["reviews"].get(annotation_id, {})
        response = {
            "reference_commentary": item["reference_commentary"],
            "candidates": [
                {"candidate_id": x["candidate_id"], "source": x["source"], "text": x["text"]}
                for x in item["anonymous_candidates"]
            ],
        }
        if review.get("grounding_complete"):
            response["after_completion"] = {
                "candidate_sources": {
                    x["candidate_id"]: x["source"] for x in item["anonymous_candidates"]
                },
                "selection_stratum": item["selection_stratum"],
                "selection_role": item["selection_role"],
                "reference_derived_event": item["reference_derived_event"],
                "existing_reference_audit": item["existing_reference_audit"],
                "notice": "这些字段只用于抽样与事后核对，不是真值，也没有参与刚才的视频事实标注。",
            }
        return response

    def save_review(self, annotation_id: str, incoming: dict) -> dict:
        if annotation_id not in self.items:
            raise KeyError(annotation_id)
        if not isinstance(incoming, dict):
            raise ValueError("review must be a JSON object")
        with REVIEW_LOCK:
            document = self.load_reviews()
            old = document["reviews"].get(annotation_id, {})
            new = {
                "video_fact": incoming.get("video_fact", {}),
                "video_fact_complete": bool(incoming.get("video_fact_complete")),
                "candidate_reviews": incoming.get("candidate_reviews", {}),
                "grounding_complete": bool(incoming.get("grounding_complete")),
                "post_reveal_note": incoming.get("post_reveal_note", ""),
            }
            if not isinstance(new["video_fact"], dict) or not isinstance(new["candidate_reviews"], dict):
                raise ValueError("invalid review object")

            if old.get("video_fact_locked_at"):
                if new["video_fact"] != old.get("video_fact") or not new["video_fact_complete"]:
                    raise PermissionError("视频事实已经锁定；如需订正，请写在“揭示后备注”中")
                new["video_fact_locked_at"] = old["video_fact_locked_at"]
            elif new["video_fact_complete"]:
                self._validate_fact(new["video_fact"])
                new["video_fact_locked_at"] = utc_now()

            if old.get("grounding_locked_at"):
                if new["candidate_reviews"] != old.get("candidate_reviews") or not new["grounding_complete"]:
                    raise PermissionError("候选文本评价已经锁定")
                new["grounding_locked_at"] = old["grounding_locked_at"]
            elif new["grounding_complete"]:
                self._validate_candidates(new["candidate_reviews"])
                new["grounding_locked_at"] = utc_now()

            new["updated_at"] = utc_now()
            document["reviews"][annotation_id] = new
            document["updated_at"] = utc_now()
            atomic_write_json(self.reviews_path, document)
            return new

    @staticmethod
    def _validate_fact(fact: dict) -> None:
        required = ("primary_event", "action", "result", "actor_role", "target_role", "review_confidence")
        missing = [key for key in required if not fact.get(key)]
        if missing:
            raise ValueError("视频事实尚未填完：" + ", ".join(missing))
        for slot in ("action", "result", "actor_role", "target_role"):
            value = fact.get(slot)
            if not isinstance(value, dict) or value.get("observability") not in {"clear", "partial", "not_observable"}:
                raise ValueError(f"{slot} 缺少可见程度")
            if value["observability"] != "not_observable" and not str(value.get("value", "")).strip():
                raise ValueError(f"{slot} 可见时必须写明内容")

    @staticmethod
    def _validate_candidates(candidates: dict) -> None:
        for candidate_id in ("candidate_a", "candidate_b"):
            row = candidates.get(candidate_id)
            if not isinstance(row, dict):
                raise ValueError(f"缺少 {candidate_id} 的评价")
            if not row.get("overall_support"):
                raise ValueError(f"缺少 {candidate_id} 的总体视频支持度")

    def export_csv(self) -> bytes:
        reviews = self.load_reviews()["reviews"]
        output = io.StringIO()
        columns = [
            "annotation_id", "dataset_index", "review_position", "video_fact_complete", "grounding_complete",
            "primary_event", "action_observability", "action", "result_observability", "result",
            "actor_role_observability", "actor_role", "target_role_observability", "target_role",
            "video_confidence", "evidence_note", "candidate_id", "candidate_source", "candidate_text",
            "overall_support", "core_event", "candidate_action", "candidate_result", "candidate_actor_role",
            "candidate_target_role", "unsupported_claims", "candidate_confidence", "candidate_note",
            "post_reveal_note",
        ]
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for item in self.packet["items"]:
            review = reviews.get(item["annotation_id"], {})
            fact = review.get("video_fact", {})
            base = {
                "annotation_id": item["annotation_id"],
                "dataset_index": item["dataset_index"],
                "review_position": item["review_position"],
                "video_fact_complete": bool(review.get("video_fact_complete")),
                "grounding_complete": bool(review.get("grounding_complete")),
                "primary_event": fact.get("primary_event", ""),
                "video_confidence": fact.get("review_confidence", ""),
                "evidence_note": fact.get("evidence_note", ""),
                "post_reveal_note": review.get("post_reveal_note", ""),
            }
            for slot in ("action", "result", "actor_role", "target_role"):
                value = fact.get(slot, {})
                base[f"{slot}_observability"] = value.get("observability", "")
                base[slot] = value.get("value", "")
            by_id = {x["candidate_id"]: x for x in item["anonymous_candidates"]}
            candidate_reviews = review.get("candidate_reviews", {})
            for candidate_id in ("candidate_a", "candidate_b"):
                candidate = by_id[candidate_id]
                judgment = candidate_reviews.get(candidate_id, {})
                row = dict(base)
                row.update(
                    {
                        "candidate_id": candidate_id,
                        "candidate_source": candidate["source"],
                        "candidate_text": candidate["text"],
                        "overall_support": judgment.get("overall_support", ""),
                        "core_event": judgment.get("core_event", ""),
                        "candidate_action": judgment.get("action", ""),
                        "candidate_result": judgment.get("result", ""),
                        "candidate_actor_role": judgment.get("actor_role", ""),
                        "candidate_target_role": judgment.get("target_role", ""),
                        "unsupported_claims": judgment.get("unsupported_claims", ""),
                        "candidate_confidence": judgment.get("confidence", ""),
                        "candidate_note": judgment.get("note", ""),
                    }
                )
                writer.writerow(row)
        return output.getvalue().encode("utf-8-sig")


def make_handler(app: ReviewApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SoccerMasterReview/1.0"

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

        def send_bytes(self, body: bytes, content_type: str, status: int = 200, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if extra:
                for key, value in extra.items():
                    self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def send_json(self, value: object, status: int = 200) -> None:
            self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_HEAD(self) -> None:
            self.do_GET()

        def do_GET(self) -> None:
            path = unquote(urlparse(self.path).path)
            try:
                if path == "/":
                    self.send_bytes(app.index_path.read_bytes(), "text/html; charset=utf-8")
                elif path == "/teacher":
                    self.send_bytes(app.teacher_page_path.read_bytes(), "text/html; charset=utf-8")
                elif path == "/adaptive":
                    self.send_bytes(app.adaptive_page_path.read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/items":
                    self.send_json(app.public_items())
                elif path == "/api/reviews":
                    self.send_json(app.load_reviews())
                elif path == "/api/teacher-drafts":
                    self.send_json(app.teacher_drafts)
                elif path == "/api/teacher-reviews":
                    self.send_json(app.load_teacher_reviews())
                elif path == "/api/adaptive-results":
                    self.send_json(app.adaptive_results)
                elif path == "/api/adaptive-reviews":
                    self.send_json(app.load_adaptive_reviews())
                elif path == "/api/adaptive-export.json":
                    self.send_bytes(app.adaptive_reviews_path.read_bytes(), "application/json; charset=utf-8", extra={"Content-Disposition": "attachment; filename=adaptive_luna_reviews.json"})
                elif path == "/api/teacher-export.json":
                    self.send_bytes(app.teacher_reviews_path.read_bytes(), "application/json; charset=utf-8", extra={"Content-Disposition": "attachment; filename=teacher_draft_reviews.json"})
                elif path.startswith("/api/item/") and path.endswith("/reveal"):
                    annotation_id = path[len("/api/item/") : -len("/reveal")].strip("/")
                    self.send_json(app.reveal(annotation_id))
                elif path == "/api/export.json":
                    self.send_bytes(app.reviews_path.read_bytes(), "application/json; charset=utf-8", extra={"Content-Disposition": "attachment; filename=commentary_fact_reviews.json"})
                elif path == "/api/export.csv":
                    self.send_bytes(app.export_csv(), "text/csv; charset=utf-8", extra={"Content-Disposition": "attachment; filename=commentary_fact_reviews.csv"})
                elif path.startswith("/video/"):
                    self.serve_video(path[len("/video/") :])
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except KeyError:
                self.send_json({"error": "unknown annotation ID"}, 404)
            except PermissionError as exc:
                self.send_json({"error": str(exc)}, 403)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)

        def do_POST(self) -> None:
            path = unquote(urlparse(self.path).path)
            teacher_match = re.fullmatch(r"/api/teacher-reviews/([A-Za-z0-9_-]+)", path)
            adaptive_match = re.fullmatch(r"/api/adaptive-reviews/([A-Za-z0-9_-]+)", path)
            match = re.fullmatch(r"/api/reviews/([A-Za-z0-9_-]+)", path)
            if not match and not teacher_match and not adaptive_match:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid request size")
                value = json.loads(self.rfile.read(length))
                if teacher_match:
                    self.send_json(app.save_teacher_review(teacher_match.group(1), value))
                elif adaptive_match:
                    self.send_json(app.save_adaptive_review(adaptive_match.group(1), value))
                else:
                    self.send_json(app.save_review(match.group(1), value))
            except KeyError:
                self.send_json({"error": "unknown annotation ID"}, 404)
            except PermissionError as exc:
                self.send_json({"error": str(exc)}, 403)
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, 400)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)

        def serve_video(self, annotation_id: str) -> None:
            item = app.items.get(annotation_id)
            if item is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            video = Path(item["video_path"])
            if not video.is_file():
                self.send_json({"error": f"视频当前不可访问：{video}"}, 404)
                return
            size = video.stat().st_size
            start, end = 0, size - 1
            status = HTTPStatus.OK
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
                if not match:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
                if match.group(1):
                    start = int(match.group(1))
                    end = int(match.group(2)) if match.group(2) else end
                elif match.group(2):
                    suffix = int(match.group(2))
                    start = max(0, size - suffix)
                if start >= size or end < start:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
                end = min(end, size - 1)
                status = HTTPStatus.PARTIAL_CONTENT
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", mimetypes.guess_type(video.name)[0] or "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            with video.open("rb") as source:
                source.seek(start)
                remaining = length
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("For safety this review server only binds to loopback")
    app = ReviewApplication(args.packet_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Review UI: http://{args.host}:{args.port}", flush=True)
    print(f"Packet: {app.packet_dir}", flush=True)
    print(f"Reviews autosave to: {app.reviews_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
