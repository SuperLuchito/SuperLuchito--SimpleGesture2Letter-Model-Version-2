from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover - optional import at module load
    cv2 = None
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from .config import AppConfig, load_config
from .embedding import DinoEmbedder
from .hand_detector import HandDetection, HandDetector
from .logging_utils import UncertainEventLogger
from .retrieval import GalleryIndex, RetrievalHit
from .schemas import TopKItem, VLMDecision, build_inference_message
from .state_machine import HoldToCommitStateMachine
from .vlm_judge import JudgeResult, VLMJudge

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
GALLERY_DIR = BACKEND_DIR / "gallery"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
CONFIG_PATH = BACKEND_DIR / "config.yaml"
INDEX_PATH = ARTIFACTS_DIR / "faiss.index"
META_PATH = ARTIFACTS_DIR / "meta.json"
UNCERTAIN_DIR = ARTIFACTS_DIR / "uncertain_events"
VLM_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-judge")


class RuntimeContext:
    def __init__(self) -> None:
        self.lock = Lock()
        self.config = load_config(CONFIG_PATH)

        self._hand_detector: HandDetector | None = None
        self._embedder: DinoEmbedder | None = None
        self._gallery_index: GalleryIndex | None = None
        self._vlm_judge: VLMJudge | None = None
        self._event_logger: UncertainEventLogger | None = None

        self.errors: dict[str, str] = {}

    def reload_config(self) -> AppConfig:
        self.config = load_config(CONFIG_PATH)
        return self.config

    def get_hand_detector(self) -> HandDetector | None:
        with self.lock:
            if self._hand_detector is not None:
                return self._hand_detector
            try:
                cfg = self.config
                model_path = Path(cfg.hand_landmarker_model_path)
                if not model_path.is_absolute():
                    model_path = (ROOT_DIR / model_path).resolve()
                self._hand_detector = HandDetector(
                    str(model_path),
                    min_bbox_area=cfg.min_bbox_area,
                    bbox_padding=cfg.hand_bbox_padding,
                    focus_ratio=cfg.hand_focus_ratio,
                    bg_suppression=cfg.hand_bg_suppression,
                    bg_darken_factor=cfg.hand_bg_darken_factor,
                    mask_dilate_ratio=cfg.hand_mask_dilate_ratio,
                    mask_blur_sigma=cfg.hand_mask_blur_sigma,
                    max_num_hands=cfg.max_num_hands,
                )
                self.errors.pop("hand_detector", None)
            except Exception as exc:
                self.errors["hand_detector"] = str(exc)
                return None
            return self._hand_detector

    def get_embedder(self) -> DinoEmbedder | None:
        with self.lock:
            if self._embedder is not None:
                return self._embedder
            try:
                cfg = self.config
                self._embedder = DinoEmbedder(model_name=cfg.embedding_model, device=cfg.device)
                self.errors.pop("embedder", None)
            except Exception as exc:
                self.errors["embedder"] = str(exc)
                return None
            return self._embedder

    def get_gallery_index(self) -> GalleryIndex | None:
        with self.lock:
            if self._gallery_index is not None:
                return self._gallery_index
            try:
                self._gallery_index = GalleryIndex.load(INDEX_PATH, META_PATH)
                self.errors.pop("gallery_index", None)
            except Exception as exc:
                self.errors["gallery_index"] = str(exc)
                return None
            return self._gallery_index

    def get_vlm_judge(self) -> VLMJudge | None:
        if not self.config.enable_vlm_judge:
            return None
        with self.lock:
            if self._vlm_judge is not None:
                return self._vlm_judge
            try:
                cfg = self.config
                self._vlm_judge = VLMJudge(
                    base_url=cfg.vlm_base_url,
                    model=cfg.vlm_model,
                    timeout_ms=cfg.vlm_timeout_ms,
                )
                self.errors.pop("vlm", None)
            except Exception as exc:
                self.errors["vlm"] = str(exc)
                return None
            return self._vlm_judge

    def get_event_logger(self) -> UncertainEventLogger:
        with self.lock:
            if self._event_logger is None:
                self._event_logger = UncertainEventLogger(UNCERTAIN_DIR)
            return self._event_logger

    def allowed_labels(self) -> list[str]:
        index = self.get_gallery_index()
        if index and index.metadata:
            return sorted({str(item.get("letter", "")) for item in index.metadata if item.get("letter")})

        labels = []
        for p in GALLERY_DIR.iterdir() if GALLERY_DIR.exists() else []:
            if not p.is_dir():
                continue
            if p.name == self.config.none_label_dir:
                continue
            if self.config.letters_allowlist and p.name not in self.config.letters_allowlist:
                continue
            labels.append(p.name)
        return sorted(labels)

    def health(self) -> dict[str, Any]:
        cfg = self.config
        hand_ready = self.get_hand_detector() is not None
        embed_ready = self.get_embedder() is not None
        idx = self.get_gallery_index()
        index_loaded = idx is not None and idx.size > 0

        vlm_reachable = False
        vlm_message = "disabled"
        judge = self.get_vlm_judge()
        if judge is not None:
            vlm_reachable, vlm_message = judge.health()

        return {
            "ok": hand_ready and embed_ready and index_loaded,
            "config": cfg.to_dict(),
            "hand_detector_ready": hand_ready,
            "embedding_ready": embed_ready,
            "index_loaded": index_loaded,
            "index_size": idx.size if idx else 0,
            "vlm_enabled": cfg.enable_vlm_judge,
            "vlm_reachable": vlm_reachable,
            "vlm_message": vlm_message,
            "allowed_labels": self.allowed_labels(),
            "errors": self.errors,
        }


class SessionProcessor:
    def __init__(self, runtime: RuntimeContext) -> None:
        self.runtime = runtime
        cfg = runtime.config
        self.state = HoldToCommitStateMachine(
            hold_ms=cfg.hold_ms,
            cooldown_ms=cfg.cooldown_ms,
            precommit_ratio=cfg.precommit_ratio,
            uncertain_streak_frames=cfg.uncertain_streak_frames,
            switch_min_frames=getattr(cfg, "switch_min_frames", 3),
        )

        self.cached_vlm_candidate_key: str | None = None
        self.cached_vlm_result: JudgeResult | None = None
        self.last_vlm_call_ms: int = 0
        self.last_vlm_decision: VLMDecision = VLMDecision()
        self.last_vlm_decision_ms: int = 0
        self.pending_vlm_future: Future[tuple[VLMDecision, bytes | None]] | None = None
        self.pending_vlm_candidate_key: str | None = None
        self.pending_vlm_trigger: str = ""
        self.pending_vlm_context: dict[str, Any] | None = None
        self.pending_vlm_query_crop_bgr: np.ndarray | None = None
        self.pending_vlm_topk_items: list[TopKItem] = []

    def _none_message(self, *, detection: HandDetection, now_ms: int, topk_items: list[TopKItem] | None = None):
        return build_inference_message(
            status="NONE",
            letter="NONE",
            score=0.0,
            confidence=0.0,
            hand_present=detection.hand_present,
            bbox_norm=list(detection.bbox_norm),
            hold_elapsed_ms=0,
            hold_target_ms=self.runtime.config.hold_ms,
            text_value=self.state.text_value,
            committed_now=False,
            topk=topk_items or [],
            vlm=VLMDecision(),
            sim1=0.0,
            sim2=0.0,
            margin=0.0,
            uncertain=False,
            cooldown_left_ms=self.state.cooldown_left_ms(now_ms),
        )

    def _run_vlm(
        self,
        *,
        query_rgb: np.ndarray,
        hits: list[RetrievalHit],
        trigger: str,
        sim1: float,
        sim2: float,
        margin: float,
    ) -> tuple[VLMDecision, bytes | None]:
        judge = self.runtime.get_vlm_judge()
        if judge is None:
            return VLMDecision(used=False, reason="VLM judge disabled or unavailable"), None

        try:
            result, collage_jpeg = judge.judge(
                query_rgb=query_rgb,
                topk_hits=hits,
                allowed_labels=set(self.runtime.allowed_labels()),
                trigger=trigger,
            )
            vlm = VLMDecision(
                used=True,
                letter=result.letter,
                confidence=result.confidence,
                reason=result.reason,
                trigger=trigger,
            )
            return vlm, collage_jpeg
        except Exception as exc:
            return VLMDecision(used=False, reason=f"VLM error: {exc}", trigger=trigger), None

    def _clear_pending_vlm(self) -> None:
        self.pending_vlm_future = None
        self.pending_vlm_candidate_key = None
        self.pending_vlm_trigger = ""
        self.pending_vlm_context = None
        self.pending_vlm_query_crop_bgr = None
        self.pending_vlm_topk_items = []

    def _collect_pending_vlm(self, now_ms: int, current_candidate_key: str | None) -> VLMDecision | None:
        future = self.pending_vlm_future
        if future is None or not future.done():
            return None

        if self.pending_vlm_candidate_key != current_candidate_key:
            self._clear_pending_vlm()
            return None

        try:
            decision, collage_jpeg = future.result()
        except Exception as exc:
            decision = VLMDecision(used=False, reason=f"VLM future error: {exc}", trigger=self.pending_vlm_trigger)
            collage_jpeg = None

        self.cached_vlm_result = JudgeResult(
            letter=decision.letter,
            confidence=decision.confidence,
            reason=decision.reason,
            raw="",
            used=decision.used,
            trigger=decision.trigger,
        )
        self.last_vlm_decision = decision
        self.last_vlm_decision_ms = now_ms
        self.state.mark_vlm_called()

        cfg = self.runtime.config
        if cfg.log_uncertain_events and self.pending_vlm_context and self.pending_vlm_query_crop_bgr is not None:
            logger = self.runtime.get_event_logger()
            try:
                logger.log_event(
                    query_crop_bgr=self.pending_vlm_query_crop_bgr,
                    collage_jpeg_bytes=collage_jpeg,
                    payload={
                        "timestamp_ms": now_ms,
                        **self.pending_vlm_context,
                        "post_vlm": decision.to_dict(),
                        "topk": [item.to_dict() for item in self.pending_vlm_topk_items],
                    },
                )
            except Exception:
                pass

        self._clear_pending_vlm()
        return decision

    def _schedule_vlm(
        self,
        *,
        now_ms: int,
        query_rgb: np.ndarray,
        query_crop_bgr: np.ndarray,
        hits: list[RetrievalHit],
        topk_items: list[TopKItem],
        trigger: str,
        candidate_key: str | None,
        pre_vlm: dict[str, Any],
    ) -> bool:
        if self.pending_vlm_future is not None:
            return False
        if candidate_key is None:
            return False

        self.pending_vlm_future = VLM_EXECUTOR.submit(
            self._run_vlm,
            query_rgb=query_rgb.copy(),
            hits=hits,
            trigger=trigger,
            sim1=float(pre_vlm.get("sim1", 0.0)),
            sim2=float(pre_vlm.get("sim2", 0.0)),
            margin=float(pre_vlm.get("margin", 0.0)),
        )
        self.pending_vlm_candidate_key = candidate_key
        self.pending_vlm_trigger = trigger
        self.pending_vlm_context = {
            "trigger": trigger,
            "pre_vlm": pre_vlm,
        }
        self.pending_vlm_query_crop_bgr = query_crop_bgr.copy()
        self.pending_vlm_topk_items = [TopKItem(letter=i.letter, score=i.score, exemplar_path=i.exemplar_path) for i in topk_items]
        self.last_vlm_call_ms = now_ms
        return True

    def process_frame(self, frame_bgr: np.ndarray, now_ms: int) -> dict[str, Any]:
        cfg = self.runtime.config

        if cv2 is None:
            detection = HandDetection(False, (0.0, 0.0, 0.0, 0.0), (0, 0, 0, 0), None)
            return self._none_message(detection=detection, now_ms=now_ms)

        hand_detector = self.runtime.get_hand_detector()
        if hand_detector is None:
            detection = HandDetection(False, (0.0, 0.0, 0.0, 0.0), (0, 0, 0, 0), None)
            return self._none_message(detection=detection, now_ms=now_ms)

        detection = hand_detector.detect(frame_bgr, now_ms)

        if self.state.in_cooldown(now_ms):
            cooldown_letter = "NONE"
            cooldown_score = 0.0
            cooldown_confidence = 0.0
            cooldown_topk: list[TopKItem] = []
            sim1 = 0.0
            sim2 = 0.0
            margin = 0.0
            uncertain = False
            hand_present = bool(detection.hand_present)
            bbox_norm = list(detection.bbox_norm) if hand_present else [0.0, 0.0, 0.0, 0.0]

            if detection.hand_present and detection.crop_bgr is not None:
                embedder = self.runtime.get_embedder()
                gallery_index = self.runtime.get_gallery_index()
                if embedder is not None and gallery_index is not None and gallery_index.size > 0:
                    query_rgb = cv2.cvtColor(detection.crop_bgr, cv2.COLOR_BGR2RGB)
                    query_vec = embedder.embed_rgb(query_rgb)
                    hits = gallery_index.search(query_vec, k=cfg.retrieval_k)
                    cooldown_topk = [
                        TopKItem(letter=hit.letter, score=hit.score, exemplar_path=hit.exemplar_path)
                        for hit in hits
                    ]
                    if hits:
                        sim1 = float(hits[0].score)
                        sim2 = float(hits[1].score) if len(hits) > 1 else -1.0
                        margin = float(sim1 - sim2)
                        uncertain = (sim1 < cfg.sim_vlm_th) or (margin < cfg.margin_th)
                        cooldown_letter = hits[0].letter
                        cooldown_score = sim1
                        cooldown_confidence = sim1

            return build_inference_message(
                status="COOLDOWN",
                letter=cooldown_letter,
                score=cooldown_score,
                confidence=cooldown_confidence,
                hand_present=hand_present,
                bbox_norm=bbox_norm,
                hold_elapsed_ms=0,
                hold_target_ms=cfg.hold_ms,
                text_value=self.state.text_value,
                committed_now=False,
                topk=cooldown_topk,
                vlm=VLMDecision(),
                sim1=sim1,
                sim2=sim2,
                margin=margin,
                uncertain=uncertain,
                cooldown_left_ms=self.state.cooldown_left_ms(now_ms),
            )

        if not detection.hand_present or detection.crop_bgr is None:
            self.state.clear_candidate()
            self.cached_vlm_candidate_key = None
            self.cached_vlm_result = None
            return self._none_message(detection=detection, now_ms=now_ms)

        embedder = self.runtime.get_embedder()
        gallery_index = self.runtime.get_gallery_index()
        if embedder is None or gallery_index is None or gallery_index.size == 0:
            self.state.clear_candidate()
            self.cached_vlm_candidate_key = None
            self.cached_vlm_result = None
            return self._none_message(detection=detection, now_ms=now_ms)

        query_rgb = cv2.cvtColor(detection.crop_bgr, cv2.COLOR_BGR2RGB)
        query_vec = embedder.embed_rgb(query_rgb)
        hits = gallery_index.search(query_vec, k=cfg.retrieval_k)

        topk_items = [TopKItem(letter=hit.letter, score=hit.score, exemplar_path=hit.exemplar_path) for hit in hits]
        if not hits:
            self.state.clear_candidate()
            self.cached_vlm_candidate_key = None
            self.cached_vlm_result = None
            return self._none_message(detection=detection, now_ms=now_ms, topk_items=topk_items)

        sim1 = hits[0].score
        sim2 = hits[1].score if len(hits) > 1 else -1.0
        margin = sim1 - sim2

        if sim1 < cfg.sim_none:
            self.state.clear_candidate()
            self.cached_vlm_candidate_key = None
            self.cached_vlm_result = None
            return build_inference_message(
                status="NONE",
                letter="NONE",
                score=float(sim1),
                confidence=0.0,
                hand_present=False,
                bbox_norm=[0.0, 0.0, 0.0, 0.0],
                hold_elapsed_ms=0,
                hold_target_ms=cfg.hold_ms,
                text_value=self.state.text_value,
                committed_now=False,
                topk=topk_items,
                vlm=VLMDecision(),
                sim1=float(sim1),
                sim2=float(sim2),
                margin=float(margin),
                uncertain=False,
                cooldown_left_ms=self.state.cooldown_left_ms(now_ms),
            )

        hold = self.state.update_candidate(hits[0].letter, now_ms)
        candidate_letter = self.state.candidate_letter or hits[0].letter
        candidate_key = self.state.candidate_key
        if candidate_key != self.cached_vlm_candidate_key:
            self.cached_vlm_candidate_key = candidate_key
            self.cached_vlm_result = None

        uncertain = (sim1 < cfg.sim_vlm_th) or (margin < cfg.margin_th)
        self.state.update_uncertain(uncertain)

        if now_ms - self.last_vlm_decision_ms <= 4000:
            vlm_decision = self.last_vlm_decision
        else:
            vlm_decision = VLMDecision()

        vlm_interval_ms = int(getattr(cfg, "vlm_min_interval_ms", 1800))
        vlm_allowed_now = (now_ms - self.last_vlm_call_ms) >= vlm_interval_ms

        ready_vlm = self._collect_pending_vlm(now_ms, candidate_key)
        if ready_vlm is not None:
            vlm_decision = ready_vlm
        elif self.pending_vlm_future is not None:
            vlm_decision = VLMDecision(
                used=False,
                letter="NONE",
                confidence=0.0,
                reason="pending",
                trigger=self.pending_vlm_trigger,
            )

        should_call_vlm, trigger = self.state.should_call_vlm(
            hold_elapsed_ms=hold.hold_elapsed_ms,
            is_uncertain=uncertain,
        )

        if (
            cfg.enable_vlm_judge
            and should_call_vlm
            and vlm_allowed_now
            and self.pending_vlm_future is None
            and detection.crop_bgr is not None
        ):
            started = self._schedule_vlm(
                now_ms=now_ms,
                query_rgb=query_rgb,
                query_crop_bgr=detection.crop_bgr,
                hits=hits,
                topk_items=topk_items,
                trigger=trigger,
                candidate_key=candidate_key,
                pre_vlm={
                    "candidate": candidate_letter,
                    "sim1": float(sim1),
                    "sim2": float(sim2),
                    "margin": float(margin),
                    "uncertain": bool(uncertain),
                    "hold_elapsed_ms": int(hold.hold_elapsed_ms),
                },
            )
            if started:
                vlm_decision = VLMDecision(
                    used=False,
                    letter="NONE",
                    confidence=0.0,
                    reason="pending",
                    trigger=trigger,
                )

        commit_now = False
        final_letter = candidate_letter
        final_conf = sim1

        if hold.hold_elapsed_ms >= cfg.hold_ms:
            if uncertain and cfg.enable_vlm_judge and self.runtime.get_vlm_judge() is not None:
                verdict = self.cached_vlm_result
                if verdict is None:
                    if vlm_allowed_now and self.pending_vlm_future is None and detection.crop_bgr is not None:
                        self._schedule_vlm(
                            now_ms=now_ms,
                            query_rgb=query_rgb,
                            query_crop_bgr=detection.crop_bgr,
                            hits=hits,
                            topk_items=topk_items,
                            trigger="commit_gate",
                            candidate_key=candidate_key,
                            pre_vlm={
                                "candidate": candidate_letter,
                                "sim1": float(sim1),
                                "sim2": float(sim2),
                                "margin": float(margin),
                                "uncertain": bool(uncertain),
                                "hold_elapsed_ms": int(hold.hold_elapsed_ms),
                            },
                        )
                        vlm_decision = VLMDecision(
                            used=False,
                            letter="NONE",
                            confidence=0.0,
                            reason="pending",
                            trigger="commit_gate",
                        )

                if verdict and verdict.letter != "NONE" and verdict.confidence >= cfg.vlm_min_confidence:
                    final_letter = verdict.letter
                    final_conf = verdict.confidence
                    commit_now = True
                else:
                    commit_now = False
            else:
                commit_now = True

        if commit_now:
            self.state.commit(final_letter, now_ms)
            status = "COMMITTED"
        else:
            status = "CANDIDATE"

        return build_inference_message(
            status=status,
            letter=final_letter if status == "COMMITTED" else candidate_letter,
            score=float(sim1),
            confidence=float(final_conf if status == "COMMITTED" else sim1),
            hand_present=True,
            bbox_norm=list(detection.bbox_norm),
            hold_elapsed_ms=hold.hold_elapsed_ms,
            hold_target_ms=cfg.hold_ms,
            text_value=self.state.text_value,
            committed_now=commit_now,
            topk=topk_items,
            vlm=vlm_decision,
            sim1=float(sim1),
            sim2=float(sim2),
            margin=float(margin),
            uncertain=bool(uncertain),
            cooldown_left_ms=self.state.cooldown_left_ms(now_ms),
        )


runtime = RuntimeContext()

app = FastAPI(title="RSL Static Dactyl MVP")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/gallery_files", StaticFiles(directory=str(GALLERY_DIR)), name="gallery_files")


@app.on_event("startup")
async def startup_event() -> None:
    runtime.reload_config()


@app.get("/")
def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(runtime.health())


@app.get("/api/gallery")
def api_gallery() -> JSONResponse:
    payload: dict[str, list[str]] = {}
    if not GALLERY_DIR.exists():
        return JSONResponse(payload)

    for label_dir in sorted(p for p in GALLERY_DIR.iterdir() if p.is_dir()):
        urls = []
        for img in sorted(label_dir.rglob("*")):
            if not img.is_file():
                continue
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            rel_path = img.relative_to(GALLERY_DIR).as_posix()
            urls.append(f"/gallery_files/{rel_path}")
        payload[label_dir.name] = urls
    return JSONResponse(payload)


@app.get("/gallery")
def gallery_inspector() -> HTMLResponse:
    html = """
<!doctype html>
<html lang=\"ru\">
<head>
  <meta charset=\"utf-8\" />
  <title>Gallery Inspector</title>
  <style>
    body { font-family: sans-serif; padding: 16px; background: #101318; color: #f3f5f7; }
    .topbar { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
    .back-btn {
      display: inline-flex;
      align-items: center;
      text-decoration: none;
      color: #eaf2f8;
      background: #27303d;
      border: 1px solid #41506a;
      padding: 8px 12px;
      border-radius: 8px;
      font-weight: 600;
    }
    .back-btn:hover { background: #2f3b4b; }
    .grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }
    .tile { background: #1c2129; border-radius: 10px; padding: 8px; }
    img { width: 100%; border-radius: 6px; display: block; }
    h1 { margin: 0; }
    h2 { margin-top: 24px; }
  </style>
</head>
<body>
  <div class=\"topbar\">
    <a class=\"back-btn\" href=\"/\" id=\"backBtn\">← Назад</a>
    <h1>Эталоны галереи</h1>
  </div>
  <div id=\"content\"></div>
  <script>
    document.getElementById('backBtn').addEventListener('click', (event) => {
      event.preventDefault();
      if (window.history.length > 1) {
        window.history.back();
      } else {
        window.location.href = '/';
      }
    });

    async function run() {
      const res = await fetch('/api/gallery');
      const data = await res.json();
      const root = document.getElementById('content');
      for (const [label, images] of Object.entries(data)) {
        const h = document.createElement('h2');
        h.textContent = `${label} (${images.length})`;
        root.appendChild(h);
        const grid = document.createElement('div');
        grid.className = 'grid';
        for (const url of images) {
          const tile = document.createElement('div');
          tile.className = 'tile';
          const img = document.createElement('img');
          img.src = url;
          tile.appendChild(img);
          grid.appendChild(tile);
        }
        root.appendChild(grid);
      }
    }
    run();
  </script>
</body>
</html>
"""
    return HTMLResponse(html)


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session = SessionProcessor(runtime)

    while True:
        try:
            packet = await websocket.receive()
        except WebSocketDisconnect:
            break
        except Exception:
            break

        if packet["type"] == "websocket.disconnect":
            break

        if packet.get("text") is not None:
            try:
                data = json.loads(packet["text"])
            except json.JSONDecodeError:
                continue

            if data.get("type") == "control" and data.get("action") == "clear_text":
                session.state.clear_text()
                await websocket.send_json({"type": "ack", "action": "clear_text"})
            continue

        frame_bytes = packet.get("bytes")
        if not frame_bytes:
            continue

        if cv2 is None:
            await websocket.send_json(
                {
                    "status": "NONE",
                    "letter": "NONE",
                    "error": "opencv-python is not installed",
                }
            )
            continue

        np_buf = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue

        now_ms = int(time.monotonic() * 1000)
        payload = await asyncio.to_thread(session.process_frame, frame_bgr, now_ms)
        try:
            await websocket.send_json(payload)
        except WebSocketDisconnect:
            break
        except RuntimeError:
            break
        except Exception:
            break
