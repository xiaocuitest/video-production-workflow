#!/usr/bin/env python3
"""Friendly local control panel for the talking-head video workflow."""

from __future__ import annotations

import json
import http.client
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from pipeline import MOTION_TEMPLATE_BY_ID, motion_template_catalog, render_html


PROJECT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT / "web"
INPUT_DIR = PROJECT / "input" / "jobs"
OUTPUT_DIR = PROJECT / "output"
ALLOWED_VIDEO = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
ALLOWED_IMAGE = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
ENV_PATH = PROJECT / ".env.local"
PREVIEW_URL = "http://127.0.0.1:3002/#project/video-production-workflow"
PREVIEW_EMBED_URL = "http://127.0.0.1:3002/api/projects/video-production-workflow/preview"
PROVIDER_KEY_NAMES = {
    "ark": "ARK_API_KEY",
    "pexels": "PEXELS_API_KEY",
    "openai": "OPENAI_API_KEY",
}
PROVIDER_LABELS = {
    "ARK_API_KEY": "火山方舟",
    "PEXELS_API_KEY": "Pexels",
    "OPENAI_API_KEY": "OpenAI",
}

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024 * 1024

state_lock = threading.Lock()
credential_lock = threading.Lock()
preview_process: subprocess.Popen[str] | None = None
active_process: subprocess.Popen[str] | None = None
credential_cache: dict[str, str] | None = None


def latest_final_video() -> str:
    candidates = sorted(OUTPUT_DIR.glob("final-*.mp4"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        return ""
    summary_path = OUTPUT_DIR / "summary.json"
    if summary_path.exists() and candidates[0].stat().st_mtime < summary_path.stat().st_mtime:
        return ""
    return candidates[0].name


def hyperframes_player_bundle() -> Path | None:
    """Find the player shipped with the same pinned HyperFrames CLI version."""
    candidates = []
    npx_root = Path.home() / ".npm" / "_npx"
    for bundle in npx_root.glob("*/node_modules/hyperframes/dist/hyperframes-player.global.js"):
        package_json = bundle.parent.parent / "package.json"
        try:
            package = json.loads(package_json.read_text(encoding="utf-8"))
            if package.get("version") == "0.8.10":
                candidates.append(bundle)
        except (OSError, ValueError):
            continue
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


@app.after_request
def protect_sensitive_responses(response):
    if request.path in {"/api/status", "/api/provider-keys"}:
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


def initial_state() -> dict[str, Any]:
    summary_path = OUTPUT_DIR / "summary.json"
    summary = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {
        "phase": "ready" if summary else "idle",
        "progress": 100 if summary else 0,
        "message": "成片方案已经准备好" if summary else "选择视频后即可开始",
        "detail": "",
        "logs": [],
        "summary": summary,
        "has_previous_result": bool(summary),
        "preview_url": PREVIEW_URL,
        "preview_embed_url": PREVIEW_EMBED_URL,
        "final_video": latest_final_video(),
        "updated_at": time.time(),
    }


job_state = initial_state()


def update_state(**changes: Any) -> None:
    with state_lock:
        job_state.update(changes)
        job_state["updated_at"] = time.time()


def append_log(line: str) -> None:
    clean = redact_secret_text(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)).strip()
    if not clean:
        return
    with state_lock:
        logs = [*job_state.get("logs", []), clean][-28:]
        job_state["logs"] = logs
        job_state["detail"] = clean[-180:]
        job_state["updated_at"] = time.time()


def public_state() -> dict[str, Any]:
    with state_lock:
        payload = dict(job_state)
        payload["logs"] = list(job_state.get("logs", []))
    payload["summary"] = enrich_summary(payload.get("summary") or {})
    workflow = load_agent_workflow()
    if workflow and payload.get("phase") == "processing" and float(payload.get("progress", 0) or 0) >= 84:
        for agent in workflow.get("agents", []):
            if agent.get("id") == "qa":
                agent.update({"status": "running", "detail": "正在执行 HyperFrames 运行、布局、动效和对比度检查", "progress": 80})
        workflow["active_agent"] = "qa"
        workflow["status"] = "running"
    payload["agent_workflow"] = workflow
    payload["artifacts"] = artifact_manifest()
    payload["preview_running"] = preview_healthy()
    provider_env = workflow_env()
    payload["provider_status"] = {
        "ark": bool(provider_env.get("ARK_API_KEY", "").strip()),
        "pexels": bool(provider_env.get("PEXELS_API_KEY", "").strip()),
        "openai": bool(provider_env.get("OPENAI_API_KEY", "").strip()),
    }
    return payload


def artifact_manifest() -> list[dict[str, str]]:
    items = [
        ("cover-9x16.png", "竖版封面", "image"),
        ("cover-3x4.png", "3:4 封面", "image"),
        ("subtitles.srt", "字幕文件", "file"),
        ("transcript.txt", "逐字稿", "file"),
        ("publish-copy.md", "发布文案", "file"),
        ("motion-plan.json", "动效计划", "file"),
        ("motion-template-manifest.json", "Motion 模板方案", "file"),
        ("director-plan.json", "智能导演决策", "file"),
        ("media-sources.json", "素材来源", "file"),
        ("cover-copy.json", "封面文案提炼", "file"),
        ("cover-manifest.json", "封面模板说明", "file"),
        ("cover-qa.json", "封面完整性质检", "file"),
        ("agent-workflow.json", "多 Agent 制作记录", "file"),
        ("workflow-qa.json", "视听一致性质检", "file"),
    ]
    result = []
    for filename, label, kind in items:
        path = OUTPUT_DIR / filename
        if path.exists():
            result.append({
                "name": filename, "label": label, "kind": kind,
                "url": f"/output/{filename}?v={int(path.stat().st_mtime)}",
            })
    music_preview = next(iter(sorted(OUTPUT_DIR.glob("bgm-preview.*"))), None)
    if music_preview:
        result.append({
            "name": music_preview.name, "label": "背景音乐试听", "kind": "audio",
            "url": f"/output/{music_preview.name}?v={int(music_preview.stat().st_mtime)}",
        })
    mix_preview = next(iter(sorted(OUTPUT_DIR.glob("program-mix-preview.*"))), None)
    if mix_preview:
        result.append({
            "name": mix_preview.name, "label": "人声与配乐实际混音", "kind": "audio-mix",
            "url": f"/output/{mix_preview.name}?v={int(mix_preview.stat().st_mtime)}",
        })
    final_video = job_state.get("final_video", "") or latest_final_video()
    if final_video and (OUTPUT_DIR / final_video).exists():
        result.append({"name": final_video, "label": "最终视频", "kind": "video", "url": f"/output/{final_video}"})
    return result


def load_agent_workflow() -> dict[str, Any]:
    path = OUTPUT_DIR / "agent-workflow.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return {}
        # The durable ledger stores low-level run provenance.  Adapt it to the
        # small, friendly status contract consumed by the control panel.
        order = ("content", "director", "media", "motion", "audio", "cover", "assembly", "qa")
        labels = {
            "content": "等待识别与理解完整内容", "director": "等待制定全片导演方案",
            "media": "等待检索或生成配套素材", "motion": "等待设计统一 Motion",
            "audio": "等待设计配乐与声音节奏", "cover": "等待提炼封面方案",
            "assembly": "等待各专项成果", "qa": "等待视听一致性检查",
        }
        latest: dict[str, dict[str, Any]] = {}
        for run in value.get("runs", []):
            if not isinstance(run, dict):
                continue
            actor = str(run.get("actor_id", ""))
            if actor in order:
                latest[actor] = run
        agents = []
        for actor in order:
            run = latest.get(actor, {})
            metadata = run.get("metadata") if isinstance(run.get("metadata"), dict) else {}
            agents.append({
                "id": actor,
                "status": run.get("status", "pending"),
                "detail": metadata.get("detail") or metadata.get("label") or run.get("capability") or labels[actor],
                "progress": 100 if run.get("status") == "succeeded" else 0,
            })
        active = next((item["id"] for item in agents if item["status"] == "running"), "")
        qa_summary = (((value.get("manifest") or {}).get("metadata") or {}).get("qa") or {})
        warnings = int(qa_summary.get("warnings", 0) or 0)
        return {
            **value, "active_agent": active, "agents": agents,
            "message": f"制作完成，另有 {warnings} 条降级或复核提示" if warnings else "各专项成果均已通过装配",
            "detail": "系统没有用占位 Demo 补齐缺失素材；可在视听一致性质检中查看具体原因。" if warnings else "可以打开视频预览进行最终确认。",
        }
    except (OSError, ValueError):
        return {}


def enrich_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep older completed jobs compatible with the new template picker."""
    result = dict(summary)
    composition_path = OUTPUT_DIR / "composition-data.json"
    composition: dict[str, Any] = {}
    if composition_path.exists():
        try:
            composition = json.loads(composition_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    selected = str(result.get("motion_template") or composition.get("motion_template") or "expert")
    if selected not in MOTION_TEMPLATE_BY_ID:
        selected = "expert"
    recommended = str(result.get("recommended_motion_template") or composition.get("recommended_motion_template") or selected)
    if recommended not in MOTION_TEMPLATE_BY_ID:
        recommended = selected
    result["motion_template"] = selected
    result["recommended_motion_template"] = recommended
    result["motion_templates"] = motion_template_catalog(selected, recommended)
    return result


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.18):
            return True
    except OSError:
        return False


def preview_healthy() -> bool:
    """Require a real Studio response, not merely a process holding the port."""
    try:
        connection = http.client.HTTPConnection("127.0.0.1", 3002, timeout=2.5)
        connection.request("GET", "/")
        response = connection.getresponse()
        response.read(64)
        return 200 <= response.status < 500
    except (OSError, http.client.HTTPException):
        return False
    finally:
        try:
            connection.close()
        except (NameError, OSError):
            pass


def _read_local_env() -> dict[str, str]:
    if not ENV_PATH.exists() or ENV_PATH.is_symlink():
        return {}
    try:
        os.chmod(ENV_PATH, 0o600)
        result: dict[str, str] = {}
        for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
        return result
    except OSError:
        return {}


def _write_local_env(values: dict[str, str]) -> None:
    """Atomic 0600 fallback storage; never served by Flask or copied to jobs."""
    if ENV_PATH.is_symlink():
        raise RuntimeError("本机凭据文件状态异常，已拒绝写入。")
    if not values:
        ENV_PATH.unlink(missing_ok=True)
        return
    temp_path = ENV_PATH.with_name(f"{ENV_PATH.name}.tmp")
    if temp_path.is_symlink():
        raise RuntimeError("本机凭据临时文件状态异常，已拒绝写入。")
    descriptor = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("".join(f"{key}={value}\n" for key, value in sorted(values.items())))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, ENV_PATH)
        os.chmod(ENV_PATH, 0o600)
    finally:
        temp_path.unlink(missing_ok=True)


def _load_provider_keys_unlocked() -> dict[str, str]:
    global credential_cache
    if credential_cache is not None:
        return credential_cache
    local_values = _read_local_env()
    credentials = {
        env_name: local_values.get(env_name, "").strip()
        for env_name in PROVIDER_KEY_NAMES.values()
        if local_values.get(env_name, "").strip()
    }
    credential_cache = credentials
    return credential_cache


def provider_credentials() -> dict[str, str]:
    with credential_lock:
        return dict(_load_provider_keys_unlocked())


def validate_provider_updates(payload: dict[str, Any]) -> dict[str, str]:
    updates: dict[str, str] = {}
    for public_name, env_name in PROVIDER_KEY_NAMES.items():
        value = str(payload.get(public_name, "")).strip()
        if not value:
            continue
        if len(value) < 8 or len(value) > 4096 or any(character in value for character in "\r\n\x00"):
            raise ValueError(f"{PROVIDER_LABELS[env_name]} Key 格式不完整，请重新检查。")
        updates[env_name] = value
    return updates


def save_provider_keys(updates: dict[str, str]) -> None:
    global credential_cache
    with credential_lock:
        current = dict(_load_provider_keys_unlocked())
        local_values = _read_local_env()
        local_values.update(updates)
        _write_local_env(local_values)
        current.update(updates)
        credential_cache = current


def clear_provider_keys(env_names: set[str]) -> None:
    global credential_cache
    with credential_lock:
        current = dict(_load_provider_keys_unlocked())
        local_values = _read_local_env()
        for env_name in env_names:
            local_values.pop(env_name, None)
            current.pop(env_name, None)
        _write_local_env(local_values)
        credential_cache = current


def provider_status_payload() -> dict[str, bool]:
    credentials = provider_credentials()
    return {
        public_name: bool(credentials.get(env_name, "").strip())
        for public_name, env_name in PROVIDER_KEY_NAMES.items()
    }


def redact_secret_text(value: str) -> str:
    clean = value
    for secret in provider_credentials().values():
        if len(secret) >= 8:
            clean = clean.replace(secret, "[凭据已隐藏]")
    return clean


def workflow_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(provider_credentials())
    env["PATH"] = f"{PROJECT / 'bin'}:/opt/homebrew/bin:/usr/local/bin:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_stream(command: list[str], on_line=None) -> int:
    global active_process
    process = subprocess.Popen(
        command, cwd=PROJECT, env=workflow_env(), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    active_process = process
    assert process.stdout is not None
    for line in process.stdout:
        append_log(line)
        if on_line:
            on_line(line)
    code = process.wait()
    active_process = None
    return code


def generation_line(line: str) -> None:
    if "1/5" in line:
        update_state(progress=28, message="正在识别你说的内容")
    elif "2/5" in line:
        update_state(progress=48, message="正在判断剪辑节奏")
    elif "3/5" in line:
        update_state(progress=62, message="正在挑选封面画面")
    elif "4/5" in line:
        update_state(progress=76, message="正在生成字幕和图文动效")
    elif "5/5" in line:
        update_state(progress=84, message="正在检查最终画面")


def load_summary() -> dict[str, Any]:
    path = OUTPUT_DIR / "summary.json"
    return enrich_summary(json.loads(path.read_text(encoding="utf-8")) if path.exists() else {})


def run_generation(video: Path, support_dir: Path, config_path: Path, mode: str) -> None:
    try:
        for stale in (OUTPUT_DIR / "agent-workflow.json", OUTPUT_DIR / "workflow-qa.json"):
            stale.unlink(missing_ok=True)
        update_state(phase="processing", progress=18, message="素材已收到，准备开始", logs=[], final_video="")
        command = [
            sys.executable, "-u", str(PROJECT / "scripts" / "pipeline.py"),
            "--input", str(video), "--mode", mode, "--project", str(PROJECT),
            "--config", str(config_path), "--support-dir", str(support_dir),
        ]
        if run_stream(command, generation_line) != 0:
            raise RuntimeError("视频分析没有完成，请展开错误详情查看原因。")
        update_state(progress=88, message="正在做字幕、布局和动效检查")
        if run_stream(["npm", "run", "check"]) != 0:
            raise RuntimeError("画面检查未通过，请展开错误详情查看原因。")
        summary = load_summary()
        update_state(
            phase="ready", progress=100, message="成片方案已经准备好",
            detail="请先查看视频预览和封面，确认后再导出。",
            summary=summary, has_previous_result=True,
        )
    except Exception as exc:
        append_log(str(exc))
        update_state(phase="error", progress=0, message="这次没有生成成功", detail=str(exc))


def drain_preview(process: subprocess.Popen[str]) -> None:
    if not process.stdout:
        return
    for line in process.stdout:
        append_log(f"预览：{line}")


def ensure_preview() -> bool:
    global preview_process
    if preview_healthy():
        return True
    manager = PROJECT / "scripts" / "service_manager.py"
    preview_process = subprocess.Popen(
        [sys.executable, str(manager), "preview"], cwd=PROJECT, env=workflow_env(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    threading.Thread(target=drain_preview, args=(preview_process,), daemon=True).start()
    for _ in range(120):
        if preview_healthy():
            return True
        if preview_process.poll() is not None:
            return False
        time.sleep(0.25)
    return False


def render_final() -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    filename = f"final-{stamp}.mp4"
    try:
        update_state(phase="rendering", progress=92, message="正在导出最终视频", detail="长视频导出需要一些时间，请不要关闭操作台。")
        command = [
            "npm", "run", "render", "--", "--output", f"output/{filename}",
            "--fps", "24", "--quality", "high",
        ]
        if run_stream(command) != 0:
            raise RuntimeError("最终视频导出失败，请展开错误详情查看原因。")
        update_state(
            phase="complete", progress=100, message="最终视频已经导出",
            detail="可以直接播放或下载。", final_video=filename,
        )
    except Exception as exc:
        append_log(str(exc))
        update_state(phase="error", progress=0, message="导出没有完成", detail=str(exc))


@app.get("/")
def home():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/vendor/hyperframes-player.global.js")
def hyperframes_player_script():
    bundle = hyperframes_player_bundle()
    if bundle is None:
        return "HyperFrames player bundle is unavailable", 404
    return send_from_directory(bundle.parent, bundle.name, mimetype="text/javascript")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/status")
def status():
    return jsonify(public_state())


def request_is_local() -> bool:
    return (request.remote_addr or "") in {"127.0.0.1", "::1"}


@app.post("/api/provider-keys")
def store_provider_keys():
    if not request_is_local():
        return jsonify({"error": "本机凭据只能从本机操作台修改。"}), 403
    if not request.is_json:
        return jsonify({"error": "凭据请求格式不正确。"}), 415
    try:
        updates = validate_provider_updates(request.get_json(silent=True) or {})
        if not updates:
            return jsonify({"error": "请至少填写一个需要保存的 Key。"}), 400
        save_provider_keys(updates)
        return jsonify({"ok": True, "provider_status": provider_status_payload()})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 500


@app.delete("/api/provider-keys")
def delete_provider_keys():
    if not request_is_local():
        return jsonify({"error": "本机凭据只能从本机操作台修改。"}), 403
    if not request.is_json:
        return jsonify({"error": "凭据请求格式不正确。"}), 415
    payload = request.get_json(silent=True) or {}
    requested = payload.get("providers") or list(PROVIDER_KEY_NAMES)
    if not isinstance(requested, list):
        return jsonify({"error": "请选择需要清空的 Key。"}), 400
    env_names = {PROVIDER_KEY_NAMES[name] for name in requested if name in PROVIDER_KEY_NAMES}
    if not env_names:
        return jsonify({"error": "没有找到需要清空的 Key。"}), 400
    try:
        clear_provider_keys(env_names)
        return jsonify({"ok": True, "provider_status": provider_status_payload()})
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/motion-templates")
def motion_templates():
    return jsonify({"templates": motion_template_catalog("", "")})


@app.post("/api/start")
def start_job():
    if job_state.get("phase") in {"processing", "rendering"}:
        return jsonify({"error": "当前任务还在处理中，请稍候。"}), 409
    video = request.files.get("video")
    if not video or not video.filename:
        return jsonify({"error": "请先选择一条视频。"}), 400
    video_ext = Path(video.filename).suffix.lower()
    if video_ext not in ALLOWED_VIDEO:
        return jsonify({"error": "目前支持 MP4、MOV、M4V、WebM 和 MKV。"}), 400

    job_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_dir = INPUT_DIR / job_id
    support_dir = job_dir / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / f"video{video_ext}"
    update_state(phase="uploading", progress=8, message="正在接收视频", detail=video.filename)
    video.save(video_path)

    uploaded_support_count = 0
    for index, media in enumerate(request.files.getlist("media")[:12]):
        ext = Path(media.filename or "").suffix.lower()
        if ext in ALLOWED_IMAGE | ALLOWED_VIDEO:
            # Preserve meaningful Chinese words such as “顾客体验” in the saved
            # name; the director uses them to route uploads to the right sentence.
            original_stem = Path(media.filename or "").stem
            safe = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "-", original_stem).strip("-_")[:36]
            safe = safe or secure_filename(original_stem)[:36] or f"media-{index + 1}"
            media.save(support_dir / f"{index:02d}-{safe}{ext}")
            uploaded_support_count += 1

    bgm_mode = request.form.get("bgm_mode", "generated")
    if bgm_mode not in {"generated", "upload", "none"}:
        bgm_mode = "generated"
    bgm_path = ""
    bgm = request.files.get("bgm")
    if bgm_mode == "upload":
        ext = Path(bgm.filename or "").suffix.lower() if bgm else ""
        if not bgm or ext not in ALLOWED_AUDIO:
            return jsonify({"error": "选择“使用自己的音乐”后，请上传 MP3、WAV、M4A、AAC、FLAC 或 OGG。"}), 400
        saved_bgm = job_dir / f"bgm{ext}"
        bgm.save(saved_bgm)
        bgm_path = str(saved_bgm)

    smart_media = request.form.get("smart_media") == "on"
    provider_env = workflow_env()
    has_media_provider = bool(
        provider_env.get("ARK_API_KEY", "").strip()
        or provider_env.get("PEXELS_API_KEY", "").strip()
        or (provider_env.get("OPENAI_API_KEY", "").strip() and request.form.get("allow_ai_images") == "on")
    )
    if smart_media and uploaded_support_count == 0 and not has_media_provider:
        return jsonify({"error": "你已开启配套画面，但还没有上传素材或配置取材接口。最省事的是填写一个火山方舟 API Key；也可以填写 Pexels，或填写 OpenAI Key 并勾选允许 AI 生图。"}), 400

    accent = request.form.get("accent", "#FFD83D")
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", accent):
        accent = "#FFD83D"
    keywords = [item.strip() for item in re.split(r"[,，、\s]+", request.form.get("keywords", "")) if item.strip()]
    config = {
        "mode": request.form.get("mode", "auto"),
        "title": request.form.get("title", "").strip()[:42],
        "subtitle": request.form.get("subtitle", "").strip()[:70],
        "series": request.form.get("series", "观点拆解").strip()[:16] or "观点拆解",
        "speaker": request.form.get("speaker", "").strip()[:20],
        "speaker_title": request.form.get("speaker_title", "").strip()[:28],
        "accent": accent,
        "keywords": keywords[:16],
        "bgm_mode": bgm_mode,
        "bgm_path": bgm_path,
        "bgm_volume": 0.38,
        "director_enabled": request.form.get("director_enabled") == "on",
        "director_provider": "ark",
        "director_model": request.form.get("director_model", "doubao-seed-2-0-lite-260215").strip()[:120],
        "smart_media": smart_media,
        "smart_media_count": 5 if request.form.get("motion_template") == "editorial" else 3,
        "scene_generation": request.form.get("scene_generation", "auto") if request.form.get("scene_generation", "auto") in {"auto", "image", "video", "none"} else "auto",
        "ark_image_model": request.form.get("ark_image_model", "doubao-seedream-5-0-lite-260128").strip()[:120],
        "ark_video_model": request.form.get("ark_video_model", "doubao-seedance-1-5-pro-251215").strip()[:120],
        "allow_ai_images": request.form.get("allow_ai_images") == "on",
        "cover_template": "headline",
        "motion_template": request.form.get("motion_template", "auto") if request.form.get("motion_template", "auto") in ({"auto"} | set(MOTION_TEMPLATE_BY_ID)) else "auto",
        "silence_threshold_seconds": 0.85,
        "cover_time_seconds": None,
        "model": "small",
    }
    mode = config["mode"] if config["mode"] in {"auto", "keep", "recut"} else "auto"
    config_path = job_dir / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    threading.Thread(target=run_generation, args=(video_path, support_dir, config_path, mode), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.post("/api/motion-template")
def select_motion_template():
    if job_state.get("phase") in {"processing", "rendering", "uploading"}:
        return jsonify({"error": "当前任务还在处理中，请稍候。"}), 409
    payload = request.get_json(silent=True, force=True) or {}
    selected = str(payload.get("template", "")).strip()
    if selected not in MOTION_TEMPLATE_BY_ID:
        return jsonify({"error": "没有找到这个 Motion 方案。"}), 400
    composition_path = OUTPUT_DIR / "composition-data.json"
    if not composition_path.exists():
        return jsonify({"error": "请先生成一次成片方案。"}), 409
    try:
        composition = json.loads(composition_path.read_text(encoding="utf-8"))
        summary = load_summary()
        recommended = str(summary.get("recommended_motion_template") or selected)
        if recommended not in MOTION_TEMPLATE_BY_ID:
            recommended = selected
        composition["motion_template"] = selected
        composition["recommended_motion_template"] = recommended
        composition_path.write_text(json.dumps(composition, ensure_ascii=False, indent=2), encoding="utf-8")
        (PROJECT / "index.html").write_text(render_html(composition), encoding="utf-8")

        templates = motion_template_catalog(selected, recommended)
        manifest = {"selected": selected, "recommended": recommended, "templates": templates}
        (OUTPUT_DIR / "motion-template-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        motion_plan_path = OUTPUT_DIR / "motion-plan.json"
        if motion_plan_path.exists():
            motion_plan = json.loads(motion_plan_path.read_text(encoding="utf-8"))
            motion_plan.update({"motion_template": selected, "recommended_motion_template": recommended, "motion_templates": templates})
            motion_plan_path.write_text(json.dumps(motion_plan, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.update({"motion_template": selected, "recommended_motion_template": recommended, "motion_templates": templates})
        (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        update_state(summary=summary)
        return jsonify({"ok": True, "template": selected, "label": MOTION_TEMPLATE_BY_ID[selected]["label"], "preview_url": PREVIEW_URL})
    except (OSError, ValueError, KeyError) as exc:
        return jsonify({"error": f"Motion 方案切换失败：{exc}"}), 500


@app.post("/api/cover")
def select_cover():
    payload = request.get_json(silent=True, force=True) or {}
    selected = str(payload.get("template", "")).strip()
    manifest_path = OUTPUT_DIR / "cover-manifest.json"
    if not manifest_path.exists():
        return jsonify({"error": "请先生成一次封面方案。"}), 409
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    valid = {item.get("id") for item in manifest.get("templates", [])}
    if selected not in valid:
        return jsonify({"error": "没有找到这个封面模板。"}), 400
    qa_path = OUTPUT_DIR / "cover-qa.json"
    if not qa_path.exists():
        return jsonify({"error": "封面还没有通过完整性质检，请重新生成。"}), 409
    qa_report = json.loads(qa_path.read_text(encoding="utf-8"))
    approved_files = {item.get("file") for item in qa_report.get("items", []) if item.get("pass") is True}
    required_files = {f"cover-{selected}-9x16.png", f"cover-{selected}-3x4.png"}
    if qa_report.get("pass") is not True or not required_files.issubset(approved_files):
        return jsonify({"error": "该封面存在文字缺失或越界，系统已阻止选用。"}), 409
    source_vertical = OUTPUT_DIR / f"cover-{selected}-9x16.png"
    source_square = OUTPUT_DIR / f"cover-{selected}-3x4.png"
    if not source_vertical.exists() or not source_square.exists():
        return jsonify({"error": "这个模板的封面还没有生成完成。"}), 409
    shutil.copy2(source_vertical, OUTPUT_DIR / "cover-9x16.png")
    shutil.copy2(source_square, OUTPUT_DIR / "cover-3x4.png")
    manifest["selected"] = selected
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = load_summary()
    summary["cover_template"] = selected
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    update_state(summary=summary)
    return jsonify({"ok": True, "template": selected, "cover_url": f"/output/cover-9x16.png?v={int(time.time())}"})


@app.post("/api/preview")
def preview():
    if job_state.get("phase") not in {"ready", "complete", "idle"}:
        return jsonify({"error": "请等待成片方案生成完成。"}), 409
    ok = ensure_preview()
    if not ok:
        return jsonify({"error": "预览没有启动成功，请查看错误详情。"}), 500
    return jsonify({"ok": True, "url": PREVIEW_URL, "embed_url": PREVIEW_EMBED_URL})


@app.post("/api/render")
def render():
    payload = request.get_json(silent=True, force=True) or {}
    if payload.get("approved") is not True:
        return jsonify({"error": "请先确认你已经看过视频预览。"}), 400
    if job_state.get("phase") not in {"ready", "complete"}:
        return jsonify({"error": "请先完成生成和预览。"}), 409
    threading.Thread(target=render_final, daemon=True).start()
    return jsonify({"ok": True})


@app.get("/output/<path:filename>")
def output_file(filename: str):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=request.args.get("download") == "1")


if __name__ == "__main__":
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    app.run(host="127.0.0.1", port=5088, debug=False, threaded=True)
