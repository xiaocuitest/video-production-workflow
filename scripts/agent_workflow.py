#!/usr/bin/env python3
"""Serializable multi-agent contracts for the video production workflow.

This module deliberately has no dependency on Flask, HyperFrames, or a model SDK.
It is the boundary between specialist agents/skills and the existing renderer:

* every specialist execution is recorded;
* every editorial decision becomes a :class:`CueContract`;
* component dependencies are checked before assembly;
* incomplete scene bundles degrade safely instead of rendering half a design;
* the complete state and assembly manifest are JSON serializable.

The current rule functions in ``pipeline.py`` can be wrapped with
``AgentWorkflow.execute_skill``.  A future remote/model agent uses the same run
record API through ``begin_agent`` / ``complete_run`` / ``fail_run``.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = "video-agent-workflow/v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _identifier(prefix: str, *parts: Any) -> str:
    material = "|".join(str(part) for part in parts if part is not None)
    if not material:
        material = uuid.uuid4().hex
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _jsonable(value: Any) -> Any:
    """Return a conservative JSON-safe copy without leaking live callables."""
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _compact_payload(value: Any, limit: int = 20_000) -> Any:
    """Keep execution traces useful without duplicating an entire transcript."""
    safe = _jsonable(value)
    encoded = json.dumps(safe, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) <= limit:
        return safe
    return {
        "truncated": True,
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "characters": len(encoded),
        "preview": encoded[: min(1200, limit)],
    }


class ActorKind(str, enum.Enum):
    SKILL = "skill"
    AGENT = "agent"
    HUMAN = "human"
    SYSTEM = "system"


class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class CueType(str, enum.Enum):
    HOOK = "hook"
    CALLOUT = "callout"
    KNOWLEDGE = "knowledge"
    SCENE = "scene"
    CHAPTER = "chapter"
    CAPTION_EMPHASIS = "caption_emphasis"
    LOWER_THIRD = "lower_third"
    CAMERA = "camera"
    MUSIC = "music"


class CueStatus(str, enum.Enum):
    PROPOSED = "proposed"
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class AssetStatus(str, enum.Enum):
    REQUESTED = "requested"
    GENERATING = "generating"
    READY = "ready"
    REJECTED = "rejected"
    FAILED = "failed"
    MISSING = "missing"


class SpeakerRole(str, enum.Enum):
    """How the original speaker participates in one complete visual package."""

    FULL = "full"
    CUTOUT = "cutout"
    CIRCLE = "circle"
    CARD = "card"
    NONE = "none"


class TextRole(str, enum.Enum):
    NONE = "none"
    HOOK = "hook"
    CALLOUT = "callout"
    SUMMARY = "summary"
    CHAPTER = "chapter"
    LABEL = "label"
    LOWER_THIRD = "lower_third"
    EMPHASIS = "emphasis"


class CaptionMode(str, enum.Enum):
    NORMAL = "normal"
    COMPACT = "compact"
    ABOVE_SAFE_AREA = "above_safe_area"
    HIDDEN = "hidden"


MEDIA_VISUAL_MODES = frozenset({
    "media_background",
    "media_fullscreen",
    "media_fullscreen_with_speaker_pip",
    "media_half",
    "generated_image",
    "generated_video",
    "stock_video",
})


@dataclasses.dataclass(slots=True)
class ArtifactRef:
    artifact_id: str
    role: str
    path: str = ""
    producer_run_id: str = ""
    media_type: str = "application/json"
    checksum: str = ""
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def create(cls, role: str, path: str | Path = "", **kwargs: Any) -> "ArtifactRef":
        return cls(
            artifact_id=_identifier("artifact", role, path),
            role=role,
            path=str(path),
            **kwargs,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            artifact_id=str(data.get("artifact_id") or _identifier("artifact", data.get("role"), data.get("path"))),
            role=str(data.get("role", "artifact")),
            path=str(data.get("path", "")),
            producer_run_id=str(data.get("producer_run_id", "")),
            media_type=str(data.get("media_type", "application/json")),
            checksum=str(data.get("checksum", "")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclasses.dataclass(slots=True)
class AssetRef:
    asset_id: str
    kind: str
    path: str = ""
    status: AssetStatus = AssetStatus.MISSING
    provider: str = ""
    model: str = ""
    source_url: str = ""
    caption: str = ""
    semantic_tags: list[str] = dataclasses.field(default_factory=list)
    semantic_score: float | None = None
    duration: float | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status is AssetStatus.READY and bool(self.path or self.source_url)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None, *, default_kind: str = "image") -> "AssetRef | None":
        if not data:
            return None
        path = str(data.get("path", ""))
        source_url = str(data.get("source_url", ""))
        raw_status = str(data.get("status") or ("ready" if path or source_url else "missing"))
        try:
            status = AssetStatus(raw_status)
        except ValueError:
            status = AssetStatus.MISSING
        kind = str(data.get("kind") or data.get("type") or default_kind)
        tags = data.get("semantic_tags") or data.get("tags") or []
        return cls(
            asset_id=str(data.get("asset_id") or _identifier("asset", kind, path, source_url, data.get("query"))),
            kind=kind,
            path=path,
            status=status,
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            source_url=source_url,
            caption=str(data.get("caption") or data.get("query") or ""),
            semantic_tags=[str(item) for item in tags],
            semantic_score=(float(data["semantic_score"]) if data.get("semantic_score") is not None else None),
            duration=(float(data["duration"]) if data.get("duration") is not None else None),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclasses.dataclass(slots=True)
class AgentRunRecord:
    run_id: str
    actor_id: str
    actor_kind: ActorKind
    stage: str
    capability: str
    status: RunStatus = RunStatus.PENDING
    provider: str = "local"
    model: str = ""
    version: str = ""
    started_at: str = ""
    finished_at: str = ""
    inputs: Any = dataclasses.field(default_factory=dict)
    outputs: Any = dataclasses.field(default_factory=dict)
    input_artifacts: list[str] = dataclasses.field(default_factory=list)
    output_artifacts: list[str] = dataclasses.field(default_factory=list)
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentRunRecord":
        return cls(
            run_id=str(data["run_id"]),
            actor_id=str(data.get("actor_id", "unknown")),
            actor_kind=ActorKind(str(data.get("actor_kind", "agent"))),
            stage=str(data.get("stage", "")),
            capability=str(data.get("capability", "")),
            status=RunStatus(str(data.get("status", "pending"))),
            provider=str(data.get("provider", "local")),
            model=str(data.get("model", "")),
            version=str(data.get("version", "")),
            started_at=str(data.get("started_at", "")),
            finished_at=str(data.get("finished_at", "")),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            input_artifacts=[str(item) for item in data.get("input_artifacts", [])],
            output_artifacts=[str(item) for item in data.get("output_artifacts", [])],
            metrics=dict(data.get("metrics") or {}),
            error=dict(data["error"]) if data.get("error") else None,
            metadata=dict(data.get("metadata") or {}),
        )


@dataclasses.dataclass(slots=True)
class CueContract:
    """One editorial intent and all components required to render it safely."""

    cue_id: str
    cue_type: CueType
    start: float
    end: float
    spoken_quote: str = ""
    editorial_text: str = ""
    semantic_intent: str = ""
    visual_mode: str = "speaker_hold"
    asset_requirement: str = ""
    forbidden_visuals: list[str] = dataclasses.field(default_factory=list)
    semantic_tags: list[str] = dataclasses.field(default_factory=list)
    asset: AssetRef | None = None
    speaker_pip: bool = False
    fullscreen: bool = False
    alignment: str = ""
    confidence: float = 0.0
    owner_run_id: str = ""
    dependencies: list[str] = dataclasses.field(default_factory=list)
    fallback_visual_mode: str = "speaker_hold"
    status: CueStatus = CueStatus.PROPOSED
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def requires_asset(self) -> bool:
        return self.visual_mode in MEDIA_VISUAL_MODES or "asset.ready" in self.dependencies

    @property
    def requires_spoken_alignment(self) -> bool:
        return self.cue_type in {CueType.HOOK, CueType.CALLOUT, CueType.CAPTION_EMPHASIS}

    def ensure_dependencies(self) -> None:
        # Canonical ordering is also the assembly ordering.  In particular,
        # ``pip.background_ready`` must remain after asset/background gates.
        required = ["timing.valid"]
        if self.requires_spoken_alignment:
            required.append("speech.aligned")
        if self.visual_mode in MEDIA_VISUAL_MODES or "asset.ready" in self.dependencies:
            required.extend([
                "asset.ready", "asset.semantic_match", "asset.no_contradiction",
                "background.ready",
            ])
        if self.speaker_pip or "pip.background_ready" in self.dependencies:
            required.append("pip.background_ready")
        required.extend(item for item in self.dependencies if item not in required)
        self.dependencies = list(dict.fromkeys(required))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CueContract":
        cue = cls(
            cue_id=str(data["cue_id"]),
            cue_type=CueType(str(data.get("cue_type", "scene"))),
            start=float(data.get("start", 0)),
            end=float(data.get("end", data.get("start", 0))),
            spoken_quote=str(data.get("spoken_quote", "")),
            editorial_text=str(data.get("editorial_text", "")),
            semantic_intent=str(data.get("semantic_intent", "")),
            visual_mode=str(data.get("visual_mode", "speaker_hold")),
            asset_requirement=str(data.get("asset_requirement", "")),
            forbidden_visuals=[str(item) for item in data.get("forbidden_visuals", [])],
            semantic_tags=[str(item) for item in data.get("semantic_tags", [])],
            asset=AssetRef.from_mapping(data.get("asset")),
            speaker_pip=bool(data.get("speaker_pip", False)),
            fullscreen=bool(data.get("fullscreen", False)),
            alignment=str(data.get("alignment", "")),
            confidence=float(data.get("confidence", 0)),
            owner_run_id=str(data.get("owner_run_id", "")),
            dependencies=[str(item) for item in data.get("dependencies", [])],
            fallback_visual_mode=str(data.get("fallback_visual_mode", "speaker_hold")),
            status=CueStatus(str(data.get("status", "proposed"))),
            metadata=dict(data.get("metadata") or {}),
        )
        cue.ensure_dependencies()
        return cue


@dataclasses.dataclass(slots=True)
class TransitionSpec:
    """Seek-safe transition instruction for the assembly layer."""

    name: str = "cut"
    duration: float = 0.0
    easing: str = "none"
    delay: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | str | None, *, default: "TransitionSpec | None" = None) -> "TransitionSpec":
        if isinstance(data, str):
            return cls(name=data)
        if isinstance(data, Mapping):
            return cls(
                name=str(data.get("name", "cut")),
                duration=max(0.0, float(data.get("duration", 0) or 0)),
                easing=str(data.get("easing", "none")),
                delay=max(0.0, float(data.get("delay", 0) or 0)),
            )
        return dataclasses.replace(default) if default else cls()


@dataclasses.dataclass(slots=True)
class ScenePackage:
    """Atomic visual package consumed by a renderer, never a loose component list.

    ``background_role`` identifies the layer occupying the canvas before any
    speaker treatment enters.  ``media_role`` says how the optional asset is
    used. PiP-like speaker roles may only be assembled after a complete
    alternate background (verified media, graphic, or source-blur board),
    expressed in ``dependency_order`` and the generated assembly checklist.
    """

    package_id: str
    cue_id: str
    start: float
    end: float
    background_role: str = "source"
    media_role: str = "none"
    media: AssetRef | None = None
    speaker_role: SpeakerRole = SpeakerRole.FULL
    text_role: TextRole = TextRole.NONE
    text: str = ""
    caption_mode: CaptionMode = CaptionMode.NORMAL
    entry_transition: TransitionSpec = dataclasses.field(default_factory=TransitionSpec)
    exit_transition: TransitionSpec = dataclasses.field(default_factory=TransitionSpec)
    dependency_order: list[str] = dataclasses.field(default_factory=list)
    dependency_status: dict[str, bool] = dataclasses.field(default_factory=dict)
    status: CueStatus = CueStatus.PROPOSED
    degraded_from: dict[str, Any] = dataclasses.field(default_factory=dict)
    degradation_reason: list[str] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def background_ready(self) -> bool:
        if self.background_role in {"source", "source_blur", "graphic", "solid"}:
            return True
        return self.background_role == "media" and bool(self.media and self.media.ready)

    @property
    def speaker_is_pip(self) -> bool:
        return self.speaker_role in {SpeakerRole.CUTOUT, SpeakerRole.CIRCLE, SpeakerRole.CARD}

    def assembly_steps(self) -> list[dict[str, Any]]:
        """Return deterministic component order for this package.

        The renderer may parallelize unrelated work, but must honor ``after``.
        A circle/cutout/card speaker is never emitted before ``background.ready``.
        """
        steps: list[dict[str, Any]] = [
            {
                "order": 10,
                "component": "background",
                "action": "mount_media" if self.background_role == "media" else "keep_source",
                "role": self.background_role,
                "asset_id": self.media.asset_id if self.media else "",
                "after": ["timing.valid"],
                "ready": self.background_ready,
                "transition": _jsonable(self.entry_transition),
            }
        ]
        if self.media_role != "none" and self.background_role != "media":
            steps.append({
                "order": 20, "component": "media", "action": "mount",
                "role": self.media_role, "asset_id": self.media.asset_id if self.media else "",
                "after": ["background.ready", "asset.ready"],
                "ready": bool(self.media and self.media.ready),
            })
        speaker_after = ["background.ready"] if self.speaker_is_pip else ["timing.valid"]
        steps.append({
            "order": 30,
            "component": "speaker",
            "action": "hide" if self.speaker_role is SpeakerRole.NONE else "show",
            "role": self.speaker_role.value,
            "after": speaker_after,
            "ready": self.background_ready if self.speaker_is_pip else True,
        })
        if self.text_role is not TextRole.NONE:
            steps.append({
                "order": 40, "component": "text", "action": "show",
                "role": self.text_role.value, "text": self.text,
                "after": ["background.ready", "speaker.ready"], "ready": True,
            })
        steps.append({
            "order": 50, "component": "captions",
            "action": "hide" if self.caption_mode is CaptionMode.HIDDEN else "show",
            "role": self.caption_mode.value, "after": ["background.ready"], "ready": True,
        })
        steps.append({
            "order": 90, "component": "package", "action": "exit",
            "role": "restore_source_and_full_speaker", "after": ["package.duration"],
            "ready": True, "transition": _jsonable(self.exit_transition),
        })
        return sorted(steps, key=lambda item: int(item["order"]))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenePackage":
        return cls(
            package_id=str(data["package_id"]), cue_id=str(data["cue_id"]),
            start=float(data.get("start", 0)), end=float(data.get("end", data.get("start", 0))),
            background_role=str(data.get("background_role", "source")),
            media_role=str(data.get("media_role", "none")),
            media=AssetRef.from_mapping(data.get("media")),
            speaker_role=SpeakerRole(str(data.get("speaker_role", "full"))),
            text_role=TextRole(str(data.get("text_role", "none"))),
            text=str(data.get("text", "")),
            caption_mode=CaptionMode(str(data.get("caption_mode", "normal"))),
            entry_transition=TransitionSpec.from_mapping(data.get("entry_transition")),
            exit_transition=TransitionSpec.from_mapping(data.get("exit_transition")),
            dependency_order=[str(item) for item in data.get("dependency_order", [])],
            dependency_status={str(key): bool(value) for key, value in dict(data.get("dependency_status") or {}).items()},
            status=CueStatus(str(data.get("status", "proposed"))),
            degraded_from=dict(data.get("degraded_from") or {}),
            degradation_reason=[str(item) for item in data.get("degradation_reason", [])],
            metadata=dict(data.get("metadata") or {}),
        )


@dataclasses.dataclass(slots=True)
class GateResult:
    gate_id: str
    cue_id: str
    dependency: str
    passed: bool
    blocking: bool
    message: str
    checked_at: str = dataclasses.field(default_factory=_utc_now)
    evidence: dict[str, Any] = dataclasses.field(default_factory=dict)
    fallback_applied: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GateResult":
        return cls(
            gate_id=str(data["gate_id"]), cue_id=str(data["cue_id"]),
            dependency=str(data["dependency"]), passed=bool(data["passed"]),
            blocking=bool(data.get("blocking", True)), message=str(data.get("message", "")),
            checked_at=str(data.get("checked_at", _utc_now())),
            evidence=dict(data.get("evidence") or {}),
            fallback_applied=str(data.get("fallback_applied", "")),
        )


@dataclasses.dataclass(slots=True)
class AssemblyManifest:
    manifest_id: str
    workflow_id: str
    generated_at: str
    video: dict[str, Any]
    policy: dict[str, Any]
    cues: list[dict[str, Any]]
    excluded_cues: list[dict[str, Any]]
    tracks: dict[str, list[str]]
    gate_summary: dict[str, Any]
    execution_summary: dict[str, Any]
    artifacts: list[dict[str, Any]]
    scene_packages: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    assembly_checklist: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AssemblyManifest":
        return cls(
            manifest_id=str(data.get("manifest_id", "")),
            workflow_id=str(data.get("workflow_id", "")),
            generated_at=str(data.get("generated_at", "")),
            video=dict(data.get("video") or {}),
            policy=dict(data.get("policy") or {}),
            cues=list(data.get("cues") or []),
            excluded_cues=list(data.get("excluded_cues") or []),
            tracks={str(key): [str(item) for item in value] for key, value in dict(data.get("tracks") or {}).items()},
            gate_summary=dict(data.get("gate_summary") or {}),
            execution_summary=dict(data.get("execution_summary") or {}),
            artifacts=list(data.get("artifacts") or []),
            scene_packages=list(data.get("scene_packages") or []),
            assembly_checklist=list(data.get("assembly_checklist") or []),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclasses.dataclass(slots=True)
class WorkflowState:
    workflow_id: str
    project_id: str
    schema_version: str = SCHEMA_VERSION
    status: str = "building"
    created_at: str = dataclasses.field(default_factory=_utc_now)
    updated_at: str = dataclasses.field(default_factory=_utc_now)
    video: dict[str, Any] = dataclasses.field(default_factory=dict)
    runs: list[AgentRunRecord] = dataclasses.field(default_factory=list)
    cues: list[CueContract] = dataclasses.field(default_factory=list)
    gates: list[GateResult] = dataclasses.field(default_factory=list)
    scene_packages: list[ScenePackage] = dataclasses.field(default_factory=list)
    artifacts: list[ArtifactRef] = dataclasses.field(default_factory=list)
    manifest: AssemblyManifest | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        self.updated_at = _utc_now()
        return _jsonable(self)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "WorkflowState":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        state = cls(
            workflow_id=str(data["workflow_id"]),
            project_id=str(data.get("project_id", "video")),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            status=str(data.get("status", "building")),
            created_at=str(data.get("created_at", _utc_now())),
            updated_at=str(data.get("updated_at", _utc_now())),
            video=dict(data.get("video") or {}),
            runs=[AgentRunRecord.from_dict(item) for item in data.get("runs", [])],
            cues=[CueContract.from_dict(item) for item in data.get("cues", [])],
            gates=[GateResult.from_dict(item) for item in data.get("gates", [])],
            scene_packages=[ScenePackage.from_dict(item) for item in data.get("scene_packages", [])],
            artifacts=[ArtifactRef.from_dict(item) for item in data.get("artifacts", [])],
            metadata=dict(data.get("metadata") or {}),
        )
        if data.get("manifest"):
            state.manifest = AssemblyManifest.from_dict(data["manifest"])
        return state


class DependencyGate:
    """Built-in deterministic checks; model/human QC can add explicit evidence."""

    def __init__(self, *, semantic_threshold: float = 0.68) -> None:
        self.semantic_threshold = semantic_threshold

    def check(self, cue: CueContract, dependency: str, video: Mapping[str, Any]) -> GateResult:
        passed = True
        message = "依赖已满足"
        evidence: dict[str, Any] = {}

        if dependency == "timing.valid":
            duration = float(video.get("duration", 0) or 0)
            passed = cue.start >= 0 and cue.end > cue.start and (duration <= 0 or cue.end <= duration + 0.001)
            message = "时间窗有效" if passed else "时间窗越界或持续时间无效"
            evidence = {"start": cue.start, "end": cue.end, "video_duration": duration}
        elif dependency == "speech.aligned":
            passed = bool(cue.spoken_quote.strip()) and cue.alignment not in {"missing", "unaligned"}
            message = "文字已绑定口播原话" if passed else "Hook/Callout 没有可验证的口播原话或时间对齐"
            evidence = {"alignment": cue.alignment, "spoken_quote": cue.spoken_quote}
        elif dependency == "asset.ready":
            passed = bool(cue.asset and cue.asset.ready)
            message = "配套素材已就绪" if passed else "配套素材尚未生成、下载或验收"
            evidence = {"asset_id": cue.asset.asset_id if cue.asset else "", "status": cue.asset.status.value if cue.asset else "missing"}
        elif dependency == "asset.semantic_match":
            if not cue.asset:
                passed = False
            elif cue.asset.semantic_score is not None:
                passed = cue.asset.semantic_score >= self.semantic_threshold
            else:
                cue_tags = set(cue.semantic_tags)
                asset_tags = set(cue.asset.semantic_tags)
                passed = bool(cue_tags & asset_tags) if cue_tags and asset_tags else bool(cue.asset.metadata.get("semantic_approved"))
            message = "素材语义与口播一致" if passed else "素材尚未通过语义匹配质检"
            evidence = {
                "required_tags": cue.semantic_tags,
                "asset_tags": cue.asset.semantic_tags if cue.asset else [],
                "semantic_score": cue.asset.semantic_score if cue.asset else None,
                "threshold": self.semantic_threshold,
            }
        elif dependency == "asset.no_contradiction":
            contradictions = list((cue.asset.metadata if cue.asset else {}).get("detected_contradictions", []))
            forbidden_hits = list((cue.asset.metadata if cue.asset else {}).get("forbidden_visual_hits", []))
            passed = bool(cue.asset) and not contradictions and not forbidden_hits
            message = "未发现画面矛盾" if passed else "素材与口播存在矛盾或命中禁用画面"
            evidence = {"contradictions": contradictions, "forbidden_hits": forbidden_hits, "forbidden_visuals": cue.forbidden_visuals}
        elif dependency == "background.ready":
            if cue.visual_mode in MEDIA_VISUAL_MODES:
                passed = bool(cue.asset and cue.asset.ready)
                role = "media"
            else:
                passed = True
                role = "source"
            message = "场景背景层已先于其他组件就绪" if passed else "配套背景层未就绪，不能进入人物或文字组件"
            evidence = {"background_role": role, "asset_ready": bool(cue.asset and cue.asset.ready)}
        elif dependency == "pip.background_ready":
            package = cue.metadata.get("scene_package") if isinstance(cue.metadata.get("scene_package"), Mapping) else {}
            background_role = str(package.get("background_role", "source"))
            complete_graphic = background_role in {"source_blur", "graphic", "solid", "editorial_canvas"}
            complete_media = background_role == "media" and bool(cue.asset and cue.asset.ready and cue.visual_mode in MEDIA_VISUAL_MODES)
            passed = bool(cue.speaker_pip and (complete_graphic or complete_media))
            message = "人物画中画与配套背景已组成完整场景包" if passed else "人物画中画不能脱离已就绪的配套背景单独出现"
            evidence = {"speaker_pip": cue.speaker_pip, "visual_mode": cue.visual_mode, "background_role": background_role, "asset_ready": bool(cue.asset and cue.asset.ready)}
        else:
            explicit = cue.metadata.get("dependency_evidence", {}).get(dependency)
            passed = bool(explicit and explicit.get("passed"))
            message = str((explicit or {}).get("message") or f"依赖 {dependency} 尚无验收证据")
            evidence = dict(explicit or {})

        return GateResult(
            gate_id=_identifier("gate", cue.cue_id, dependency),
            cue_id=cue.cue_id,
            dependency=dependency,
            passed=passed,
            blocking=True,
            message=message,
            evidence=evidence,
        )


class AgentWorkflow:
    """Small orchestration facade intended to be called directly by pipeline.py."""

    def __init__(self, state: WorkflowState, *, semantic_threshold: float = 0.68) -> None:
        self.state = state
        self.gate = DependencyGate(semantic_threshold=semantic_threshold)

    @classmethod
    def create(
        cls,
        project_id: str,
        *,
        source: str | Path = "",
        duration: float = 0.0,
        workflow_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "AgentWorkflow":
        identity = workflow_id or _identifier("workflow", project_id, source, uuid.uuid4().hex)
        state = WorkflowState(
            workflow_id=identity,
            project_id=project_id,
            video={"source": str(source), "duration": float(duration)},
            metadata=dict(metadata or {}),
        )
        return cls(state)

    @classmethod
    def load(cls, path: str | Path) -> "AgentWorkflow":
        return cls(WorkflowState.load(path))

    def begin_run(
        self,
        *,
        actor_id: str,
        actor_kind: ActorKind | str,
        stage: str,
        capability: str,
        provider: str = "local",
        model: str = "",
        version: str = "",
        inputs: Any = None,
        input_artifacts: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        kind = actor_kind if isinstance(actor_kind, ActorKind) else ActorKind(actor_kind)
        run_id = _identifier("run", self.state.workflow_id, actor_id, capability, len(self.state.runs), uuid.uuid4().hex)
        self.state.runs.append(AgentRunRecord(
            run_id=run_id, actor_id=actor_id, actor_kind=kind, stage=stage,
            capability=capability, status=RunStatus.RUNNING, provider=provider,
            model=model, version=version, started_at=_utc_now(),
            inputs=_compact_payload(inputs), input_artifacts=list(input_artifacts),
            metadata=dict(metadata or {}),
        ))
        return run_id

    def _run(self, run_id: str) -> AgentRunRecord:
        try:
            return next(item for item in self.state.runs if item.run_id == run_id)
        except StopIteration as exc:
            raise KeyError(f"Unknown workflow run: {run_id}") from exc

    def complete_run(
        self,
        run_id: str,
        *,
        outputs: Any = None,
        output_artifacts: Sequence[str] = (),
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        run = self._run(run_id)
        run.status = RunStatus.SUCCEEDED
        run.finished_at = _utc_now()
        run.outputs = _compact_payload(outputs)
        run.output_artifacts = list(output_artifacts)
        run.metrics.update(dict(metrics or {}))

    def fail_run(self, run_id: str, error: BaseException, *, reraise: bool = False) -> None:
        run = self._run(run_id)
        run.status = RunStatus.FAILED
        run.finished_at = _utc_now()
        run.error = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))[-8000:],
        }
        if reraise:
            raise error

    def execute_skill(
        self,
        skill_name: str,
        function: Callable[..., Any],
        *,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
        stage: str,
        version: str = "rules-v1",
        inputs: Any = None,
        input_artifacts: Sequence[str] = (),
        output_artifacts: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Execute an existing local rule function and record its provenance."""
        run_id = self.begin_run(
            actor_id=skill_name, actor_kind=ActorKind.SKILL, stage=stage,
            capability=skill_name, provider="local-rules", version=version,
            inputs=inputs if inputs is not None else {"args": args, "kwargs": kwargs or {}},
            input_artifacts=input_artifacts, metadata=metadata,
        )
        try:
            result = function(*args, **dict(kwargs or {}))
        except Exception as exc:
            self.fail_run(run_id, exc)
            raise
        self.complete_run(run_id, outputs=result, output_artifacts=output_artifacts)
        return result

    def begin_agent(
        self,
        agent_id: str,
        *,
        stage: str,
        capability: str,
        provider: str,
        model: str,
        inputs: Any = None,
        input_artifacts: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Start a future model-agent run without imposing a specific SDK."""
        return self.begin_run(
            actor_id=agent_id, actor_kind=ActorKind.AGENT, stage=stage,
            capability=capability, provider=provider, model=model, inputs=inputs,
            input_artifacts=input_artifacts, metadata=metadata,
        )

    def add_artifact(self, artifact: ArtifactRef) -> ArtifactRef:
        existing = next((item for item in self.state.artifacts if item.artifact_id == artifact.artifact_id), None)
        if existing:
            return existing
        self.state.artifacts.append(artifact)
        return artifact

    def add_cue(self, cue: CueContract) -> CueContract:
        cue.ensure_dependencies()
        existing = next((item for item in self.state.cues if item.cue_id == cue.cue_id), None)
        if existing:
            self.state.cues[self.state.cues.index(existing)] = cue
        else:
            self.state.cues.append(cue)
        self.state.cues.sort(key=lambda item: (item.start, item.end, item.cue_id))
        return cue

    def ingest_director_plan(self, plan: Mapping[str, Any], *, owner_run_id: str = "") -> list[CueContract]:
        """Adapt the current pipeline director-plan shape to the unified contract."""
        created: list[CueContract] = []

        def add(cue_type: CueType, item: Mapping[str, Any], index: int) -> None:
            start = float(item.get("start", 0) or 0)
            default_duration = 3.2 if cue_type is not CueType.CHAPTER else 2.0
            end = float(item.get("end", start + float(item.get("duration", default_duration))) or start + default_duration)
            phrases = item.get("phrases") if isinstance(item.get("phrases"), Sequence) and not isinstance(item.get("phrases"), str) else []
            text = str(item.get("text") or item.get("title") or item.get("caption") or "".join(str(value) for value in phrases))
            if cue_type is CueType.SCENE:
                requested = str(item.get("type", "image"))
                visual_mode = "generated_video" if requested == "video" else "generated_image"
                fallback = "none"
            elif cue_type is CueType.CHAPTER:
                visual_mode, fallback = "chapter_card", "speaker_hold"
            elif cue_type is CueType.HOOK:
                visual_mode, fallback = "hook_card", "hook_card"
            elif cue_type is CueType.CALLOUT:
                visual_mode, fallback = "callout_card", "callout_card"
            elif cue_type is CueType.KNOWLEDGE:
                visual_mode, fallback = "knowledge_board", "speaker_hold"
            elif cue_type is CueType.LOWER_THIRD:
                visual_mode, fallback = "lower_third", "none"
            else:
                visual_mode, fallback = cue_type.value, "none"
            cue = CueContract(
                cue_id=_identifier("cue", cue_type.value, round(start, 3), index, text),
                cue_type=cue_type,
                start=start,
                end=end,
                spoken_quote=text if cue_type in {CueType.HOOK, CueType.CALLOUT, CueType.CAPTION_EMPHASIS} else "",
                editorial_text=str(item.get("editorial_text") or text),
                semantic_intent=str(item.get("semantic_intent") or item.get("reason") or text),
                visual_mode=visual_mode,
                asset_requirement=str(item.get("prompt", "")),
                forbidden_visuals=[str(value) for value in item.get("forbidden_visuals", [])],
                semantic_tags=[str(value) for value in item.get("semantic_tags", [])],
                alignment=str(item.get("alignment", "")),
                confidence=float(item.get("confidence", 0) or 0),
                owner_run_id=owner_run_id,
                fallback_visual_mode=fallback,
                fullscreen=cue_type is CueType.CHAPTER,
                metadata={"legacy": _jsonable(item)},
            )
            self.add_cue(cue)
            created.append(cue)

        if isinstance(plan.get("hook"), Mapping):
            add(CueType.HOOK, plan["hook"], 0)
        routing = (
            ("callouts", CueType.CALLOUT),
            ("knowledge_cards", CueType.KNOWLEDGE),
            ("scene_assets", CueType.SCENE),
            ("chapters", CueType.CHAPTER),
            ("caption_emphasis", CueType.CAPTION_EMPHASIS),
            ("lower_thirds", CueType.LOWER_THIRD),
        )
        for key, cue_type in routing:
            for index, item in enumerate(plan.get(key, [])):
                if isinstance(item, Mapping):
                    add(cue_type, item, index)
        return created

    def bind_asset(
        self,
        cue_id: str,
        asset: AssetRef | Mapping[str, Any],
        *,
        visual_mode: str | None = None,
        speaker_pip: bool | None = None,
        semantic_tags: Iterable[str] = (),
    ) -> CueContract:
        cue = next((item for item in self.state.cues if item.cue_id == cue_id), None)
        if cue is None:
            raise KeyError(f"Unknown cue: {cue_id}")
        cue.asset = asset if isinstance(asset, AssetRef) else AssetRef.from_mapping(asset)
        if visual_mode:
            cue.visual_mode = visual_mode
        if speaker_pip is not None:
            cue.speaker_pip = speaker_pip
        if semantic_tags:
            cue.semantic_tags = list(dict.fromkeys([*cue.semantic_tags, *[str(item) for item in semantic_tags]]))
        cue.ensure_dependencies()
        return cue

    def evaluate_gates(self, *, apply_fallbacks: bool = True) -> list[GateResult]:
        """Evaluate every declared dependency and enforce component bundles."""
        self.state.gates = []
        for cue in self.state.cues:
            cue.ensure_dependencies()
            results = [self.gate.check(cue, dependency, self.state.video) for dependency in cue.dependencies]
            failed = [item for item in results if item.blocking and not item.passed]
            if failed and apply_fallbacks and cue.fallback_visual_mode:
                original = cue.visual_mode
                cue.metadata["degraded_from"] = original
                cue.metadata["gate_failures"] = [item.dependency for item in failed]
                cue.visual_mode = cue.fallback_visual_mode
                cue.speaker_pip = False
                cue.fullscreen = cue.cue_type is CueType.CHAPTER and cue.fallback_visual_mode == "chapter_card"
                cue.status = CueStatus.SKIPPED if cue.visual_mode == "none" else CueStatus.DEGRADED
                for result in failed:
                    result.fallback_applied = cue.fallback_visual_mode
            elif failed:
                cue.status = CueStatus.BLOCKED
            else:
                cue.status = CueStatus.READY
            self.state.gates.extend(results)
        return self.state.gates

    def normalize_scene_package(self, cue_or_id: CueContract | str) -> ScenePackage:
        """Normalize one cue into a complete V2 visual package.

        Optional specialist decisions can be supplied in
        ``cue.metadata['scene_package']`` using the same field names. Invalid or
        incomplete PiP decisions are ignored in favor of the safe full-speaker
        fallback.
        """
        if isinstance(cue_or_id, CueContract):
            cue = cue_or_id
        else:
            cue = next((item for item in self.state.cues if item.cue_id == cue_or_id), None)
            if cue is None:
                raise KeyError(f"Unknown cue: {cue_or_id}")

        results = [item for item in self.state.gates if item.cue_id == cue.cue_id]
        dependency_status = {item.dependency: item.passed for item in results}
        failures = [item.dependency for item in results if item.blocking and not item.passed]
        original_mode = str(cue.metadata.get("degraded_from") or cue.visual_mode)
        # A director ``scene`` may intentionally degrade to a complete graphic
        # concept world.  Cue type alone therefore does not imply a missing
        # media bundle; only an actual media visual mode does.
        was_media = original_mode in MEDIA_VISUAL_MODES
        timing_valid = dependency_status.get("timing.valid", cue.end > cue.start >= 0)
        media_ready = bool(cue.asset and cue.asset.ready)
        media_dependencies_ok = all(dependency_status.get(name, False) for name in (
            "asset.ready", "asset.semantic_match", "asset.no_contradiction", "background.ready",
        )) if was_media else True
        complete_media = was_media and media_ready and media_dependencies_ok and timing_valid

        text_role_by_type = {
            CueType.HOOK: TextRole.HOOK,
            CueType.CALLOUT: TextRole.CALLOUT,
            CueType.KNOWLEDGE: TextRole.SUMMARY,
            CueType.CHAPTER: TextRole.CHAPTER,
            CueType.SCENE: TextRole.LABEL,
            CueType.CAPTION_EMPHASIS: TextRole.EMPHASIS,
            CueType.LOWER_THIRD: TextRole.LOWER_THIRD,
        }
        text_role = text_role_by_type.get(cue.cue_type, TextRole.NONE)
        text = cue.editorial_text or cue.spoken_quote
        background_role = "source"
        media_role = "none"
        speaker_role = SpeakerRole.FULL
        caption_mode = CaptionMode.NORMAL
        entry = TransitionSpec("cut", 0.0, "none")
        exit_ = TransitionSpec("cut", 0.0, "none")

        if complete_media:
            if original_mode == "media_half":
                background_role, media_role = "source", "upper_half"
                speaker_role = SpeakerRole.FULL
            else:
                background_role, media_role = "media", "background"
                speaker_role = SpeakerRole.CIRCLE if cue.speaker_pip else SpeakerRole.NONE
            caption_mode = CaptionMode.ABOVE_SAFE_AREA
            entry = TransitionSpec("media_reveal", 0.38, "power2.out")
            exit_ = TransitionSpec("media_release", 0.28, "power2.inOut")
        elif was_media:
            # Atomic degradation: remove the whole media/PiP/text-label bundle.
            # The source video remains authoritative and the speaker returns full.
            background_role, media_role = "source", "none"
            speaker_role, text_role = SpeakerRole.FULL, TextRole.NONE
            caption_mode = CaptionMode.NORMAL
            text = ""
        elif cue.cue_type is CueType.CHAPTER:
            background_role, speaker_role = "graphic", SpeakerRole.NONE
            caption_mode = CaptionMode.HIDDEN
            entry = TransitionSpec("chapter_in", 0.34, "power3.out")
            exit_ = TransitionSpec("chapter_out", 0.26, "power2.in")
        elif cue.cue_type is CueType.HOOK:
            background_role = "graphic"
            speaker_role = SpeakerRole.CIRCLE
            caption_mode = CaptionMode.HIDDEN
            entry = TransitionSpec("hook_in", 0.34, "circ.out")
            exit_ = TransitionSpec("hook_out", 0.20, "power3.in")
        elif cue.cue_type is CueType.CALLOUT:
            background_role = "source_blur"
            speaker_role = SpeakerRole.CIRCLE
            caption_mode = CaptionMode.HIDDEN
            entry = TransitionSpec("callout_in", 0.28, "power3.out")
            exit_ = TransitionSpec("callout_out", 0.20, "power2.in")
        elif cue.cue_type is CueType.KNOWLEDGE:
            background_role = "source_blur"
            speaker_role = SpeakerRole.CARD
            caption_mode = CaptionMode.NORMAL
            entry = TransitionSpec("knowledge_in", 0.34, "expo.out")
            exit_ = TransitionSpec("knowledge_out", 0.22, "power2.in")
        elif cue.cue_type is CueType.LOWER_THIRD:
            entry = TransitionSpec("lower_third_in", 0.24, "power3.out")
            exit_ = TransitionSpec("lower_third_out", 0.18, "power2.in")

        overrides = cue.metadata.get("scene_package")
        if isinstance(overrides, Mapping):
            background_role = str(overrides.get("background_role", background_role))
            media_role = str(overrides.get("media_role", media_role))
            try:
                speaker_role = SpeakerRole(str(overrides.get("speaker_role", speaker_role.value)))
            except ValueError:
                pass
            try:
                text_role = TextRole(str(overrides.get("text_role", text_role.value)))
            except ValueError:
                pass
            try:
                caption_mode = CaptionMode(str(overrides.get("caption_mode", caption_mode.value)))
            except ValueError:
                pass
            entry = TransitionSpec.from_mapping(overrides.get("entry_transition"), default=entry)
            exit_ = TransitionSpec.from_mapping(overrides.get("exit_transition"), default=exit_)

        degraded_from: dict[str, Any] = {}
        status = cue.status
        if was_media and not complete_media:
            degraded_from = {
                "visual_mode": original_mode,
                "speaker_role": "circle" if cue.speaker_pip else "none",
                "background_role": "media",
                "asset_id": cue.asset.asset_id if cue.asset else "",
            }
            status = CueStatus.BLOCKED if not timing_valid else CueStatus.DEGRADED

        # Final invariant: PiP requires a complete alternate background. It may
        # be verified media, a deliberate graphic world, or a blurred knowledge
        # board — never the unchanged source frame that caused the old mismatch.
        accepted_pip_background = (
            (background_role == "media" and media_ready and media_dependencies_ok)
            or background_role in {"source_blur", "graphic", "solid"}
        )
        if speaker_role in {SpeakerRole.CIRCLE, SpeakerRole.CUTOUT, SpeakerRole.CARD} and not accepted_pip_background:
            degraded_from.update({"speaker_role": speaker_role.value, "background_role": background_role})
            background_role, media_role = "source", "none"
            speaker_role, text_role = SpeakerRole.FULL, TextRole.NONE
            caption_mode = CaptionMode.NORMAL
            text = ""
            status = CueStatus.BLOCKED if not timing_valid else CueStatus.DEGRADED
            if "pip.background_ready" not in failures:
                failures.append("pip.background_ready")

        dependency_order = list(cue.dependencies)
        if "background.ready" not in dependency_order:
            insert_at = dependency_order.index("pip.background_ready") if "pip.background_ready" in dependency_order else len(dependency_order)
            dependency_order.insert(insert_at, "background.ready")
        if speaker_role in {SpeakerRole.CIRCLE, SpeakerRole.CUTOUT, SpeakerRole.CARD} and "pip.background_ready" not in dependency_order:
            dependency_order.append("pip.background_ready")

        return ScenePackage(
            package_id=_identifier("package", cue.cue_id), cue_id=cue.cue_id,
            start=cue.start, end=cue.end, background_role=background_role,
            media_role=media_role, media=cue.asset if complete_media else None,
            speaker_role=speaker_role, text_role=text_role, text=text,
            caption_mode=caption_mode, entry_transition=entry,
            exit_transition=exit_, dependency_order=dependency_order,
            dependency_status=dependency_status, status=status,
            degraded_from=degraded_from, degradation_reason=failures,
            metadata={"cue_type": cue.cue_type.value, "visual_mode": cue.visual_mode},
        )

    def normalize_scene_packages(self, *, refresh_gates: bool = False) -> list[ScenePackage]:
        """Normalize all cues and persist their atomic visual packages."""
        if refresh_gates or not self.state.gates:
            self.evaluate_gates()
        self.state.scene_packages = [self.normalize_scene_package(cue) for cue in self.state.cues]
        self.state.scene_packages.sort(key=lambda item: (item.start, item.end, item.package_id))
        return self.state.scene_packages

    def assembly_checklist(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        """Return renderer-facing ordered steps and explicit blockers per package."""
        if refresh or not self.state.scene_packages:
            self.normalize_scene_packages(refresh_gates=refresh)
        checklist: list[dict[str, Any]] = []
        for package in self.state.scene_packages:
            steps = package.assembly_steps()
            pip_step = next((item for item in steps if item["component"] == "speaker" and item["role"] in {"circle", "cutout", "card"}), None)
            if pip_step and ("background.ready" not in pip_step["after"] or not package.background_ready):
                raise ValueError(f"Unsafe PiP assembly order in package {package.package_id}")
            checklist.append({
                "package_id": package.package_id,
                "cue_id": package.cue_id,
                "start": package.start,
                "end": package.end,
                "status": package.status.value,
                "assemble": package.status is not CueStatus.BLOCKED,
                "dependency_order": list(package.dependency_order),
                "blockers": list(package.degradation_reason),
                "steps": steps,
            })
        return checklist

    def build_manifest(self, *, metadata: Mapping[str, Any] | None = None) -> AssemblyManifest:
        """Freeze the safe-to-render scene bundles into one assembly manifest."""
        if not self.state.gates:
            self.evaluate_gates()
        packages = self.state.scene_packages or self.normalize_scene_packages()
        checklist = self.assembly_checklist()
        included = [cue for cue in self.state.cues if cue.status in {CueStatus.READY, CueStatus.DEGRADED} and cue.visual_mode != "none"]
        excluded = [cue for cue in self.state.cues if cue not in included]
        tracks = {
            "background": [item.cue_id for item in packages if item.background_role == "media" and item.background_ready],
            "speaker_overlay": [item.cue_id for item in packages if item.speaker_is_pip and item.background_ready],
            "editorial_overlay": [cue.cue_id for cue in included if cue.visual_mode not in MEDIA_VISUAL_MODES],
            "audio": [cue.cue_id for cue in included if cue.cue_type is CueType.MUSIC],
        }
        failed = [item for item in self.state.gates if not item.passed]
        manifest = AssemblyManifest(
            manifest_id=_identifier("assembly", self.state.workflow_id, len(self.state.cues), _utc_now()),
            workflow_id=self.state.workflow_id,
            generated_at=_utc_now(),
            video=_jsonable(self.state.video),
            policy={
                "scene_bundle_atomic": True,
                "speaker_pip_requires_ready_background": True,
                "component_order": ["background", "media", "speaker", "text", "captions", "exit_restore"],
                "semantic_gate_threshold": self.gate.semantic_threshold,
                "missing_media_fallback": "source background + full speaker; never render a partial package or demo placeholder",
            },
            cues=[_jsonable(item) for item in included],
            excluded_cues=[{
                "cue_id": item.cue_id,
                "status": item.status.value,
                "reason": item.metadata.get("gate_failures", []),
            } for item in excluded],
            tracks=tracks,
            gate_summary={
                "total": len(self.state.gates),
                "passed": len(self.state.gates) - len(failed),
                "failed": len(failed),
                "failures": [_jsonable(item) for item in failed],
            },
            execution_summary={
                "total": len(self.state.runs),
                "succeeded": sum(item.status is RunStatus.SUCCEEDED for item in self.state.runs),
                "failed": sum(item.status is RunStatus.FAILED for item in self.state.runs),
                "runs": [{
                    "run_id": item.run_id, "actor_id": item.actor_id,
                    "actor_kind": item.actor_kind.value, "stage": item.stage,
                    "capability": item.capability, "status": item.status.value,
                    "provider": item.provider, "model": item.model,
                } for item in self.state.runs],
            },
            artifacts=[_jsonable(item) for item in self.state.artifacts],
            scene_packages=[_jsonable(item) for item in packages],
            assembly_checklist=checklist,
            metadata=dict(metadata or {}),
        )
        self.state.manifest = manifest
        self.state.status = "ready" if not any(cue.status is CueStatus.BLOCKED for cue in self.state.cues) else "needs_attention"
        return manifest

    def save(self, path: str | Path) -> Path:
        return self.state.save(path)


__all__ = [
    "SCHEMA_VERSION",
    "ActorKind",
    "RunStatus",
    "CueType",
    "CueStatus",
    "AssetStatus",
    "SpeakerRole",
    "TextRole",
    "CaptionMode",
    "ArtifactRef",
    "AssetRef",
    "AgentRunRecord",
    "CueContract",
    "TransitionSpec",
    "ScenePackage",
    "GateResult",
    "AssemblyManifest",
    "WorkflowState",
    "DependencyGate",
    "AgentWorkflow",
]
