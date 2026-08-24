#!/usr/bin/env python3
"""Structured editorial QA for the multi-agent video workflow.

The checker is intentionally dependency-free.  It validates the hand-off between
the director plan, material director, motion director, and timeline assembler and
returns machine-readable findings with an explicit repair owner.

Exit codes:
    0  QA passed (warnings may still be present)
    1  QA failed because at least one error was found
    2  Input or invocation error
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


OWNER_TIMELINE = "timeline_assembler"
OWNER_DIRECTOR = "chief_director"
OWNER_CONTENT = "content_analyst"
OWNER_MEDIA = "media_director"
OWNER_MOTION = "motion_designer"

EPSILON = 0.04
SEMANTIC_TERMS = (
    "餐饮", "市场", "竞争", "内卷", "战略", "团队", "创始人", "决策", "初心",
    "品牌", "连锁", "成本", "效率", "利润", "定位", "价值", "特色", "体验",
    "顾客", "客户", "消费者", "人群", "快餐", "中餐", "门店", "菜单", "菜品",
    "网红菜", "胖东来", "服务", "后厨", "员工", "供应链", "价格", "品质",
    "规模化", "标准化", "取舍", "选择", "大众", "高毛利", "薄利多销",
)
STOP_CHARS = set("的是了和与及在要就也而不是有都一个这个那种通过对于如果最很更让把被")


@dataclass(frozen=True)
class Window:
    family: str
    index: int | str
    start: float
    end: float
    item: dict[str, Any]
    source: str = "composition"

    @property
    def location(self) -> str:
        return f"{self.source}.{self.family}[{self.index}]"


@dataclass
class Finding:
    id: str
    severity: str
    check: str
    message: str
    owner_agent: str
    location: str
    repair: str
    evidence: dict[str, Any]


class Auditor:
    def __init__(
        self,
        composition: dict[str, Any],
        director: dict[str, Any],
        project: Path,
        workflow: dict[str, Any] | None = None,
    ):
        self.composition = composition
        self.director = director
        self.project = project
        self.workflow = workflow or {}
        self.duration = number(composition.get("duration"))
        self.findings: list[Finding] = []
        self._finding_counter = 0
        self.stats: Counter[str] = Counter()

    def add(
        self,
        severity: str,
        check: str,
        message: str,
        owner: str,
        location: str,
        repair: str,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        self._finding_counter += 1
        self.findings.append(Finding(
            id=f"QA-{self._finding_counter:03d}",
            severity=severity,
            check=check,
            message=message,
            owner_agent=owner,
            location=location,
            repair=repair,
            evidence=evidence or {},
        ))
        self.stats[check] += 1

    def run(self) -> dict[str, Any]:
        if self.duration is None or self.duration <= 0:
            self.add(
                "error", "time_range", "成片 duration 缺失或不是正数。", OWNER_TIMELINE,
                "composition.duration", "写入经过剪辑映射后的正数时长。",
                {"value": self.composition.get("duration")},
            )
        self.check_ranges()
        self.check_composition_exclusivity()
        self.check_director_exclusivity()
        self.check_media_assets_and_pip()
        self.check_scene_binding()
        self.check_asset_reuse()
        self.check_scene_packages_v2()
        self.check_completion_quality()
        severity_counts = Counter(item.severity for item in self.findings)
        passed = severity_counts["error"] == 0
        return {
            "schema_version": "video-workflow-qa/1.1",
            "pass": passed,
            "status": "pass" if passed else "fail",
            "summary": {
                "errors": severity_counts["error"],
                "warnings": severity_counts["warning"],
                "infos": severity_counts["info"],
                "findings": len(self.findings),
                "duration": self.duration,
                "checks_with_findings": dict(sorted(self.stats.items())),
                "repair_owners": dict(sorted(Counter(
                    item.owner_agent for item in self.findings if item.severity in {"error", "warning"}
                ).items())),
            },
            "findings": [asdict(item) for item in self.findings],
        }

    def check_ranges(self) -> None:
        groups = (
            ("hook", singleton(self.composition.get("hook"))),
            ("chapters", list_of_dicts(self.composition.get("chapters"))),
            ("visual_beats", list_of_dicts(self.composition.get("visual_beats"))),
            ("lower_thirds", list_of_dicts(self.composition.get("lower_thirds"))),
            ("camera_beats", list_of_dicts(self.composition.get("camera_beats"))),
            ("captions", list_of_dicts(self.composition.get("captions"))),
        )
        for family, items in groups:
            for index, item in enumerate(items):
                location = f"composition.{family}[{index}]"
                start = number(item.get("start"))
                end = cue_end(item)
                if start is None or end is None or not math.isfinite(start) or not math.isfinite(end):
                    self.add(
                        "error", "time_range", "时间字段缺失或不是有限数字。", OWNER_TIMELINE,
                        location, "补齐 start 与 duration/end，并使用秒为单位的数字。",
                        {"start": item.get("start"), "duration": item.get("duration"), "end": item.get("end")},
                    )
                    continue
                if start < -EPSILON or end <= start + EPSILON or (self.duration and end > self.duration + EPSILON):
                    self.add(
                        "error", "time_range", "时间段超出成片范围或长度无效。", OWNER_TIMELINE,
                        location, "重新吸附到成片时间轴，并裁剪到 [0, duration] 内。",
                        {"start": start, "end": end, "composition_duration": self.duration},
                    )
                spoken_end = number(item.get("spoken_end"))
                if spoken_end is not None and (spoken_end < start - EPSILON or spoken_end > end + 1.25):
                    self.add(
                        "warning", "speech_alignment", "spoken_end 与视觉 cue 的时间窗不一致。", OWNER_CONTENT,
                        location, "用字级时间戳重新吸附 cue，视觉尾部最多保留约 1.25 秒。",
                        {"start": start, "visual_end": end, "spoken_end": spoken_end},
                    )

    def check_composition_exclusivity(self) -> None:
        hook = windows("hook", singleton(self.composition.get("hook")))
        chapters = windows("chapters", list_of_dicts(self.composition.get("chapters")))
        beats = windows("visual_beats", list_of_dicts(self.composition.get("visual_beats")))
        lower = windows("lower_thirds", list_of_dicts(self.composition.get("lower_thirds")))

        for left, right in overlap_pairs(hook + chapters + beats):
            if left.family == right.family == "visual_beats":
                severity = "error"
                reason = "两个 Callout/素材场景同时占用主视觉。"
            elif {left.family, right.family} <= {"hook", "chapters", "visual_beats"}:
                severity = "error"
                reason = "Hook、Callout/素材与全屏章节必须互斥。"
            else:
                continue
            self.add(
                severity, "exclusive_cues", reason, OWNER_TIMELINE,
                f"{left.location} <> {right.location}",
                "由总导演确定唯一主视觉，装配器移动、缩短或删除优先级较低的 cue。",
                overlap_evidence(left, right),
            )

        for card in lower:
            for primary in hook + chapters:
                if overlaps(card, primary):
                    self.add(
                        "error", "exclusive_cues", "身份条不能覆盖 Hook 或全屏章节。", OWNER_MOTION,
                        f"{card.location} <> {primary.location}",
                        "把身份条移动到纯口播段，或在全屏主视觉期间隐藏。",
                        overlap_evidence(card, primary),
                    )
            for beat in beats:
                if overlaps(card, beat):
                    intentional_opening_identity = beat.item.get("kind") == "media" and card.start < 4.2
                    self.add(
                        "info" if intentional_opening_identity else "warning", "overlay_collision",
                        "开场身份条叠加在已验收素材的安全区。" if intentional_opening_identity else "身份条与 Callout/素材场景同时出现，可能争夺注意力。", OWNER_MOTION,
                        f"{card.location} <> {beat.location}",
                        "已作为开场人物背书保留。" if intentional_opening_identity else "检查版式安全区；优先将身份条移动到无 Callout 的口播段。",
                        overlap_evidence(card, beat),
                    )

    def check_director_exclusivity(self) -> None:
        hook = windows("hook", singleton(self.director.get("hook")), source="director")
        chapters = windows("chapters", list_of_dicts(self.director.get("chapters")), source="director")
        callouts = windows("callouts", list_of_dicts(self.director.get("callouts")), source="director")
        for left, right in overlap_pairs(hook + chapters + callouts):
            self.add(
                "warning", "director_exclusivity",
                "导演候选计划同时安排了 Hook、Callout 或章节；最终装配必须记录取舍。", OWNER_DIRECTOR,
                f"{left.location} <> {right.location}",
                "总导演必须在计划阶段给主视觉排他优先级，不能依赖装配阶段静默丢弃。",
                overlap_evidence(left, right),
            )

    def check_media_assets_and_pip(self) -> None:
        cues: list[tuple[str, dict[str, Any]]] = []
        hook = self.composition.get("hook")
        if isinstance(hook, dict):
            cues.append(("composition.hook", hook))
        cues.extend(
            (f"composition.visual_beats[{index}]", item)
            for index, item in enumerate(list_of_dicts(self.composition.get("visual_beats")))
        )
        for location, cue in cues:
            asset = cue.get("asset")
            is_media = cue.get("kind") == "media"
            pip_requested = bool(
                cue.get("speaker_pip") or cue.get("show_speaker_pip") or cue.get("pip")
            )
            # Current editorial renderer implies PiP whenever a Hook or media cue has an asset.
            pip_implied = bool(asset) and (location == "composition.hook" or is_media)
            if is_media and not asset:
                self.add(
                    "error", "media_asset", "media cue 没有配套 asset，不能进入装配。", OWNER_MEDIA,
                    location, "素材导演补齐已验证的图片/视频；失败时将 cue 降级为 quote 并禁止 PiP。",
                    {"kind": cue.get("kind"), "text": cue_text(cue)},
                )
            if asset and not real_asset(asset, self.project):
                self.add(
                    "error", "media_asset", "asset 记录存在，但文件缺失、为空或类型不受支持。", OWNER_MEDIA,
                    f"{location}.asset", "重新取材并在装配前验证 path、kind 与文件大小。",
                    {"asset": asset},
                )
            if pip_requested and not real_asset(asset, self.project):
                self.add(
                    "error", "pip_dependency", "主讲人 PiP 被请求，但没有真实背景素材。", OWNER_MOTION,
                    location, "只有在背景素材通过验收后才启用 PiP；否则回退为全画幅口播。",
                    {"pip_requested": True, "asset": asset},
                )
            if pip_implied and not real_asset(asset, self.project):
                self.add(
                    "error", "pip_dependency", "隐式 PiP 场景的背景素材不可用。", OWNER_MOTION,
                    location, "素材验证成功后再创建圆形主讲人层。", {"asset": asset},
                )

    def check_scene_binding(self) -> None:
        scenes = windows("scene_assets", list_of_dicts(self.director.get("scene_assets")), source="director")
        smart_media_enabled = bool(self.composition.get("smart_media_enabled", True))
        media_cues = [
            item for item in windows("visual_beats", list_of_dicts(self.composition.get("visual_beats")))
            if item.item.get("kind") == "media"
        ]
        hook_items = windows("hook", singleton(self.composition.get("hook")))
        if hook_items and hook_items[0].item.get("asset"):
            media_cues.append(hook_items[0])

        matched_scene_locations: set[str] = set()
        for cue in media_cues:
            candidates = [scene for scene in scenes if expanded_overlap(cue, scene, 0.85)]
            if not candidates:
                self.add(
                    "error", "scene_binding", "素材 cue 找不到同时间段的导演 scene。", OWNER_DIRECTOR,
                    cue.location, "总导演为 cue 指定 scene_id，装配器按 ID 绑定，禁止按文件顺序绑定。",
                    {"cue_start": cue.start, "cue_end": cue.end, "cue_text": cue_text(cue.item)},
                )
                continue
            ranked = sorted(
                ((semantic_score(cue.item, scene.item), scene) for scene in candidates),
                key=lambda pair: (pair[0], overlap_amount(cue, pair[1])), reverse=True,
            )
            score, scene = ranked[0]
            matched_scene_locations.add(scene.location)
            if score <= 0:
                self.add(
                    "error", "scene_semantics", "素材画面与导演场景没有可解释的语义关联。", OWNER_MEDIA,
                    f"{cue.location} -> {scene.location}",
                    "素材导演根据 spoken_quote/scene caption 重新取材；保留 scene_id 和命中关键词作为证据。",
                    {
                        "cue_text": cue_text(cue.item),
                        "scene_text": cue_text(scene.item),
                        "cue_window": [cue.start, cue.end],
                        "scene_window": [scene.start, scene.end],
                    },
                )
            elif score < 2:
                self.add(
                    "warning", "scene_semantics", "素材与场景只有较弱的语义命中，需要人工复核。", OWNER_MEDIA,
                    f"{cue.location} -> {scene.location}",
                    "补充更具体的 caption/prompt，或换成能直接证明当前口播的素材。",
                    {"semantic_score": score, "cue_text": cue_text(cue.item), "scene_text": cue_text(scene.item)},
                )

        workflow_scene_cues = [item for item in workflow_cues(self.workflow) if str(item.get("cue_type", "")) == "scene"]
        for scene in scenes:
            if scene.location not in matched_scene_locations:
                resolution = ""
                if 0 <= scene.index < len(workflow_scene_cues):
                    resolution = str(dict(workflow_scene_cues[scene.index].get("metadata") or {}).get("resolution", ""))
                if resolution.startswith("merged_into_"):
                    self.add(
                        "info", "scene_delivery", "导演场景已合并到同时间段的更高优先级主视觉。", OWNER_DIRECTOR,
                        scene.location, "无需重复叠加素材；保留 merged_package_id 供追溯。",
                        {"resolution": resolution, "scene_text": cue_text(scene.item), "window": [scene.start, scene.end]},
                    )
                    continue
                if not smart_media_enabled:
                    self.add(
                        "info", "scene_delivery", "用户未开启智能配套素材，本场景仅保留为导演建议。", OWNER_MEDIA,
                        scene.location, "无需补齐真实素材；若后续开启智能配套素材，再按场景提示词取材并验收。",
                        {"resolution": "media_disabled_by_user", "scene_text": cue_text(scene.item), "window": [scene.start, scene.end]},
                    )
                    continue
                if resolution == "semantic_graphic_fallback":
                    complete_fallback = next((
                        cue for cue in windows("visual_beats", list_of_dicts(self.composition.get("visual_beats")))
                        if cue.item.get("kind") == "context" and str(cue.item.get("speaker_role", "full")) == "full"
                        and expanded_overlap(cue, scene, 1.2)
                    ), None)
                    self.add(
                        "info" if complete_fallback else "warning", "scene_delivery",
                        "真实素材未通过验收，已用完整语义图形场景交付。" if complete_fallback else "真实素材未通过验收，已用语义图形场景完整降级。", OWNER_MEDIA,
                        scene.location, "如需真实画面，补充或生成素材并通过像素级语义验收后再替换。",
                        {"resolution": resolution, "scene_text": cue_text(scene.item), "window": [scene.start, scene.end]},
                    )
                    continue
                self.add(
                    "warning", "scene_delivery", "导演要求的场景没有进入最终时间线。", OWNER_MEDIA,
                    scene.location,
                    "确认是取材失败、被互斥规则删除还是主动降级；将结果和原因回传总导演。",
                    {"scene_text": cue_text(scene.item), "window": [scene.start, scene.end]},
                )

    def check_asset_reuse(self) -> None:
        uses: defaultdict[str, list[Window]] = defaultdict(list)
        candidates = windows("visual_beats", list_of_dicts(self.composition.get("visual_beats")))
        candidates.extend(windows("hook", singleton(self.composition.get("hook"))))
        for cue in candidates:
            asset = cue.item.get("asset")
            path = asset.get("path") if isinstance(asset, dict) else None
            if path:
                uses[str(path)].append(cue)
        for path, cue_uses in uses.items():
            if len(cue_uses) < 2:
                continue
            for left, right in all_pairs(cue_uses):
                score = semantic_score(left.item, right.item)
                if score <= 0 and abs(left.start - right.start) > 1:
                    self.add(
                        "error", "asset_reuse", "同一素材被复用于两个不相干的场景。", OWNER_MEDIA,
                        f"{left.location} <> {right.location}",
                        "为每个 scene_id 提供独立素材；只有语义一致且总导演明确批准时才允许复用。",
                        {
                            "asset_path": path,
                            "left_text": cue_text(left.item),
                            "right_text": cue_text(right.item),
                            "semantic_score": score,
                        },
                    )

    def check_completion_quality(self) -> None:
        """Block structurally valid previews that still read as unfinished demos."""
        speaker = str(self.composition.get("speaker", "")).strip()
        speaker_title = str(self.composition.get("speaker_title", "")).strip()
        lower = list_of_dicts(self.composition.get("lower_thirds"))
        if not speaker or not speaker_title or not lower:
            self.add(
                "error", "identity_delivery", "主讲人姓名、身份标签或人物信息动效未完整交付。", OWNER_CONTENT,
                "composition.identity", "从用户输入或明确自我介绍中取得姓名；职衔无可靠来源时使用非资历性的节目角色标签，并生成开场身份条。",
                {"speaker": speaker, "speaker_title": speaker_title, "lower_thirds": len(lower)},
            )

        scenes = list_of_dicts(self.director.get("scene_assets"))
        media_beats = [
            item for item in list_of_dicts(self.composition.get("visual_beats"))
            if item.get("kind") == "media" and real_asset(item.get("asset"), self.project)
        ]
        hook = self.composition.get("hook") if isinstance(self.composition.get("hook"), dict) else {}
        if hook.get("asset") and real_asset(hook.get("asset"), self.project):
            media_beats.append(hook)
        if bool(self.composition.get("smart_media_enabled", True)) and scenes and not media_beats:
            self.add(
                "error", "scene_delivery", "导演规划了真实配套素材，但最终采用数为 0，当前只能视为降级预览。", OWNER_MEDIA,
                "composition.visual_beats", "按完整场景提示词重新生成/检索并通过真实像素验收；没有素材时禁止用空泛场景标签冒充成片。",
                {"planned_scenes": len(scenes), "adopted_media": 0},
            )

        chapter_titles = [str(item.get("title", "")) for item in list_of_dicts(self.composition.get("chapters"))]
        if any("第二" in title for title in chapter_titles) and not any("第一" in title for title in chapter_titles):
            self.add(
                "error", "chapter_structure", "章节只出现“第二点”，论证结构不完整。", OWNER_DIRECTOR,
                "composition.chapters", "补齐第一论点章节，或同时取消数字编号。", {"chapters": chapter_titles},
            )

        suspicious = (
            "大众名声", "高可担", "玻璃多销", "人群食物", "监视不下去", "为命题",
            "胖冻来", "人群一百", "好养毛", "下排桌", "中间两头站多头", "将来打击",
        )
        bad_captions = [
            {"start": item.get("start"), "text": str(item.get("text", ""))}
            for item in list_of_dicts(self.composition.get("captions"))
            if any(term in str(item.get("text", "")) for term in suspicious)
        ]
        if bad_captions:
            self.add(
                "error", "caption_semantics", "字幕仍包含明显行业术语错识别，不能作为成片交付。", OWNER_CONTENT,
                "composition.captions", "重新校订低置信片段，并让内容分析基于校正稿重新运行。", {"samples": bad_captions[:6]},
            )

        primary = sorted(
            [*windows("visual_beats", list_of_dicts(self.composition.get("visual_beats"))),
             *windows("chapters", list_of_dicts(self.composition.get("chapters"))),
             *windows("hook", singleton(self.composition.get("hook"))),
             *windows("lower_thirds", list_of_dicts(self.composition.get("lower_thirds"))),
             *windows("camera_beats", list_of_dicts(self.composition.get("camera_beats")))],
            key=lambda item: item.start,
        )
        if self.duration and primary:
            gaps: list[tuple[float, float]] = []
            cursor = 0.0
            for item in primary:
                if item.start - cursor > 12:
                    gaps.append((round(cursor, 2), round(item.start, 2)))
                cursor = max(cursor, item.end)
            if self.duration - cursor > 12:
                gaps.append((round(cursor, 2), round(self.duration, 2)))
            if gaps:
                self.add(
                    "warning", "visual_rhythm", "存在超过 12 秒没有内容化视觉变化的口播死区。", OWNER_MOTION,
                    "composition.visual_beats", "在论证机制、案例或对比处增加信息图/素材，保持每 3–8 秒有一次有意义变化。", {"gaps": gaps},
                )

    def check_scene_packages_v2(self) -> None:
        """Validate atomic V2 packages without making legacy artifacts unusable."""
        packages, package_source = workflow_packages(self.workflow, self.composition)
        cues = workflow_cues(self.workflow)
        cue_by_id = {str(item.get("cue_id", "")): item for item in cues if item.get("cue_id")}
        checklist = workflow_checklist(self.workflow)

        if not packages:
            self.check_legacy_fullscreen(cues)
            self.add(
                "warning", "scene_package_v2",
                "当前产物没有 V2 scene_packages，无法验证组件级进出场原子性。", OWNER_TIMELINE,
                package_source,
                "下一次装配输出 scene_packages 与 assembly_checklist；旧产物仍可继续使用。",
                {"legacy_compatible": True},
            )
            return

        for index, package in enumerate(packages):
            location = f"{package_source}[{index}]"
            package_id = str(package.get("package_id", "")).strip()
            cue_id = str(package.get("cue_id", "")).strip()
            cue = cue_by_id.get(cue_id, {})
            start, end = number(package.get("start")), cue_end(package)
            speaker_role = str(package.get("speaker_role", "full")).lower()
            pip_like = speaker_role in {"circle", "card", "cutout"}
            background_role = str(package.get("background_role", "source")).lower()
            media_role = str(package.get("media_role", "none")).lower()
            media = package.get("media")
            status = str(package.get("status", "proposed")).lower()

            missing_identity = [name for name, value in (("package_id", package_id), ("cue_id", cue_id)) if not value]
            if missing_identity:
                self.add(
                    "error", "scene_package_completeness", "scene package 缺少稳定身份字段。", OWNER_TIMELINE,
                    location, "补齐 package_id 与 cue_id，装配和返修必须使用同一个 cue 关联。",
                    {"missing": missing_identity},
                )
            if cue_id and cues and not cue:
                self.add(
                    "error", "scene_package_binding", "scene package 引用了不存在的 cue_id。", OWNER_DIRECTOR,
                    location, "重新绑定 manifest.cues 中的有效 cue_id，禁止仅按时间猜测。",
                    {"cue_id": cue_id},
                )
            if start is None or end is None or end <= start or start < 0 or (self.duration and end > self.duration + EPSILON):
                self.add(
                    "error", "scene_package_timing", "scene package 的完整时间窗缺失或越界。", OWNER_TIMELINE,
                    location, "填写 start/end，并限制在成片时间轴内。",
                    {"start": package.get("start"), "end": package.get("end"), "duration": self.duration},
                )
            elif cue:
                cue_start, cue_end_value = number(cue.get("start")), cue_end(cue)
                if cue_start is not None and cue_end_value is not None and (
                    abs(start - cue_start) > .35 or abs(end - cue_end_value) > .55
                ):
                    self.add(
                        "error", "scene_package_timing", "package 时间窗没有与所属 cue 对齐。", OWNER_TIMELINE,
                        location, "以 cue 的字级时间戳为基准重新生成 package 时间窗。",
                        {"package_window": [start, end], "cue_window": [cue_start, cue_end_value]},
                    )

            semantic_payload = cue_text(package) or cue_text(cue) or " ".join(str(value) for value in package.get("semantic_tags", []))
            if not semantic_payload.strip():
                self.add(
                    "error", "scene_package_semantics", "scene package 没有语义意图或可追溯文字。", OWNER_CONTENT,
                    location, "写入 spoken_quote/semantic_intent/text 或 semantic_tags，供素材与口播复核。",
                    {"cue_id": cue_id},
                )

            media_required = background_role == "media" or media_role != "none"
            if media_required and not v2_ready_asset(media, self.project):
                self.add(
                    "error", "scene_package_asset", "scene package 需要素材，但 media 未达到 ready 且可渲染状态。", OWNER_MEDIA,
                    location, "绑定 status=ready 的本地图片/视频并完成语义验收；否则降级整个 package。",
                    {"background_role": background_role, "media_role": media_role, "media": media},
                )
            if media_required and isinstance(media, dict):
                media_metadata = dict(media.get("metadata") or {})
                score = number(media.get("semantic_score"))
                visual_validation = str(media_metadata.get("visual_validation", "")).lower()
                semantic_evidence = bool(
                    score is not None or media.get("semantic_tags")
                    or media_metadata.get("semantic_approved") or media_metadata.get("semantic_evidence")
                )
                contradictions = list(media_metadata.get("detected_contradictions") or [])
                forbidden_hits = list(media_metadata.get("forbidden_visual_hits") or [])
                if not semantic_evidence:
                    self.add(
                        "error", "scene_package_semantics", "素材缺少语义匹配验收证据。", OWNER_MEDIA,
                        location, "写入 semantic_score/tags 或 semantic_evidence，证明素材支持当前口播。",
                        {"asset_id": media.get("asset_id", "")},
                    )
                elif visual_validation != "vision-approved":
                    self.add(
                        "error", "scene_package_semantics", "素材只完成了元数据路由，没有经过真实画面验收。", OWNER_MEDIA,
                        location, "抽取实际图片/视频帧做视觉语义与矛盾检查；通过后写入 visual_validation=vision-approved。",
                        {"visual_validation": visual_validation or "missing", "semantic_evidence": media_metadata.get("semantic_evidence")},
                    )
                elif score is not None and score < .78:
                    self.add(
                        "error", "scene_package_semantics", "素材语义匹配分低于装配门槛。", OWNER_MEDIA,
                        location, "重新取材或生成，直至语义分达到 0.78 且不存在画面矛盾。",
                        {"semantic_score": score, "threshold": .78},
                    )
                if contradictions or forbidden_hits:
                    self.add(
                        "error", "scene_package_semantics", "素材命中画面矛盾或禁用视觉。", OWNER_MEDIA,
                        location, "替换素材并重新执行矛盾检测，禁止带问题素材进入场景包。",
                        {"contradictions": contradictions, "forbidden_hits": forbidden_hits},
                    )
            if status in {"ready", "degraded"} and media_required and not isinstance(media, dict):
                self.add(
                    "error", "scene_package_asset", "已交付 package 声明需要素材，但没有 media 对象。", OWNER_MEDIA,
                    location, "素材失败时不得只交付人物组件，应整体降级或跳过。", {"status": status},
                )

            self.check_package_pip(package, location, checklist)
            self.check_package_exit(package, location, checklist)
            self.check_package_fullscreen(package, cue, location)
            self.check_package_caption_safety(package, cue, location)

    def check_package_pip(
        self, package: dict[str, Any], location: str, checklist: list[dict[str, Any]],
    ) -> None:
        speaker_role = str(package.get("speaker_role", "full")).lower()
        if speaker_role not in {"circle", "card", "cutout"}:
            return
        media = package.get("media")
        background_role = str(package.get("background_role", "source")).lower()
        dependency_status = dict(package.get("dependency_status") or {})
        graphic_backgrounds = {"source_blur", "graphic", "solid", "editorial_canvas"}
        background_ready = dependency_status.get("background.ready") is not False and (
            (background_role == "media" and v2_ready_asset(media, self.project))
            or background_role in graphic_backgrounds
        )
        if not background_ready:
            self.add(
                "error", "scene_package_pip", "circle/card/cutout 人物没有绑定 ready 的完整背景场景。", OWNER_MOTION,
                location, "先完成素材背景或明确的 graphic/source_blur 背景并通过 background.ready，再挂载人物。",
                {
                    "speaker_role": speaker_role,
                    "background_role": background_role,
                    "background_ready": dependency_status.get("background.ready"),
                },
            )

        entry = package_entry_evidence(package, checklist)
        delta = entry.get("delta")
        explicit_order = bool(entry.get("ordered"))
        if delta is not None and not (.08 <= delta <= .22) and not explicit_order:
            self.add(
                "error", "scene_package_entry_order", "背景没有比人物 PiP 提前约 0.1–0.2 秒进入。", OWNER_MOTION,
                location, "把背景 entry_at 提前 0.1–0.2 秒，或声明 background→speaker 的 entry_order。",
                entry,
            )
        elif entry.get("declared") and not explicit_order and delta is None:
            self.add(
                "error", "scene_package_entry_order", "已声明的 entry_order 没有保证背景先于人物 PiP。", OWNER_MOTION,
                location, "按 background→speaker 排序，或让 speaker step 显式 after=background.ready。",
                entry,
            )
        elif delta is None and not explicit_order:
            self.add(
                "warning", "scene_package_entry_order", "PiP package 未提供可验证的入场顺序。", OWNER_MOTION,
                location, "输出 entry_order/dependency_order，或记录 background_at 与 speaker_at。",
                {**entry, "legacy_compatible": True},
            )
        text_role = str(package.get("text_role", "none")).lower()
        text_at = component_time(package, "text")
        speaker_at = component_time(package, "speaker")
        if text_role != "none" and text_at is not None and speaker_at is not None and text_at < speaker_at + .07:
            self.add(
                "error", "scene_package_entry_order", "主视觉文字早于人物 PiP 完成入场。", OWNER_MOTION,
                location, "把文字 entry_at 放到人物之后至少 0.07 秒，保持背景→人物→文字。",
                {"speaker_at": speaker_at, "text_at": text_at, "delta": round(text_at - speaker_at, 3)},
            )

    def check_package_exit(
        self, package: dict[str, Any], location: str, checklist: list[dict[str, Any]],
    ) -> None:
        if str(package.get("speaker_role", "full")).lower() not in {"circle", "card", "cutout"} and str(package.get("media_role", "none")) == "none":
            return
        evidence = package_exit_evidence(package, checklist)
        if evidence.get("explicit") and not evidence.get("safe"):
            self.add(
                "error", "scene_package_exit_order", "scene package 的退出顺序可能残留人物、文字或素材层。", OWNER_MOTION,
                location, "先清理文字/PiP，再卸载素材背景，最后显式 restore_source_and_full_speaker。",
                evidence,
            )
        elif not evidence.get("explicit"):
            self.add(
                "warning", "scene_package_exit_order", "scene package 没有声明退出清理与源画面恢复。", OWNER_MOTION,
                location, "输出 exit_order 或 package exit step，并声明 restore_source_and_full_speaker。",
                {**evidence, "legacy_compatible": True},
            )

    def check_package_fullscreen(self, package: dict[str, Any], cue: dict[str, Any], location: str) -> None:
        metadata = package.get("metadata") if isinstance(package.get("metadata"), dict) else {}
        visual_mode = str(package.get("visual_mode") or metadata.get("visual_mode") or "").lower()
        text_role = str(package.get("text_role", "none")).lower()
        is_fullscreen = bool(package.get("fullscreen", False)) or text_role == "chapter" or "fullscreen" in visual_mode
        if not is_fullscreen:
            return
        cue_type = str(cue.get("cue_type") or package.get("cue_type") or metadata.get("cue_type") or "")
        start, end = number(package.get("start")), cue_end(package)
        duration = (end - start) if start is not None and end is not None else None
        if cue_type != "chapter":
            self.add(
                "error", "fullscreen_policy", "fullscreen 只能用于 chapter package。", OWNER_DIRECTOR,
                location, "把普通场景改为半屏/背景素材；全屏只保留章节切换。",
                {"cue_type": cue_type, "duration": duration},
            )
        if duration is not None and not (1.2 - EPSILON <= duration <= 2.2 + EPSILON):
            self.add(
                "error", "fullscreen_duration", "V2 全屏章节时长不在 1.2–2.2 秒。", OWNER_TIMELINE,
                location, "将全屏章节压缩到 1.2–2.2 秒并重新检查语音衔接。",
                {"duration": round(duration, 3)},
            )

    def check_legacy_fullscreen(self, cues: list[dict[str, Any]]) -> None:
        legacy: list[tuple[str, dict[str, Any]]] = [
            (f"workflow.cues[{index}]", cue) for index, cue in enumerate(cues) if cue.get("fullscreen")
        ]
        if not legacy:
            legacy = [
                (f"composition.chapters[{index}]", chapter)
                for index, chapter in enumerate(list_of_dicts(self.composition.get("chapters")))
            ]
        for location, cue in legacy:
            start, end = number(cue.get("start")), cue_end(cue)
            duration = end - start if start is not None and end is not None else None
            cue_type = str(cue.get("cue_type", "chapter"))
            if cue_type != "chapter" or duration is None or not (1.2 - EPSILON <= duration <= 2.2 + EPSILON):
                self.add(
                    "warning", "legacy_fullscreen_policy", "旧版全屏 cue 不符合 V2 的 chapter/1.2–2.2 秒规则。", OWNER_TIMELINE,
                    location, "下次重新装配时迁移为 V2 短章节 package；不阻断本次旧产物。",
                    {"cue_type": cue_type, "duration": duration, "legacy_compatible": True},
                )

    def check_package_caption_safety(
        self, package: dict[str, Any], cue: dict[str, Any], location: str,
    ) -> None:
        cue_type = str(cue.get("cue_type") or package.get("cue_type") or "")
        caption_mode = str(package.get("caption_mode", "normal")).lower()
        metadata = dict(package.get("metadata") or {})
        covered = {
            str(item).lower() for item in (
                package.get("covered_zones") or metadata.get("covered_zones") or metadata.get("occluded_zones") or []
            )
        }
        caption_ratio = first_number(
            package.get("caption_occlusion_ratio"), metadata.get("caption_occlusion_ratio"),
            nested_value(package.get("caption_zone"), "occlusion_ratio"),
        )
        highlight_ratio = first_number(
            package.get("highlight_occlusion_ratio"), metadata.get("highlight_occlusion_ratio"),
            nested_value(package.get("highlight_zone"), "occlusion_ratio"),
        )
        caption_hidden = caption_mode == "hidden" or "caption" in covered or zone_hidden(package.get("caption_zone"))
        highlight_hidden = "highlight" in covered or zone_hidden(package.get("highlight_zone"))
        text_role = str(package.get("text_role", "")).lower()
        start, end = number(package.get("start")), cue_end(package)
        takeover_duration = (end - start) if start is not None and end is not None else None
        deliberate_text_takeover = (
            cue_type in {"hook", "callout"}
            and text_role in {"hook", "callout", "compare", "stat"}
            and takeover_duration is not None and takeover_duration <= 5.3
        )
        if cue_type != "chapter" and not deliberate_text_takeover and (caption_hidden or (caption_ratio is not None and caption_ratio >= .98)):
            self.add(
                "error", "caption_safety", "场景组件完全遮挡或隐藏了字幕安全区。", OWNER_MOTION,
                location, "改用 compact/above_safe_area，或缩小素材与人物层，保留字幕可读区。",
                {"caption_mode": caption_mode, "covered_zones": sorted(covered), "occlusion_ratio": caption_ratio},
            )
        if highlight_hidden or (highlight_ratio is not None and highlight_ratio >= .98):
            self.add(
                "error", "highlight_safety", "场景组件完全遮挡了重点高亮区。", OWNER_MOTION,
                location, "移动 Callout/人物/素材边界，为高亮词保留独立安全区。",
                {"covered_zones": sorted(covered), "occlusion_ratio": highlight_ratio},
            )
        has_safety_evidence = any(
            key in package or key in metadata for key in (
                "caption_zone", "highlight_zone", "covered_zones", "occluded_zones",
                "caption_occlusion_ratio", "highlight_occlusion_ratio", "safe_zones",
            )
        )
        if not has_safety_evidence and (
            str(package.get("background_role", "source")) == "media"
            or str(package.get("speaker_role", "full")) in {"circle", "card", "cutout"}
        ):
            self.add(
                "warning", "caption_safety", "scene package 没有字幕/高亮安全区验收证据。", OWNER_MOTION,
                location, "输出 caption_zone、highlight_zone 或遮挡比例；旧 package 暂不阻断。",
                {"legacy_compatible": True},
            )


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def singleton(value: Any) -> list[dict[str, Any]]:
    return [value] if isinstance(value, dict) and value else []


def list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def cue_end(item: dict[str, Any]) -> float | None:
    start = number(item.get("start"))
    if start is None:
        return None
    if number(item.get("duration")) is not None:
        return start + float(item["duration"])
    return number(item.get("end"))


def windows(family: str, items: list[dict[str, Any]], source: str = "composition") -> list[Window]:
    result: list[Window] = []
    for index, item in enumerate(items):
        start, end = number(item.get("start")), cue_end(item)
        if start is not None and end is not None and end > start:
            result.append(Window(family, index, start, end, item, source))
    return result


def overlaps(left: Window, right: Window, tolerance: float = EPSILON) -> bool:
    return min(left.end, right.end) - max(left.start, right.start) > tolerance


def expanded_overlap(left: Window, right: Window, tolerance: float) -> bool:
    return min(left.end + tolerance, right.end + tolerance) >= max(left.start - tolerance, right.start - tolerance)


def overlap_amount(left: Window, right: Window) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def overlap_pairs(items: list[Window]) -> Iterable[tuple[Window, Window]]:
    for left, right in all_pairs(items):
        if overlaps(left, right):
            yield left, right


def all_pairs(items: list[Window]) -> Iterable[tuple[Window, Window]]:
    for index, left in enumerate(items):
        for right in items[index + 1:]:
            yield left, right


def overlap_evidence(left: Window, right: Window) -> dict[str, Any]:
    return {
        "left": {"location": left.location, "start": left.start, "end": left.end},
        "right": {"location": right.location, "start": right.start, "end": right.end},
        "overlap_seconds": round(overlap_amount(left, right), 3),
    }


def cue_text(item: dict[str, Any]) -> str:
    return " ".join(str(item.get(key, "")) for key in (
        "text", "spoken_quote", "editorial_text", "semantic_intent", "caption", "query",
        "scene_prompt", "prompt", "title",
    ) if item.get(key)).strip()


def semantic_tokens(item: dict[str, Any]) -> set[str]:
    text = re.sub(r"\s+", "", cue_text(item).lower())
    tokens = {term for term in SEMANTIC_TERMS if term in text}
    # Chinese bigrams give coverage for topics not present in the curated business vocabulary.
    cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)
    tokens.update(
        cleaned[index:index + 2]
        for index in range(max(0, len(cleaned) - 1))
        if not any(char in STOP_CHARS for char in cleaned[index:index + 2])
    )
    return tokens


def semantic_score(left: dict[str, Any], right: dict[str, Any]) -> int:
    shared = semantic_tokens(left) & semantic_tokens(right)
    return sum(2 if token in SEMANTIC_TERMS else 1 for token in shared)


def resolve_asset_path(asset: dict[str, Any], project: Path) -> Path | None:
    raw = asset.get("path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw)
    return path if path.is_absolute() else project / path


def real_asset(asset: Any, project: Path) -> bool:
    if not isinstance(asset, dict) or asset.get("kind") not in {"image", "video"}:
        return False
    path = resolve_asset_path(asset, project)
    if not path or not path.is_file() or path.stat().st_size <= 0:
        return False
    allowed = {
        "image": {".jpg", ".jpeg", ".png", ".webp", ".avif"},
        "video": {".mp4", ".mov", ".m4v", ".webm"},
    }
    return path.suffix.lower() in allowed[str(asset["kind"])]


def v2_ready_asset(asset: Any, project: Path) -> bool:
    return isinstance(asset, dict) and str(asset.get("status", "")).lower() == "ready" and real_asset(asset, project)


def workflow_packages(
    workflow: dict[str, Any], composition: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    direct = list_of_dicts(workflow.get("scene_packages"))
    if direct:
        return direct, "workflow.scene_packages"
    manifest = workflow.get("manifest") if isinstance(workflow.get("manifest"), dict) else {}
    frozen = list_of_dicts(manifest.get("scene_packages"))
    if frozen:
        return frozen, "workflow.manifest.scene_packages"
    embedded = list_of_dicts(composition.get("scene_packages"))
    if embedded:
        return embedded, "composition.scene_packages"
    return [], "workflow.scene_packages"


def workflow_cues(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    direct = list_of_dicts(workflow.get("cues"))
    if direct:
        return direct
    manifest = workflow.get("manifest") if isinstance(workflow.get("manifest"), dict) else {}
    return list_of_dicts(manifest.get("cues"))


def workflow_checklist(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = workflow.get("manifest") if isinstance(workflow.get("manifest"), dict) else {}
    return list_of_dicts(workflow.get("assembly_checklist")) or list_of_dicts(manifest.get("assembly_checklist"))


def package_steps(package: dict[str, Any], checklist: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = list_of_dicts(package.get("assembly_steps")) or list_of_dicts(package.get("steps"))
    package_id = str(package.get("package_id", ""))
    for item in checklist:
        if package_id and str(item.get("package_id", "")) != package_id:
            continue
        nested = list_of_dicts(item.get("steps")) or list_of_dicts(item.get("assembly_steps"))
        if nested:
            result.extend(nested)
        elif item.get("component"):
            result.append(item)
    return result


def normalize_order(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, dict):
            label = item.get("component") or item.get("role") or item.get("action") or item.get("dependency")
        else:
            label = item
        if label:
            result.append(str(label).lower())
    return result


def component_time(package: dict[str, Any], component: str) -> float | None:
    metadata = package.get("metadata") if isinstance(package.get("metadata"), dict) else {}
    values = (
        package.get(f"{component}_at"), package.get(f"{component}_start"),
        nested_value(package.get(component), "entry_at"), nested_value(package.get(component), "start"),
        metadata.get(f"{component}_at"), metadata.get(f"{component}_start"),
        nested_value(metadata.get(component), "entry_at"), nested_value(metadata.get(component), "start"),
    )
    return first_number(*values)


def package_entry_evidence(package: dict[str, Any], checklist: list[dict[str, Any]]) -> dict[str, Any]:
    order = normalize_order(package.get("entry_order"))
    dependency_order = normalize_order(package.get("dependency_order"))
    steps = sorted(package_steps(package, checklist), key=lambda item: number(item.get("order")) or 0)
    step_order = [str(item.get("component", "")).lower() for item in steps if item.get("component")]
    ordered = before_component(order, "background", "speaker") or before_component(step_order, "background", "speaker")
    if not ordered:
        ordered = before_dependency(dependency_order, "background.ready", ("speaker.ready", "pip.background_ready"))
    if not ordered:
        speaker_steps = [item for item in steps if str(item.get("component", "")).lower() == "speaker"]
        ordered = any("background.ready" in [str(value) for value in item.get("after", [])] for item in speaker_steps)
    background_at = component_time(package, "background")
    speaker_at = component_time(package, "speaker")
    delta = round(speaker_at - background_at, 3) if background_at is not None and speaker_at is not None else None
    return {
        "ordered": ordered,
        "declared": bool(order or dependency_order or steps),
        "entry_order": order,
        "dependency_order": dependency_order,
        "step_order": step_order,
        "background_at": background_at,
        "speaker_at": speaker_at,
        "delta": delta,
    }


def package_exit_evidence(package: dict[str, Any], checklist: list[dict[str, Any]]) -> dict[str, Any]:
    order = normalize_order(package.get("exit_order"))
    steps = package_steps(package, checklist)
    exit_steps = [
        item for item in steps
        if str(item.get("action", "")).lower() in {"exit", "unmount", "hide", "restore"}
        or str(item.get("component", "")).lower() in {"package", "restore_source"}
    ]
    metadata = package.get("metadata") if isinstance(package.get("metadata"), dict) else {}
    restore = bool(
        package.get("restore_source") or metadata.get("restore_source")
        or str(package.get("exit_state", "")).lower() in {"source", "source_restored", "restore_source_and_full_speaker"}
        or any("restore_source" in str(item.get("role", "")).lower() for item in exit_steps)
        or any("restore_source" in value for value in order)
    )
    explicit = bool(order or exit_steps or "restore_source" in package or "exit_state" in package)
    speaker_index = next((index for index, value in enumerate(order) if "speaker" in value or "pip" in value), None)
    background_index = next((index for index, value in enumerate(order) if "background" in value or "media" in value), None)
    order_safe = not order or (
        background_index is not None and (speaker_index is None or speaker_index < background_index)
    )
    return {
        "explicit": explicit,
        "safe": explicit and restore and order_safe,
        "exit_order": order,
        "restore_source": restore,
        "order_safe": order_safe,
        "exit_steps": exit_steps,
    }


def before_component(order: list[str], left: str, right: str) -> bool:
    left_index = next((index for index, value in enumerate(order) if left in value), None)
    right_index = next((index for index, value in enumerate(order) if right in value or (right == "speaker" and "pip" in value)), None)
    return left_index is not None and right_index is not None and left_index < right_index


def before_dependency(order: list[str], left: str, right: tuple[str, ...]) -> bool:
    left_index = next((index for index, value in enumerate(order) if value == left), None)
    right_index = next((index for index, value in enumerate(order) if value in right), None)
    return left_index is not None and right_index is not None and left_index < right_index


def nested_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def first_number(*values: Any) -> float | None:
    return next((parsed for parsed in (number(value) for value in values) if parsed is not None), None)


def zone_hidden(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    visible = value.get("visible")
    covered = value.get("covered")
    occlusion = number(value.get("occlusion_ratio"))
    return visible is False or covered is True or (occlusion is not None and occlusion >= .98)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} 文件不存在：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} 不是合法 JSON：{path} ({exc})") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} 顶层必须是 JSON object：{path}")
    return value


def audit_files(
    composition_path: str | Path = "output/composition-data.json",
    director_plan_path: str | Path = "output/director-plan.json",
    project: str | Path = ".",
    workflow_path: str | Path | None = None,
) -> dict[str, Any]:
    """Public integration API: load workflow artifacts and return a QA report."""
    project_path = Path(project).resolve()
    composition = read_json(Path(composition_path), "composition")
    director = read_json(Path(director_plan_path), "director plan")
    candidate = Path(workflow_path) if workflow_path else project_path / "output" / "agent-workflow.json"
    workflow = read_json(candidate, "agent workflow") if candidate.is_file() else {}
    return Auditor(composition, director, project_path, workflow).run()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多 Agent 视频时间线结构化质检")
    parser.add_argument("--composition", type=Path, default=Path("output/composition-data.json"))
    parser.add_argument("--director-plan", type=Path, default=Path("output/director-plan.json"))
    parser.add_argument("--workflow", type=Path, help="V2 agent-workflow.json；默认读取项目 output 目录")
    parser.add_argument("--project", type=Path, default=Path("."), help="用于解析 asset 相对路径")
    parser.add_argument("--output", type=Path, help="同时把 JSON 报告写入该路径")
    parser.add_argument("--compact", action="store_true", help="stdout 输出紧凑 JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        report = audit_files(args.composition, args.director_plan, args.project, args.workflow)
    except (OSError, ValueError) as exc:
        print(json.dumps({"pass": False, "status": "input_error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=None if args.compact else 2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
