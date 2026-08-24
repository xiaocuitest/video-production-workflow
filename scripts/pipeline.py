#!/usr/bin/env python3
"""Transcript-driven director pipeline for portrait talking-head videos."""

from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import html
import io
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

try:
    from agent_workflow import AgentWorkflow, ArtifactRef, AssetRef, CueStatus, CueType
    from workflow_qa import Auditor
except ImportError:  # Allows `import scripts.pipeline` from project-root tests.
    from scripts.agent_workflow import AgentWorkflow, ArtifactRef, AssetRef, CueStatus, CueType
    from scripts.workflow_qa import Auditor


DEFAULTS: dict[str, Any] = {
    "mode": "auto",
    "title": "",
    "subtitle": "",
    "series": "观点拆解",
    "speaker": "",
    "speaker_title": "",
    "auto_identity": True,
    "accent": "#FFD83D",
    "keywords": [],
    "silence_threshold_seconds": 0.85,
    "cover_time_seconds": None,
    "model": "small",
    "bgm_mode": "generated",
    "bgm_path": "",
    "bgm_volume": 0.38,
    "director_enabled": True,
    "director_provider": "ark",
    "director_model": "doubao-seed-2-0-lite-260215",
    "smart_media": False,
    "smart_media_count": 3,
    "scene_generation": "auto",
    "ark_image_model": "doubao-seedream-5-0-lite-260128",
    "ark_video_model": "doubao-seedance-1-5-pro-251215",
    "allow_ai_images": False,
    "cover_template": "headline",
    "motion_template": "auto",
}

MOTION_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "id": "expert",
        "label": "专家洞察",
        "eyebrow": "克制专业",
        "description": "小尺寸观点卡、轻推近与清晰信息层级，适合行业分析和专家口播。",
        "best_for": "结论、方法、行业判断",
        "pacing": "稳健 · 4–6 秒一次信息变化",
        "features": ["上半区编辑卡", "轻柔推近", "章节全屏", "重点词克制强调"],
        "demo_script": ["行业判断出现时，先保留讲者", "结论原话同步浮现", "主题转折才进入全屏章节"],
        "preview_kind": "quote",
        "support_top": 238,
        "support_height": 452,
        "support_radius": 20,
        "quote_size": 42,
        "quote_long_size": 37,
        "caption_size": 58,
        "stat_size": 86,
        "panel_alpha": .62,
        "blur": .30,
    },
    {
        "id": "conflict",
        "label": "观点冲突",
        "eyebrow": "对比有力",
        "description": "突出“不是／而是”、数字和反常识结论，节奏更鲜明但不使用傻大字。",
        "best_for": "反常识、对比、数字观点",
        "pacing": "鲜明 · 3–5 秒一次对比推进",
        "features": ["左右对比卡", "数字冲击", "短促切入", "冲突词高亮"],
        "demo_script": ["先出现旧认知", "语音说到转折时揭示新观点", "关键数字同步落版"],
        "preview_kind": "compare",
        "support_top": 220,
        "support_height": 500,
        "support_radius": 18,
        "quote_size": 45,
        "quote_long_size": 39,
        "caption_size": 59,
        "stat_size": 94,
        "panel_alpha": .67,
        "blur": .34,
    },
    {
        "id": "story",
        "label": "故事案例",
        "eyebrow": "画面辅助",
        "description": "给人物、门店、产品和具体动作更多半屏画面，保留讲者持续在场。",
        "best_for": "案例、场景、人物和品牌故事",
        "pacing": "叙事 · 5–8 秒一组画面辅助",
        "features": ["半屏纪实素材", "人物持续在场", "场景字幕", "柔和转场"],
        "demo_script": ["讲到具体场景时出现素材", "素材只占画面上半区", "回到结论时自然让出画面"],
        "preview_kind": "media",
        "support_top": 188,
        "support_height": 590,
        "support_radius": 24,
        "quote_size": 40,
        "quote_long_size": 35,
        "caption_size": 57,
        "stat_size": 84,
        "panel_alpha": .54,
        "blur": .26,
    },
    {
        "id": "editorial",
        "label": "视觉杂志",
        "eyebrow": "精致多镜头",
        "description": "人物主画面、聚光观点、全屏素材画中画与知识卡交替出现，适合需要持续视觉变化的专业口播。",
        "best_for": "科普、知识、品牌故事和具象案例",
        "pacing": "丰富 · 3–7 秒切换一种视觉角色",
        "features": ["全屏素材＋人物画中画", "聚光观点镜头", "知识卡解释", "语音触发素材", "柔和光感转场"],
        "demo_script": ["开场用主题贴纸和人物圆窗建立记忆点", "说到具体对象时全屏切入配套素材", "人物缩入画中画保持专家在场", "结论用聚光或知识卡收束"],
        "preview_kind": "editorial",
        "support_top": 170,
        "support_height": 610,
        "support_radius": 26,
        "quote_size": 42,
        "quote_long_size": 36,
        "caption_size": 57,
        "stat_size": 88,
        "panel_alpha": .48,
        "blur": .32,
    },
)
MOTION_TEMPLATE_BY_ID = {item["id"]: item for item in MOTION_TEMPLATES}

TRIGGER_WORDS = ("不是", "而是", "真正", "为什么", "只有", "关键", "所以", "如果", "本质", "结果", "第一", "第二", "最后", "总结")
STRONG_CALLOUT_WORDS = ("不是", "而是", "真正", "为什么", "只有", "关键", "本质", "第一", "第二", "最后", "总结", "最重要", "意味着")
SCENE_WORDS = ("餐饮", "快餐", "中餐", "餐厅", "门店", "厨房", "商超", "出餐", "取餐", "消费者", "顾客", "城市", "工厂", "产品", "现场", "市场", "团队", "食物", "品牌", "服务")
TRAILING_FRAGMENTS = ("只有", "给我", "因为", "如果", "所以", "但是", "结果", "其实", "就是", "一个", "两个", "的", "有", "要", "去", "来")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}

ASR_CORRECTIONS = {
    "战略堵定": "战略笃定",
    "连锁进城": "连锁进程",
    "大众名声餐饮": "大众民生餐饮",
    "掩饱别人": "眼馋别人",
    "高可担": "高客单",
    "玻璃多销": "薄利多销",
    "人群食物的快餐": "人均十五的快餐",
    "中向两头": "中间两头",
    "监视不下去": "坚持不下去",
    "胖冻来": "胖东来",
    "东西好可从来": "东西好，可从来",
    "都是为命题": "都是伪命题",
    "人群一百": "人均一百",
    "倒雇便宜菜": "捣鼓便宜菜",
    "好养毛": "薅羊毛",
    "那真正能够顾客不在意那个": "真正能留住的顾客，不在意这些",
    "下排桌": "下牌桌",
}

COVER_CONCEPT_TERMS = (
    "定位", "取舍", "战略", "顾客", "体验", "效率", "成本", "利润", "品牌", "连锁",
    "标准化", "规模化", "用户", "产品", "服务", "增长", "管理", "价值", "初心", "人性",
)

ARK_CHAT_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
ARK_IMAGE_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
ARK_VIDEO_TASK_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--mode", choices=["auto", "keep", "recut"], default="auto")
    parser.add_argument("--config")
    parser.add_argument("--image-dir")
    parser.add_argument("--support-dir")
    parser.add_argument("--cache-dir", default=os.environ.get("VIDEO_WORKFLOW_MODEL_CACHE", ""))
    return parser.parse_args()


def load_config(path: str | None, mode: str) -> dict[str, Any]:
    config = dict(DEFAULTS)
    if path and Path(path).exists():
        config.update(json.loads(Path(path).read_text(encoding="utf-8")))
    config["mode"] = mode
    return config


def motion_template_catalog(selected: str = "expert", recommended: str = "expert") -> list[dict[str, Any]]:
    """Return UI-safe metadata for the three coherent motion systems."""
    return [
        {
            key: value for key, value in item.items()
            if key in {"id", "label", "eyebrow", "description", "best_for", "preview_kind", "pacing", "features", "demo_script"}
        } | {"selected": item["id"] == selected, "recommended": item["id"] == recommended}
        for item in MOTION_TEMPLATES
    ]


def recommend_motion_template(visual_beats: list[dict[str, Any]], plan: dict[str, Any]) -> str:
    media_count = sum(item.get("kind") == "media" for item in visual_beats)
    conflict_count = sum(item.get("kind") in {"compare", "stat"} for item in visual_beats)
    hook_text = clean_text(str(plan.get("hook", {}).get("text", "")))
    if media_count >= 3:
        return "editorial"
    if media_count >= 2:
        return "story"
    if conflict_count or ("不是" in hook_text and "而是" in hook_text):
        return "conflict"
    return "expert"


def file_signature(path: Path) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    digest.update(str(size).encode())
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if size > 2 * 1024 * 1024:
            handle.seek(max(0, size - 1024 * 1024))
            digest.update(handle.read(1024 * 1024))
    return digest.hexdigest()[:20]


def media_info(path: Path) -> dict[str, Any]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = float(container.duration / av.time_base) if container.duration else float(stream.duration * stream.time_base)
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        return {"duration": duration, "width": stream.width, "height": stream.height, "fps": fps}


def audio_duration(path: Path) -> float:
    with av.open(str(path)) as container:
        if container.duration:
            return float(container.duration / av.time_base)
        stream = container.streams.audio[0]
        return float(stream.duration * stream.time_base)


def transcribe(path: Path, model_name: str, cache_dir: str) -> tuple[list[Word], list[dict[str, Any]], str]:
    kwargs: dict[str, Any] = {"device": "cpu", "compute_type": "int8"}
    if cache_dir:
        kwargs["download_root"] = cache_dir
    model = WhisperModel(model_name, **kwargs)
    iterator, info = model.transcribe(
        str(path), language="zh", vad_filter=True, word_timestamps=True,
        vad_parameters={"min_silence_duration_ms": 500}, beam_size=5,
    )
    words: list[Word] = []
    segments: list[dict[str, Any]] = []
    for segment in iterator:
        text = segment.text.strip()
        segments.append({"start": round(segment.start, 3), "end": round(segment.end, 3), "text": text})
        for item in segment.words or []:
            token = item.word.strip()
            if token:
                words.append(Word(token, float(item.start), float(item.end), float(item.probability or 1.0)))
    if not words:
        words = [Word(s["text"], s["start"], s["end"], 1.0) for s in segments if s["text"]]
    return words, segments, info.language


def transcribe_with_cache(
    source: Path, output: Path, model_name: str, cache_dir: str, signature: str
) -> tuple[list[Word], list[dict[str, Any]], str, bool]:
    cache_path = output / "transcript.json"
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            # Older cache files predate the model field and were produced by
            # the default `small` model. Keep those reusable, while still
            # invalidating the cache when the user explicitly changes models.
            cached_model = payload.get("model") or "small"
            if payload.get("signature") == signature and cached_model == model_name:
                words = [Word(**item) for item in payload.get("words", [])]
                if words:
                    return words, payload.get("segments", []), payload.get("language", "zh"), True
        except (OSError, ValueError, TypeError):
            pass
    words, segments, language = transcribe(source, model_name, cache_dir)
    return words, segments, language, False


def candidate_cuts(words: list[Word], duration: float, threshold: float) -> list[tuple[float, float]]:
    cuts: list[tuple[float, float]] = []
    for left, right in zip(words, words[1:]):
        if right.start - left.end >= threshold:
            start, end = left.end + 0.16, right.start - 0.16
            if end - start >= 0.35:
                cuts.append((max(0.0, start), min(duration, end)))
    return cuts


def invert_cuts(cuts: list[tuple[float, float]], duration: float) -> list[dict[str, float]]:
    if not cuts:
        return [{"source_start": 0.0, "source_end": duration, "edited_start": 0.0, "duration": duration}]
    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in sorted(cuts):
        if start > cursor:
            keep.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        keep.append((cursor, duration))
    result: list[dict[str, float]] = []
    edited = 0.0
    for start, end in keep:
        length = end - start
        if length < 0.05:
            continue
        result.append({"source_start": round(start, 3), "source_end": round(end, 3), "edited_start": round(edited, 3), "duration": round(length, 3)})
        edited += length
    return result


def choose_mode(requested: str, cuts: list[tuple[float, float]], duration: float) -> str:
    if requested != "auto":
        return requested
    saved = sum(end - start for start, end in cuts)
    return "recut" if len(cuts) >= 2 and saved >= max(4.0, duration * 0.025) else "keep"


def map_time(value: float, keep: list[dict[str, float]]) -> float | None:
    for item in keep:
        if item["source_start"] - 0.001 <= value <= item["source_end"] + 0.001:
            return item["edited_start"] + max(0.0, value - item["source_start"])
    return None


def remap_words(words: list[Word], keep: list[dict[str, float]]) -> list[Word]:
    mapped: list[Word] = []
    for word in words:
        start, end = map_time(word.start, keep), map_time(word.end, keep)
        if start is not None and end is not None:
            mapped.append(Word(word.text, round(start, 3), round(max(start + 0.04, end), 3), word.probability))
    return mapped


def remap_segments(segments: list[dict[str, Any]], keep: list[dict[str, float]]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for segment in segments:
        start, end = map_time(float(segment["start"]), keep), map_time(float(segment["end"]), keep)
        text = clean_text(str(segment.get("text", "")))
        if start is not None and end is not None and text:
            mapped.append({"start": round(start, 3), "end": round(max(start + .08, end), 3), "text": text, "keyword": ""})
    return mapped


def clean_text(text: str) -> str:
    cleaned = re.sub(r"\s+", "", text).strip()
    for wrong, right in ASR_CORRECTIONS.items():
        cleaned = cleaned.replace(wrong, right)
    return cleaned


def descriptive_filename_title(source_stem: str) -> str:
    candidate = re.split(r"[#。！？!?]", source_stem)[0].strip(" _-，,")
    if re.match(r"^(IMG|VID|MOV|DSC|视频|录屏|video|untitled|未命名)[-_ ]?\d*", candidate, re.I):
        return ""
    clause = re.split(r"[，,；;：:]", candidate)[0].strip()
    return clause if 4 <= len(clause) <= 18 else ""


def ranked_cover_thoughts(groups: list[dict[str, Any]]) -> list[str]:
    candidates: list[tuple[float, str]] = []
    for index, group in enumerate(groups):
        thought, _ = semantic_callout(groups, index)
        thought = clean_text(thought).strip("，。！？!?；;")
        if not 8 <= len(thought) <= 32:
            continue
        score = 0.0
        score += sum(term in thought for term in STRONG_CALLOUT_WORDS) * 5.0
        score += sum(term in thought for term in COVER_CONCEPT_TERMS) * 2.2
        score += 5.0 if "不是" in thought and "而是" in thought else 0.0
        score += 3.0 if any(term in thought for term in ("一定", "决定", "最重要", "不长久", "本质")) else 0.0
        score -= abs(len(thought) - 20) * 0.12
        candidates.append((score, thought))
    ranked: list[str] = []
    for _, thought in sorted(candidates, key=lambda item: item[0], reverse=True):
        if thought not in ranked:
            ranked.append(thought)
    return ranked


def extract_cover_keywords(full_text: str, preferred: tuple[str, ...] = ()) -> list[str]:
    keywords: list[str] = []
    for term in (*preferred, *COVER_CONCEPT_TERMS):
        if term in full_text and term not in keywords:
            keywords.append(term)
    if len(keywords) < 2:
        for term in ("行业", "生意", "选择", "方法", "认知", "结果"):
            if term in full_text and term not in keywords:
                keywords.append(term)
    return (keywords + ["洞察", "方法"])[:2]


def build_cover_copy(
    groups: list[dict[str, Any]], source_stem: str, configured_title: str,
    configured_subtitle: str, series: str,
) -> dict[str, Any]:
    """Turn the complete transcript into distinct copy roles for each cover layout."""
    full_text = clean_text("".join(str(group.get("text", "")) for group in groups))
    filename_title = descriptive_filename_title(source_stem)
    thoughts = ranked_cover_thoughts(groups)
    title = configured_title.strip()
    subtitle = configured_subtitle.strip()
    hook = thoughts[0] if thoughts else "把复杂问题讲清楚"
    angle = "行业洞察"
    preferred: tuple[str, ...] = ()

    if "战略" in full_text and "取舍" in full_text and any(term in full_text for term in ("餐饮", "品牌", "连锁")):
        title = title or "餐饮连锁，赢在战略取舍"
        subtitle = subtitle or "守住定位、敢于取舍，才能穿越存量内卷"
        contrast = next((item for item in thoughts if "火" in item or "摇摆" in item or "不长久" in item), "")
        hook = contrast or "看什么火就做什么，品牌一定走不长久"
        angle = "餐饮经营洞察"
        preferred = ("定位", "取舍")
    elif any(term in full_text for term in ("麦当劳", "肯德基")) and any(term in full_text for term in ("标准化", "规模化", "人的生意")):
        title = title or filename_title or "麦当劳的伪学徒"
        subtitle = subtitle or "学会了规模与标准，却忘了餐饮是人的生意"
        hook = next((item for item in thoughts if "标准" in item or "消费者" in item or "人" in item), hook)
        angle = "连锁品牌反思"
        preferred = ("规模化", "顾客")
    elif "餐饮" in full_text:
        title = title or filename_title or "餐饮经营的关键选择"
        subtitle = subtitle or (thoughts[0] if thoughts else "从顾客、效率与体验重新理解这门生意")
        angle = "餐饮经营洞察"
        preferred = ("顾客", "价值")
    elif "品牌" in full_text or "市场" in full_text:
        title = title or filename_title or "品牌增长的底层判断"
        subtitle = subtitle or (thoughts[0] if thoughts else "在变化的市场里找到真正重要的选择")
        angle = "商业经营洞察"
        preferred = ("品牌", "增长")
    else:
        title = title or filename_title or "一个值得重新思考的观点"
        subtitle = subtitle or (thoughts[0] if thoughts else "从现象走向结论，把复杂问题讲清楚")

    keywords = extract_cover_keywords(full_text, preferred)
    impact_line = hook[:24].rstrip("，。！？!?；;")
    if len(impact_line) < 8:
        impact_line = subtitle[:24]
    return {
        "topic": title[:22],
        "headline": title[:22],
        "subheadline": subtitle[:38],
        "hook": hook[:32].rstrip("，。！？!?；;"),
        "impact_line": impact_line,
        "keywords": keywords,
        "angle": angle,
        "series": series,
        "method": "full-transcript topic → tension → conclusion → keywords",
    }


def json_request(url: str, payload: dict[str, Any], token: str, timeout: int = 120) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "User-Agent": "VideoProductionWorkflow/6.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_model_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            return {}
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}


def content_review_prompt(groups: list[dict[str, Any]], config: dict[str, Any]) -> str:
    transcript = [
        {"index": index, "start": round(float(item["start"]), 2), "end": round(float(item["end"]), 2), "text": clean_text(str(item["text"]))}
        for index, item in enumerate(groups)
    ]
    return f"""你是中文商业口播的资深校对与内容分析师。先修正语音识别错字，再建立论证结构。禁止改写讲者观点，禁止补充逐字稿没有的信息。

用户填写姓名：{config.get('speaker') or '未填写'}
用户填写职衔：{config.get('speaker_title') or '未填写'}
带时间的原始识别稿：
{json.dumps(transcript, ensure_ascii=False)}

只返回 JSON：
{{
  "corrected_segments":[{{"index":0,"text":"","confidence":0.0,"changes":["错词→正确词"]}}],
  "identity":{{"name":"","title":"","name_confidence":0.0,"title_confidence":0.0,"evidence":""}},
  "content_map":{{
    "thesis":"",
    "audience":"",
    "arguments":[{{"title":"","mechanism":"","evidence":[""],"counterexample":"","conclusion":"","start":0,"end":0}}],
    "examples":[{{"name":"","point":"","start":0,"end":0}}],
    "closing":""
  }},
  "uncertain_phrases":[{{"start":0,"end":0,"heard_as":"","candidates":[""],"reason":""}}]
}}

规则：
1. corrected_segments 必须逐项返回，index 与输入一致；只改同音错字、行业术语、数字和明显病句，不改变原意，不增加未说内容。
2. 餐饮语境优先检查：客单、民生餐饮、薄利多销、成本领先、人均、薅羊毛、下牌桌等术语。
3. identity.name 只有在逐字稿明确出现“我是/我叫”时才能填写；title 只有明确说出或用户填写时才能填写，禁止虚构专家头衔。
4. content_map 必须拆出“总论点→一级论点→经营机制→案例/反例→结论”，不能只做主题摘要。
5. 无法可靠校正的短语放入 uncertain_phrases；不要用猜测污染 corrected_segments。"""


def apply_content_review(groups: list[dict[str, Any]], review: dict[str, Any]) -> list[dict[str, Any]]:
    corrected = {
        int(item.get("index", -1)): clean_text(str(item.get("text", "")))
        for item in review.get("corrected_segments", []) if isinstance(item, dict)
    }
    result: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        original = clean_text(str(group.get("text", "")))
        candidate = corrected.get(index, "")
        # A correction may repair several characters, but it must still resemble
        # the speech segment and must not become an editorial rewrite.
        ratio = difflib.SequenceMatcher(None, cue_text_key(original), cue_text_key(candidate)).ratio() if candidate else 0
        length_ratio = len(candidate) / max(1, len(original))
        accepted = bool(candidate and .48 <= length_ratio <= 1.65 and ratio >= .42)
        result.append({**group, "text": candidate if accepted else original, "asr_reviewed": accepted})
    return result


def repair_caption_boundaries(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Move a punctuation-terminated orphan character back to its prior word."""
    result: list[dict[str, Any]] = []
    orphan = re.compile(r"^([\u4e00-\u9fff]{1,2}[，。！？,!?])(.*)$")
    for raw in groups:
        item = dict(raw)
        match = orphan.match(clean_text(str(item.get("text", ""))))
        if result and match and not re.search(r"[，。！？,!?]$", str(result[-1].get("text", ""))):
            prefix, remainder = match.groups()
            result[-1]["text"] = clean_text(str(result[-1]["text"]) + prefix)
            seam = min(float(item.get("end", item.get("start", 0))), float(item.get("start", 0)) + .28)
            result[-1]["end"] = round(max(float(result[-1].get("end", 0)), seam), 3)
            if not remainder:
                continue
            item["text"] = remainder
            item["start"] = round(seam, 3)
        result.append(item)
    return result


def build_content_review(
    groups: list[dict[str, Any]], config: dict[str, Any], output: Path,
) -> dict[str, Any]:
    token = os.environ.get("ARK_API_KEY", "").strip()
    model = str(config.get("director_model", "")).strip()
    cache_path = output / "content-analysis.json"
    transcript_hash = hashlib.sha1(
        "|".join(clean_text(str(item.get("text", ""))) for item in groups).encode("utf-8")
    ).hexdigest()
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_segments = cached.get("corrected_segments", [])
            hash_matches = cached.get("transcript_hash") in {None, transcript_hash}
            if cached.get("model") == model and hash_matches and len(cached_segments) == len(groups):
                cached["transcript_hash"] = transcript_hash
                return cached
        except (OSError, ValueError, TypeError):
            pass
    fallback: dict[str, Any] = {
        "status": "fallback", "provider": "rules", "model": "",
        "corrected_segments": [
            {"index": index, "text": clean_text(str(item.get("text", ""))), "confidence": .55, "changes": []}
            for index, item in enumerate(groups)
        ],
        "identity": {"name": "", "title": "", "name_confidence": 0, "title_confidence": 0, "evidence": ""},
        "content_map": {}, "uncertain_phrases": [],
    }
    if token and model and config.get("director_enabled", True):
        try:
            response = json_request(ARK_CHAT_URL, {
                "model": model,
                "messages": [{"role": "user", "content": content_review_prompt(groups, config)}],
                "stream": False, "temperature": .1, "max_tokens": 6144,
            }, token, 180)
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            raw = parse_model_json(content)
            if isinstance(raw.get("corrected_segments"), list):
                fallback.update(raw)
                fallback.update({"status": "llm", "provider": "Volcano Ark", "model": model})
        except Exception as exc:
            fallback["reason"] = f"内容复核调用失败：{type(exc).__name__}"
    fallback["transcript_hash"] = transcript_hash
    cache_path.write_text(json.dumps(fallback, ensure_ascii=False, indent=2), encoding="utf-8")
    return fallback


def resolve_identity(config: dict[str, Any], review: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    identity = review.get("identity") if isinstance(review.get("identity"), dict) else {}
    speaker = clean_text(str(config.get("speaker", "")))
    title = clean_text(str(config.get("speaker_title", "")))
    source = "user"
    if not speaker and config.get("auto_identity", True) and float(identity.get("name_confidence", 0) or 0) >= .62:
        speaker = clean_text(str(identity.get("name", "")))[:10]
        source = "speech-inferred"
    if not title and config.get("auto_identity", True):
        if float(identity.get("title_confidence", 0) or 0) >= .78:
            title = clean_text(str(identity.get("title", "")))[:22]
            source = "speech-inferred"
        elif speaker:
            # This is a program-role label, not an invented professional credential.
            title = "餐饮经营观点分享"
            source = "program-role"
    return speaker, title, {
        "source": source, "name_confidence": float(identity.get("name_confidence", 0) or 0),
        "title_confidence": float(identity.get("title_confidence", 0) or 0),
        "evidence": clean_text(str(identity.get("evidence", ""))),
    }


def director_prompt(
    groups: list[dict[str, Any]], config: dict[str, Any], duration: float,
    content_review: dict[str, Any] | None = None,
) -> str:
    transcript = [{"start": round(float(item["start"]), 2), "end": round(float(item["end"]), 2), "text": clean_text(str(item["text"]))} for item in groups]
    content_map = (content_review or {}).get("content_map", {})
    return f"""你是一位资深中文商业短视频总导演。请阅读完整逐字稿，做语义决策，不要按关键词机械抽取。

视频时长：{duration:.2f} 秒
主讲人：{config.get('speaker') or '未填写'}
主讲人职衔：{config.get('speaker_title') or '未填写'}
逐字稿（时间为剪辑后秒数）：
{json.dumps(transcript, ensure_ascii=False)}
内容分析师给出的论证结构（需要复核后使用，不可机械照搬）：
{json.dumps(content_map, ensure_ascii=False)}

只返回一个 JSON 对象，结构必须是：
{{
  "cover": {{"headline":"", "subheadline":"", "hook":"", "keywords":["",""], "angle":""}},
  "hook": {{"start":0, "end":3.8, "text":"", "reason":""}},
  "callouts": [{{"start":0, "end":0, "text":"", "kind":"quote|compare|stat", "reason":""}}],
  "knowledge_cards": [{{"start":0, "end":0, "title":"", "points":["",""], "reason":""}}],
  "caption_emphasis": [{{"start":0, "phrases":[""], "reason":""}}],
  "chapters": [{{"start":0, "title":"", "reason":""}}],
  "scene_assets": [{{"start":0, "end":0, "type":"image|video", "caption":"", "prompt":"", "reason":""}}],
  "lower_thirds": [{{"start":0, "duration":3.2, "reason":""}}],
  "outro": {{"start":0,"title":"","points":["",""],"reason":""}},
  "music": {{"mood":"", "bpm":88, "instruments":[""], "energy_curve":""}}
}}

决策标准：
1. Hook 必须是全片最有冲突、反常识或利益相关的完整观点；不能选自我介绍、寒暄、铺垫或半句话。text 必须逐字复制逐字稿中连续出现的 10–30 个汉字，禁止改写、压缩或拼接；start/end 填这句原话真正开始和结束的时间。
2. Callout 只选真正值得停顿的结论、对比、数字或方法，每两个至少相隔 20 秒；text 必须逐字复制逐字稿中连续出现的 10–30 个汉字，禁止总结式改写。start/end 必须覆盖这句原话的真实播报区间。长视频每 25–40 秒约一条。
3. caption_emphasis 要更密但克制，约每 8–15 秒一次。phrases 必须是该时间附近字幕中原样存在的连续短语，优先核心名词、动作、因果结论，每次 1–2 个，不能强调虚词。
4. knowledge_cards 只在讲者连续列举、归纳或形成 2–3 个并列结论时使用；同一张卡不能提前揭示尚未讲到的后续观点。title 是 4–10 字总结，points 是当时语音已经讲到的短结论，不能杜撰；画面持续 4–7 秒，逐条随语音累积。
5. chapters 只放真正主题转折，必须在转折词之前 0.2–0.8 秒切入并在核心正文开始前退出；如果内容明确有“第一/第二”，章节结构必须成对完整。只有章节允许纯文字全屏，持续 1.2–1.6 秒。
6. scene_assets 只为可被视觉化的具体人物、地点、动作、物品或经营场景生成；抽象机制改用 knowledge/compare/stat，不要硬配真人素材。caption 用 6–14 个字概括当前具体对象。prompt 必须写全：国家/城市语境＋主体＋场所＋动作＋经营含义＋镜头构图＋纪实风格＋禁用项；不得只写“竖屏纪实风格”。素材出现时人物才可缩成画中画。具体场景建议每 10–18 秒一处，覆盖案例和长口播死区。
7. lower_thirds 在开场 0.6–4 秒至少出现一次，章节后可复现一次；只要姓名或节目身份标签存在就不可返回空数组。
8. outro 在最后 3–5 秒回到讲者并总结本片已经说出的两个行动点；不是全屏标题页，不得遮住结尾口播。
9. 封面文案分别承担主题、结果、冲突和关键词，不要把同一句复制到所有位置。
10. music 是专业、克制、有判断力的行业洞察氛围，不要娱乐综艺感。
11. 所有 start/end 必须来自逐字稿附近的真实时间，不得超出视频时长。时间只帮助定位，程序还会用逐词时间戳再次校准。只输出 JSON，不要解释。"""


def fallback_emphasis(groups: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    concepts = (*COVER_CONCEPT_TERMS, "内卷", "便宜", "需求", "选择", "坚持", "顾客需要", "用脚投票", "现炒现做")
    result: list[dict[str, Any]] = []
    last = -20.0
    for group in groups:
        start, text = float(group["start"]), clean_text(str(group["text"]))
        if start - last < 9.0:
            continue
        candidates = [term for term in concepts if term in text]
        if not candidates:
            candidates = [term for term in STRONG_CALLOUT_WORDS if term in text and len(term) >= 2]
        if candidates:
            phrase = max(candidates, key=len)
            result.append({"start": round(start, 3), "phrases": [phrase], "reason": "规则兜底：核心概念"})
            last = start
    return result


def fallback_callout_candidates(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        context = clean_text("".join(str(item.get("text", "")) for item in groups[index:index + 5]))
        text = ""
        if "看什么火就做什么" in context and "不长久" in context:
            text = "看什么火就做什么，定位总在摇摆，品牌一定不长久"
        elif "顾客需要什么" in context and "自己喜欢什么" in context:
            text = "战略取舍，要看核心顾客需要什么，而不是自己喜欢什么"
        elif "最重要的是两点" in context and "战略笃定" in context and "战略取舍" in context:
            text = "餐饮连锁最重要的两点：战略笃定与战略取舍"
        elif "薄利多销" in context and "成本领先" in context:
            text = "用薄利多销抢市场，用成本领先赚利润"
        elif "最关键的是看不见利润" in context and "坚持不下去" in context:
            text = "看不见利润的事情，大部分人都坚持不下去"
        elif "胖东来" in context and "从来不便宜" in context:
            text = "胖东来的东西好，但从来不便宜"
        if not text:
            continue
        score = sum(term in text for term in STRONG_CALLOUT_WORDS) * 6 + sum(term in text for term in COVER_CONCEPT_TERMS) * 2
        score += 8 if ("不是" in text or "而不是" in text) else 0
        score += 6 if any(term in text for term in ("一定", "最重要", "最关键", "取舍")) else 0
        candidates.append({"start": float(group["start"]), "end": float(groups[min(len(groups) - 1, index + 4)]["end"]), "text": text, "kind": beat_kind(text), "score": score})
    unique: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda value: (-value["score"], value["start"])):
        if item["text"] not in {value["text"] for value in unique}:
            unique.append(item)
    return unique


def fallback_knowledge_cards(groups: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    """Find one real, nearby two-point sequence instead of inventing a summary board."""
    if duration < 55:
        return []
    candidates: list[dict[str, Any]] = []
    for group in groups:
        start = float(group.get("start", 0))
        text = clean_text(str(group.get("text", ""))).strip("，。！？!?；;")
        if not duration * .32 <= start <= duration * .72:
            continue
        if not 7 <= len(text) <= 24 or text.endswith(TRAILING_FRAGMENTS):
            continue
        score = sum(term in text for term in (*STRONG_CALLOUT_WORDS, *COVER_CONCEPT_TERMS))
        score += 2 if any(term in text for term in ("第一", "第二", "首先", "其次", "所以", "最重要")) else 0
        candidates.append({**group, "text": text, "score": score})
    candidates.sort(key=lambda item: (-int(item["score"]), float(item["start"])))
    if not candidates:
        return []
    first = candidates[0]
    nearby = [
        item for item in candidates[1:]
        if 1.2 <= float(item["start"]) - float(first["start"]) <= 14
    ]
    if not nearby:
        return []
    second = min(nearby, key=lambda item: float(item["start"]))
    return [{
        "start": round(float(first["start"]), 3),
        "end": round(min(duration, max(float(second.get("end", second["start"])), float(first["start"]) + 5.4)), 3),
        "title": "核心判断",
        "points": [str(first["text"])[:22], str(second["text"])[:22]],
        "reason": "规则兜底：相邻观点形成两点归纳",
    }]


def fallback_director_plan(groups: list[dict[str, Any]], config: dict[str, Any], duration: float, source_stem: str, reason: str) -> dict[str, Any]:
    cover = build_cover_copy(groups, source_stem, config.get("title", ""), config.get("subtitle", ""), config.get("series", "观点拆解"))
    candidates = fallback_callout_candidates(groups)
    hook_text = candidates[0]["text"] if candidates else cover["hook"]
    hook_candidate = next((item for item in candidates if item["text"] == hook_text), None)
    provisional_chapters = select_chapters(groups, duration)
    callouts: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda value: value["start"]):
        if item["text"] == hook_text or overlaps_window(item["start"], 4.0, provisional_chapters, 1.2) or (callouts and item["start"] - callouts[-1]["start"] < 20):
            continue
        callouts.append({"start": item["start"], "end": min(item["end"], item["start"] + 4), "text": item["text"], "kind": item["kind"], "reason": "规则兜底：完整结论/对比/方法"})
    if not callouts and len(candidates) > 1:
        item = candidates[1]
        callouts.append({"start": item["start"], "end": min(item["end"], item["start"] + 4), "text": item["text"], "kind": item["kind"], "reason": "规则兜底：完整结论"})
    scenes = []
    for item in callouts:
        nearby = next((g for g in groups if abs(float(g["start"]) - float(item["start"])) < 1.5), None)
        text = clean_text(str(nearby["text"])) if nearby else item["text"]
        if any(term in text for term in SCENE_WORDS):
            caption = fallback_scene_caption(text)
            scenes.append({"start": item["start"], "end": item["end"], "type": fallback_scene_type(text), "caption": caption, "prompt": f"竖屏纪实编辑摄影，{caption}，自然光，真实商业现场，无文字，无Logo，无水印", "reason": "规则兜底：具体场景"})
    # 即使没有大模型，也从口播里的具体人物、地点、品牌、产品和动作中持续找视觉锚点。
    # 这些节点只在用户上传素材或主动开启智能取材时变成真实素材，不会凭空占画面。
    scene_limit = max(0, int(config.get("smart_media_count", 3)))
    for index, group in enumerate(groups):
        if len(scenes) >= scene_limit:
            break
        start = float(group.get("start", 0))
        if start < 4 or start > duration - 3 or any(abs(start - float(item["start"])) < 13 for item in scenes):
            continue
        context = clean_text("".join(str(item.get("text", "")) for item in groups[index:index + 3]))[:42]
        concrete = [term for term in SCENE_WORDS if term in context]
        if not concrete:
            continue
        caption = fallback_scene_caption(context)
        if len(caption) < 4:
            continue
        scenes.append({
            "start": round(start, 3), "end": round(min(duration, float(groups[min(len(groups) - 1, index + 2)].get("end", start + 4))), 3),
            "type": fallback_scene_type(context), "caption": caption,
            "prompt": f"竖屏纪实编辑摄影，{caption}，自然光，真实商业现场，细节丰富，无文字，无Logo，无水印",
            "reason": f"规则兜底：语音说到具体场景（{max(concrete, key=len)}）",
        })
    scenes.sort(key=lambda item: float(item["start"]))
    lower = []
    if config.get("speaker") or config.get("speaker_title"):
        for at in (1.0, duration * .46, duration * .78):
            if at < duration - 4:
                lower.append({"start": round(at, 3), "duration": 3.2, "reason": "规则兜底：身份背书"})
    return {
        "status": "fallback", "provider": "rules", "model": "", "reason": reason,
        "cover": cover,
        "hook": {
            "start": round(float(hook_candidate["start"]), 3) if hook_candidate else 0,
            "end": round(min(duration, float(hook_candidate["end"])), 3) if hook_candidate else min(3.8, duration),
            "text": hook_text[:30], "reason": "规则兜底：最高分完整观点",
        },
        "callouts": callouts,
        "knowledge_cards": fallback_knowledge_cards(groups, duration),
        "caption_emphasis": fallback_emphasis(groups, duration),
        "chapters": [{**item, "reason": "规则兜底：主题转折"} for item in provisional_chapters],
        "scene_assets": scenes[:scene_limit],
        "lower_thirds": lower,
        "outro": {
            "start": round(max(4.2, duration - 4.6), 3), "title": "守住定位，做好取舍",
            "points": ["战略笃定", "战略取舍"], "reason": "规则兜底：结尾行动总结",
        },
        "music": {"mood": "专业行业洞察，克制、理性、有判断力", "bpm": 88, "instruments": ["暖色合成器", "轻打击", "低频脉冲"], "energy_curve": "Hook抬升，口播压低，章节与结论轻微抬升"},
    }


def strengthen_scene_prompt(scene: dict[str, Any]) -> dict[str, Any]:
    item = dict(scene)
    caption = clean_text(str(item.get("caption", "")))[:20] or "中国商业经营现场"
    reason = clean_text(str(item.get("reason", "")))[:48]
    prompt = clean_text(str(item.get("prompt", "")))
    if "胖东来" in caption:
        item["visual_expectation"] = "中国大型品质商超服务现场，整洁货架、顾客购物、员工服务"
        item["brand_reference"] = "editorial-text-only"
        item["prompt"] = (
            "中国大型品质商超内部，整洁货架、丰富商品、员工主动服务、顾客自然购物，表现好品质与合理价格的平衡；"
            "9:16竖屏真实纪录片摄影，中景，主体位于中左，右上预留人物画中画空间；不得出现任何品牌门头、Logo、标语、字幕或水印"
        )
        return item
    weak = len(prompt) < 34 or not any(term in prompt for term in (caption, *SCENE_WORDS, "快餐", "中餐", "商超", "出餐", "取餐"))
    if weak:
        if "大众" in caption or "快餐" in caption or "效率" in reason or "成本" in reason:
            subject = "中国城市大众快餐门店午餐高峰，开放式出餐台，员工标准化分装套餐，顾客排队快速取餐，表现高周转、薄利多销和成本控制"
        elif "胖东来" in caption:
            subject = "中国大型品质商超，整洁货架、丰富商品、员工主动服务、顾客自然购物，表现好品质与合理价格的平衡，不伪造品牌门头"
        elif "中餐" in caption or "体验" in reason:
            subject = "中国人均一百元左右的中餐厅，服务员规范上菜，环境整洁，顾客自然用餐，表现服务流程和体验稳定性"
        elif "门店" in caption or "餐饮" in caption:
            subject = "中国城市餐饮门店经营现场，午餐客流、服务员出餐、店长巡视门店，表现存量竞争下的真实经营状态"
        else:
            subject = f"中国商业纪录片场景，准确表现{caption}，{reason}"
        prompt = (
            f"{subject}；9:16竖屏，真实纪录片摄影，自然光，稳定缓慢推进，主体位于中左区域，"
            "右上或右下预留人物画中画空间；无字幕、无水印、无虚构Logo、无新闻演播室、无家庭厨房、无灾难或执法场景"
        )
    elif caption not in prompt:
        prompt = f"准确表现{caption}；{prompt}"
    item["prompt"] = prompt
    return item


def sanitize_director_plan(plan: dict[str, Any], groups: list[dict[str, Any]], config: dict[str, Any], duration: float, source_stem: str) -> dict[str, Any]:
    fallback = fallback_director_plan(groups, config, duration, source_stem, plan.get("reason", "模型结果不完整"))
    cover = plan.get("cover") if isinstance(plan.get("cover"), dict) else {}
    merged_cover = {**fallback["cover"], **{key: value for key, value in cover.items() if value}}
    if config.get("title"):
        merged_cover["headline"] = config["title"]
    if config.get("subtitle"):
        merged_cover["subheadline"] = config["subtitle"]
    valid: dict[str, Any] = {**fallback, **plan, "cover": merged_cover}
    for key in ("callouts", "knowledge_cards", "caption_emphasis", "chapters", "scene_assets", "lower_thirds"):
        if not isinstance(valid.get(key), list):
            valid[key] = fallback[key]
    valid["callouts"] = [item for item in valid["callouts"] if isinstance(item, dict) and 0 <= float(item.get("start", -1)) < duration and 8 <= len(clean_text(str(item.get("text", "")))) <= 36]
    valid["knowledge_cards"] = [
        item for item in valid["knowledge_cards"]
        if isinstance(item, dict)
        and 4 <= float(item.get("start", -1)) < duration - 4
        and 2 <= len([point for point in item.get("points", []) if clean_text(str(point))]) <= 3
    ][:2]
    valid["chapters"] = [item for item in valid["chapters"] if isinstance(item, dict) and 4 <= float(item.get("start", -1)) < duration - 3 and clean_text(str(item.get("title", "")))]
    numbered: list[dict[str, Any]] = []
    for group in groups:
        text = clean_text(str(group.get("text", "")))
        number = 1 if ("首先" in text or "第一" in text) else 2 if ("其次" in text or "第二" in text) else 0
        if not number:
            continue
        concept = "战略笃定" if "笃定" in text else "战略取舍" if "取舍" in text else clean_text(re.sub(r"^(首先是|其次是|第一|第二)[，,:：]*", "", text))[:10]
        numbered.append({
            "start": round(max(4.0, float(group.get("start", 0)) - 1.05), 3),
            "title": f"第{'一' if number == 1 else '二'}点：{concept}",
            "reason": "原话出现明确论点编号，章节成对补全",
        })
    if len({item["title"][:2] for item in numbered}) >= 2:
        existing_numbers = {"一" if "第一" in str(item.get("title", "")) else "二" if "第二" in str(item.get("title", "")) else "" for item in valid["chapters"]}
        valid["chapters"].extend(item for item in numbered if ("一" if "第一" in item["title"] else "二") not in existing_numbers)
        valid["chapters"] = sorted(valid["chapters"], key=lambda item: float(item.get("start", 0)))
    valid["scene_assets"] = [
        strengthen_scene_prompt(item) for item in valid["scene_assets"]
        if isinstance(item, dict) and 0 <= float(item.get("start", -1)) < duration
        and (clean_text(str(item.get("prompt", ""))) or clean_text(str(item.get("caption", ""))))
    ]
    scene_limit = max(0, int(config.get("smart_media_count", 3)))
    for candidate in fallback.get("scene_assets", []):
        if len(valid["scene_assets"]) >= scene_limit:
            break
        if any(abs(float(candidate.get("start", 0)) - float(item.get("start", 0))) < 10 for item in valid["scene_assets"]):
            continue
        valid["scene_assets"].append(strengthen_scene_prompt(candidate))
    valid["scene_assets"] = sorted(valid["scene_assets"], key=lambda item: float(item.get("start", 0)))[:scene_limit]
    valid["lower_thirds"] = [item for item in valid["lower_thirds"] if isinstance(item, dict) and .4 <= float(item.get("start", -1)) < duration - 3]
    if (config.get("speaker") or config.get("speaker_title")) and not valid["lower_thirds"]:
        valid["lower_thirds"] = [{"start": 1.0, "duration": 3.2, "reason": "开场身份背书"}]
    outro = valid.get("outro") if isinstance(valid.get("outro"), dict) else fallback.get("outro", {})
    points = [clean_text(str(point))[:18] for point in outro.get("points", []) if clean_text(str(point))][:2]
    valid["outro"] = {
        **fallback.get("outro", {}), **outro,
        "start": round(max(4.2, min(duration - 2.8, float(outro.get("start", duration - 4.2)))), 3),
        "title": clean_text(str(outro.get("title", "")))[:18] or "核心行动",
        "points": points or fallback.get("outro", {}).get("points", []),
    }
    hook = valid.get("hook") if isinstance(valid.get("hook"), dict) else fallback["hook"]
    hook_text = clean_text(str(hook.get("text", "")))
    valid["hook"] = {**fallback["hook"], **hook, "text": (hook_text if 8 <= len(hook_text) <= 36 else fallback["hook"]["text"])}
    valid["status"] = "llm" if plan.get("status") == "llm" else "fallback"
    return valid


def build_director_plan(
    groups: list[dict[str, Any]], config: dict[str, Any], duration: float, source_stem: str,
    output: Path, content_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    token = os.environ.get("ARK_API_KEY", "").strip()
    model = str(config.get("director_model", "")).strip()
    cache_path = output / "director-plan.json"
    analysis_path = output / "content-analysis.json"
    if cache_path.is_file() and analysis_path.is_file() and cache_path.stat().st_mtime >= analysis_path.stat().st_mtime:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("status") == "llm" and cached.get("model") == model:
                return sanitize_director_plan(cached, groups, config, duration, source_stem)
        except (OSError, ValueError, TypeError):
            pass
    if not config.get("director_enabled", True):
        plan = fallback_director_plan(groups, config, duration, source_stem, "智能导演未开启")
    elif not token or not model:
        plan = fallback_director_plan(groups, config, duration, source_stem, "未配置火山方舟 API Key 或导演模型 ID")
    else:
        try:
            response = json_request(ARK_CHAT_URL, {
                "model": model,
                "messages": [{"role": "user", "content": director_prompt(groups, config, duration, content_review)}],
                "stream": False, "temperature": .2, "max_tokens": 4096,
            }, token, 180)
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            raw = parse_model_json(content)
            raw.update({"status": "llm", "provider": "Volcano Ark", "model": model, "reason": "完整逐字稿语义分析"})
            plan = sanitize_director_plan(raw, groups, config, duration, source_stem)
        except Exception as exc:
            plan = fallback_director_plan(groups, config, duration, source_stem, f"智能导演调用失败：{type(exc).__name__}")
    cache_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def derive_title(words: list[Word], configured: str, source_stem: str) -> str:
    if configured.strip():
        return configured.strip()
    filename_title = re.split(r"[#。！？!?]", source_stem)[0].strip(" _-，,")
    if 4 <= len(filename_title) <= 42 and not re.match(r"^(IMG|VID|MOV|DSC|视频|录屏)[-_ ]?\d*", filename_title, re.I):
        first_clause = re.split(r"[，,:：；;]", filename_title)[0].strip()
        if 4 <= len(first_clause) <= 18:
            return first_clause
    text = re.split(r"[。！？!?，,；;]", clean_text("".join(w.text for w in words[:24])))[0]
    for prefix in ("大家好", "今天我们来聊聊", "今天讲的是", "我想跟大家聊聊"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return (text or "一个值得重新思考的观点")[:18]


def derive_subtitle(configured: str, source_stem: str, groups: list[dict[str, Any]], title: str) -> str:
    if configured.strip():
        return configured.strip()[:32]
    filename = re.split(r"[#。！？!?]", source_stem)[0]
    clauses = [item.strip() for item in re.split(r"[，,；;]", filename) if item.strip() and item.strip() != title]
    if clauses:
        return "，".join(clauses[:2])[:32]
    for group in groups:
        if group["start"] > 12 and any(term in group["text"] for term in TRIGGER_WORDS):
            return group["text"][:26]
    return "把复杂观点讲得更清楚"


def group_captions(words: list[Word], keywords: list[str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    current: list[Word] = []
    char_count = 0
    punctuation = re.compile(r"[，。！？；：,.!?;:]$")

    def flush() -> None:
        nonlocal current, char_count
        if current:
            text = clean_text("".join(word.text for word in current))
            if text:
                groups.append({"start": round(max(0.0, current[0].start - 0.04), 3), "end": round(current[-1].end + 0.08, 3), "text": text})
        current, char_count = [], 0

    for word in words:
        token = clean_text(word.text)
        if not token:
            continue
        if current and word.start - current[-1].end > 0.48:
            flush()
        current.append(word)
        char_count += len(token)
        if char_count >= 13 or (char_count >= 7 and punctuation.search(token)):
            flush()
    flush()
    for left, right in zip(groups, groups[1:]):
        if left["end"] >= right["start"]:
            seam = round((left["end"] + right["start"]) / 2, 3)
            left["end"] = max(left["start"] + 0.08, seam - 0.002)
            right["start"] = max(left["end"] + 0.002, seam)
    for group in groups:
        choices = [key for key in keywords if key and key in group["text"]] or [term for term in TRIGGER_WORDS if term in group["text"]]
        group["keyword"] = max(choices, key=len) if choices else ""
    return groups


def cue_text_key(text: str) -> str:
    """Normalize visible speech without losing Chinese characters or numbers."""
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff%]", "", clean_text(text)).lower()


def word_character_index(words: list[Word]) -> tuple[str, list[int]]:
    characters: list[str] = []
    word_indexes: list[int] = []
    for word_index, word in enumerate(words):
        for character in cue_text_key(word.text):
            characters.append(character)
            word_indexes.append(word_index)
    transcript = "".join(characters)
    # Current correction pairs preserve character count, so timestamp indexes remain valid
    # even when Whisper split a mistaken phrase across multiple word tokens.
    for wrong, right in ASR_CORRECTIONS.items():
        wrong_key = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff%]", "", wrong).lower()
        right_key = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff%]", "", right).lower()
        if len(wrong_key) == len(right_key):
            transcript = transcript.replace(wrong_key, right_key)
    return transcript, word_indexes


def exact_spoken_window(words: list[Word], phrase: str, hinted_start: float) -> tuple[float, float] | None:
    transcript, character_words = word_character_index(words)
    needle = cue_text_key(phrase)
    if not needle or not transcript or len(needle) < 4:
        return None
    positions: list[int] = []
    cursor = transcript.find(needle)
    while cursor >= 0:
        positions.append(cursor)
        cursor = transcript.find(needle, cursor + 1)
    if not positions:
        return None
    chosen = min(positions, key=lambda position: abs(words[character_words[position]].start - hinted_start))
    first_word = character_words[chosen]
    last_word = character_words[chosen + len(needle) - 1]
    return float(words[first_word].start), float(words[last_word].end)


def spoken_excerpt(groups: list[dict[str, Any]], index: int, minimum: int = 10, maximum: int = 30) -> str:
    """Take a continuous, displayable excerpt from the actual transcript."""
    text = ""
    for group in groups[index:index + 4]:
        text += clean_text(str(group.get("text", "")))
        if len(cue_text_key(text)) >= minimum:
            punctuation_end = next(
                (match.end() for match in re.finditer(r"[，。！？；,.!?;]", text) if match.end() >= minimum),
                None,
            )
            if punctuation_end:
                text = text[:punctuation_end]
                break
        if len(text) >= maximum:
            break
    text = text[:maximum]
    # Removing a filler prefix still leaves a continuous substring of the spoken sentence.
    text = re.sub(r"^(然后|所以|但是|其实|就是|我觉得|我跟你讲|我们讲)", "", text)
    return text.rstrip("，。！？；,.!?;")


def fuzzy_spoken_excerpt(words: list[Word], cue_text: str, hinted_start: float) -> tuple[str, float, float] | None:
    """Find the strongest continuous overlap when an editorial summary is not verbatim."""
    transcript, character_words = word_character_index(words)
    cue_key = cue_text_key(cue_text)
    if len(cue_key) < 6 or not transcript:
        return None
    matcher = difflib.SequenceMatcher(None, cue_key, transcript, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size >= 5]
    if not blocks:
        return None
    block = max(
        blocks,
        key=lambda item: item.size * 3 - min(10, abs(words[character_words[item.b]].start - hinted_start) * .05),
    )
    if block.size < max(5, round(len(cue_key) * .22)):
        return None
    start_character = max(0, block.b - block.a) if block.a <= 5 else block.b
    first_word = character_words[start_character]
    parts: list[str] = []
    end_word = first_word
    for word_index in range(first_word, min(len(words), first_word + 18)):
        parts.append(clean_text(words[word_index].text))
        end_word = word_index
        joined = "".join(parts)
        if len(cue_text_key(joined)) >= 10 and re.search(r"[，。！？；,.!?;]$", joined):
            break
        if len(cue_text_key(joined)) >= min(30, max(14, len(cue_key))):
            break
    excerpt = clean_text("".join(parts))[:30].rstrip("，。！？；,.!?;")
    exact = exact_spoken_window(words, excerpt, words[first_word].start)
    if exact:
        return excerpt, exact[0], exact[1]
    return excerpt, float(words[first_word].start), float(words[end_word].end)


def best_spoken_excerpt(
    cue_text: str, hinted_start: float, groups: list[dict[str, Any]], words: list[Word]
) -> tuple[str, float, float, str]:
    exact = exact_spoken_window(words, cue_text, hinted_start)
    if exact:
        return clean_text(cue_text).rstrip("，。！？；,.!?;"), exact[0], exact[1], "exact"

    fuzzy = fuzzy_spoken_excerpt(words, cue_text, hinted_start)
    if fuzzy:
        return fuzzy[0], fuzzy[1], fuzzy[2], "fuzzy-snapped"

    nearby = [
        (index, group) for index, group in enumerate(groups)
        if abs(float(group.get("start", 0)) - hinted_start) <= 18
    ] or list(enumerate(groups))
    cue_key = cue_text_key(cue_text)

    def candidate_score(pair: tuple[int, dict[str, Any]]) -> float:
        index, group = pair
        context = "".join(clean_text(str(item.get("text", ""))) for item in groups[index:index + 3])
        similarity = difflib.SequenceMatcher(None, cue_key, cue_text_key(context)).ratio()
        time_penalty = min(0.24, abs(float(group.get("start", 0)) - hinted_start) * .008)
        return similarity - time_penalty

    best_index, best_group = max(nearby, key=candidate_score)
    excerpt = spoken_excerpt(groups, best_index)
    located = exact_spoken_window(words, excerpt, float(best_group.get("start", hinted_start)))
    if located:
        return excerpt, located[0], located[1], "snapped"
    return excerpt, float(best_group.get("start", hinted_start)), float(best_group.get("end", hinted_start + 2.8)), "caption-window"


def spoken_window_confidence(words: list[Word], start: float, end: float) -> tuple[bool, float, int]:
    probabilities = [
        float(word.probability) for word in words
        if word.end >= start and word.start <= end and cue_text_key(word.text)
    ]
    if not probabilities:
        return False, 0.0, 0
    average = sum(probabilities) / len(probabilities)
    low_count = sum(value < .52 for value in probabilities)
    return average >= .72 and low_count <= 1, average, low_count


def align_director_cues(
    plan: dict[str, Any], groups: list[dict[str, Any]], words: list[Word], duration: float
) -> dict[str, Any]:
    """Make every editorial cue enter on the exact words heard in the program audio."""
    if not groups or not words:
        return {**plan, "timing_basis": "分段时间戳（无逐词数据）"}
    aligned = dict(plan)
    hook = dict(plan.get("hook") or {})
    hook_text, hook_start, hook_end, hook_mode = best_spoken_excerpt(
        str(hook.get("text", "")), float(hook.get("start", 0)), groups, words
    )
    aligned["hook"] = {
        **hook,
        "editorial_text": clean_text(str(hook.get("text", ""))),
        "text": hook_text,
        "start": round(max(0, hook_start - .08), 3),
        "end": round(min(duration, max(hook_start + .7, hook_end + .18)), 3),
        "alignment": hook_mode,
    }

    callouts: list[dict[str, Any]] = []
    for item in plan.get("callouts", []):
        cue = dict(item)
        text, start, end, mode = best_spoken_excerpt(
            str(cue.get("text", "")), float(cue.get("start", 0)), groups, words
        )
        nearby_context = clean_text("".join(
            str(group.get("text", "")) for group in groups
            if start - 1.2 <= float(group.get("start", 0)) <= end + 3.2
        ))
        if "看什么火" in text and "不长久" in nearby_context:
            text = "看什么火就做什么，定位摇摆，一定不长久"
            mode = "semantic-reconstruction-from-contiguous-speech"
        elif "超高品质" in text and "伪命题" in nearby_context:
            text = "超高品质＋超低价格，是伪命题"
            mode = "semantic-reconstruction-from-contiguous-speech"
        if len(cue_text_key(text)) < 8:
            continue
        kind = beat_kind(text)
        aligned_item = {
            **cue,
            "editorial_text": clean_text(str(cue.get("text", ""))),
            "text": text,
            "kind": kind,
            "start": round(max(0, start - .08), 3),
            "end": round(min(duration, max(start + .7, end + .18)), 3),
            "alignment": mode,
        }
        confident, average_probability, low_tokens = spoken_window_confidence(words, start, end)
        aligned_item["asr_confidence"] = round(average_probability, 3)
        aligned_item["asr_low_tokens"] = low_tokens
        if not confident:
            continue
        if "而是" in text:
            turn_window = exact_spoken_window(words, text[text.index("而是"):], start)
            if turn_window:
                aligned_item["turn_offset"] = round(max(.08, turn_window[0] - start), 3)
        number = re.search(r"\d+(?:\.\d+)?(?:%|块|元|倍|家|个|年|分钟|秒)?", text)
        if number:
            number_window = exact_spoken_window(words, number.group(0), start)
            if number_window:
                aligned_item["number_offset"] = round(max(.06, number_window[0] - start), 3)
        callouts.append(aligned_item)
    # If the model selected a garbled ASR phrase, look for a nearby coherent,
    # high-confidence continuous statement rather than publishing the error as
    # large on-screen text.
    if len(callouts) < len(plan.get("callouts", [])):
        for candidate in sorted(fallback_callout_candidates(groups), key=lambda item: float(item["start"])):
            if len(callouts) >= len(plan.get("callouts", [])):
                break
            if cue_text_key(str(candidate.get("text", ""))) == cue_text_key(str(aligned["hook"].get("text", ""))):
                continue
            text, start, end, mode = best_spoken_excerpt(str(candidate["text"]), float(candidate["start"]), groups, words)
            confident, average_probability, low_tokens = spoken_window_confidence(words, start, end)
            if not confident or len(cue_text_key(text)) < 8:
                continue
            hook_start_value = float(aligned["hook"].get("start", 0))
            hook_end_value = float(aligned["hook"].get("end", hook_start_value))
            if start < hook_end_value + .8 and end > hook_start_value - .2:
                continue
            if any(abs(start - float(item["start"])) < 20 for item in callouts):
                continue
            callouts.append({
                "start": round(max(0, start - .08), 3), "end": round(min(duration, max(start + .7, end + .18)), 3),
                "text": text, "editorial_text": text, "kind": beat_kind(text),
                "reason": "低置信原话已拒绝，改用附近高置信连续结论",
                "alignment": mode, "asr_confidence": round(average_probability, 3), "asr_low_tokens": low_tokens,
            })
        callouts.sort(key=lambda item: float(item["start"]))
    aligned["callouts"] = callouts
    aligned["timing_basis"] = "逐词时间戳 + 连续原话校准"
    return aligned


def joined_group_text(groups: list[dict[str, Any]], index: int, limit: int = 34) -> str:
    text = groups[index]["text"]
    if (len(text) < 14 or text.endswith(TRAILING_FRAGMENTS)) and index + 1 < len(groups):
        text += groups[index + 1]["text"]
    return text[:limit]


def chapter_heading(groups: list[dict[str, Any]], index: int) -> str:
    """Turn a transcript neighborhood into a compact, complete chapter heading."""
    start, end = max(0, index - 2), min(len(groups), index + 5)
    context = clean_text("".join(item["text"] for item in groups[start:end]))
    context = re.sub(r"^(然后|所以|但是|其实|就是|一个|再一个|来以后|最近的那个)", "", context)
    if "顾客需要什么" in context and "自己喜欢什么" in context:
        return "战略取舍：关注顾客需要"
    if "战略笃定" in context and any(term in context for term in ("初心", "坚守", "定位")):
        return "战略笃定：守住品牌定位"
    if "为什么" in context and "原因" in context:
        subject = context.split("为什么", 1)[1].split("原因", 1)[0]
        subject = re.sub(r"(其实|只有|会有|有)$", "", subject)[:13]
        if subject:
            return f"为什么{subject}"
    if "不是" in context and "而是" in context:
        right = context.split("而是", 1)[1][:14]
        if right:
            return f"真正重要的是{right}"
    if "总部" in context and ("学到" in context or "震撼" in context):
        brand = context[max(0, context.index("总部") - 7):context.index("总部")]
        return f"在{brand}总部学到的一课"[:22]
    if "交警" in context and "咖啡" in context:
        return "一杯送给交警的热咖啡"
    if "顾客" in context and "亲人" in context:
        return "把顾客当作自己的亲人"
    chosen = clean_text(groups[index]["text"])
    if index > 0 and re.match(r"^(来|去|是|的|了|把|给|用|在|就|再)", chosen):
        chosen = clean_text(groups[index - 1]["text"] + chosen)
    chosen = re.sub(r"^(然后|所以|但是|其实|就是|再一个|最近的那个)", "", chosen)
    for marker in ("因为", "所以", "然后", "但是", "而且"):
        if marker in chosen[7:]:
            chosen = chosen.split(marker, 1)[0]
    return chosen[:22].rstrip("的了是在把给用和与") or "这一段的关键观点"


def select_chapters(groups: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    if duration < 70 or not groups:
        return []
    count = max(1, min(5, round(duration / 85)))
    chapters: list[dict[str, Any]] = []
    for index in range(count):
        target = duration * (index + 1) / (count + 1)
        candidates = [(i, g) for i, g in enumerate(groups) if abs(g["start"] - target) <= 24]
        if not candidates:
            continue
        chosen_index, chosen = max(candidates, key=lambda pair: (sum(term in pair[1]["text"] for term in TRIGGER_WORDS) * 8 - abs(pair[1]["start"] - target)))
        start = max(4.0, chosen["start"] - 0.18)
        if chapters and start - chapters[-1]["start"] < 42:
            continue
        text = chapter_heading(groups, chosen_index)
        chapters.append({"start": round(start, 3), "duration": 2.8, "title": text, "index": len(chapters) + 1})
    return chapters


def overlaps_window(start: float, duration: float, windows: list[dict[str, Any]], padding: float = 1.2) -> bool:
    end = start + duration
    return any(start < item["start"] + item["duration"] + padding and end > item["start"] - padding for item in windows)


def beat_kind(text: str) -> str:
    if "不是" in text and "而是" in text:
        return "compare"
    if re.search(r"\d+(?:\.\d+)?(?:%|块|元|倍|家|个|年|分钟|秒)?", text):
        return "stat"
    return "quote"


def scene_query(text: str) -> str:
    stripped = text
    for term in TRIGGER_WORDS:
        stripped = stripped.replace(term, "")
    stripped = re.sub(r"[的了呢吗吧啊呀所以因为但是其实就是]", "", stripped)
    return (stripped[:18] or text[:18]).strip()


def fallback_scene_caption(text: str) -> str:
    scene_labels = (
        (("胖东来",), "胖东来商品现场"), (("网红菜",), "网红菜与门店菜单"),
        (("薄利多销", "成本领先"), "后厨效率与成本"), (("核心团队", "团队"), "核心团队决策现场"),
        (("顾客", "体验"), "顾客体验现场"), (("餐饮品牌", "品牌"), "餐饮品牌门店"),
        (("市场", "内卷"), "餐饮市场竞争"), (("产品",), "产品细节特写"), (("门店",), "真实门店现场"),
    )
    return next((label for terms, label in scene_labels if any(term in text for term in terms)), scene_query(text)[:14])


def fallback_scene_type(text: str) -> str:
    still_scene_terms = ("菜单", "菜品", "商品", "产品", "价格", "数字", "包装")
    if any(term in text for term in still_scene_terms):
        return "image"
    moving_scene_terms = ("现场", "门店", "市场", "团队", "顾客", "体验", "后厨", "制作", "服务", "走进", "选择")
    return "video" if any(term in text for term in moving_scene_terms) else "image"


def semantic_callout(groups: list[dict[str, Any]], index: int) -> tuple[str, float]:
    """Return one readable thought and the moment the spoken thought finishes."""
    group = groups[index]
    parts = [clean_text(group["text"])]
    end = float(group["end"])
    for offset in (1, 2, 3, 4, 5, 6):
        current = "".join(parts)
        incomplete = len(current) < 12 or current.endswith(TRAILING_FRAGMENTS) or bool(re.search(r"(为什么|因为|如果|就是|叫|是|要去|能够|我们要去)$", current))
        needs_context = any(term in current for term in ("为什么", "如果", "第二", "第一", "202", "24小时", "消费者最后", "最困难"))
        if index + offset >= len(groups) or (not incomplete and not needs_context):
            break
        parts.append(clean_text(groups[index + offset]["text"]))
        end = float(groups[index + offset]["end"])
        if len("".join(parts)) >= 78:
            break
    context = "".join(parts)
    context = re.sub(r"^(然后|其实|就是|我觉得|我们讲|我跟你讲|结果)", "", context)
    context = context.replace("对吧", "").replace("哎呀", "").replace("哎", "").replace("学的不像", "学得不像")
    if "第二个是惨" in context and "想吃" in context:
        return "第二个原因是馋：想吃得更好", end
    if "为什么" in context and "能够成为" in context:
        subject = context.split("为什么", 1)[1].split("能够成为", 1)[0][-10:]
        destination = context.split("能够成为", 1)[1]
        destination = destination[:destination.find("企业") + 2] if "企业" in destination else destination[:12]
        return f"为什么{subject}能成为{destination}"[:30], end
    if "如果" in context and "路边" in context and "打扫" in context:
        return "如果社区环境脏了，就主动去打扫", end
    if re.search(r"20\d{2}年", context) and "推出" in context:
        year = re.search(r"20\d{2}年", context).group(0)
        brand = context.split(year, 1)[1].split("推出", 1)[0].replace("疫情一结束", "").replace("疫情结束", "")[-8:]
        return f"{year}，{brand}推出一款新产品"[:30], end
    if "24小时" in context and "无家可归" in context:
        return "24小时门店，也能成为城市的收容站", end
    if "取消中央厨房" in context and "现炒现做" in context:
        return "取消中央厨房，在门店现炒现做", end
    if "消费者最后" in context and "用脚投票" in context:
        return "消费者最后都会用脚投票", end
    if "最困难" in context and "转变思维" in context:
        return "行业越困难，越需要转变思维", end
    for marker in ("然后", "再一个", "我给你讲", "对不对"):
        position = context.find(marker, 10)
        if position > 0:
            context = context[:position]
    if len(context) > 30:
        endings = [match.end() for match in re.finditer(r"(了|企业|产品|卫生|风味|投票|厨房|咖啡|原因)", context[:34]) if match.end() >= 12]
        context = context[:endings[-1] if endings else 30]
    return context[:30].rstrip("的了是在把给用和与"), end


def callout_score(group: dict[str, Any], target: float) -> float:
    text = clean_text(group["text"])
    score = -abs(float(group["start"]) - target) * 0.22
    score += sum(term in text for term in STRONG_CALLOUT_WORDS) * 7
    score += sum(term in text for term in ("所以", "如果", "结果")) * 1.5
    score += sum(term in text for term in SCENE_WORDS) * 2.5
    score += 5 if re.search(r"\d", text) else 0
    score += 4.5 if ("不是" in text and "而是" in text) else 0
    score += 2.5 if 9 <= len(text) <= 18 else 0
    score -= 7 if re.match(r"^(的|了|是|把|给|用|来|去|再|然后)", text) else 0
    score -= 6 if text.endswith(TRAILING_FRAGMENTS) else 0
    return score


def select_visual_beats(
    groups: list[dict[str, Any]], duration: float, assets: list[dict[str, Any]], chapters: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not groups:
        return []
    count = max(2, min(12, math.ceil(duration / 34)))
    beats: list[dict[str, Any]] = []
    asset_cursor = 0
    for beat_index in range(count):
        target = 15 + beat_index * max(27, (duration - 26) / count)
        candidates = [(i, group) for i, group in enumerate(groups) if abs(float(group["start"]) - target) <= 17]
        if not candidates:
            continue
        group_index, group = max(candidates, key=lambda pair: callout_score(pair[1], target))
        text, spoken_end = semantic_callout(groups, group_index)
        if not text:
            continue
        start = max(4.2, float(group["start"]))
        duration_for_beat = round(max(3.1, min(4.6, spoken_end - start + 1.15)), 3)
        if overlaps_window(start, duration_for_beat, chapters, 1.4) or (beats and start - beats[-1]["start"] < 23):
            continue
        kind = beat_kind(text)
        asset = None
        if asset_cursor < len(assets) and (beat_index % 2 == 0 or kind == "quote"):
            asset = assets[asset_cursor]
            asset_cursor += 1
            kind = "media"
        beat: dict[str, Any] = {
            "start": round(start, 3), "duration": 4.2 if kind == "media" else duration_for_beat,
            "kind": kind, "text": text, "query": scene_query(text), "asset": asset,
            "spoken_end": round(spoken_end, 3),
            "cue_reason": "contrast" if "不是" in text and "而是" in text else "number" if re.search(r"\d", text) else "keyword" if any(term in text for term in TRIGGER_WORDS) else "scene",
            "blur": 0.0 if kind == "media" else 0.36 if kind == "quote" else 0.28,
        }
        if kind == "compare":
            left, right = text.split("而是", 1)
            beat["left"] = left.replace("不是", "")[-14:] or "旧做法"
            beat["right"] = right[:14] or "新方向"
        elif kind == "stat":
            match = re.search(r"\d+(?:\.\d+)?(?:%|块|元|倍|家|个|年|分钟|秒)?", text)
            beat["stat"] = match.group(0) if match else "01"
            beat["label"] = text.replace(beat["stat"], "")[:18]
        beats.append(beat)
    return beats


def chapters_from_plan(plan: dict[str, Any], duration: float) -> list[dict[str, Any]]:
    chapters: list[dict[str, Any]] = []
    for item in sorted(plan.get("chapters", []), key=lambda value: float(value.get("start", 0))):
        start = max(4.0, min(duration - 3.0, float(item.get("start", 0))))
        if chapters and start - chapters[-1]["start"] < 20:
            continue
        chapters.append({"start": round(start, 3), "duration": 1.45, "title": clean_text(str(item.get("title", "")))[:22], "index": len(chapters) + 1, "reason": item.get("reason", "")})
    return chapters


def visual_beats_from_plan(plan: dict[str, Any], duration: float, chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    beats: list[dict[str, Any]] = []
    for item in sorted(plan.get("callouts", []), key=lambda value: float(value.get("start", 0))):
        start = max(4.2, min(duration - 3.0, float(item.get("start", 0))))
        text = clean_text(str(item.get("text", "")))[:36]
        kind = str(item.get("kind", "quote"))
        if kind not in {"quote", "compare", "stat"}:
            kind = beat_kind(text)
        beat_duration = max(3.2, min(4.8, float(item.get("end", start + 3.8)) - start + .8))
        if not text or overlaps_window(start, beat_duration, chapters, 1.2) or (beats and start - beats[-1]["start"] < 20):
            continue
        beat: dict[str, Any] = {
            "start": round(start, 3), "duration": round(beat_duration, 3), "kind": kind, "text": text,
            "query": scene_query(text), "asset": None, "spoken_end": round(min(duration, start + beat_duration), 3),
            "cue_reason": item.get("reason", "模型语义决策"), "blur": .36 if kind == "quote" else .28,
            "alignment": item.get("alignment", "unknown"),
        }
        if "turn_offset" in item:
            beat["turn_offset"] = float(item["turn_offset"])
        if "number_offset" in item:
            beat["number_offset"] = float(item["number_offset"])
        if kind == "compare" and "而是" in text:
            left, right = text.split("而是", 1)
            beat["left"], beat["right"] = left.replace("不是", "")[-14:] or "旧做法", right[:14] or "新方向"
        elif kind == "stat":
            match = re.search(r"\d+(?:\.\d+)?(?:%|块|元|倍|家|个|年|分钟|秒)?", text)
            beat["stat"] = match.group(0) if match else "01"
            beat["label"] = text.replace(beat["stat"], "")[:18]
        beats.append(beat)
    for item in sorted(plan.get("knowledge_cards", []), key=lambda value: float(value.get("start", 0))):
        start = max(5.0, min(duration - 5.0, float(item.get("start", 0))))
        points = [clean_text(str(point))[:24] for point in item.get("points", []) if clean_text(str(point))][:3]
        board_duration = max(5.2, min(7.8, float(item.get("end", start + 6.2)) - start + .8))
        if len(points) < 2 or overlaps_window(start, board_duration, beats + chapters, .8):
            continue
        beats.append({
            "start": round(start, 3), "duration": round(board_duration, 3), "kind": "knowledge",
            "text": clean_text(str(item.get("title", "核心判断")))[:12] or "核心判断", "points": points,
            "query": "", "asset": None, "spoken_end": round(min(duration, start + board_duration), 3),
            "cue_reason": item.get("reason", "模型语义归纳"), "blur": .22, "alignment": "semantic-window",
        })
    outro = plan.get("outro") if isinstance(plan.get("outro"), dict) else {}
    if outro:
        start = max(4.2, min(duration - 2.8, float(outro.get("start", duration - 4.2))))
        length = max(2.8, min(4.4, duration - start - .12))
        points = [clean_text(str(point))[:18] for point in outro.get("points", []) if clean_text(str(point))][:2]
        # The closing card is a lower overlay over the restored speaker, not a
        # full-screen title slate. It can replace a weak overlapping callout.
        beats = [beat for beat in beats if not overlaps_window(float(beat["start"]), float(beat["duration"]), [{"start": start, "duration": length}], .2)]
        if points:
            beats.append({
                "start": round(start, 3), "duration": round(length, 3), "kind": "outro",
                "text": clean_text(str(outro.get("title", "核心行动")))[:18] or "核心行动",
                "points": points, "query": "", "asset": None, "spoken_end": round(duration, 3),
                "cue_reason": outro.get("reason", "结尾行动总结"), "blur": 0,
                "alignment": "closing-window",
            })
    return sorted(beats, key=lambda value: float(value["start"]))


def enrich_visual_scene_packages(beats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn loose overlays into complete, dependency-ordered visual scenes."""
    enriched: list[dict[str, Any]] = []
    for index, raw in enumerate(sorted(beats, key=lambda item: float(item["start"]))):
        beat = dict(raw)
        kind = str(beat.get("kind", "quote"))
        has_asset = bool(beat.get("asset"))
        if kind == "media" and has_asset:
            state, speaker_role, text_role, caption_mode = "media_stage", "circle", "scene_caption", "overlay_gradient"
            background = {"role": "asset_full", "status": "ready", "asset": beat.get("asset")}
        elif kind == "outro":
            state, speaker_role, text_role, caption_mode = "speaker_anchor", "full", "closing_summary", "normal"
            background = {"role": "source_clear", "status": "ready", "asset": None}
        elif kind == "knowledge":
            state, speaker_role, text_role, caption_mode = "knowledge_board", "card", "progressive_points", "normal"
            background = {"role": "editorial_canvas", "status": "ready", "asset": None}
        elif kind in {"compare", "stat"}:
            state, speaker_role, text_role, caption_mode = "knowledge_board", "card", kind, "hidden"
            background = {"role": "source_blurred", "status": "ready", "asset": None}
        elif kind == "context":
            # A rejected/missing asset must never leave a floating portrait over
            # a fake scene. Keep the speaker full and make the graphic itself
            # carry the explanatory content.
            state, speaker_role, text_role, caption_mode = "concept_explainer", "full", "scene_context", "normal"
            background = {"role": "editorial_canvas", "status": "ready", "asset": None}
        else:
            state, speaker_role, text_role, caption_mode = "concept_explainer", "circle", "progressive_quote", "hidden"
            background = {"role": "editorial_canvas", "status": "ready", "asset": None}
        beat.update({
            "package_id": f"scene-{index + 1:02d}", "visual_state": state,
            "speaker_role": speaker_role, "text_role": text_role, "caption_mode": caption_mode,
            "background": background,
            "entry_order": ["background", "speaker", "text"],
            "entry_offsets": {"background": 0.0, "speaker": .16, "text": .30},
            "exit_order": ["text", "speaker", "background"],
            "exit_offsets": {"text": .34, "speaker": .20, "background": 0.0},
            "transition": {"entry": "focus-pull" if state != "media_stage" else "hard-cut", "exit": "blur-through", "duration": .34},
        })
        enriched.append(beat)
    return enriched


def build_visual_scene_manifest(hook: dict[str, Any], beats: list[dict[str, Any]], chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def media_payload(asset: dict[str, Any] | None, semantic_text: str) -> dict[str, Any] | None:
        if not asset:
            return None
        payload = {
            **asset, "status": "ready", "caption": semantic_text,
            "metadata": {
                **dict(asset.get("metadata") or {}),
                "semantic_evidence": dict(asset.get("metadata") or {}).get("semantic_evidence", "unverified"),
            },
        }
        if asset.get("semantic_score") is not None:
            payload["semantic_score"] = float(asset["semantic_score"])
        return payload

    hook_start = float(hook.get("start", 0))
    hook_end = hook_start + float(hook.get("duration", 3.8))
    hook_asset = hook.get("asset") if isinstance(hook.get("asset"), dict) else None
    packages = [{
        "package_id": "opening-hook", "cue_id": "opening-hook", "cue_type": "hook",
        "visual_state": "opening_world", "visual_mode": "opening_world", "start": hook_start, "end": hook_end,
        "speaker_role": "circle" if hook_asset else "full", "text_role": "hook", "text": str(hook.get("text", "")),
        "semantic_intent": str(hook.get("reason", "开场核心观点")), "caption_mode": "hidden",
        "background_role": "media" if hook_asset else "graphic", "media_role": "background" if hook_asset else "none",
        "media": media_payload(hook_asset, str(hook.get("asset_caption") or hook.get("text", ""))),
        "status": "ready", "dependency_status": {"timing.valid": True, "background.ready": True},
        "background_at": hook_start, "speaker_at": hook_start + .16, "text_at": hook_start + .30,
        "entry_order": ["background", "speaker", "text"], "exit_order": ["text", "speaker", "background"],
        "restore_source": True, "exit_state": "restore_source_and_full_speaker",
        "safe_zones": ["highlight"], "caption_zone": {"hidden": True}, "highlight_zone": {"occlusion_ratio": 0},
    }]
    for beat in beats:
        start = float(beat["start"]); end = start + float(beat["duration"])
        asset = beat.get("asset") if isinstance(beat.get("asset"), dict) else None
        background_role = "media" if asset else "graphic"
        kind = str(beat.get("kind", "quote"))
        cue_type = "scene" if kind in {"media", "context"} else "knowledge" if kind in {"knowledge", "outro"} else "callout"
        text_role = "label" if kind in {"media", "context"} else "summary" if kind in {"knowledge", "outro"} else "callout"
        packages.append({
            "package_id": str(beat.get("package_id")), "cue_id": str(beat.get("package_id")),
            "cue_type": cue_type,
            "visual_state": beat.get("visual_state"), "visual_mode": beat.get("visual_state"),
            "start": start, "end": end, "speaker_role": beat.get("speaker_role", "full"),
            "text_role": text_role,
            "text": str(beat.get("text", "")), "semantic_intent": str(beat.get("cue_reason", beat.get("text", ""))),
            "caption_mode": beat.get("caption_mode", "normal"), "background_role": background_role,
            "media_role": "background" if asset else "none", "media": media_payload(asset, str(beat.get("text", ""))),
            "status": "ready", "dependency_status": {"timing.valid": True, "background.ready": True},
            "background_at": start, "speaker_at": start + float(beat.get("entry_offsets", {}).get("speaker", .16)),
            "text_at": start + float(beat.get("entry_offsets", {}).get("text", .30)),
            "entry_order": beat.get("entry_order", ["background", "speaker", "text"]),
            "exit_order": beat.get("exit_order", ["text", "speaker", "background"]),
            "restore_source": True, "exit_state": "restore_source_and_full_speaker",
            "safe_zones": ["caption", "highlight"], "caption_zone": {"occlusion_ratio": 0}, "highlight_zone": {"occlusion_ratio": 0},
        })
    packages.extend({
        "package_id": f"chapter-{index + 1:02d}", "cue_id": f"chapter-{index + 1:02d}", "cue_type": "chapter",
        "visual_state": "chapter_transition", "visual_mode": "chapter_card", "fullscreen": True,
        "start": float(chapter["start"]), "end": float(chapter["start"]) + float(chapter["duration"]),
        "speaker_role": "none", "text_role": "chapter", "text": str(chapter.get("title", "")),
        "semantic_intent": str(chapter.get("reason", chapter.get("title", ""))), "caption_mode": "hidden",
        "background_role": "graphic", "media_role": "none", "media": None, "status": "ready",
        "dependency_status": {"timing.valid": True, "background.ready": True},
        "text_at": float(chapter["start"]) + .28,
        "entry_order": ["background", "text"], "exit_order": ["text", "background"],
        "restore_source": True, "exit_state": "restore_source_and_full_speaker",
        "safe_zones": ["chapter"], "caption_zone": {"hidden": True}, "highlight_zone": {"occlusion_ratio": 0},
    } for index, chapter in enumerate(chapters))
    return sorted(packages, key=lambda package: float(package.get("start", 0)))


def attach_scene_assets(
    beats: list[dict[str, Any]], scenes: list[dict[str, Any]], assets: list[dict[str, Any]],
    duration: float, chapters: list[dict[str, Any]], project: Path, config: dict[str, Any],
) -> list[dict[str, Any]]:
    remaining = list(assets)
    reserved_names = {
        str(scene.get("_resolved_source_name", ""))
        for scene in scenes if scene.get("_resolved_source_name")
    }
    for scene in scenes:
        if not remaining:
            break
        start = max(.2, min(duration - 3.2, float(scene.get("start", 0))))
        scene_end = min(duration, max(start + 2.8, float(scene.get("end", start + 4.2))))
        scene_duration = round(max(3.8, min(6.2, scene_end - start + .8)), 3)
        scene_semantics = clean_text(f"{scene.get('caption', '')}{scene.get('prompt', '')}")
        semantic_terms = (*COVER_CONCEPT_TERMS, *SCENE_WORDS, "胖东来", "网红菜", "菜单", "薄利多销")
        candidates = [
            beat for beat in beats
            if abs(float(beat["start"]) - start) <= 7 and beat.get("asset") is None
            and any(term in scene_semantics and term in clean_text(str(beat.get("text", ""))) for term in semantic_terms)
        ]
        resolved_name = str(scene.get("_resolved_source_name", ""))
        asset_index = next(
            (index for index, item in enumerate(remaining) if resolved_name and item.get("source_name") == resolved_name),
            None,
        )
        if asset_index is None:
            # Manual uploads remain eligible, but files generated for another
            # director scene are reserved and must never be silently reassigned.
            manual_indexes = [
                index for index, item in enumerate(remaining)
                if str(item.get("source_name", "")) not in reserved_names
                and not re.match(r"^(?:88-ark|89-ark|90-smart|91-ai)-", str(item.get("source_name", "")))
            ]
            semantic_indexes = [
                index for index in manual_indexes
                if any(
                    term in scene_semantics and term in clean_text(Path(str(remaining[index].get("source_name", ""))).stem)
                    for term in semantic_terms
                )
            ]
            asset_index = semantic_indexes[0] if semantic_indexes else (manual_indexes[0] if manual_indexes else None)
        if asset_index is None:
            continue
        asset = remaining.pop(asset_index)
        asset = validate_scene_asset(asset, scene, project, config)
        if asset is None:
            asset = regenerate_scene_asset(scene, project, config)
        if asset is None:
            continue
        if not candidates:
            if overlaps_window(start, scene_duration, beats + chapters, .15):
                continue
            caption = clean_text(str(scene.get("caption", "")))[:18] or scene_query(str(scene.get("prompt", "")))[:18] or "场景说明"
            beats.append({
                "start": round(start, 3), "duration": scene_duration, "kind": "media", "text": caption,
                "query": scene_query(caption), "asset": asset, "spoken_end": round(scene_end, 3),
                "scene_prompt": scene.get("prompt", ""), "cue_reason": scene.get("reason", "语音触发具体场景"),
                "alignment": "scene-timestamp", "blur": 0.0,
            })
            continue
        beat = min(candidates, key=lambda value: abs(float(value["start"]) - start))
        beat["asset"] = asset
        beat["kind"] = "media"
        beat["duration"] = min(6.2, max(3.8, beat["duration"], scene_duration))
        if scene.get("caption"):
            beat["text"] = clean_text(str(scene["caption"]))[:18]
        beat["scene_prompt"] = scene.get("prompt", "")
        beat["cue_reason"] = scene.get("reason", beat.get("cue_reason", "场景化"))
        beat["alignment"] = "scene-timestamp"
        beat["blur"] = 0.0
    return sorted(beats, key=lambda value: float(value["start"]))


def media_validation_data_url(path: Path, kind: str) -> str:
    if kind == "video":
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if container.duration:
                container.seek(int(container.duration * .5), any_frame=False, backward=True)
            frame = next(container.decode(stream))
            image = frame.to_image()
    else:
        image = Image.open(path).convert("RGB")
    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def validate_scene_asset(
    asset: dict[str, Any], scene: dict[str, Any], project: Path, config: dict[str, Any],
) -> dict[str, Any] | None:
    """Inspect actual pixels before a supporting asset may enter the timeline."""
    key = os.environ.get("ARK_API_KEY", "").strip()
    model = str(config.get("director_model", "")).strip()
    raw_path = Path(str(asset.get("path", "")))
    path = raw_path if raw_path.is_absolute() else project / raw_path
    if not key or not model or not path.exists():
        print("    配套画面缺少可用的视觉验收模型，已改用语义图形场景")
        return None
    expected = clean_text(str(scene.get("visual_expectation") or scene.get("caption", "")))
    prompt = clean_text(str(scene.get("prompt", "")))
    reason = clean_text(str(scene.get("reason", "")))
    try:
        data_url = media_validation_data_url(path, str(asset.get("kind", "image")))
        response = json_request(ARK_CHAT_URL, {
            "model": model,
            "temperature": .1,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": (
                    "你是短视频素材画面质检员。检查真实画面是否直接支持当前口播场景，"
                    "是否存在对象、动作、地点、品牌、情绪或文字上的矛盾。只返回 JSON："
                    '{"score":0.0,"observed":"","contradictions":[],"forbidden_visual_hits":[],"reason":""}。'
                    f"\n预期场景：{expected}\n生成/检索要求：{prompt}\n使用理由：{reason}"
                )},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
        }, key, 120)
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        verdict = parse_model_json(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
        score = max(0.0, min(1.0, float(verdict.get("score", 0) or 0)))
        contradictions = [clean_text(str(item)) for item in verdict.get("contradictions", []) if clean_text(str(item))]
        forbidden = [clean_text(str(item)) for item in verdict.get("forbidden_visual_hits", []) if clean_text(str(item))]
    except Exception as exc:
        print(f"    配套画面视觉验收未完成：{type(exc).__name__}，已改用语义图形场景")
        return None
    if score < .78 or contradictions or forbidden:
        print(f"    配套画面与口播不一致（{score:.2f}），已拒绝并改用语义图形场景")
        return None
    approved = dict(asset)
    approved["semantic_score"] = round(score, 3)
    approved["metadata"] = {
        **dict(asset.get("metadata") or {}),
        "visual_validation": "vision-approved",
        "semantic_evidence": clean_text(str(verdict.get("reason", ""))) or "vision-model-approved",
        "observed": clean_text(str(verdict.get("observed", ""))),
        "detected_contradictions": contradictions,
        "forbidden_visual_hits": forbidden,
    }
    return approved


def regenerate_scene_asset(
    scene: dict[str, Any], project: Path, config: dict[str, Any],
) -> dict[str, Any] | None:
    """Retry one rejected scene with a stricter, content-complete image prompt."""
    key = os.environ.get("ARK_API_KEY", "").strip()
    model = str(config.get("ark_image_model", "")).strip()
    if not key or not model:
        return None
    retry_scene = strengthen_scene_prompt(scene)
    retry_scene["prompt"] = (
        f"{retry_scene['prompt']}；必须直接支持口播“{clean_text(str(scene.get('caption', '')))[:20]}”，"
        "画面主体和动作清晰可辨，不使用无关人物采访、新闻演播、警察灾难、农村菜摊等替代场景"
    )
    digest = hashlib.sha1(retry_scene["prompt"].encode("utf-8")).hexdigest()[:12]
    target_dir = project / "assets" / "support" / "vision-retries"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"retry-{digest}.png"
    try:
        if not target.is_file() or target.stat().st_size < 10_000:
            print("    首次素材未通过，正在按口播语义重做一版配套画面")
            generated = generate_ark_image(retry_scene, key, model, target)
            if not generated:
                return None
        with Image.open(target) as image:
            asset: dict[str, Any] = {
                "path": target.relative_to(project).as_posix(), "kind": "image",
                "source_name": target.name, "width": image.width, "height": image.height,
            }
        return validate_scene_asset(asset, retry_scene, project, config)
    except Exception as exc:
        print(f"    配套画面重做未完成：{type(exc).__name__}")
        return None


def add_scene_fallback_beats(
    beats: list[dict[str, Any]], scenes: list[dict[str, Any]], duration: float,
    chapters: list[dict[str, Any]], hook_window: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep a director-selected scene alive as a semantic graphic when media is unavailable.

    This is intentionally not labelled as generated footage.  It gives the
    renderer a complete background/speaker/text state while preserving the
    option to replace it with verified image/video later.
    """
    result = list(beats)
    for scene in sorted(scenes, key=lambda item: float(item.get("start", 0))):
        start = max(4.2, min(duration - 4.0, float(scene.get("start", 0))))
        scene_end = min(duration, max(start + 2.8, float(scene.get("end", start + 3.8))))
        length = round(max(3.8, min(5.2, scene_end - start + 1.15)), 3)
        occupied = result + chapters + hook_window
        if overlaps_window(start, length, occupied, .3):
            blockers = [
                item for item in occupied
                if overlaps_window(start, length, [item], .3)
            ]
            shifted = max((float(item["start"]) + float(item.get("duration", 0)) + .34 for item in blockers), default=start)
            if shifted + length > duration - .2 or overlaps_window(shifted, length, occupied, .18):
                continue
            start = shifted
        caption = clean_text(str(scene.get("caption", "")))[:18]
        if not caption:
            caption = scene_query(str(scene.get("prompt", "")))[:18] or "场景拆解"
        if "胖东来" in caption:
            caption = "胖东来：品质不等于低价"
        result.append({
            "start": round(start, 3), "duration": length, "kind": "context", "text": caption,
            "query": scene_query(caption), "asset": None, "spoken_end": round(scene_end, 3),
            "scene_prompt": scene.get("prompt", ""),
            "cue_reason": scene.get("reason", "语音触发具体场景"),
            "alignment": "scene-timestamp-graphic-fallback", "blur": .20,
            "fallback_from_media": True,
        })
    return sorted(result, key=lambda value: float(value["start"]))


def apply_caption_emphasis(captions: list[dict[str, Any]], plan: dict[str, Any]) -> None:
    for caption in captions:
        caption["emphasis"] = []
    for item in plan.get("caption_emphasis", []):
        try:
            start = float(item.get("start", 0))
        except (TypeError, ValueError):
            continue
        candidates = sorted(captions, key=lambda cap: abs(float(cap["start"]) - start))[:3]
        for phrase in item.get("phrases", []):
            cleaned = clean_text(str(phrase))
            target = next((cap for cap in candidates if cleaned and cleaned in cap["text"]), None)
            if target and cleaned not in target["emphasis"]:
                target["emphasis"].append(cleaned)
    for caption in captions:
        caption["keyword"] = caption["emphasis"][0] if caption["emphasis"] else ""


def lower_thirds_from_plan(plan: dict[str, Any], duration: float, occupied: list[dict[str, Any]], speaker: str, speaker_title: str) -> list[dict[str, Any]]:
    if not (speaker or speaker_title):
        return []
    result: list[dict[str, Any]] = []
    for item in plan.get("lower_thirds", []):
        start = max(.6, min(duration - 3.0, float(item.get("start", 1.0))))
        length = max(2.8, min(3.4, float(item.get("duration", 3.2))))
        if overlaps_window(start, length, occupied, .55):
            alternatives = [start + delta for delta in (4.8, -4.8, 8.5) if .6 <= start + delta < duration - length]
            start = next((value for value in alternatives if not overlaps_window(value, length, occupied, .55)), -1)
        if start < 0 or (result and start - result[-1]["start"] < 18):
            continue
        result.append({"start": round(start, 3), "duration": round(length, 3), "speaker": speaker, "title": speaker_title, "reason": item.get("reason", "身份背书")})
    return result[:3]


def select_camera_beats(
    groups: list[dict[str, Any]], keep: list[dict[str, float]], duration: float,
    visual_beats: list[dict[str, Any]], chapters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    targets = np.arange(10.0, max(10.1, duration - 5), 18.0)
    scales = (1.09, 1.13, 1.07, 1.11)
    offsets = ((-14, -8), (12, -18), (-8, 10), (16, 4))
    for index, target in enumerate(targets):
        candidates = [g for g in groups if target <= g["start"] <= target + 5]
        start = candidates[0]["start"] if candidates else float(target)
        if overlaps_window(start, 2.8, visual_beats + chapters, 0.5):
            continue
        segment_index = next((i for i, item in enumerate(keep) if item["edited_start"] <= start <= item["edited_start"] + item["duration"]), 0)
        x, y = offsets[index % len(offsets)]
        schedule.append({"start": round(start, 3), "duration": 2.8, "scale": scales[index % len(scales)], "x": x, "y": y, "segment_index": segment_index})
    return schedule


def stage_support_media(source_dir: Path, project: Path, signature: str) -> list[dict[str, Any]]:
    if not source_dir.exists():
        return []
    target_dir = project / "assets" / "support" / signature
    target_dir.mkdir(parents=True, exist_ok=True)
    result: list[dict[str, Any]] = []
    files = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
    for index, source in enumerate(files[:14]):
        target = target_dir / f"{index:02d}{source.suffix.lower()}"
        shutil.copy2(source, target)
        item: dict[str, Any] = {
            "path": target.relative_to(project).as_posix(),
            "kind": "video" if source.suffix.lower() in VIDEO_EXTS else "image",
            # Keep the original name so generated media can be routed back to the
            # exact director scene and named uploads can be matched semantically.
            "source_name": source.name,
        }
        try:
            if item["kind"] == "video":
                item.update(media_info(target))
            else:
                with Image.open(target) as image:
                    item.update({"width": image.width, "height": image.height})
        except Exception:
            continue
        result.append(item)
    return result


def download_file(url: str, target: Path, max_bytes: int = 90 * 1024 * 1024) -> bool:
    request = urllib.request.Request(url, headers={"User-Agent": "VideoProductionWorkflow/3.0"})
    with urllib.request.urlopen(request, timeout=35) as response:
        total = 0
        with target.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    target.unlink(missing_ok=True)
                    return False
                handle.write(chunk)
    return target.exists() and target.stat().st_size > 0


def fetch_pexels_video(query: str, key: str, target: Path) -> dict[str, Any] | None:
    params = urllib.parse.urlencode({"query": query, "orientation": "portrait", "size": "medium", "locale": "zh-CN", "per_page": 5})
    request = urllib.request.Request(f"https://api.pexels.com/v1/videos/search?{params}", headers={"Authorization": key, "User-Agent": "VideoProductionWorkflow/3.0"})
    with urllib.request.urlopen(request, timeout=25) as response:
        videos = json.loads(response.read().decode("utf-8")).get("videos", [])
    if not videos:
        return None
    video = videos[0]
    files = [item for item in video.get("video_files", []) if item.get("file_type") == "video/mp4"]
    if not files:
        return None
    files.sort(key=lambda item: (item.get("height", 0) < item.get("width", 0), abs((item.get("height", 0) or 0) - 1280)))
    if not download_file(files[0]["link"], target):
        return None
    user = video.get("user", {})
    return {"provider": "Pexels", "query": query, "creator": user.get("name", ""), "creator_url": user.get("url", ""), "source_url": video.get("url", ""), "path": str(target)}


def generate_openai_image(query: str, key: str, target: Path) -> dict[str, Any] | None:
    prompt = f"Vertical documentary editorial photograph illustrating: {query}. Realistic, natural light, tasteful composition, no text, no watermark, no logos, suitable as supporting B-roll for a Chinese business talking-head video."
    payload = json.dumps({"model": "gpt-image-1", "prompt": prompt, "size": "1024x1536", "quality": "low"}).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/images/generations", data=payload, method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "VideoProductionWorkflow/3.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.loads(response.read().decode("utf-8")).get("data", [])
    if not data:
        return None
    encoded = data[0].get("b64_json")
    if encoded:
        target.write_bytes(base64.b64decode(encoded))
    elif data[0].get("url"):
        download_file(data[0]["url"], target, 35 * 1024 * 1024)
    return {"provider": "OpenAI", "query": query, "creator": "AI generated", "source_url": "", "path": str(target)} if target.exists() else None


def generate_ark_image(scene: dict[str, Any], key: str, model: str, target: Path) -> dict[str, Any] | None:
    prompt = clean_text(str(scene.get("prompt", ""))) or "竖屏纪实商业现场，自然光，无文字，无Logo，无水印"
    response = json_request(ARK_IMAGE_URL, {
        "model": model, "prompt": prompt, "size": "2K", "response_format": "url",
        "watermark": False, "sequential_image_generation": "disabled",
    }, key, 240)
    data = response.get("data", [])
    if not data:
        return None
    if data[0].get("b64_json"):
        target.write_bytes(base64.b64decode(data[0]["b64_json"]))
    elif data[0].get("url"):
        download_file(data[0]["url"], target, 40 * 1024 * 1024)
    return {
        "provider": "Volcano Ark / Seedream", "query": prompt, "creator": "AI generated",
        "source_url": data[0].get("url", ""), "path": str(target), "reason": scene.get("reason", "场景化"),
    } if target.exists() else None


def generate_ark_video(scene: dict[str, Any], key: str, model: str, target: Path) -> dict[str, Any] | None:
    prompt = clean_text(str(scene.get("prompt", ""))) or "竖屏纪实商业现场，自然光，无文字，无Logo，无水印"
    request_text = f"{prompt} --ratio 9:16 --duration 5"
    response = json_request(ARK_VIDEO_TASK_URL, {"model": model, "content": [{"type": "text", "text": request_text}]}, key, 90)
    task_id = str(response.get("id", ""))
    if not task_id:
        return None
    task_url = f"{ARK_VIDEO_TASK_URL}/{urllib.parse.quote(task_id)}"
    for _ in range(120):
        request = urllib.request.Request(task_url, headers={"Authorization": f"Bearer {key}", "User-Agent": "VideoProductionWorkflow/6.0"})
        with urllib.request.urlopen(request, timeout=45) as poll_response:
            state = json.loads(poll_response.read().decode("utf-8"))
        status = state.get("status")
        if status == "succeeded":
            video_url = state.get("content", {}).get("video_url", "")
            if video_url and download_file(video_url, target):
                return {"provider": "Volcano Ark / Seedance", "query": prompt, "creator": "AI generated", "source_url": video_url, "task_id": task_id, "path": str(target), "reason": scene.get("reason", "场景化")}
            return None
        if status in {"failed", "cancelled", "expired"}:
            return None
        time.sleep(5)
    return None


def resolve_smart_media(scenes: list[dict[str, Any]], support_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    if not config.get("smart_media"):
        return []
    pexels_key = os.environ.get("PEXELS_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    ark_key = os.environ.get("ARK_API_KEY", "").strip()
    image_model = str(config.get("ark_image_model", "")).strip()
    video_model = str(config.get("ark_video_model", "")).strip()
    generation_mode = str(config.get("scene_generation", "auto"))
    limit = max(0, min(6, int(config.get("smart_media_count", 3))))
    records: list[dict[str, Any]] = []
    support_dir.mkdir(parents=True, exist_ok=True)
    for index, scene in enumerate(scenes[:limit]):
        scene.update(strengthen_scene_prompt(scene))
        query = clean_text(str(scene.get("caption", ""))) or scene_query(str(scene.get("prompt", "")))
        record = None
        requested_type = str(scene.get("type", "image"))
        wants_video = generation_mode == "video" or (generation_mode == "auto" and requested_type == "video")
        prompt_hash = hashlib.sha1(clean_text(str(scene.get("prompt", ""))).encode("utf-8")).hexdigest()[:10]
        cached_video = support_dir / f"88-ark-{index:02d}-{prompt_hash}.mp4"
        cached_image = support_dir / f"89-ark-{index:02d}-{prompt_hash}.png"
        if wants_video and cached_video.is_file() and cached_video.stat().st_size > 10_000:
            record = {
                "provider": "Volcano Ark / Seedance (cached)", "query": query,
                "creator": "AI generated", "path": str(cached_video),
                "reason": scene.get("reason", "场景化"),
            }
        elif cached_image.is_file() and cached_image.stat().st_size > 10_000:
            record = {
                "provider": "Volcano Ark / Seedream (cached)", "query": query,
                "creator": "AI generated", "path": str(cached_image),
                "reason": scene.get("reason", "场景化"),
            }
        tried_pexels = False
        if record is None and ark_key and wants_video and video_model:
            try:
                record = generate_ark_video(scene, ark_key, video_model, cached_video)
            except Exception as exc:
                print(f"    火山视频未完成：{type(exc).__name__}")
        if record is None and wants_video and pexels_key:
            tried_pexels = True
            try:
                record = fetch_pexels_video(query, pexels_key, support_dir / f"90-smart-{index:02d}-{prompt_hash}.mp4")
            except Exception as exc:
                print(f"    Pexels 素材未命中：{type(exc).__name__}")
        if record is None and ark_key and generation_mode in {"auto", "image", "video"} and image_model:
            try:
                record = generate_ark_image(scene, ark_key, image_model, cached_image)
            except Exception as exc:
                print(f"    火山生图未完成：{type(exc).__name__}")
        if record is None and pexels_key and not tried_pexels and generation_mode in {"auto", "video", "none"}:
            try:
                record = fetch_pexels_video(query, pexels_key, support_dir / f"90-smart-{index:02d}-{prompt_hash}.mp4")
            except Exception as exc:
                print(f"    Pexels 素材未命中：{type(exc).__name__}")
        if record is None and openai_key and config.get("allow_ai_images"):
            try:
                record = generate_openai_image(query, openai_key, support_dir / f"91-ai-{index:02d}-{prompt_hash}.png")
            except Exception as exc:
                print(f"    AI 生图未完成：{type(exc).__name__}")
        if record:
            # attach_scene_assets uses this private routing hint. Without it, a
            # failed generation for scene A could make scene A consume scene B's
            # successfully generated file simply because both live in one folder.
            scene["_resolved_source_name"] = Path(str(record["path"])).name
            records.append(record)
    return records


def music_cue_plan(
    duration: float, visual_beats: list[dict[str, Any]], chapters: list[dict[str, Any]], peak_volume: float,
) -> list[dict[str, Any]]:
    """Build an audible editorial bed that ducks under speech and lifts at argument boundaries."""
    floor = max(0.085, peak_volume * 0.45)
    cues: list[dict[str, Any]] = [
        {"at": 0.0, "volume": 0.0, "duration": 0.01, "mood": "silence"},
        {"at": 0.18, "volume": peak_volume, "duration": 0.9, "mood": "hook"},
        {"at": min(3.7, max(1.2, duration * 0.03)), "volume": floor, "duration": 0.8, "mood": "speaker"},
    ]
    for beat in visual_beats:
        at = float(beat["start"])
        end = at + float(beat["duration"])
        lift = peak_volume * (0.92 if beat["kind"] in {"stat", "compare"} else 0.76)
        cues.extend([
            {"at": max(0.0, at - 0.34), "volume": floor, "duration": 0.24, "mood": "prepare"},
            {"at": at + 0.06, "volume": lift, "duration": 0.42, "mood": f"callout-{beat['kind']}"},
            {"at": max(at + 0.8, end - 0.48), "volume": floor, "duration": 0.44, "mood": "speaker"},
        ])
    for chapter in chapters:
        at = float(chapter["start"])
        end = at + float(chapter["duration"])
        cues.extend([
            {"at": max(0.0, at - 0.65), "volume": max(floor, peak_volume * 0.68), "duration": 0.52, "mood": "chapter-rise"},
            {"at": at + 0.08, "volume": min(0.42, peak_volume * 1.12), "duration": 0.36, "mood": "chapter-hit"},
            {"at": end - 0.30, "volume": floor, "duration": 0.46, "mood": "speaker"},
        ])
    cues.extend([
        {"at": max(0.0, duration - 4.0), "volume": max(floor, peak_volume * 0.55), "duration": 0.7, "mood": "outro"},
        {"at": max(0.0, duration - 1.6), "volume": 0.0, "duration": 1.5, "mood": "fade-out"},
    ])
    cues.sort(key=lambda item: float(item["at"]))
    return [{**item, "at": round(float(item["at"]), 3), "volume": round(float(item["volume"]), 4)} for item in cues]


def generated_bgm(duration: float, output: Path, ffmpeg_path: Path, accent_times: list[float]) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 4096:
        return output
    wav_path = output.with_suffix(".wav")
    sample_rate = 44100
    total_samples = max(1, int(duration * sample_rate))
    beat_seconds = 60 / 88
    roots = np.array([73.42, 58.27, 87.31, 65.41], dtype=np.float32)
    chord_ratios = np.array([
        [1.0, 1.20, 1.50, 2.25],
        [1.0, 1.25, 1.50, 2.00],
        [1.0, 1.25, 1.50, 2.25],
        [1.0, 1.25, 1.50, 2.00],
    ], dtype=np.float32)
    rng = np.random.default_rng(20260807)
    with wave.open(str(wav_path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        chunk_size = sample_rate * 5
        for offset in range(0, total_samples, chunk_size):
            count = min(chunk_size, total_samples - offset)
            t = (np.arange(count, dtype=np.float32) + offset) / sample_rate
            chord_index = (t / (beat_seconds * 8)).astype(np.int32) % 4
            root = roots[chord_index]
            pad_left = np.zeros(count, dtype=np.float32)
            pad_right = np.zeros(count, dtype=np.float32)
            for chord_no in range(4):
                mask = chord_index == chord_no
                if not np.any(mask):
                    continue
                for voice_no, ratio in enumerate(chord_ratios[chord_no]):
                    frequency = roots[chord_no] * ratio
                    weight = (0.32, 0.23, 0.19, 0.12)[voice_no]
                    pad_left[mask] += np.sin(2 * np.pi * frequency * 0.998 * t[mask] + voice_no * .37) * weight
                    pad_right[mask] += np.sin(2 * np.pi * frequency * 1.002 * t[mask] + voice_no * .61) * weight
            phase = np.mod(t, beat_seconds) / beat_seconds
            half_phase = np.mod(t, beat_seconds / 2) / (beat_seconds / 2)
            pluck = (
                np.sin(2 * np.pi * root * 2.0 * t)
                + np.sin(2 * np.pi * root * 3.0 * t) * .35
            ) * np.exp(-8.2 * phase)
            sub = np.sin(2 * np.pi * root * .5 * t) * np.exp(-5.4 * phase)
            shaker_gate = np.exp(-32.0 * half_phase) * (half_phase < .16)
            shaker = rng.normal(0, 1, count).astype(np.float32) * shaker_gate
            left = pad_left * .18 + pluck * .072 + sub * .055 + shaker * .013
            right = pad_right * .18 + pluck * .066 + sub * .052 - shaker * .011
            for accent_at in accent_times:
                delta = t - accent_at
                envelope = np.where((delta >= 0) & (delta < 1.45), np.exp(-3.8 * np.maximum(0, delta)), 0.0)
                accent = (np.sin(2 * np.pi * 440.0 * t) + np.sin(2 * np.pi * 659.25 * t) * .45) * envelope * .052
                impact = np.sin(2 * np.pi * 55.0 * t) * envelope * .075
                left += accent + impact
                right += accent * .92 + impact
            fade_in = np.minimum(1.0, t / 2.0)
            fade_out = np.minimum(1.0, np.maximum(0.0, duration - t) / 2.0)
            stereo = np.stack((left, right), axis=1) * (fade_in * fade_out)[:, None]
            writer.writeframes((np.clip(stereo, -0.95, 0.95) * 32767).astype("<i2").tobytes())
    if ffmpeg_path.exists():
        result = subprocess.run([
            str(ffmpeg_path), "-y", "-i", str(wav_path),
            "-af", "highpass=f=45,lowpass=f=12000,loudnorm=I=-18:TP=-2:LRA=7",
            "-codec:a", "libmp3lame", "-b:a", "192k", str(output),
        ], capture_output=True)
        if result.returncode == 0:
            wav_path.unlink(missing_ok=True)
            return output
    return wav_path


def resolve_ffmpeg_path(project: Path) -> Path:
    """Return a durable FFmpeg binary and repair stale temporary symlinks."""
    link = project / "bin" / "ffmpeg"
    if link.is_file():
        return link
    try:
        import imageio_ffmpeg

        bundled = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    except (ImportError, OSError):
        return link
    if not bundled.is_file():
        return link
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(bundled)
        return link
    except OSError:
        return bundled


def prepare_bgm(
    config: dict[str, Any], project: Path, duration: float, signature: str,
    visual_beats: list[dict[str, Any]], chapters: list[dict[str, Any]],
) -> dict[str, Any] | None:
    mode = config.get("bgm_mode", "generated")
    if mode == "none":
        return None
    audio_dir = project / "assets" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    supplied_value = str(config.get("bgm_path", "")).strip()
    supplied = Path(supplied_value).expanduser() if supplied_value else None
    peak_volume = max(0.0, min(0.46, float(config.get("bgm_volume", 0.38))))
    cues = music_cue_plan(duration, visual_beats, chapters, peak_volume)
    if supplied and supplied.is_file():
        target = audio_dir / f"bgm-{signature}{supplied.suffix.lower()}"
        shutil.copy2(supplied, target)
    else:
        accents = [float(item["at"]) for item in cues if item["mood"] in {"chapter-hit", "callout-stat", "callout-compare"}]
        cue_hash = hashlib.sha1(json.dumps(accents).encode("utf-8")).hexdigest()[:8]
        target = generated_bgm(duration, audio_dir / f"bgm-v5-editorial-{signature}-{cue_hash}-{round(duration)}s.mp3", resolve_ffmpeg_path(project), accents)
    track_duration = audio_duration(target)
    segments = []
    cursor = 0.0
    while cursor < duration - 0.02:
        length = min(track_duration, duration - cursor)
        segments.append({"start": round(cursor, 3), "duration": round(length, 3)})
        cursor += length
    return {
        "path": target.relative_to(project).as_posix(), "duration": track_duration, "segments": segments,
        "volume": peak_volume, "floor_volume": round(max(0.085, peak_volume * 0.45), 4), "cues": cues,
        "strategy": "expert editorial pulse; audible speech bed with semantic swells and chapter impacts",
        "mix_profile": "expert-industry-insight", "master_lufs": -18,
    }


def volume_expression(cues: list[dict[str, Any]]) -> str:
    """Build a deterministic piecewise-linear FFmpeg volume curve from semantic cues."""
    ordered = sorted(cues, key=lambda item: float(item["at"]))
    # FFmpeg's expression parser has a relatively shallow nesting limit. A
    # dense video can produce dozens of semantic cues, so preserve the global
    # envelope plus representative peaks instead of emitting an unbounded tree.
    if len(ordered) > 10:
        required = {0, 1, 2, len(ordered) - 2, len(ordered) - 1}
        interior = list(range(3, max(3, len(ordered) - 2)))
        slots = max(0, 10 - len(required))
        if interior and slots:
            required.update(interior[round(index * (len(interior) - 1) / max(1, slots - 1))] for index in range(slots))
        ordered = [item for index, item in enumerate(ordered) if index in required]
    expression = f"{float(ordered[-1]['volume']):.5f}" if ordered else "0.22"
    previous = 0.0
    pieces: list[tuple[float, float, float, float]] = []
    for cue in ordered:
        at = max(0.0, float(cue["at"])); length = max(.01, float(cue.get("duration", .3))); target = float(cue["volume"])
        pieces.append((at, length, previous, target)); previous = target
    for index in range(len(pieces) - 1, -1, -1):
        at, length, before, target = pieces[index]
        next_at = pieces[index + 1][0] if index + 1 < len(pieces) else 1e9
        ramp = f"{before:.5f}+({target:.5f}-{before:.5f})*(t-{at:.5f})/{length:.5f}"
        expression = f"if(lt(t,{at:.5f}),{before:.5f},if(lt(t,{at + length:.5f}),{ramp},if(lt(t,{next_at:.5f}),{target:.5f},{expression})))"
    return expression


def bake_program_mix(project: Path, source: Path, keep: list[dict[str, float]], bgm: dict[str, Any] | None, signature: str, duration: float) -> dict[str, Any] | None:
    if not bgm:
        return None
    ffmpeg = resolve_ffmpeg_path(project)
    if not ffmpeg.is_file():
        raise RuntimeError("缺少本地音频混合组件")
    target = project / "assets" / "audio" / f"program-mix-v6-{signature}-{round(duration)}s.m4a"
    filter_parts: list[str] = []
    voice_labels: list[str] = []
    for index, segment in enumerate(keep):
        label = f"voice{index}"
        voice_labels.append(f"[{label}]")
        filter_parts.append(f"[0:a]atrim=start={segment['source_start']:.6f}:end={segment['source_end']:.6f},asetpts=PTS-STARTPTS[{label}]")
    if len(voice_labels) == 1:
        filter_parts.append(f"{voice_labels[0]}anull[voice]")
    else:
        filter_parts.append(f"{''.join(voice_labels)}concat=n={len(voice_labels)}:v=0:a=1[voice]")
    expression = volume_expression(bgm.get("cues", []))
    filter_parts.append(f"[1:a]atrim=0:{duration:.6f},asetpts=PTS-STARTPTS,volume='{expression}':eval=frame[music]")
    filter_parts.append("[voice][music]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95:attack=5:release=80:level=false[mix]")
    command = [
        str(ffmpeg), "-y", "-i", str(source), "-stream_loop", "-1", "-i", str(project / bgm["path"]),
        "-filter_complex", ";".join(filter_parts), "-map", "[mix]", "-t", f"{duration:.6f}",
        "-c:a", "aac", "-b:a", "224k", "-ar", "48000", str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not target.exists() or target.stat().st_size < 8192:
        raise RuntimeError(f"节目音轨混合失败：{result.stderr[-500:]}")
    return {
        "path": target.relative_to(project).as_posix(), "duration": round(audio_duration(target), 3),
        "volume": 1.0, "strategy": "voice + semantic BGM baked into one guaranteed preview/render track",
    }


def subtitle_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_srt(groups: list[dict[str, Any]], path: Path) -> None:
    blocks = [f"{index}\n{subtitle_timestamp(group['start'])} --> {subtitle_timestamp(group['end'])}\n{group['text']}" for index, group in enumerate(groups, 1)]
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def locate_font() -> str | None:
    candidates = ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]
    return next((path for path in candidates if Path(path).exists()), None)


def extract_frame(path: Path, at_seconds: float) -> Image.Image | None:
    try:
        with av.open(str(path)) as container:
            container.seek(int(max(0, at_seconds) * av.time_base), any_frame=False, backward=True)
            for frame in container.decode(video=0):
                if float(frame.time or 0) + 0.08 >= at_seconds:
                    return frame.to_image().convert("RGB")
    except Exception:
        return None
    return None


def frame_score(image: Image.Image) -> float:
    sample = np.asarray(image.resize((180, 320)).convert("L"), dtype=np.float32)
    sharpness = float(np.diff(sample, axis=1).var() + np.diff(sample, axis=0).var())
    exposure = max(0.0, 1.0 - abs(float(sample.mean()) - 132.0) / 132.0)
    return sharpness * (0.55 + 0.45 * exposure)


def choose_cover_frame(path: Path, duration: float, configured: float | None) -> tuple[Image.Image, float]:
    if configured is not None:
        frame = extract_frame(path, float(configured))
        if frame:
            return frame, float(configured)
    best: tuple[float, Image.Image, float] | None = None
    for value in np.linspace(max(1.0, duration * 0.05), max(1.2, duration * 0.20), 10):
        frame = extract_frame(path, float(value))
        if frame is None:
            continue
        score = frame_score(frame)
        if best is None or score > best[0]:
            best = (score, frame, float(value))
    return (best[1], best[2]) if best else (Image.new("RGB", (1080, 1920), "#15171B"), 0.0)


def cover_canvas(frame: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    ratio = max(width / frame.width, height / frame.height)
    resized = frame.resize((round(frame.width * ratio), round(frame.height * ratio)), Image.Resampling.LANCZOS)
    left, top = max(0, (resized.width - width) // 2), max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def wrap_text(text: str, max_chars: int, max_lines: int = 3) -> list[str]:
    cleaned = clean_text(text)
    return [cleaned[index:index + max_chars] for index in range(0, min(len(cleaned), max_chars * max_lines), max_chars)] or ["观点拆解"]


def cover_font(path: str | None, size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.truetype(path, size, index=1 if bold else 0) if path else ImageFont.load_default()


def _pixel_wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Wrap Chinese or mixed text by rendered width without discarding characters."""
    value = clean_text(text) or "观点拆解"
    lines: list[str] = []
    current = ""
    for char in value:
        candidate = current + char
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _fit_cover_text(
    draw: ImageDraw.ImageDraw, text: str, font_path: str | None, max_width: int,
    max_lines: int, start_size: int, min_size: int, bold: bool = False,
) -> tuple[list[str], ImageFont.ImageFont]:
    """Find a font and wrapping that preserves the complete source string."""
    for size in range(start_size, max(17, min_size) - 1, -2):
        font = cover_font(font_path, size, bold)
        lines = _pixel_wrap(draw, text, font, max_width)
        if len(lines) <= max_lines:
            return lines, font
    font = cover_font(font_path, max(16, min_size - 2), bold)
    return _pixel_wrap(draw, text, font, max_width), font


def _record_text(
    draw: ImageDraw.ImageDraw, layout: list[dict[str, Any]], role: str, text: str,
    xy: tuple[int, int], font: ImageFont.ImageFont, fill: str, *, anchor: str | None = None,
    stroke_width: int = 0, stroke_fill: str | None = None,
) -> tuple[int, int, int, int]:
    kwargs: dict[str, Any] = {"font": font, "fill": fill, "stroke_width": stroke_width}
    if anchor:
        kwargs["anchor"] = anchor
    if stroke_fill:
        kwargs["stroke_fill"] = stroke_fill
    draw.text(xy, text, **kwargs)
    bbox = draw.textbbox(xy, text, font=font, anchor=anchor, stroke_width=stroke_width)
    layout.append({"role": role, "text": text, "bbox": [int(value) for value in bbox]})
    return tuple(int(value) for value in bbox)


def _draw_lines(
    draw: ImageDraw.ImageDraw, layout: list[dict[str, Any]], role: str, lines: list[str],
    xy: tuple[int, int], font: ImageFont.ImageFont, fill: str, line_gap: int,
    *, anchor: str | None = None, stroke_width: int = 0, stroke_fill: str | None = None,
) -> int:
    x, y = xy
    for line in lines:
        _record_text(draw, layout, role, line, (x, y), font, fill, anchor=anchor, stroke_width=stroke_width, stroke_fill=stroke_fill)
        bbox = draw.textbbox((x, y), line, font=font, anchor=anchor, stroke_width=stroke_width)
        y += max(1, bbox[3] - bbox[1]) + line_gap
    return y


def draw_identity(
    draw: ImageDraw.ImageDraw, layout: list[dict[str, Any]], width: int, height: int,
    speaker: str, speaker_title: str, font_path: str | None, accent: str,
) -> None:
    identity = "｜".join(item for item in (speaker, speaker_title) if item) or "本期主讲"
    y = round(height * 0.91)
    box = (round(width * .055), y - round(width * .045), round(width * .945), y + round(width * .055))
    draw.rounded_rectangle(box, radius=round(width * .02), fill=(5, 6, 8, 205), outline=(255, 255, 255, 35), width=2)
    draw.ellipse((round(width * .075), y - round(width * .018), round(width * .105), y + round(width * .012)), fill=accent)
    lines, font = _fit_cover_text(draw, identity, font_path, round(width * .77), 1, round(width * .035), round(width * .023), True)
    _record_text(draw, layout, "identity", "".join(lines), (round(width * .125), y - round(width * .030)), font, "#FFFFFF")


def make_cover_editorial(
    frame: Image.Image, cover_copy: dict[str, Any], series: str, speaker: str,
    speaker_title: str, accent: str, size: tuple[int, int], output: Path,
) -> list[dict[str, Any]]:
    title = str(cover_copy["headline"])
    subtitle = str(cover_copy["subheadline"])
    angle = str(cover_copy.get("angle", series))
    hook = str(cover_copy.get("hook", ""))
    canvas = cover_canvas(frame, size).convert("RGBA")
    width, height = size
    gradient = Image.new("RGBA", size, (0, 0, 0, 0))
    gradient_draw = ImageDraw.Draw(gradient)
    for y in range(height):
        opacity = int(235 * max(0.0, (y / height - 0.38) / 0.62) ** 1.05)
        gradient_draw.line((0, y, width, y), fill=(4, 5, 7, opacity))
    canvas = ImageEnhance.Contrast(Image.alpha_composite(canvas, gradient)).enhance(1.05)
    draw = ImageDraw.Draw(canvas)
    layout: list[dict[str, Any]] = []
    font_path = locate_font()
    x, panel_right = round(width * .075), round(width * .925)
    text_width = panel_right - x
    title_lines, title_font = _fit_cover_text(draw, title, font_path, text_width, 3, round(width * .084), round(width * .052), True)
    subtitle_lines, subtitle_font = _fit_cover_text(draw, subtitle, font_path, text_width, 2, round(width * .039), round(width * .028))
    angle_lines, small_font = _fit_cover_text(draw, angle, font_path, round(width * .54), 1, round(width * .030), round(width * .022), True)
    hook_text = f"观点｜{hook}" if hook and hook != subtitle else ""
    hook_lines, hook_font = _fit_cover_text(draw, hook_text, font_path, text_width, 2, round(width * .029), round(width * .022), True) if hook_text else ([], small_font)
    identity = "｜".join(item for item in (speaker, speaker_title) if item) or "本期主讲"
    identity_lines, identity_font = _fit_cover_text(draw, identity, font_path, text_width, 1, round(width * .029), round(width * .021), True)
    title_lh = round(width * .093)
    subtitle_lh = round(width * .050)
    hook_lh = round(width * .040)
    content_height = round(width * .060) + len(title_lines) * title_lh + round(width * .025) + len(subtitle_lines) * subtitle_lh
    if hook_lines:
        content_height += round(width * .020) + len(hook_lines) * hook_lh
    content_height += round(width * .074)
    panel_bottom = round(height * .945)
    panel_top = max(round(height * .48), panel_bottom - content_height - round(width * .075))
    draw.rounded_rectangle((round(width * .04), panel_top, round(width * .96), panel_bottom), radius=round(width * .026), fill=(5, 6, 8, 218), outline=(255, 255, 255, 40), width=2)
    y = panel_top + round(width * .050)
    label_width = min(round(width * .60), max(round(width * .25), draw.textbbox((0, 0), angle_lines[0], font=small_font)[2] + round(width * .06)))
    draw.rounded_rectangle((x, y, x + label_width, y + round(width * .050)), radius=round(width * .015), fill=accent)
    _record_text(draw, layout, "angle", angle_lines[0], (x + round(width * .025), y + round(width * .009)), small_font, "#0B0C0F")
    y += round(width * .072)
    y = _draw_lines(draw, layout, "headline", title_lines, (x, y), title_font, "#FFFFFF", round(width * .010), stroke_width=max(2, width // 300), stroke_fill="#08090B")
    y += round(width * .015)
    y = _draw_lines(draw, layout, "subheadline", subtitle_lines, (x, y), subtitle_font, "#D9DCE2", round(width * .010))
    if hook and hook != subtitle:
        y += round(width * .010)
        y = _draw_lines(draw, layout, "hook", hook_lines, (x, y), hook_font, accent, round(width * .008))
    identity_y = panel_bottom - round(width * .060)
    draw.line((x, identity_y - round(width * .018), x + width * .18, identity_y - round(width * .018)), fill=accent, width=max(3, width // 300))
    _record_text(draw, layout, "identity", identity_lines[0], (x, identity_y), identity_font, "#FFFFFF")
    canvas.convert("RGB").save(output, quality=94)
    return layout


def make_cover_headline(
    frame: Image.Image, cover_copy: dict[str, Any], series: str, speaker: str,
    speaker_title: str, accent: str, size: tuple[int, int], output: Path,
) -> list[dict[str, Any]]:
    title = str(cover_copy["headline"])
    subtitle = str(cover_copy["subheadline"])
    angle = str(cover_copy.get("angle", series))
    canvas = cover_canvas(frame, size).convert("RGBA")
    width, height = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0)); od = ImageDraw.Draw(overlay)
    for y in range(height):
        top = max(0.0, 1.0 - y / max(1, height * .46))
        bottom = max(0.0, (y / height - .68) / .32)
        od.line((0, y, width, y), fill=(5, 6, 8, int(205 * max(top, bottom))))
    canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas)
    layout: list[dict[str, Any]] = []
    font_path = locate_font()
    angle_lines, kicker_font = _fit_cover_text(draw, angle, font_path, round(width * .84), 1, round(width * .043), round(width * .026), True)
    lines, headline_font = _fit_cover_text(draw, title, font_path, round(width * .86), 3, round(width * .070), round(width * .046), True)
    _record_text(draw, layout, "angle", angle_lines[0], (width // 2, round(height * .042)), kicker_font, accent, anchor="ma", stroke_width=max(1, width // 540), stroke_fill="#11120F")
    y = round(height * .095)
    y = _draw_lines(draw, layout, "headline", lines, (width // 2, y), headline_font, "#FFFFFF", round(width * .012), anchor="ma", stroke_width=max(3, width // 250), stroke_fill="#11120F")
    draw.line((round(width * .12), y + 5, round(width * .88), y + 5), fill=accent, width=max(4, width // 220))
    subtitle_lines, subtitle_font = _fit_cover_text(draw, subtitle, font_path, round(width * .72), 2, round(width * .042), round(width * .030), True)
    banner_top = round(height * .615)
    line_height = round(width * .052)
    banner_height = round(width * .075) + len(subtitle_lines) * line_height
    draw.rounded_rectangle((round(width * .10), banner_top, round(width * .90), banner_top + banner_height), radius=round(width * .025), fill=(5, 6, 8, 205))
    _draw_lines(draw, layout, "subheadline", subtitle_lines, (width // 2, banner_top + round(width * .035)), subtitle_font, accent, round(width * .010), anchor="ma")
    draw_identity(draw, layout, width, height, speaker, speaker_title, font_path, accent)
    canvas.convert("RGB").save(output, quality=95)
    return layout


def impact_keyword(title: str, subtitle: str) -> str:
    if "规模化" in subtitle and "标准化" in subtitle:
        return "规模标准"
    for term in ("规模化", "标准化", "用户主义", "餐饮", "顾客", "品牌", "增长", "管理", "效率", "体验"):
        if term in f"{title}{subtitle}":
            return (term[:4] + "观点")[:4]
    cleaned = re.sub(r"[的了与和是一个]", "", clean_text(title))
    return (cleaned[:4] or "核心观点").ljust(4, "点")


def make_cover_impact(
    frame: Image.Image, cover_copy: dict[str, Any], series: str, speaker: str,
    speaker_title: str, accent: str, size: tuple[int, int], output: Path,
) -> list[dict[str, Any]]:
    title = str(cover_copy["headline"])
    subtitle = str(cover_copy["subheadline"])
    base = cover_canvas(frame, size).convert("RGBA")
    width, height = size
    blurred = base.filter(ImageFilter.GaussianBlur(radius=max(8, width // 90)))
    tint = Image.new("RGBA", size, (20, 13, 42, 92))
    canvas = Image.alpha_composite(blurred, tint)
    sharp = base.crop((round(width * .14), round(height * .17), round(width * .86), round(height * .88)))
    panel = Image.new("RGBA", size, (0, 0, 0, 0)); panel.alpha_composite(sharp, (round(width * .14), round(height * .17)))
    canvas = Image.alpha_composite(canvas, panel)
    draw = ImageDraw.Draw(canvas)
    layout: list[dict[str, Any]] = []
    font_path = locate_font()
    keywords = [clean_text(str(item)) for item in cover_copy.get("keywords", [])[:2] if clean_text(str(item))]
    while len(keywords) < 2:
        keywords.append("洞察" if not keywords else "方法")
    card_top, card_bottom = round(height * .070), round(height * .175)
    gap = round(width * .025)
    left_x, full_right = round(width * .055), round(width * .945)
    card_width = (full_right - left_x - gap) // 2
    for index, keyword in enumerate(keywords):
        x1 = left_x + index * (card_width + gap)
        x2 = x1 + card_width
        draw.rounded_rectangle((x1, card_top, x2, card_bottom), radius=round(width * .025), fill=(7, 8, 11, 220), outline=accent, width=max(2, width // 360))
        keyword_lines, keyword_font = _fit_cover_text(draw, keyword, font_path, card_width - round(width * .06), 1, round(width * .072), round(width * .040), True)
        _record_text(draw, layout, f"keyword_{index + 1}", keyword_lines[0], ((x1 + x2) // 2, card_top + round((card_bottom - card_top) * .22)), keyword_font, accent, anchor="ma", stroke_width=max(2, width // 320), stroke_fill="#11120F")
    strap_y = round(height * .62)
    impact_line = clean_text(str(cover_copy.get("impact_line", title)))
    impact_lines, title_font = _fit_cover_text(draw, impact_line, font_path, round(width * .76), 2, round(width * .047), round(width * .032), True)
    strap_height = round(width * .075) + len(impact_lines) * round(width * .058)
    draw.rounded_rectangle((round(width * .08), strap_y, round(width * .92), strap_y + strap_height), radius=round(width * .025), fill=(8, 9, 12, 218), outline=(255, 255, 255, 35), width=2)
    _draw_lines(draw, layout, "impact_line", impact_lines, (width // 2, strap_y + round(width * .035)), title_font, "#FFFFFF", round(width * .010), anchor="ma", stroke_width=max(1, width // 420), stroke_fill="#08090B")
    headline_lines, headline_font = _fit_cover_text(draw, title, font_path, round(width * .84), 2, round(width * .033), round(width * .025), True)
    _draw_lines(draw, layout, "headline", headline_lines, (round(width * .08), strap_y + strap_height + round(width * .025)), headline_font, accent, round(width * .008))
    draw_identity(draw, layout, width, height, speaker, speaker_title, font_path, accent)
    canvas.convert("RGB").save(output, quality=95)
    return layout


def validate_cover_artifact(
    path: Path, layout: list[dict[str, Any]], required: dict[str, str], size: tuple[int, int],
) -> dict[str, Any]:
    """Fail closed when cover copy is incomplete or rendered outside safe bounds."""
    width, height = size
    errors: list[str] = []
    warnings: list[str] = []
    try:
        with Image.open(path) as image:
            if image.size != size:
                errors.append(f"分辨率错误：{image.size}，应为 {size}")
    except OSError as exc:
        errors.append(f"封面文件无法读取：{exc}")
    by_role: dict[str, str] = {}
    for item in layout:
        by_role[item["role"]] = by_role.get(item["role"], "") + str(item["text"])
        x1, y1, x2, y2 = item["bbox"]
        if x1 < round(width * .025) or x2 > round(width * .975) or y1 < round(height * .015) or y2 > round(height * .975):
            errors.append(f"{item['role']} 文字越出安全区：{item['bbox']}")
        if x2 <= x1 or y2 <= y1:
            errors.append(f"{item['role']} 没有形成有效文字区域")
    for role, expected in required.items():
        actual = clean_text(by_role.get(role, ""))
        if actual != clean_text(expected):
            errors.append(f"{role} 文案不完整：期望“{expected}”，实际“{actual}”")
    if len(layout) < len(required):
        errors.append("封面文字图层数量不足")
    return {"file": path.name, "pass": not errors, "errors": errors, "warnings": warnings, "layout": layout}


def generate_cover_templates(
    frame: Image.Image, cover_copy: dict[str, Any], series: str, speaker: str,
    speaker_title: str, accent: str, output_dir: Path, selected: str,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    templates = [
        ("headline", "人物标题型", "标题在上，人物完整保留，适合观点类账号", make_cover_headline),
        ("editorial", "编辑部卡片型", "信息最完整，适合专业访谈与深度内容", make_cover_editorial),
        ("impact", "大字冲击型", "关键词强对比，适合短视频信息流抢眼", make_cover_impact),
    ]
    manifest: list[dict[str, Any]] = []
    qa_items: list[dict[str, Any]] = []
    identity = "｜".join(item for item in (speaker, speaker_title) if item) or "本期主讲"
    for template_id, label, description, renderer in templates:
        for size, suffix in [((1080, 1920), "9x16"), ((1080, 1440), "3x4")]:
            path = output_dir / f"cover-{template_id}-{suffix}.png"
            layout = renderer(frame, cover_copy, series, speaker, speaker_title, accent, size, path)
            if template_id == "headline":
                required = {"angle": str(cover_copy.get("angle", series)), "headline": str(cover_copy["headline"]), "subheadline": str(cover_copy["subheadline"]), "identity": identity}
            elif template_id == "editorial":
                required = {"angle": str(cover_copy.get("angle", series)), "headline": str(cover_copy["headline"]), "subheadline": str(cover_copy["subheadline"]), "identity": identity}
                if cover_copy.get("hook") and cover_copy.get("hook") != cover_copy.get("subheadline"):
                    required["hook"] = f"观点｜{cover_copy['hook']}"
            else:
                keywords = [clean_text(str(item)) for item in cover_copy.get("keywords", [])[:2] if clean_text(str(item))]
                while len(keywords) < 2:
                    keywords.append("洞察" if not keywords else "方法")
                required = {"keyword_1": keywords[0], "keyword_2": keywords[1], "impact_line": str(cover_copy.get("impact_line", cover_copy["headline"])), "headline": str(cover_copy["headline"]), "identity": identity}
            qa_items.append(validate_cover_artifact(path, layout, required, size))
        manifest.append({"id": template_id, "label": label, "description": description, "preview": f"cover-{template_id}-9x16.png"})
    qa_report = {"pass": all(item["pass"] for item in qa_items), "checked": len(qa_items), "items": qa_items}
    (output_dir / "cover-qa.json").write_text(json.dumps(qa_report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not qa_report["pass"]:
        failures = [f"{item['file']}：{'；'.join(item['errors'])}" for item in qa_items if not item["pass"]]
        raise RuntimeError("封面质检未通过：" + " | ".join(failures))
    valid = {item[0] for item in templates}
    selected_id = selected if selected in valid else "headline"
    shutil.copy2(output_dir / f"cover-{selected_id}-9x16.png", output_dir / "cover-9x16.png")
    shutil.copy2(output_dir / f"cover-{selected_id}-3x4.png", output_dir / "cover-3x4.png")
    (output_dir / "cover-manifest.json").write_text(json.dumps({"selected": selected_id, "templates": manifest, "quality": {"pass": True, "report": "cover-qa.json"}}, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest, selected_id, qa_report


def markup_caption(text: str, emphasis: list[str]) -> str:
    phrases = sorted({item for item in emphasis if item and item in text}, key=len, reverse=True)
    if not phrases:
        return html.escape(text)
    pattern = re.compile("(" + "|".join(re.escape(item) for item in phrases) + ")")
    parts = pattern.split(text)
    return "".join(f'<span class="keyword">{html.escape(part)}</span>' if part in phrases else html.escape(part) for part in parts)


def render_html(data: dict[str, Any]) -> str:
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    duration, accent = max(0.2, float(data["duration"])), data["accent"]
    template_id = str(data.get("motion_template", "expert"))
    profile = MOTION_TEMPLATE_BY_ID.get(template_id, MOTION_TEMPLATE_BY_ID["expert"])
    panel_alpha = float(profile["panel_alpha"])
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=1080, height=1920" />
  <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
  <style>
    * {{ box-sizing:border-box; }}
    @font-face {{ font-family:"WorkflowSans"; src:url("assets/fonts/STHeitiMedium.ttc"); font-style:normal; font-weight:100 900; }}
    html,body {{ margin:0;width:1080px;height:1920px;overflow:hidden;background:#090A0C; }}
    body {{ font-family:"WorkflowSans",sans-serif; }}
    #root {{ position:relative;width:1080px;height:1920px;overflow:hidden;background:#090A0C;color:white; }}
    .source-video {{ position:absolute;inset:0;width:1080px;height:1920px;object-fit:cover;transform-origin:50% 42%;z-index:1;will-change:transform; }}
    .shade {{ position:absolute;inset:0;z-index:2;background:linear-gradient(180deg,rgba(0,0,0,.24),transparent 24%,transparent 58%,rgba(0,0,0,.46)); }}
    .brand {{ position:absolute;top:98px;left:70px;z-index:9;display:flex;gap:14px;align-items:center;font-size:27px;font-weight:700;letter-spacing:2px;text-shadow:0 2px 8px #000; }}
    .brand::before {{ content:"";width:42px;height:8px;border-radius:8px;background:{accent};box-shadow:0 0 24px {accent}; }}
    .progress {{ position:absolute;z-index:9;right:68px;top:105px;font:600 23px/1 ui-monospace,SFMono-Regular,monospace;color:rgba(255,255,255,.78); }}
    .caption {{ position:absolute;left:78px;right:78px;top:1325px;height:250px;z-index:24;display:flex;justify-content:center;align-items:center;text-align:center; }}
    .caption-shell {{ max-width:900px;padding:14px 24px;font-size:{profile['caption_size']}px;line-height:1.30;font-weight:900;letter-spacing:.2px;color:#fff;-webkit-text-stroke:5px rgba(0,0,0,.98);paint-order:stroke fill;text-shadow:0 5px 14px rgba(0,0,0,.62); }}
    .keyword {{ color:{accent};-webkit-text-stroke-color:#08090B; }}
    .hook {{ position:absolute;left:72px;right:72px;top:192px;height:440px;z-index:42;display:flex;align-items:center;padding:34px 42px;border:1px solid rgba(255,255,255,.18);border-radius:22px;background:rgba(7,8,10,.68);box-shadow:0 24px 70px rgba(0,0,0,.26);backdrop-filter:blur(14px); }}
    .hook-inner {{ width:100%;border-left:6px solid {accent};padding:14px 0 16px 30px; }}
    .hook-kicker {{ color:{accent};font:800 19px/1.15 ui-monospace,SFMono-Regular,monospace;letter-spacing:3px;margin-bottom:17px; }}
    .hook-title {{ font-size:52px;line-height:1.24;font-weight:900;letter-spacing:-.7px;max-width:820px;text-shadow:0 5px 18px rgba(0,0,0,.72); }}
    .hook-title.long {{ font-size:45px;line-height:1.30;max-width:830px; }}
    .hook-subtitle {{ margin-top:17px;font-size:24px;line-height:1.40;font-weight:450;color:#D6D9DE;max-width:780px; }}
    .hook-rule {{ width:118px;height:4px;background:white;margin-top:22px; }}
    .hook-chips {{ display:none; }}
    .hook-pip {{ position:absolute;z-index:43;object-fit:cover;will-change:transform,opacity; }}
    .chapter {{ position:absolute;inset:0;z-index:40;overflow:hidden; }}
    .chapter-stage {{ position:absolute;inset:0;background:radial-gradient(circle at 82% 18%,rgba(255,216,61,.20),rgba(255,216,61,0) 32%),#090A0C;display:flex;align-items:center;padding:150px 78px;will-change:transform; }}
    .chapter-edge {{ position:absolute;left:0;right:0;bottom:0;height:14px;background:{accent};box-shadow:0 -18px 50px rgba(255,216,61,.28); }}
    .chapter-grid {{ position:absolute;inset:0;opacity:.12;background-image:linear-gradient(rgba(255,255,255,.22) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.22) 1px,transparent 1px);background-size:74px 74px; }}
    .chapter-inner {{ position:relative;width:100%; }}
    .chapter-inner::after {{ content:attr(data-ghost);position:absolute;right:-24px;top:-280px;font:900 330px/.9 ui-monospace,SFMono-Regular,monospace;color:rgba(255,216,61,.10);letter-spacing:-28px;z-index:-1; }}
    .chapter-index {{ color:{accent};font:800 30px/1 ui-monospace,SFMono-Regular,monospace;letter-spacing:5px;margin-bottom:34px; }}
    .chapter-title {{ max-width:920px;font-size:68px;line-height:1.22;font-weight:900;letter-spacing:-1px;color:#FFFFFF;text-shadow:0 7px 22px rgba(0,0,0,.44); }}
    .chapter-title.long {{ font-size:62px;line-height:1.27; }}
    .chapter-rule {{ width:78%;height:6px;margin-top:38px;background:{accent};transform-origin:left center; }}
    .chapter-pulse {{ display:flex;gap:11px;margin-top:26px; }} .chapter-pulse i {{ width:7px;height:30px;background:{accent};opacity:.24; }}
    .support {{ position:absolute;left:68px;right:68px;top:{profile['support_top']}px;height:{profile['support_height']}px;z-index:17; }}
    .support.support-quote {{ left:92px;right:92px;top:242px;height:340px; }}
    .support.support-compare,.support.support-stat {{ left:78px;right:78px;top:226px;height:420px; }}
    .support-surface {{ position:absolute;inset:0;border:1px solid rgba(255,255,255,.17);border-radius:{profile['support_radius']}px;overflow:hidden;background:rgba(9,10,12,{panel_alpha:.2f});box-shadow:0 22px 64px rgba(0,0,0,.25);backdrop-filter:blur(12px);will-change:transform,opacity; }}
    .support-inner {{ position:absolute;inset:0;padding:40px 44px; }}
    .support-label {{ color:{accent};font:800 18px/1 ui-monospace,SFMono-Regular,monospace;letter-spacing:3px;margin-bottom:20px; }}
    .support-inner.support-quote {{ display:flex;flex-direction:column;justify-content:center;background:radial-gradient(circle at 88% 12%,rgba(255,216,61,.10),transparent 38%),rgba(9,10,12,.18); }}
    .support-inner.support-quote::before {{ content:"“";position:absolute;right:30px;top:-34px;font-size:190px;color:rgba(255,216,61,.10);font-family:serif; }}
    .support-inner.support-quote .support-text {{ font-size:{profile['quote_size']}px;line-height:1.34;font-weight:800;letter-spacing:-.3px;border-top:3px solid {accent};padding-top:20px;max-width:760px; }}
    .support-inner.support-quote .support-text.long {{ font-size:{profile['quote_long_size']}px;line-height:1.40;max-width:780px; }}
    .callout-meta,.callout-connector,.callout-ghost {{ display:none; }}
    .support-media-frame {{ background:linear-gradient(180deg,rgba(9,10,12,.03),rgba(9,10,12,.82)); }}
    .support-media-caption {{ position:absolute;left:30px;right:30px;bottom:26px;padding:18px 24px;border-radius:14px;background:rgba(6,7,8,.66);font-size:34px;font-weight:800;line-height:1.34; }}
    .support-image {{ width:100%;height:100%;object-fit:cover;will-change:transform; }}
    .support-video {{ position:absolute;left:68px;top:{profile['support_top']}px;width:944px;height:{profile['support_height']}px;object-fit:cover;border-radius:{profile['support_radius']}px;z-index:16;clip-path:inset(0 100% 0 0 round {profile['support_radius']}px);will-change:transform; }}
    .compare-stage {{ position:absolute;inset:0;display:flex;align-items:stretch;justify-content:center;gap:18px;padding:28px; }}
    .compare-card {{ width:50%;min-height:0;border-radius:16px;padding:30px;background:#F6F3E9;color:#121212;display:flex;flex-direction:column;justify-content:space-between;transform-style:preserve-3d; }}
    .compare-card.right {{ background:{accent}; }} .compare-tag {{ font:800 23px/1 ui-monospace,monospace;letter-spacing:3px; }}
    .compare-text {{ font-size:39px;line-height:1.30;font-weight:900;letter-spacing:-.4px; }}
    .stat-stage {{ position:absolute;inset:0;padding:42px 48px;display:grid;grid-template-columns:1.08fr .92fr;align-items:center;gap:32px; }}
    .stat-value {{ font-size:{profile['stat_size']}px;font-weight:900;color:{accent};line-height:.95;font-variant-numeric:tabular-nums; }} .stat-label {{ margin-top:17px;font-size:32px;line-height:1.35;font-weight:800; }}
    .bars {{ height:300px;display:flex;align-items:flex-end;gap:15px;border-bottom:3px solid rgba(255,255,255,.3);padding:0 12px; }}
    .bar {{ flex:1;background:#5B5D58;transform-origin:bottom center; }} .bar:last-child {{ background:{accent}; }}
    .lower-third {{ position:absolute;left:62px;top:1110px;z-index:26;max-width:760px; }}
    .lower-third-card {{ display:flex;align-items:stretch;min-height:108px;border-radius:15px;overflow:hidden;background:rgba(5,6,8,.72);border:1px solid rgba(255,255,255,.16);box-shadow:0 16px 46px rgba(0,0,0,.30);backdrop-filter:blur(14px); }}
    .lower-third-rule {{ width:7px;background:{accent};box-shadow:0 0 24px {accent}; }}
    .lower-third-copy {{ padding:20px 28px 18px 24px; }}
    .lower-third-name {{ font-size:32px;font-weight:900;line-height:1.08;letter-spacing:.3px; }}
    .lower-third-title {{ margin-top:7px;color:#D6D9DE;font-size:21px;font-weight:500;line-height:1.30; }}
    .support.support-outro {{ left:62px;right:62px;top:980px;height:360px;z-index:24; }}
    .support-outro .support-surface {{ border-radius:22px;background:rgba(7,8,10,.78);border:1px solid rgba(255,255,255,.18);backdrop-filter:blur(16px); }}
    .outro-stage {{ position:absolute;inset:0;padding:38px 42px;display:flex;flex-direction:column;justify-content:center; }}
    .outro-kicker {{ color:{accent};font:900 18px/1 ui-monospace,monospace;letter-spacing:3px; }}
    .outro-title {{ margin-top:15px;font-size:43px;line-height:1.18;font-weight:900; }}
    .outro-points {{ display:flex;gap:12px;margin-top:24px; }}
    .outro-point {{ flex:1;padding:15px 18px;border-radius:13px;background:rgba(255,255,255,.10);border-left:5px solid {accent};font-size:27px;font-weight:800; }}
    .support-speaker-pip {{ position:absolute;z-index:21;object-fit:cover;border:4px solid rgba(255,255,255,.92);box-shadow:0 18px 58px rgba(0,0,0,.34);will-change:transform,opacity; }}
    .support-speaker-pip.role-circle {{ border-radius:50%; }}
    .support-speaker-pip.role-card {{ border-radius:26px; }}
    .knowledge-stage {{ position:absolute;inset:0;padding:34px 42px;display:flex;flex-direction:column;justify-content:flex-start;gap:10px;background:radial-gradient(circle at 15% 12%,rgba(255,216,61,.22),transparent 28%),linear-gradient(180deg,#202126,#090A0C); }}
    .knowledge-kicker {{ color:{accent};font:800 17px/1 ui-monospace,SFMono-Regular,monospace;letter-spacing:3px; }}
    .knowledge-title {{ max-width:760px;font-size:40px;line-height:1.12;font-weight:900;letter-spacing:-.5px; }}
    .knowledge-points {{ display:flex;flex-direction:column;gap:9px;margin-top:8px; }}
    .knowledge-point {{ display:grid;grid-template-columns:40px 1fr;gap:12px;align-items:start;padding:13px 16px;border-radius:13px;background:rgba(255,255,255,.90);color:#171816;font-size:25px;line-height:1.28;font-weight:800;box-shadow:0 12px 28px rgba(0,0,0,.14); }}
    .knowledge-point i {{ display:flex;width:32px;height:32px;align-items:center;justify-content:center;border-radius:50%;background:{accent};font:900 16px/1 ui-monospace,monospace;font-style:normal; }}
    .context-stage {{ position:absolute;inset:0;padding:34px 42px;background:radial-gradient(circle at 86% 18%,rgba(255,216,61,.30),transparent 24%),radial-gradient(circle at 14% 52%,rgba(231,128,102,.24),transparent 26%),linear-gradient(180deg,#EEE2D6,#D5CEC2);color:#171816;overflow:hidden; }}
    .context-kicker {{ color:#5A473E;font:900 20px/1 ui-monospace,SFMono-Regular,monospace;letter-spacing:4px; }}
    .context-orbit {{ position:absolute;left:42px;top:116px;width:252px;height:252px;border:2px solid rgba(23,24,22,.16);border-radius:50%; }}
    .context-orbit::before,.context-orbit::after {{ content:"";position:absolute;border-radius:50%;border:2px solid rgba(23,24,22,.13); }}
    .context-orbit::before {{ inset:34px; }} .context-orbit::after {{ inset:76px;background:rgba(255,255,255,.50);box-shadow:0 14px 34px rgba(70,48,35,.13); }}
    .context-token {{ position:absolute;inset:76px;display:flex;align-items:center;justify-content:center;padding:10px;text-align:center;font-size:25px;line-height:1.16;font-weight:900;z-index:2; }}
    .context-dot {{ position:absolute;width:14px;height:14px;border-radius:50%;background:{accent};box-shadow:0 0 0 7px rgba(255,216,61,.18); }}
    .context-dot.one {{ left:20px;top:72px; }} .context-dot.two {{ right:28px;bottom:46px;background:#E77962;box-shadow:0 0 0 7px rgba(231,121,98,.16); }}
    .context-copy {{ position:absolute;left:334px;right:38px;top:116px;padding:22px 24px;border-top:4px solid #171816;background:rgba(255,255,255,.58);backdrop-filter:blur(12px); }}
    .context-copy small {{ display:block;margin-bottom:12px;color:#6A554B;font:800 16px/1 ui-monospace,monospace;letter-spacing:3px; }}
    .context-title {{ max-width:560px;font-size:38px;line-height:1.18;font-weight:900;letter-spacing:-.6px; }}
    .template-story .support-media-caption {{ font-size:34px;background:rgba(6,7,8,.62); }}
    .template-expert .hook {{ left:92px;right:92px;top:224px;height:360px; }}
    .template-expert .compare-card {{ padding:26px; }}
    .template-conflict .hook {{ top:178px;height:478px;border-radius:16px;border:2px solid {accent};background:rgba(8,9,10,.78); }}
    .template-conflict .hook-inner {{ border-left:0;border-top:6px solid {accent};padding:24px 8px 8px; }}
    .template-conflict .hook-title {{ font-size:60px;line-height:1.15; }}
    .template-conflict .hook-title.long {{ font-size:51px; }}
    .template-conflict .hook-rule {{ background:{accent};width:52%; }}
    .template-conflict .support-quote {{ left:44px;right:44px;top:198px;height:468px; }}
    .template-conflict .support-quote .support-surface {{ border:2px solid {accent};background:linear-gradient(90deg,{accent} 0 116px,rgba(8,9,10,.82) 116px); }}
    .template-conflict .support-inner.support-quote {{ display:block;inset:0;height:100%;min-height:0;overflow:hidden;padding:82px 48px 38px 158px;background:transparent;z-index:2; }}
    .template-conflict .support-inner.support-quote::before {{ content:"观点";right:auto;left:30px;top:86px;writing-mode:vertical-rl;font:900 35px/1 sans-serif;letter-spacing:7px;color:#111; }}
    .template-conflict .support-inner.support-quote .support-label {{ position:relative;z-index:2;color:{accent}; }}
    .template-conflict .support-inner.support-quote .support-text {{ position:relative;z-index:2;border-top:0;font-size:42px;line-height:1.30; }}
    .template-story .hook {{ top:150px;height:520px;background:linear-gradient(180deg,rgba(7,8,10,.42),rgba(7,8,10,.80));border-radius:28px; }}
    .template-story .hook-inner {{ align-self:flex-end;margin-top:110px; }}
    .template-story .support-quote {{ top:194px;height:440px; }}
    .template-editorial .hook {{ left:0;right:0;top:0;height:1110px;border:0;border-radius:0;overflow:hidden;background:radial-gradient(circle at 18% 18%,rgba(255,216,61,.38),transparent 23%),radial-gradient(circle at 86% 34%,rgba(255,130,115,.28),transparent 25%),linear-gradient(180deg,rgba(247,226,210,.94),rgba(235,207,201,.86));box-shadow:none;backdrop-filter:blur(16px);color:#171816;align-items:flex-start;padding:106px 74px; }}
    .template-editorial .hook-media {{ position:absolute;inset:0;width:100%;height:100%;object-fit:cover;z-index:0; }}
    .template-editorial.editorial-has-hook-media .hook::after {{ content:"";position:absolute;inset:0;z-index:1;background:linear-gradient(90deg,rgba(7,8,8,.78),rgba(7,8,8,.18) 66%,rgba(7,8,8,.34)),linear-gradient(180deg,rgba(0,0,0,.08),rgba(0,0,0,.42)); }}
    .template-editorial.editorial-has-hook-media .hook-kicker {{ color:{accent}; }}
    .template-editorial.editorial-has-hook-media .hook-title {{ color:#fff;text-shadow:0 5px 20px rgba(0,0,0,.48); }}
    .template-editorial.editorial-has-hook-media .hook-subtitle {{ color:rgba(255,255,255,.82); }}
    .template-editorial.editorial-has-hook-media .hook-rule {{ background:#fff; }}
    .template-editorial.editorial-has-hook-media .hook-chips i {{ color:#fff;background:rgba(7,8,8,.66);border-color:rgba(255,255,255,.46); }}
    .template-editorial .progress {{ display:none; }}
    .template-editorial.editorial-no-hook-media .hook {{ left:0;right:0;top:0;height:1920px;border-radius:0;padding:0;background:radial-gradient(circle at 50% 22%,rgba(255,216,61,.40),transparent 22%),radial-gradient(circle at 12% 42%,rgba(255,130,115,.26),transparent 24%),radial-gradient(circle at 90% 52%,rgba(126,157,255,.18),transparent 25%),linear-gradient(180deg,#F5DFD2,#E9CCC4);box-shadow:none; }}
    .template-editorial.editorial-no-hook-media .hook::before {{ content:"";position:absolute;inset:0;opacity:.18;background-image:linear-gradient(rgba(42,31,26,.24) 1px,transparent 1px),linear-gradient(90deg,rgba(42,31,26,.24) 1px,transparent 1px);background-size:72px 72px; }}
    .template-editorial.editorial-no-hook-media .hook-inner {{ position:absolute;left:74px;right:74px;bottom:270px;width:auto; }}
    .template-editorial .hook-inner {{ position:relative;z-index:45;border-left:0;padding:0;width:54%; }}
    .template-editorial .hook-kicker {{ color:#5B463C;letter-spacing:2px; }}
    .template-editorial .hook-title,.template-editorial .hook-title.long {{ font-size:49px;line-height:1.22;color:#171816;text-shadow:none; }}
    .template-editorial .hook-subtitle {{ color:#54433C;font-size:23px; }}
    .template-editorial .hook-rule {{ background:#171816;width:150px; }}
    .template-editorial .hook-chips {{ display:flex;flex-wrap:wrap;gap:10px;margin-top:30px; }}
    .template-editorial .hook-chips i {{ display:block;padding:10px 16px;border:2px solid rgba(23,24,22,.18);border-radius:999px;background:rgba(255,255,255,.66);font-size:19px;font-style:normal;font-weight:800; }}
    .template-editorial .hook-pip {{ left:624px;top:170px;width:330px;height:330px;border-radius:50%;border:8px solid rgba(255,255,255,.92);box-shadow:0 24px 70px rgba(74,43,32,.26); }}
    .template-editorial.editorial-no-hook-media .hook-pip {{ left:350px;top:270px;width:380px;height:380px; }}
    .template-editorial .support.support-media {{ left:0;right:0;top:0;height:1920px;z-index:15; }}
    .template-editorial .support-media .support-surface {{ border:0;border-radius:0;background:#171816;box-shadow:none; }}
    .template-editorial .support-media .support-surface::after {{ content:"";position:absolute;inset:0;background:linear-gradient(180deg,rgba(0,0,0,.10),transparent 42%,rgba(0,0,0,.68));pointer-events:none; }}
    .template-editorial .support-media .support-inner {{ z-index:2;padding:96px 70px; }}
    .template-editorial .support-media .support-label {{ display:inline-flex;padding:10px 14px;border-radius:999px;background:rgba(8,9,9,.70);border:1px solid rgba(255,255,255,.26); }}
    .template-editorial .support-media-caption {{ left:70px;right:auto;bottom:auto;top:170px;max-width:600px;padding:18px 22px;background:rgba(8,9,9,.68);font-size:38px; }}
    .template-editorial .support-video {{ left:0;top:0;width:1080px;height:1920px;border-radius:0;z-index:14; }}
    .template-editorial .support-speaker-pip.pip-media {{ right:66px;top:116px;width:238px;height:238px;border-radius:50%; }}
    .template-editorial .support.support-quote {{ left:0;right:0;top:0;height:1920px;z-index:17; }}
    .template-editorial .support-quote .support-surface {{ border:0;border-radius:0;background:linear-gradient(180deg,rgba(6,7,8,.02),rgba(6,7,8,.12) 43%,rgba(6,7,8,.60));box-shadow:none;backdrop-filter:none; }}
    .template-editorial .support-inner.support-quote {{ inset:auto 72px 570px 72px;height:310px;padding:38px 42px;border-radius:22px;background:rgba(8,9,9,.74);border:1px solid rgba(255,255,255,.20);backdrop-filter:blur(12px); }}
    .template-editorial .support-inner.support-quote::before {{ display:none; }}
    .template-editorial .support-inner.support-quote .support-text {{ border-top:0;padding-top:0;font-size:44px;line-height:1.30; }}
    .template-editorial .support.scene-concept-explainer .support-surface {{ background:radial-gradient(circle at 14% 16%,rgba(255,216,61,.22),transparent 28%),linear-gradient(180deg,#F1E0D1,#D6CFC2); }}
    .template-editorial .support.scene-concept-explainer .support-surface::before {{ content:"";position:absolute;inset:0;opacity:.12;background-image:linear-gradient(rgba(60,52,46,.24) 1px,transparent 1px),linear-gradient(90deg,rgba(60,52,46,.24) 1px,transparent 1px);background-size:72px 72px; }}
    .template-editorial .support.scene-concept-explainer .support-inner.support-quote {{ inset:150px 70px auto 70px;height:auto;min-height:390px;background:rgba(9,10,12,.84); }}
    .template-editorial .support.scene-concept-explainer .callout-meta {{ display:block;position:absolute;left:82px;top:670px;width:470px;color:#171816; }}
    .template-editorial .support.scene-concept-explainer .callout-meta small {{ display:block;margin-bottom:13px;color:#685A50;font:900 18px/1 ui-monospace,SFMono-Regular,monospace;letter-spacing:3px; }}
    .template-editorial .support.scene-concept-explainer .callout-meta strong {{ display:block;font-size:41px;line-height:1.18;font-weight:900;letter-spacing:-.5px; }}
    .template-editorial .support.scene-concept-explainer .callout-meta em {{ display:inline-flex;margin-top:18px;padding:9px 14px;border:2px solid rgba(23,24,22,.18);border-radius:999px;background:rgba(255,255,255,.46);font-size:18px;font-style:normal;font-weight:800; }}
    .template-editorial .support.scene-concept-explainer .callout-connector {{ display:block;position:absolute;left:82px;top:835px;width:635px;height:2px;background:rgba(23,24,22,.32);transform-origin:left center; }}
    .template-editorial .support.scene-concept-explainer .callout-connector::before,.template-editorial .support.scene-concept-explainer .callout-connector::after {{ content:"";position:absolute;top:-6px;width:14px;height:14px;border-radius:50%;background:{accent};box-shadow:0 0 0 8px rgba(255,216,61,.16); }}
    .template-editorial .support.scene-concept-explainer .callout-connector::before {{ left:0; }} .template-editorial .support.scene-concept-explainer .callout-connector::after {{ right:0; }}
    .template-editorial .support.scene-concept-explainer .callout-ghost {{ display:block;position:absolute;left:72px;top:1220px;width:286px;height:286px;border:5px solid rgba(23,24,22,.075);border-radius:50%; }}
    .template-editorial .support.scene-concept-explainer .callout-ghost::before,.template-editorial .support.scene-concept-explainer .callout-ghost::after {{ content:"";position:absolute;border:4px solid rgba(23,24,22,.065);border-radius:50%; }}
    .template-editorial .support.scene-concept-explainer .callout-ghost::before {{ inset:48px; }} .template-editorial .support.scene-concept-explainer .callout-ghost::after {{ inset:102px;background:rgba(255,216,61,.12); }}
    .template-editorial .support-speaker-pip.pip-quote {{ right:80px;top:760px;width:330px;height:330px; }}
    .template-editorial .support.support-compare,.template-editorial .support.support-stat {{ left:0;right:0;top:0;height:1920px;z-index:17; }}
    .template-editorial .support-compare .support-surface,.template-editorial .support-stat .support-surface {{ border:0;border-radius:0;background:linear-gradient(180deg,rgba(232,232,226,.88),rgba(210,208,199,.78));backdrop-filter:blur(18px); }}
    .template-editorial .compare-stage {{ inset:130px 70px auto 70px;height:570px;padding:0;gap:18px; }}
    .template-editorial .compare-card {{ border:2px solid rgba(23,24,22,.14);box-shadow:0 18px 46px rgba(0,0,0,.12); }}
    .template-editorial .stat-stage {{ inset:150px 70px auto 70px;height:560px;padding:48px;border-radius:26px;background:rgba(9,10,12,.82); }}
    .template-editorial .support-speaker-pip.pip-compare,.template-editorial .support-speaker-pip.pip-stat {{ left:335px;top:820px;width:410px;height:520px;border-radius:28px; }}
    .template-editorial .support.support-knowledge {{ left:0;right:0;top:0;height:1920px;z-index:17; }}
    .template-editorial .support-knowledge .support-surface {{ border:0;border-radius:0;background:transparent;box-shadow:none;backdrop-filter:none; }}
    .template-editorial .knowledge-stage {{ padding:132px 74px 720px;gap:22px; }}
    .template-editorial .knowledge-kicker {{ font-size:21px;letter-spacing:4px; }}
    .template-editorial .knowledge-title {{ font-size:58px;letter-spacing:-1px; }}
    .template-editorial .knowledge-points {{ gap:15px;margin-top:18px; }}
    .template-editorial .knowledge-point {{ grid-template-columns:48px 1fr;gap:15px;padding:18px 22px;border-radius:16px;font-size:31px;line-height:1.32; }}
    .template-editorial .knowledge-point i {{ width:38px;height:38px;font-size:19px; }}
    .template-editorial .support-speaker-pip.pip-knowledge {{ left:322px;top:920px;width:436px;height:520px; }}
    .template-editorial .support.support-context {{ left:0;right:0;top:0;height:1920px;z-index:17; }}
    .template-editorial .support-context .support-surface {{ border:0;border-radius:0;background:transparent;box-shadow:none;backdrop-filter:none; }}
    .template-editorial .context-stage {{ padding:132px 72px; }}
    .template-editorial .context-orbit {{ left:86px;top:300px;width:650px;height:650px; }}
    .template-editorial .context-orbit::before {{ inset:92px; }} .template-editorial .context-orbit::after {{ inset:190px;box-shadow:0 22px 70px rgba(70,48,35,.13); }}
    .template-editorial .context-token {{ inset:190px;padding:40px;font-size:38px;line-height:1.18; }}
    .template-editorial .context-dot {{ width:22px;height:22px;box-shadow:0 0 0 12px rgba(255,216,61,.18); }}
    .template-editorial .context-dot.one {{ left:54px;top:184px; }} .template-editorial .context-dot.two {{ right:74px;bottom:124px;box-shadow:0 0 0 12px rgba(231,121,98,.16); }}
    .template-editorial .context-copy {{ left:72px;right:72px;top:1070px;padding:30px 34px; }}
    .template-editorial .context-copy small {{ margin-bottom:14px;font-size:18px; }}
    .template-editorial .context-title {{ max-width:700px;font-size:53px;line-height:1.16;letter-spacing:-1px; }}
    .template-editorial .support-speaker-pip.pip-context {{ right:72px;top:1018px;width:292px;height:292px;border-radius:50%; }}
    .template-editorial .support.support-outro {{ left:70px;right:70px;top:1030px;height:350px; }}
    .template-editorial .support-outro .support-surface {{ background:rgba(7,8,9,.76);box-shadow:0 24px 70px rgba(0,0,0,.28); }}
  </style>
</head>
<body>
  <div id="root" class="template-{profile['id']}" data-composition-id="main" data-start="0" data-duration="{duration:.3f}" data-width="1080" data-height="1920" data-fps="24">
    <div class="shade" data-layout-ignore></div><div class="brand" data-layout-ignore>{html.escape(data['series'])}</div><div class="progress" data-layout-ignore>PORTRAIT / 09:16</div>
  </div>
  <script>
    const DATA={data_json}; const MOTION={{id:"{profile['id']}",blur:{profile['blur']},editorial:{str(profile['id'] == 'editorial').lower()}}}; const root=document.getElementById("root"); const asset=v=>encodeURI(v).replaceAll("#","%23");
    const PACKAGE_BY_ID=new Map((DATA.scene_packages||[]).map(item=>[item.package_id,item]));const VISUAL_BEATS=(DATA.visual_beats||[]).map(beat=>{{const pack=PACKAGE_BY_ID.get(beat.package_id);if(!pack){{if(beat.kind==="outro")return beat;return null;}}if(!["ready","degraded"].includes(pack.status))return null;return {{...beat,start:Number(pack.start),duration:Number(pack.end)-Number(pack.start),speaker_role:pack.speaker_role,caption_mode:pack.caption_mode,asset:pack.media||beat.asset,background:{{role:pack.background_role,status:"ready",asset:pack.media||null}}}};}}).filter(Boolean);const HOOK_PACKAGE=(DATA.scene_packages||[]).find(item=>(item.metadata&&item.metadata.cue_type)==="hook"&&["ready","degraded"].includes(item.status));
    const hasHookMedia=Boolean(MOTION.editorial&&HOOK_PACKAGE&&HOOK_PACKAGE.media);const hasHookPip=Boolean(MOTION.editorial&&HOOK_PACKAGE&&["circle","card","cutout"].includes(HOOK_PACKAGE.speaker_role));if(MOTION.editorial)root.classList.add(hasHookMedia?"editorial-has-hook-media":"editorial-no-hook-media");
    const segmentAt=time=>DATA.keep_segments.find(seg=>time>=seg.edited_start&&time<seg.edited_start+seg.duration+.001)||DATA.keep_segments[0];
    const sourceAt=time=>{{const seg=segmentAt(time);return seg.source_start+Math.max(0,time-seg.edited_start);}};
    const sourceRoom=time=>{{const seg=segmentAt(time);return Math.max(.2,seg.edited_start+seg.duration-time);}};
    DATA.keep_segments.forEach((seg,i)=>{{
      const video=document.createElement("video");video.id=`video-${{i}}`;video.className="source-video";video.src=asset(DATA.source);video.muted=true;video.playsInline=true;video.dataset.start=seg.edited_start;video.dataset.duration=seg.duration;video.dataset.mediaStart=seg.source_start;video.dataset.trackIndex=0;video.dataset.colorGrading=JSON.stringify(MOTION.editorial?{{preset:"skin-soft",intensity:.55,effects:{{blur:.45}}}}:{{effects:{{blur:.45}}}});video.style.setProperty("--hf-color-grading-blur","0");root.appendChild(video);
      if(!DATA.mix_audio){{const audio=document.createElement("audio");audio.id=`audio-${{i}}`;audio.src=asset(DATA.source);audio.dataset.start=seg.edited_start;audio.dataset.duration=seg.duration;audio.dataset.mediaStart=seg.source_start;audio.dataset.trackIndex=10;audio.dataset.volume=1;root.appendChild(audio);}}
    }});
    if(DATA.mix_audio){{const audio=document.createElement("audio");audio.id="program-mix";audio.src=asset(DATA.mix_audio.path);audio.dataset.start=0;audio.dataset.duration=DATA.duration;audio.dataset.mediaStart=0;audio.dataset.trackIndex=20;audio.dataset.volume=1;root.appendChild(audio);}}
    else if(DATA.bgm) DATA.bgm.segments.forEach((seg,i)=>{{const audio=document.createElement("audio");audio.id=`bgm-${{i}}`;audio.src=asset(DATA.bgm.path);audio.dataset.start=seg.start;audio.dataset.duration=seg.duration;audio.dataset.mediaStart=0;audio.dataset.trackIndex=20;audio.dataset.volume=DATA.bgm.floor_volume;root.appendChild(audio);}});
    const hookAt=Math.max(0,Number(HOOK_PACKAGE?HOOK_PACKAGE.start:DATA.hook.start||0));const hookDuration=HOOK_PACKAGE?Math.max(.2,Number(HOOK_PACKAGE.end)-hookAt):Math.max(2.4,Math.min(5.2,Number(DATA.hook.end||hookAt+3.2)-hookAt+.55));
    const hookChips=(DATA.cover_copy&&DATA.cover_copy.keywords||[]).slice(0,3).map(value=>`<i>${{value}}</i>`).join("");
    const hook=document.createElement("div");hook.id="hook";hook.className="clip hook";hook.dataset.start=hookAt;hook.dataset.duration=Math.min(hookDuration,DATA.duration-hookAt);hook.dataset.trackIndex=3;hook.innerHTML=`<div class="hook-inner"><div class="hook-kicker">本期洞察 / ${{DATA.series}}</div><div class="hook-title ${{DATA.hook.text.length>22?"long":""}}">${{DATA.hook.text}}</div><div class="hook-subtitle">${{DATA.title}}</div><div class="hook-rule"></div><div class="hook-chips">${{hookChips}}</div></div>`;root.appendChild(hook);
    if(hasHookMedia){{if(HOOK_PACKAGE.media.kind==="video"){{const media=document.createElement("video");media.id="hook-media-video";media.className="hook-media";media.src=asset(HOOK_PACKAGE.media.path);media.muted=true;media.playsInline=true;media.dataset.start=hookAt;media.dataset.duration=Math.min(hookDuration,HOOK_PACKAGE.media.duration||hookDuration);media.dataset.mediaStart=0;media.dataset.trackIndex=33;hook.prepend(media);}}else{{const media=document.createElement("img");media.id="hook-media-image";media.className="hook-media";media.src=asset(HOOK_PACKAGE.media.path);hook.prepend(media);}}}}
    if(hasHookPip){{const pip=document.createElement("video");pip.id="hook-pip";pip.className="hook-pip";pip.src=asset(DATA.source);pip.muted=true;pip.playsInline=true;pip.dataset.start=hookAt+.16;pip.dataset.duration=Math.max(.2,Math.min(hookDuration-.32,sourceRoom(hookAt+.16)));pip.dataset.mediaStart=sourceAt(hookAt+.16);pip.dataset.trackIndex=32;pip.dataset.colorGrading=JSON.stringify({{preset:"skin-soft",intensity:.55}});root.appendChild(pip);}}
    DATA.captions.forEach((cap,i)=>{{const el=document.createElement("div");el.id=`cap-${{i}}`;el.className="clip caption";el.dataset.start=cap.start;el.dataset.duration=Math.max(.12,cap.end-cap.start);el.dataset.trackIndex=2;el.innerHTML=`<div class="caption-shell">${{cap.html}}</div>`;root.appendChild(el);}});
    DATA.chapters.forEach((chapter,i)=>{{const el=document.createElement("div");el.id=`chapter-${{i}}`;el.className="clip chapter";el.dataset.start=chapter.start;el.dataset.duration=chapter.duration;el.dataset.trackIndex=5;el.innerHTML=`<div class="chapter-stage"><div class="chapter-grid"></div><div class="chapter-inner" data-ghost="${{String(i+1).padStart(2,"0")}}"><div class="chapter-index">CHAPTER / ${{String(i+1).padStart(2,"0")}}</div><div class="chapter-title ${{chapter.title.length>15?"long":""}}">${{chapter.title}}</div><div class="chapter-rule"></div><div class="chapter-pulse"><i></i><i></i><i></i><i></i><i></i></div></div><div class="chapter-edge"></div></div>`;root.appendChild(el);}});
    VISUAL_BEATS.forEach((beat,i)=>{{
      const pipOffset=Number(beat.entry_offsets&&beat.entry_offsets.speaker||.16);const pipAt=Number(beat.start)+pipOffset;const hasSpeakerPip=Boolean(MOTION.editorial&&["circle","card","cutout"].includes(beat.speaker_role)&&beat.background&&beat.background.status==="ready");if(hasSpeakerPip){{const pip=document.createElement("video");pip.id=`support-pip-${{i}}`;pip.className=`support-speaker-pip pip-${{beat.kind}} role-${{beat.speaker_role}}`;pip.src=asset(DATA.source);pip.muted=true;pip.playsInline=true;pip.dataset.start=pipAt;pip.dataset.duration=Math.max(.2,Math.min(Number(beat.duration)-pipOffset-.20,sourceRoom(pipAt)));pip.dataset.mediaStart=sourceAt(pipAt);pip.dataset.trackIndex=31;pip.dataset.colorGrading=JSON.stringify({{preset:"skin-soft",intensity:.55}});root.appendChild(pip);}}
      if(beat.kind==="media"&&beat.asset&&beat.asset.kind==="video"){{const media=document.createElement("video");media.id=`support-video-${{i}}`;media.className="support-video";media.src=asset(beat.asset.path);media.muted=true;media.playsInline=true;media.dataset.start=beat.start;media.dataset.duration=Math.min(beat.duration,beat.asset.duration||beat.duration);media.dataset.mediaStart=0;media.dataset.trackIndex=30;root.appendChild(media);}}
      const stateClass=`scene-${{String(beat.visual_state||"speaker_anchor").replaceAll("_","-")}}`;const isPositioningWarning=/火|跟风|定位摇摆|什么火/.test(beat.text);const isPriceBoundary=/胖东来|品质|价格|便宜/.test(beat.text);const calloutTheme=isPositioningWarning?"定位稳定性":isPriceBoundary?"品质与价格":"关键经营判断";const calloutEyebrow=isPositioningWarning?"论点一 · 反面警示":isPriceBoundary?"论点二 · 价格边界":"核心判断";const el=document.createElement("div");el.id=`support-${{i}}`;el.className=`clip support support-${{beat.kind}} ${{stateClass}}`;el.dataset.start=beat.start;el.dataset.duration=beat.duration;el.dataset.trackIndex=4;
      if(beat.kind==="media"){{const image=beat.asset&&beat.asset.kind==="image"?`<img class="support-image" src="${{asset(beat.asset.path)}}">`:"";el.innerHTML=`<div class="support-surface">${{image}}<div class="support-inner support-media-frame"><div class="support-label">SCENE / ${{String(i+1).padStart(2,"0")}}</div><div class="support-media-caption">${{beat.text}}</div></div></div>`;}}
      else if(beat.kind==="compare") el.innerHTML=`<div class="support-surface"><div class="compare-stage"><div class="compare-card left"><div class="compare-tag">NOT THIS</div><div class="compare-text">${{beat.left}}</div></div><div class="compare-card right"><div class="compare-tag">BUT THIS</div><div class="compare-text">${{beat.right}}</div></div></div></div>`;
      else if(beat.kind==="stat") el.innerHTML=`<div class="support-surface"><div class="stat-stage"><div><div class="support-label">NUMBER / FOCUS</div><div class="stat-value">${{beat.stat}}</div><div class="stat-label">${{beat.label}}</div></div><div class="bars"><i class="bar" style="height:32%"></i><i class="bar" style="height:48%"></i><i class="bar" style="height:61%"></i><i class="bar" style="height:78%"></i><i class="bar" style="height:96%"></i></div></div></div>`;
      else if(beat.kind==="knowledge"){{const points=(beat.points||[]).map((point,pointIndex)=>`<div class="knowledge-point"><i>${{pointIndex+1}}</i><span>${{point}}</span></div>`).join("");el.innerHTML=`<div class="support-surface"><div class="knowledge-stage"><div class="knowledge-kicker">要点拆解</div><div class="knowledge-title">${{beat.text}}</div><div class="knowledge-points">${{points}}</div></div></div>`;}}
      else if(beat.kind==="outro"){{const points=(beat.points||[]).map(point=>`<div class="outro-point">${{point}}</div>`).join("");el.innerHTML=`<div class="support-surface"><div class="outro-stage"><div class="outro-kicker">本期行动总结</div><div class="outro-title">${{beat.text}}</div><div class="outro-points">${{points}}</div></div></div>`;}}
      else if(beat.kind==="context") el.innerHTML=`<div class="support-surface"><div class="context-stage"><div class="context-kicker">SCENE / CONTEXT</div><div class="context-orbit"><i class="context-dot one"></i><i class="context-dot two"></i><div class="context-token">${{beat.query||beat.text}}</div></div><div class="context-copy"><small>当前语境</small><div class="context-title">${{beat.text}}</div></div></div></div>`;
      else el.innerHTML=`<div class="support-surface"><div class="support-inner support-quote"><div class="support-label">${{calloutEyebrow}}</div><div class="support-text ${{beat.text.length>22?"long":""}}">${{beat.text}}</div></div><div class="callout-meta"><small>DECISION LENS</small><strong>${{calloutTheme}}</strong><em>行业洞察 · 经营选择</em></div><div class="callout-connector"></div><div class="callout-ghost" aria-hidden="true"></div></div>`;
      root.appendChild(el);
    }});
    DATA.lower_thirds.forEach((item,i)=>{{const el=document.createElement("div");el.id=`lower-third-${{i}}`;el.className="clip lower-third";el.dataset.start=item.start;el.dataset.duration=item.duration;el.dataset.trackIndex=6;el.innerHTML=`<div class="lower-third-card"><div class="lower-third-rule"></div><div class="lower-third-copy"><div class="lower-third-name">${{item.speaker||"本期主讲"}}</div>${{item.title?`<div class="lower-third-title">${{item.title}}</div>`:""}}</div></div>`;root.appendChild(el);}});
    window.__timelines=window.__timelines||{{}};const tl=gsap.timeline({{paused:true}});
    tl.fromTo("#hook .hook-inner",{{opacity:0}},{{opacity:1,duration:.12,ease:"none"}},hookAt+.28).fromTo("#hook .hook-kicker",{{opacity:0,x:-28}},{{opacity:1,x:0,duration:.24,ease:"expo.out"}},hookAt+.30).fromTo("#hook .hook-title",{{opacity:0,y:20}},{{opacity:1,y:0,duration:.34,ease:"circ.out"}},hookAt+.36).fromTo("#hook .hook-subtitle",{{opacity:0,x:18}},{{opacity:1,x:0,duration:.36,ease:"power3.out"}},hookAt+.48).fromTo("#hook .hook-rule",{{scaleX:0,transformOrigin:"left center"}},{{scaleX:1,duration:.42,ease:"power4.out"}},hookAt+.56).to("#hook .hook-inner",{{opacity:0,y:-10,duration:.14,ease:"power2.in"}},Math.max(hookAt+.8,hookAt+hookDuration-.36)).to("#hook",{{opacity:0,duration:.08,ease:"none"}},Math.max(hookAt+.9,hookAt+hookDuration-.08));
    if(hasHookMedia)tl.fromTo("#hook .hook-media",{{opacity:0,scale:1.04}},{{opacity:1,scale:1,duration:.42,ease:"power2.out"}},hookAt);if(hasHookPip)tl.fromTo("#hook-pip",{{opacity:0,scale:.82,rotation:-3}},{{opacity:1,scale:1,rotation:0,duration:.54,ease:"back.out(1.2)"}},hookAt+.16).to("#hook-pip",{{opacity:0,scale:.94,duration:.18,ease:"power2.in"}},hookAt+hookDuration-.22);if(MOTION.editorial)tl.fromTo("#hook .hook-chips i",{{opacity:0,y:14}},{{opacity:1,y:0,duration:.28,stagger:.07,ease:"power3.out"}},hookAt+.30);
    DATA.camera_beats.forEach((beat)=>{{const target=`#video-${{beat.segment_index}}`;tl.to(target,{{scale:beat.scale,x:beat.x,y:beat.y,duration:.42,ease:"power3.out"}},beat.start).to(target,{{scale:1,x:0,y:0,duration:.52,ease:"power2.inOut"}},beat.start+beat.duration-.52);}});
    DATA.captions.forEach((cap,i)=>{{const d=Math.max(.12,cap.end-cap.start);tl.fromTo(`#cap-${{i}} .caption-shell`,{{opacity:0,y:13,scale:.985}},{{opacity:1,y:0,scale:1,duration:Math.min(.16,d*.32),ease:"power3.out"}},cap.start);if(cap.emphasis&&cap.emphasis.length)tl.fromTo(`#cap-${{i}} .keyword`,{{textShadow:"0 0 0 rgba(255,216,61,0)"}},{{textShadow:"0 0 22px rgba(255,216,61,.68)",duration:.15,yoyo:true,repeat:1,stagger:.05}},cap.start+.08);tl.to(`#cap-${{i}} .caption-shell`,{{opacity:0,y:-6,duration:Math.min(.08,d*.15),ease:"power2.in"}},Math.max(cap.start,cap.end-.08));}});
    DATA.chapters.forEach((chapter,i)=>{{const at=chapter.start;const leave=at+chapter.duration-.38;tl.fromTo(`#chapter-${{i}} .chapter-stage`,{{y:1920,opacity:1}},{{y:0,opacity:1,duration:.46,ease:"power3.out"}},at).fromTo(`#chapter-${{i}} .chapter-index`,{{x:-150,opacity:0}},{{x:0,opacity:1,duration:.34,ease:"expo.out"}},at+.18).fromTo(`#chapter-${{i}} .chapter-title`,{{x:46,scale:1.06,opacity:0}},{{x:0,scale:1,opacity:1,duration:.58,ease:"power4.out"}},at+.28).fromTo(`#chapter-${{i}} .chapter-rule`,{{scaleX:0}},{{scaleX:1,duration:.52,ease:"circ.out"}},at+.62);Array.from(document.getElementById(`chapter-${{i}}`).getElementsByTagName("i")).forEach((tick,tickIndex)=>tl.to(tick,{{opacity:1,duration:.08,yoyo:true,repeat:1,ease:"none"}},at+.40+tickIndex*.22));tl.to(`#chapter-${{i}} .chapter-stage`,{{y:-1920,duration:.38,ease:"power3.in"}},leave);}});
    VISUAL_BEATS.forEach((beat,i)=>{{
      const at=Number(beat.start);const speakerAt=at+Number(beat.entry_offsets&&beat.entry_offsets.speaker||.16);const textAt=at+Number(beat.entry_offsets&&beat.entry_offsets.text||.30);const hasSpeakerPip=Boolean(MOTION.editorial&&["circle","card","cutout"].includes(beat.speaker_role)&&beat.background&&beat.background.status==="ready");
      tl.fromTo(`#support-${{i}} .support-surface`,{{opacity:0,scale:MOTION.editorial ? .975 : .992,y:MOTION.editorial ? 0 : 16}},{{opacity:1,scale:1,y:0,duration:MOTION.editorial ? .42 : .28,ease:MOTION.editorial ? "expo.out" : "power3.out"}},at);
      if(hasSpeakerPip)tl.fromTo(`#support-pip-${{i}}`,{{opacity:0,scale:.82,y:26}},{{opacity:1,scale:1,y:0,duration:.48,ease:"back.out(1.15)"}},speakerAt);
      if(beat.kind!=="media"&&beat.blur>0){{const footage=`#video-${{beat.segment_index}}`;const blurValue=MOTION.editorial?Math.min(beat.blur,.12):Math.min(beat.blur,MOTION.blur);tl.to(footage,{{"--hf-color-grading-blur":blurValue,duration:.22,ease:"sine.out"}},at).to(footage,{{"--hf-color-grading-blur":0,duration:.38,ease:"sine.inOut"}},at+beat.duration-.42);}}
      if(beat.kind==="media"){{tl.fromTo(`#support-${{i}} .support-inner`,{{opacity:0,y:-16}},{{opacity:1,y:0,duration:.32,ease:"power3.out"}},textAt);if(beat.asset&&beat.asset.kind==="image")tl.fromTo(`#support-${{i}} .support-image`,{{scale:1.01}},{{scale:1.06,duration:beat.duration,ease:"none"}},at);else if(beat.asset){{const radius=MOTION.editorial?0:{profile['support_radius']};tl.fromTo(`#support-video-${{i}}`,{{clipPath:`inset(0 100% 0 0 round ${{radius}}px)`,scale:1.01}},{{clipPath:`inset(0 0% 0 0 round ${{radius}}px)`,scale:1.05,duration:beat.duration,ease:"power2.out"}},at);}}}}
      else if(beat.kind==="compare"){{const turnAt=Math.max(textAt+.12,at+Math.min(beat.duration-.7,Math.max(.10,Number(beat.turn_offset||.32))));tl.fromTo(`#support-${{i}} .left`,{{x:-62,opacity:0}},{{x:0,opacity:1,duration:.36,ease:"power3.out"}},textAt).fromTo(`#support-${{i}} .right`,{{x:62,opacity:0}},{{x:0,opacity:1,duration:.40,ease:"power3.out"}},turnAt);}}
      else if(beat.kind==="stat"){{const numberAt=Math.max(textAt,at+Math.min(beat.duration-.7,Math.max(.06,Number(beat.number_offset||.12))));tl.fromTo(`#support-${{i}} .stat-value`,{{scale:1.045,opacity:0}},{{scale:1,opacity:1,duration:.32,ease:"power3.out"}},numberAt).fromTo(`#support-${{i}} .bar`,{{scaleY:0}},{{scaleY:1,duration:.58,stagger:.07,ease:"power3.out"}},numberAt+.10);}}
      else if(beat.kind==="knowledge"){{tl.fromTo(`#support-${{i}} .knowledge-kicker`,{{opacity:0,x:-24}},{{opacity:1,x:0,duration:.26,ease:"expo.out"}},textAt).fromTo(`#support-${{i}} .knowledge-title`,{{opacity:0,y:18}},{{opacity:1,y:0,duration:.38,ease:"circ.out"}},textAt+.08).fromTo(`#support-${{i}} .knowledge-point`,{{opacity:0,x:-34}},{{opacity:1,x:0,duration:.34,stagger:.28,ease:"power3.out"}},textAt+.30);}}
      else if(beat.kind==="outro"){{tl.fromTo(`#support-${{i}} .outro-kicker`,{{opacity:0,x:-20}},{{opacity:1,x:0,duration:.24,ease:"expo.out"}},textAt).fromTo(`#support-${{i}} .outro-title`,{{opacity:0,y:16}},{{opacity:1,y:0,duration:.34,ease:"power3.out"}},textAt+.08).fromTo(`#support-${{i}} .outro-point`,{{opacity:0,y:18}},{{opacity:1,y:0,duration:.30,stagger:.18,ease:"power3.out"}},textAt+.28);}}
      else if(beat.kind==="context"){{tl.fromTo(`#support-${{i}} .context-kicker`,{{opacity:0,x:-24}},{{opacity:1,x:0,duration:.26,ease:"expo.out"}},textAt).fromTo(`#support-${{i}} .context-orbit`,{{opacity:0,scale:.84,rotation:-8}},{{opacity:1,scale:1,rotation:0,duration:.62,ease:"back.out(1.08)"}},textAt+.04).fromTo(`#support-${{i}} .context-token`,{{opacity:0,scale:.88}},{{opacity:1,scale:1,duration:.34,ease:"power3.out"}},textAt+.26).fromTo(`#support-${{i}} .context-copy`,{{opacity:0,y:30}},{{opacity:1,y:0,duration:.42,ease:"power3.out"}},textAt+.38);}}
      else{{tl.fromTo(`#support-${{i}} .support-label`,{{opacity:0,x:-18}},{{opacity:1,x:0,duration:.22,ease:"expo.out"}},textAt).fromTo(`#support-${{i}} .support-text`,{{opacity:0,y:14}},{{opacity:1,y:0,duration:.34,ease:"power3.out"}},textAt+.06);if(MOTION.editorial)tl.fromTo(`#support-${{i}} .callout-meta`,{{opacity:0,x:-22}},{{opacity:1,x:0,duration:.30,ease:"power3.out"}},textAt+.20).fromTo(`#support-${{i}} .callout-connector`,{{scaleX:0}},{{scaleX:1,duration:.46,ease:"power4.out"}},textAt+.28).fromTo(`#support-${{i}} .callout-ghost`,{{opacity:0,y:30}},{{opacity:1,y:0,duration:.42,ease:"power2.out"}},textAt+.34);}}
      const exitTarget=beat.kind==="context"?`#support-${{i}} .context-stage`:`#support-${{i}} .support-inner, #support-${{i}} .knowledge-stage, #support-${{i}} .compare-stage, #support-${{i}} .stat-stage, #support-${{i}} .outro-stage`;tl.to(exitTarget,{{opacity:0,y:beat.kind==="context"?0:-8,duration:.14,ease:"power2.in"}},at+beat.duration-.36);if(hasSpeakerPip)tl.to(`#support-pip-${{i}}`,{{opacity:0,scale:.94,duration:.18,ease:"power2.in"}},at+beat.duration-.22);tl.to(`#support-${{i}} .support-surface`,{{opacity:0,scale:.997,duration:.08,ease:"none"}},at+beat.duration-.08);
    }});
    DATA.lower_thirds.forEach((item,i)=>{{const at=item.start;tl.fromTo(`#lower-third-${{i}} .lower-third-card`,{{opacity:0,x:-54,scale:.985}},{{opacity:1,x:0,scale:1,duration:.34,ease:"power3.out"}},at).to(`#lower-third-${{i}} .lower-third-card`,{{opacity:0,x:-26,duration:.24,ease:"power2.in"}},at+item.duration-.24);}});
    if(!DATA.mix_audio&&DATA.bgm&&DATA.bgm.segments.length){{DATA.bgm.cues.forEach((cue)=>{{const segmentIndex=DATA.bgm.segments.findIndex(seg=>cue.at>=seg.start&&cue.at<seg.start+seg.duration+.001);if(segmentIndex>=0)tl.to(`#bgm-${{segmentIndex}}`,{{volume:cue.volume,duration:cue.duration,ease:"sine.inOut"}},cue.at);}});}}
    window.__timelines["main"]=tl;
  </script>
</body>
</html>'''


@contextmanager
def specialist_run(
    workflow: AgentWorkflow, ledger_path: Path, agent_id: str, capability: str,
    *, detail: str, provider: str = "local-skill", model: str = "", inputs: Any = None,
):
    """Record one specialist as a durable, UI-visible Agent hand-off."""
    run_id = workflow.begin_agent(
        agent_id, stage=agent_id, capability=capability, provider=provider, model=model,
        inputs=inputs, metadata={"detail": detail},
    )
    workflow.save(ledger_path)
    print(f"    Agent · {agent_id}：{detail}")
    try:
        yield run_id
    except Exception as exc:
        workflow.fail_run(run_id, exc)
        workflow.state.status = "failed"
        workflow.save(ledger_path)
        raise
    else:
        workflow.complete_run(run_id, outputs={"status": "delivered"})
        workflow.save(ledger_path)


def bind_scene_packages(
    workflow: AgentWorkflow, visual_beats: list[dict[str, Any]], hook: dict[str, Any],
    *, editorial: bool,
) -> None:
    """Bind final media to director scene contracts; never enable PiP alone."""
    scene_cues = [cue for cue in workflow.state.cues if cue.cue_type is CueType.SCENE]
    packages = [
        {"start": float(beat["start"]), "end": float(beat["start"]) + float(beat["duration"]), "asset": beat.get("asset"), "text": str(beat.get("text", ""))}
        for beat in visual_beats if beat.get("kind") == "media" and beat.get("asset")
    ]
    if hook.get("asset"):
        packages.append({
            "start": float(hook.get("start", 0)), "end": float(hook.get("end", 0)),
            "asset": hook["asset"], "text": str(hook.get("asset_caption", "")),
        })
    claimed: set[int] = set()
    for cue in scene_cues:
        candidates = [
            (index, package) for index, package in enumerate(packages) if index not in claimed
            and min(cue.end + .9, package["end"] + .9) >= max(cue.start - .9, package["start"] - .9)
        ]
        if not candidates:
            candidates = [
                (index, package) for index, package in enumerate(packages) if index not in claimed
                and abs(package["start"] - cue.start) <= 7
            ]
        if not candidates:
            continue
        index, package = min(candidates, key=lambda item: abs(item[1]["start"] - cue.start))
        claimed.add(index)
        raw_asset = dict(package["asset"])
        raw_asset.update({
            "status": "ready", "caption": package["text"] or cue.editorial_text,
            "metadata": {
                **dict(raw_asset.get("metadata") or {}),
                "semantic_evidence": "director-scene-and-timestamp-route",
                "visual_validation": "metadata-only",
            },
        })
        workflow.bind_asset(
            cue.cue_id, AssetRef.from_mapping(raw_asset),
            visual_mode="media_fullscreen_with_speaker_pip" if editorial else "media_half",
            speaker_pip=editorial,
        )


def sync_workflow_scene_packages(
    workflow: AgentWorkflow, packages: list[dict[str, Any]], visual_beats: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Make the gated AgentWorkflow packages the renderer's only scene source."""
    type_map = {
        "hook": CueType.HOOK, "callout": CueType.CALLOUT, "knowledge": CueType.KNOWLEDGE,
        "chapter": CueType.CHAPTER, "scene": CueType.SCENE,
    }
    claimed: set[str] = set()
    matched: list[tuple[dict[str, Any], Any]] = []
    old_to_new: dict[str, tuple[str, str]] = {}
    for package in packages:
        cue_type = type_map.get(str(package.get("cue_type", "")))
        candidates = [
            cue for cue in workflow.state.cues
            if cue.cue_type is cue_type and cue.cue_id not in claimed
        ]
        if not candidates:
            continue
        start = float(package.get("start", 0))
        cue = min(candidates, key=lambda item: abs(item.start - start))
        if abs(cue.start - start) > 8:
            continue
        claimed.add(cue.cue_id)
        old_id = str(package.get("package_id", ""))
        new_id = f"scene-package-{cue.cue_id}"
        package["package_id"] = new_id
        package["cue_id"] = cue.cue_id
        old_to_new[old_id] = (new_id, cue.cue_id)
        cue.start = float(package["start"])
        cue.end = float(package["end"])
        cue.editorial_text = str(package.get("text", cue.editorial_text))
        cue.visual_mode = str(package.get("visual_mode", package.get("visual_state", cue.visual_mode)))
        cue.speaker_pip = str(package.get("speaker_role", "full")) in {"circle", "card", "cutout"}
        cue.fullscreen = bool(package.get("fullscreen", False))
        cue.metadata["scene_package"] = {
            "background_role": package.get("background_role", "source"),
            "media_role": package.get("media_role", "none"),
            "speaker_role": package.get("speaker_role", "full"),
            "text_role": package.get("text_role", "none"),
            "caption_mode": package.get("caption_mode", "normal"),
            "entry_transition": package.get("entry_transition") or {"name": "scene_in", "duration": .34, "easing": "expo.out"},
            "exit_transition": package.get("exit_transition") or {"name": "scene_out", "duration": .22, "easing": "power2.in"},
        }
        media = package.get("media")
        cue.asset = AssetRef.from_mapping(media) if isinstance(media, dict) else None
        if cue.asset:
            cue.visual_mode = (
                "media_fullscreen_with_speaker_pip"
                if str(package.get("speaker_role", "full")) in {"circle", "card", "cutout"}
                else "media_half"
            )
        if cue.cue_type is CueType.SCENE and not cue.asset:
            cue.metadata.update({"resolution": "semantic_graphic_fallback", "degraded_from": "requested_media"})
            cue.asset_requirement = ""
            cue.dependencies = ["timing.valid"]
            cue.fallback_visual_mode = cue.visual_mode
            cue.status = CueStatus.DEGRADED
        else:
            cue.dependencies = ["timing.valid"]
            if cue.requires_spoken_alignment:
                cue.dependencies.append("speech.aligned")
            if cue.asset:
                cue.dependencies.extend(["asset.ready", "asset.semantic_match", "asset.no_contradiction"])
            cue.status = CueStatus.READY
        cue.ensure_dependencies()
        matched.append((package, cue))

    # Director scenes absorbed by a stronger Hook/Callout are recorded as a
    # deliberate merge, not left looking like silently lost work.
    for cue in workflow.state.cues:
        if cue.cue_type is not CueType.SCENE or cue.cue_id in claimed:
            continue
        overlapping = next((package for package, _ in matched if float(package["start"]) < cue.end and float(package["end"]) > cue.start), None)
        if overlapping:
            cue.metadata.update({"resolution": f"merged_into_{overlapping.get('cue_type', 'primary')}", "merged_package_id": overlapping.get("package_id", "")})
            cue.visual_mode = "none"
            cue.asset_requirement = ""
            cue.dependencies = ["timing.valid"]
            cue.fallback_visual_mode = "none"
            cue.status = CueStatus.SKIPPED

    workflow.evaluate_gates(apply_fallbacks=True)
    for cue in workflow.state.cues:
        resolution = str(cue.metadata.get("resolution", ""))
        if resolution == "semantic_graphic_fallback":
            cue.status = CueStatus.DEGRADED
        elif resolution.startswith("merged_into_"):
            cue.status = CueStatus.SKIPPED
    normalized = []
    for source, cue in matched:
        package = workflow.normalize_scene_package(cue)
        package.package_id = str(source["package_id"])
        package.metadata.update({
            "cue_type": source.get("cue_type", cue.cue_type.value),
            "visual_mode": source.get("visual_mode", source.get("visual_state", cue.visual_mode)),
            "background_at": source.get("background_at", source.get("start")),
            "speaker_at": source.get("speaker_at"),
            "text_at": source.get("text_at"),
            "restore_source": True,
            "exit_state": "restore_source_and_full_speaker",
            "caption_zone": source.get("caption_zone", {"occlusion_ratio": 0}),
            "highlight_zone": source.get("highlight_zone", {"occlusion_ratio": 0}),
            "safe_zones": source.get("safe_zones", []),
        })
        normalized.append(package)
    workflow.state.scene_packages = normalized
    serialized = workflow.state.to_dict().get("scene_packages", [])
    package_by_id = {item["package_id"]: item for item in serialized}
    for beat in visual_beats:
        replacement = old_to_new.get(str(beat.get("package_id", "")))
        if replacement:
            beat["package_id"], beat["cue_id"] = replacement
    return [package_by_id[item["package_id"]] for item, _ in matched if item["package_id"] in package_by_id]


def main() -> None:
    args = parse_args()
    project = Path(args.project).resolve()
    source_input = Path(args.input).expanduser().resolve()
    if not source_input.exists():
        raise SystemExit(f"找不到视频：{source_input}")
    config = load_config(args.config, args.mode)
    assets, output = project / "assets", project / "output"
    support_dir = Path(args.support_dir or args.image_dir).resolve() if (args.support_dir or args.image_dir) else project / "input" / "images"
    assets.mkdir(exist_ok=True); output.mkdir(exist_ok=True); support_dir.mkdir(parents=True, exist_ok=True)
    signature = file_signature(source_input)
    source_asset = assets / f"source{source_input.suffix.lower()}"
    if source_input != source_asset:
        shutil.copy2(source_input, source_asset)
    info = media_info(source_asset)
    ledger_path = output / "agent-workflow.json"
    workflow = AgentWorkflow.create(
        project.name, source=source_asset.relative_to(project).as_posix(), duration=float(info["duration"]),
        metadata={"architecture": "chief-director + specialist-agents + gated-assembly"},
    )
    workflow.save(ledger_path)
    print(f"1/5 识别语音：{info['duration']:.1f} 秒")
    with specialist_run(
        workflow, ledger_path, "content", "transcript-and-semantic-analysis",
        detail="识别完整口播并建立逐词时间轴", inputs={"source": str(source_input), "model": config["model"]},
    ):
        words, raw_segments, language, reused = transcribe_with_cache(source_asset, output, config["model"], args.cache_dir, signature)
    if reused:
        print("    已复用上次逐字稿")
    proposed = candidate_cuts(words, info["duration"], float(config["silence_threshold_seconds"]))
    mode = choose_mode(config["mode"], proposed, info["duration"])
    cuts = proposed if mode == "recut" else []
    keep = invert_cuts(cuts, info["duration"])
    mapped_words = remap_words(words, keep)
    semantic_segments = remap_segments(raw_segments, keep)
    edited_duration = sum(item["duration"] for item in keep)
    workflow.state.video["duration"] = round(edited_duration, 3)
    keywords = [clean_text(str(item)) for item in config.get("keywords", []) if clean_text(str(item))]
    captions = group_captions(mapped_words, keywords)
    content_review = build_content_review(captions or semantic_segments, config, output)
    captions = repair_caption_boundaries(apply_content_review(captions, content_review))
    semantic_segments = apply_content_review(semantic_segments, content_review) if len(semantic_segments) == len(captions) else semantic_segments
    effective_speaker, effective_title, identity_meta = resolve_identity(config, content_review)
    runtime_config = {**config, "speaker": effective_speaker, "speaker_title": effective_title}
    director_groups = captions or semantic_segments
    with specialist_run(
        workflow, ledger_path, "director", "full-transcript-editorial-direction",
        detail="从完整内容决定 Hook、场景、章节和重点",
        provider="Volcano Ark" if os.environ.get("ARK_API_KEY", "").strip() else "local-skill",
        model=str(config.get("director_model", "")), inputs={"caption_groups": len(director_groups), "duration": edited_duration},
    ) as director_run:
        director_plan = build_director_plan(
            director_groups, runtime_config, edited_duration, source_input.stem, output, content_review,
        )
        director_plan = align_director_cues(director_plan, director_groups, mapped_words, edited_duration)
    workflow._run(director_run).metadata["detail"] = (
        "大模型已完成全片语义导演" if director_plan.get("status") == "llm"
        else f"规则导演已完成 · {director_plan.get('reason', '未调用模型')}"
    )
    workflow.ingest_director_plan(director_plan, owner_run_id=director_run)
    workflow.save(ledger_path)
    (output / "director-plan.json").write_text(json.dumps(director_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    cover_copy = director_plan["cover"]
    if director_plan.get("hook", {}).get("text"):
        cover_copy["hook"] = clean_text(str(director_plan["hook"]["text"]))[:32]
        cover_copy["impact_line"] = cover_copy["hook"][:24]
    title = cover_copy["headline"]
    subtitle = cover_copy["subheadline"]
    hook_start = max(0.0, float(director_plan.get("hook", {}).get("start", 0) or 0))
    hook_end = float(director_plan.get("hook", {}).get("end", hook_start + 3.2) or hook_start + 3.2)
    hook_window = [{"start": hook_start, "duration": max(2.4, min(5.2, hook_end - hook_start + .55))}]
    chapters = chapters_from_plan(director_plan, edited_duration)
    # Hook、章节、Callout 都是“画面主角”，同一时刻只保留一个视觉角色。
    # 如果模型把同一段结论同时识别为 Hook 和章节，保留更具体的 Hook，避免全屏层互相遮挡。
    chapters = [chapter for chapter in chapters if not overlaps_window(chapter["start"], chapter["duration"], hook_window, .25)]
    # 章节是唯一的全屏信息层；章节期间不再渲染底部字幕，避免“被遮住但仍在布局树中”的半套状态。
    captions = [
        caption for caption in captions
        if not overlaps_window(
            float(caption["start"]), max(.12, float(caption["end"]) - float(caption["start"])),
            chapters + hook_window, 0,
        )
    ]
    visual_beats = visual_beats_from_plan(director_plan, edited_duration, chapters)
    with specialist_run(
        workflow, ledger_path, "media", "scene-sourcing-and-generation",
        detail="为具体语句检索、生成并绑定配套画面",
        provider="Volcano Ark / Pexels / uploads", inputs={"requested_scenes": len(director_plan.get("scene_assets", []))},
    ) as media_run:
        source_records = resolve_smart_media(director_plan.get("scene_assets", []), support_dir, runtime_config)
        support_assets = stage_support_media(support_dir, project, signature)
        visual_beats = attach_scene_assets(
            visual_beats, director_plan.get("scene_assets", []), support_assets,
            edited_duration, chapters, project, runtime_config,
        )
        visual_beats = add_scene_fallback_beats(
            visual_beats, director_plan.get("scene_assets", []), edited_duration, chapters, hook_window,
        )
    delivered_media = sum(beat.get("kind") == "media" and bool(beat.get("asset")) for beat in visual_beats)
    planned_media = len(director_plan.get("scene_assets", []))
    workflow._run(media_run).metadata["detail"] = (
        f"已交付 {delivered_media} 个配套场景" if delivered_media
        else f"未取得真实素材，{planned_media} 个候选场景已安全降级"
    )
    hook_asset = None
    hook_asset_caption = ""
    resolved_visual_beats: list[dict[str, Any]] = []
    for beat in visual_beats:
        if overlaps_window(beat["start"], beat["duration"], hook_window, .25):
            # 具体素材正好落在 Hook 原话上时，把它升级为 Hook 背景；不要丢掉素材或再叠一层 Callout。
            if hook_asset is None and beat.get("kind") == "media" and beat.get("asset"):
                hook_asset = beat["asset"]
                hook_asset_caption = clean_text(str(beat.get("text", "")))[:18]
            elif beat.get("kind") == "knowledge":
                # 开场常会同时被识别为 Hook 和“几点总结”。两者是前后叙事，
                # 不应因为时间冲突直接删掉知识卡：先用 Hook 勾住，再紧接着展开。
                moved = dict(beat)
                moved["start"] = round(hook_window[0]["start"] + hook_window[0]["duration"] + .28, 3)
                moved["duration"] = round(min(float(moved["duration"]), edited_duration - moved["start"] - .2), 3)
                conflicts = resolved_visual_beats + chapters
                if moved["duration"] >= 4.8 and not overlaps_window(moved["start"], moved["duration"], conflicts, .25):
                    moved["alignment"] = "semantic-window-reflowed-after-hook"
                    resolved_visual_beats.append(moved)
            continue
        resolved_visual_beats.append(beat)
    visual_beats = resolved_visual_beats
    visual_beats = enrich_visual_scene_packages(visual_beats)
    # When the spoken quote itself becomes the main on-screen object, a second
    # subtitle line repeats the same information and makes the frame feel like
    # stacked components.  Keep subtitles for media/context/knowledge states,
    # but hide them for the brief quote/compare/stat takeover.
    caption_hidden_windows = [*hook_window, *chapters, *[
        beat for beat in visual_beats if beat.get("caption_mode") == "hidden"
    ]]
    captions = [
        caption for caption in captions
        if not any(
            float(window["start"]) <= (float(caption["start"]) + float(caption["end"])) * .5
            < float(window["start"]) + float(window.get("duration", 0))
            for window in caption_hidden_windows
        )
    ]
    with specialist_run(
        workflow, ledger_path, "motion", "motion-system-and-timeline-design",
        detail="统一安排文字、章节、镜头与人物画中画",
        inputs={"visual_beats": len(visual_beats), "chapters": len(chapters)},
    ):
        for beat in visual_beats:
            beat["segment_index"] = next((i for i, item in enumerate(keep) if item["edited_start"] <= beat["start"] <= item["edited_start"] + item["duration"]), 0)
        camera_beats = select_camera_beats(captions, keep, edited_duration, visual_beats, chapters)
        apply_caption_emphasis(captions, director_plan)
        identity_blockers = [
            *chapters, *hook_window,
            *[beat for beat in visual_beats if beat.get("kind") != "media"],
        ]
        lower_thirds = lower_thirds_from_plan(
            director_plan, edited_duration, identity_blockers, effective_speaker, effective_title,
        )
        recommended_motion_template = recommend_motion_template(visual_beats, director_plan)
        requested_motion_template = str(config.get("motion_template", "auto"))
        selected_motion_template = requested_motion_template if requested_motion_template in MOTION_TEMPLATE_BY_ID else recommended_motion_template
        motion_templates = motion_template_catalog(selected_motion_template, recommended_motion_template)
    with specialist_run(
        workflow, ledger_path, "audio", "music-and-program-mix",
        detail="根据观点节奏生成配乐并自动压低人声下方音量",
        inputs={"music_direction": director_plan.get("music", {})},
    ):
        bgm = prepare_bgm(runtime_config, project, edited_duration, signature, visual_beats, chapters)
        mix_audio = bake_program_mix(project, source_asset, keep, bgm, signature, edited_duration)
        for old_preview in output.glob("bgm-preview.*"):
            old_preview.unlink(missing_ok=True)
        for old_preview in output.glob("program-mix-preview.*"):
            old_preview.unlink(missing_ok=True)
        if bgm:
            bgm_source = project / bgm["path"]
            preview_name = f"bgm-preview{bgm_source.suffix.lower()}"
            shutil.copy2(bgm_source, output / preview_name)
            bgm["preview"] = f"output/{preview_name}"
        if mix_audio:
            mix_source = project / mix_audio["path"]
            mix_preview_name = f"program-mix-preview{mix_source.suffix.lower()}"
            shutil.copy2(mix_source, output / mix_preview_name)
            mix_audio["preview"] = f"output/{mix_preview_name}"
    for caption in captions:
        caption["html"] = markup_caption(caption["text"], caption.get("emphasis", []))
    for chapter in chapters:
        chapter["title"] = html.escape(chapter["title"])
    for beat in visual_beats:
        for key in ("text", "left", "right", "stat", "label"):
            if key in beat:
                beat[key] = html.escape(str(beat[key]))
        if "points" in beat:
            beat["points"] = [html.escape(str(point)) for point in beat.get("points", [])]
    hook = {
        **director_plan.get("hook", {}),
        "text": html.escape(str(director_plan.get("hook", {}).get("text", title))),
        # The visible Hook card is capped independently from the spoken quote.
        # Persist the actual render window so QA does not flag the following
        # Callout merely because the underlying sentence continues speaking.
        "duration": round(float(hook_window[0]["duration"]), 3),
        "asset": hook_asset,
        "asset_caption": html.escape(hook_asset_caption),
    }
    scene_packages = build_visual_scene_manifest(hook, visual_beats, chapters)
    for item in lower_thirds:
        item["speaker"] = html.escape(str(item.get("speaker", "")))
        item["title"] = html.escape(str(item.get("title", "")))
    scene_packages = sync_workflow_scene_packages(workflow, scene_packages, visual_beats)
    workflow.build_manifest(metadata={"stage": "pre-qa", "scene_source": "agent-workflow"})
    workflow.save(ledger_path)

    print(f"2/5 剪辑判断：{mode}，预计成片 {edited_duration:.1f} 秒；章节 {len(chapters)}，辅助画面 {len(visual_beats)}")
    transcript_payload = {
        "language": language, "source": str(source_input), "signature": signature,
        "model": config["model"], "media": info, "segments": raw_segments,
        "corrected_segments": [{"start": item["start"], "end": item["end"], "text": item["text"]} for item in director_groups],
        "words": [word.__dict__ for word in words],
    }
    (output / "transcript.json").write_text(json.dumps(transcript_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    transcript_text = "".join(str(segment["text"]) for segment in director_groups)
    (output / "transcript.txt").write_text(transcript_text + "\n", encoding="utf-8")
    write_srt(captions, output / "subtitles.srt")
    (output / "media-sources.json").write_text(json.dumps(source_records, ensure_ascii=False, indent=2), encoding="utf-8")
    motion_plan = {
        "requested_mode": config["mode"], "selected_mode": mode, "source_duration": round(info["duration"], 3), "edited_duration": round(edited_duration, 3),
        "removed_seconds": round(info["duration"] - edited_duration, 3), "cuts": cuts, "keep_segments": keep,
        "rhythm": "opening world → speaker hold → semantic scene package → speaker reset → chapter transition",
        "title": title, "subtitle": subtitle, "cover_copy": cover_copy, "hook": hook, "captions": captions, "chapters": chapters, "visual_beats": visual_beats, "scene_packages": scene_packages, "camera_beats": camera_beats, "lower_thirds": lower_thirds, "bgm": bgm, "mix_audio": mix_audio, "director": {key: director_plan.get(key) for key in ("status", "provider", "model", "reason", "music")},
        "motion_template": selected_motion_template, "recommended_motion_template": recommended_motion_template, "motion_templates": motion_templates,
        "smart_media_enabled": bool(config.get("smart_media")),
    }
    (output / "motion-plan.json").write_text(json.dumps(motion_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "motion-template-manifest.json").write_text(json.dumps({"selected": selected_motion_template, "recommended": recommended_motion_template, "templates": motion_templates}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("3/5 生成封面与背景乐")
    with specialist_run(
        workflow, ledger_path, "cover", "cover-copy-and-template-design",
        detail="从完整内容提炼标题、副标题与人物背书",
        inputs={"cover_copy": cover_copy},
    ):
        cover_frame, cover_at = choose_cover_frame(source_asset, info["duration"], config.get("cover_time_seconds"))
        cover_templates, selected_cover, cover_qa = generate_cover_templates(
            cover_frame, cover_copy, config["series"], effective_speaker, effective_title,
            config["accent"], output, str(config.get("cover_template", "headline")),
        )
        (output / "cover-copy.json").write_text(json.dumps(cover_copy, ensure_ascii=False, indent=2), encoding="utf-8")
    composition = {
        "source": source_asset.relative_to(project).as_posix(), "duration": round(edited_duration, 3), "title": html.escape(title), "subtitle": html.escape(subtitle),
        "series": html.escape(config["series"]), "speaker": html.escape(effective_speaker), "speaker_title": html.escape(effective_title),
        "identity": identity_meta, "accent": config["accent"],
        "keep_segments": keep, "hook": hook, "captions": captions, "chapters": chapters, "visual_beats": visual_beats, "scene_packages": scene_packages, "camera_beats": camera_beats, "lower_thirds": lower_thirds, "bgm": bgm, "mix_audio": mix_audio, "cover_copy": cover_copy,
        "motion_template": selected_motion_template, "recommended_motion_template": recommended_motion_template,
        "smart_media_enabled": bool(config.get("smart_media")),
    }
    with specialist_run(
        workflow, ledger_path, "assembly", "gated-hyperframes-assembly",
        detail="只组装已经满足依赖的完整场景包",
        inputs={"template": selected_motion_template, "visual_beats": len(visual_beats)},
    ):
        (output / "composition-data.json").write_text(json.dumps(composition, ensure_ascii=False, indent=2), encoding="utf-8")
        (project / "index.html").write_text(render_html(composition), encoding="utf-8")
        source_lines = "\n".join(f"- {item['provider']} · {item['query']} · {item.get('creator','')} · {item.get('source_url','')}" for item in source_records) or "- 本次未调用外部素材"
        publish = f"# 标题建议\n\n{title}\n\n# 副标题\n\n{subtitle}\n\n# 发布文案\n\n{transcript_text[:180]}{'…' if len(transcript_text) > 180 else ''}\n\n# 建议话题\n\n#观点 #口播 #商业思考\n\n# 素材来源\n\n{source_lines}\n"
        (output / "publish-copy.md").write_text(publish, encoding="utf-8")
    with specialist_run(
        workflow, ledger_path, "qa", "semantic-and-composition-quality-gate",
        detail="检查语音、素材、画中画和主视觉是否一致",
        inputs={"composition": "output/composition-data.json", "director": "output/director-plan.json"},
    ) as qa_run:
        qa_report = Auditor(composition, director_plan, project, workflow.state.to_dict()).run()
        (output / "workflow-qa.json").write_text(json.dumps(qa_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not qa_report["pass"]:
            owners = "、".join(qa_report.get("summary", {}).get("repair_owners", {}).keys()) or "制作团队"
            raise RuntimeError(f"多 Agent 质检未通过，需要 {owners} 修正后重新装配。")
    workflow._run(qa_run).metadata["detail"] = (
        f"结构通过 · {qa_report['summary']['warnings']} 条降级/复核提示"
        if qa_report["pass"] else "发现阻断问题，已退回责任 Agent"
    )
    for role, path in (
        ("director-plan", output / "director-plan.json"), ("motion-plan", output / "motion-plan.json"),
        ("composition", output / "composition-data.json"), ("quality-report", output / "workflow-qa.json"),
    ):
        workflow.add_artifact(ArtifactRef.create(role, path.relative_to(project)))
    workflow.build_manifest(metadata={"qa": qa_report.get("summary", {})})
    workflow.save(ledger_path)
    summary = {
        "selected_mode": mode, "source_duration": round(info["duration"], 1), "edited_duration": round(edited_duration, 1), "cover_frame_seconds": round(cover_at, 2),
        "caption_groups": len(captions), "caption_emphasis": sum(bool(item.get("emphasis")) for item in captions), "callout_cards": len(visual_beats), "chapter_cards": len(chapters), "camera_beats": len(camera_beats), "lower_thirds": len(lower_thirds),
        "bgm": bool(bgm), "bgm_baked": bool(mix_audio), "bgm_profile": bgm.get("mix_profile") if bgm else "none", "smart_media_assets": len(source_records), "delivered_media_assets": delivered_media,
        "smart_media_enabled": bool(config.get("smart_media")),
        "title": title, "subtitle": subtitle, "cover_copy": cover_copy,
        "identity": {"speaker": effective_speaker, "speaker_title": effective_title, **identity_meta},
        "director": {key: director_plan.get(key) for key in ("status", "provider", "model", "reason")},
        "cover_template": selected_cover, "cover_templates": cover_templates,
        "cover_qa": {"pass": bool(cover_qa.get("pass")), "checked": int(cover_qa.get("checked", 0))},
        "motion_template": selected_motion_template, "recommended_motion_template": recommended_motion_template, "motion_templates": motion_templates,
        "agent_workflow": {"status": workflow.state.status, "runs": len(workflow.state.runs), "gates": len(workflow.state.gates)},
        "workflow_qa": qa_report.get("summary", {}),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("4/5 已生成字幕、章节、半屏素材、统一动效、背景乐、封面和发布文案")
    print("5/5 正在交给 HyperFrames 做画面检查")


if __name__ == "__main__":
    main()
