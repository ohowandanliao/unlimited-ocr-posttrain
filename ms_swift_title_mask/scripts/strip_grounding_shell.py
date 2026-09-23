#!/usr/bin/env python3
"""Strip Unlimited-OCR grounding protocol tokens into pure-text Markdown.

评分输入(omnidocbench / agentbuilder)必须是纯文本:任何 ``<|...|>`` 协议 special token
残留都会污染 norm_pred(坑9:TextEdit 被污染到 0.5679、Total 假值 37.35 的教训)。
本脚本是剥壳的权威实现,固化在 posttrain 仓库,替代历史上 /tmp 里的一次性命令。

设计原则(2026-09-17 拍板):**尽量保守**。协议 token 是脚手架,可以删;畸形输出
(孤儿开标签、闭合块内嵌正文等)可能是模型输出问题,预处理不得擅自解释或删除,
必须保留并暴露给人工,避免悄悄"洗掉"掩盖模型退化。

处理规则:

1. 默认(保守)只删**无歧义的闭合协议块** ``<|det|>LABEL [coords]<|/det|>``
   ——块内不含 ``<`` 才视为纯标签+坐标;正文在闭合标记之后,天然保留。
2. 其余一律不删:孤儿开标签(如 ``<|det|>text [1, 2]<|det|>正文``)、坐标截断
   (``<|det|>header [786, 52,``)、无坐标标签(``<|det|>关键词 …``)、闭合块内嵌
   正文(``<|det|>table [c] …<table>…<|/det|>``)→ 原样保留,**逐文件告警并拒绝
   写盘**(评分输入不许带病,交人工复核)。
3. 人工复核后可用 ``--aggressive`` 显式清理畸形脚手架:仅删除 ``<|det|>`` +
   **已知标签白名单** + (可截断的)坐标括号、以及裸协议 token;**正文永不删**。
   该模式改变模型畸形输出的呈现形态,须在审核后使用并在记录中注明。
4. ``<PAGE>`` 页标记:GT 中没有(0/130),但 0827 契约下两轮 pred_clean 对称保留
   (18-19/129 文件)。默认**不动**以保持与 0827 基线可比;``--strip-page-markers``
   可显式删除,但删除后不能与历史基线直接对列。

模式:
  剥壳(默认)  ``--in-dir RAW --out-dir CLEAN`` 逐文件剥壳 + 自检;
              任一文件存在残留 → **不写任何文件**、非零退出(全部干净才落盘)。
  审核后清理  ``--aggressive``:在默认基础上清理畸形脚手架;若仍有残留同样拒绝落盘。
  扫描        ``--check --in-dir DIR`` 只扫描不写,报告每文件残留。
  幂等        strip(strip(x)) == strip(x),可安全重复执行。

用法示例::

    python3 strip_grounding_shell.py --in-dir evaluation/xxx_all129/pred/xxx --out-dir evaluation/xxx_all129/pred_clean/xxx
    python3 strip_grounding_shell.py --check --in-dir evaluation/xxx_all129/pred_clean/xxx
    python3 strip_grounding_shell.py --aggressive --in-dir ... --out-dir ...   # 人工审核后
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 1) 无歧义闭合协议块:块内不含 '<'(协议标签+坐标不会含 '<');含 '<' 的一律视为
#    畸形(可能内嵌正文/表格),交告警,不自动删。
RE_CLOSED_BLOCK = re.compile(r"<\|det\|>[^<]*<\|/det\|>")
# 已知协议标签白名单(--aggressive 用);观察到过的:title/text/header/footer/
# page_number/table/figure/equation 等。不在白名单的一律不动。
KNOWN_LABELS = (
    "title|text|header|footer|page_number|table|figure|equation|formula|image|"
    "graphic|caption|list|seal|watermark|paragraph|section|sidebar|footnote"
)
# 2) 畸形脚手架(--aggressive):det 开标签 + 白名单标签 + 可截断的坐标括号。
#    同行其余内容(正文)不匹配、不删;跨行不匹配。
RE_ORPHAN_SCAFFOLD = re.compile(
    rf"<\|det\|>[ \t]*(?:{KNOWN_LABELS})[ \t]*\[[^\]\n]*\]?[ \t]*"
)
# 3) 裸协议 token(--aggressive 兜底):任何残余 <|...|>。
#    2026-09-22 修复:原字符类 [A-Za-z0-9_] 认不出名字里带连字符/空格的标签
#    (实测 pwc-fullce-16k 复读段吐出 <|content-type> / <|content type> 共 45 处,
#    且常带畸形尾巴如 <|content-type>">)。放宽为 [^<>\n] 覆盖任意 token 名,
#    并以 \|?> 容忍缺失的收尾竖线。**仅用于 --aggressive 兜底清理**,保守模式不受影响。
#    另覆盖退化到只剩开头的裸 "<|"(复读段实测,后随换行/空白,无标签名)。
RE_ANY_PROTOCOL_TOKEN = re.compile(r"<\|/?[^<>\n]{0,40}\|?>|<\|(?=[\s\n]|$)")
# 扫描用:评分输入里不允许出现的任何协议 token 形态。
#    2026-09-22 修复:原规则要求必须以 |> 收尾,漏掉 <|content-type>"> 这类畸形,
#    导致自检谎报"clean"。放宽为 <| 开头且名字内无 <>(可含连字符/空格)。
RE_RESIDUE_SCAN = re.compile(r"<\|[^<>\n]{0,40}|<\|(?=[\s\n]|$)")
# <PAGE> 页标记:0827 契约保留,默认不删
RE_PAGE_MARKER = re.compile(r"<PAGE>")
# --rebuild-titles:任意 det 标签块(开标签+白名单外也认的 ASCII 标签+可截断 bbox+可缺失闭合)。
# 标题重建件只喂 title 指标,不喂 OmniDocBench。
RE_LABEL_BLOCK = re.compile(
    r"<\|det\|>[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*(\[[^\]\n]*\]?)?[ \t]*(<\|/det\|>)?"
)


def rebuild_titles(text: str) -> str:
    """标题重建:剥掉全部 det 壳+bbox;title 标签的后续内容合并为单个 ``# `` 标题行。

    用途:grounding 协议下 title 指标(只认 ATX 行)结构性塌零(见
    evaluation_pitfalls_2026-09-17.md §二);本函数把 ``<|det|>title [bbox]<|/det|>文本``
    重建为 ``# 文本``,使 title 指标恢复设计语义(即"把标题的壳和 bbox 去掉来评测")。
    grounding 标签无层级信息 → 统一一级(``#``);指标内 _normalize_levels 两侧各自平移,
    级别分按 GT 顶层标题计,内容分(0.7)为主。
    产物仅用于 title 评分(建议目录名 pred_titles/),**不得**替代 pred_clean 喂
    OmniDocBench / AgentBuilder 其他分项。幂等。
    """
    out: list[str] = []
    pos = 0
    pending_title = False

    def emit(segment: str, as_title: bool) -> None:
        if not segment:
            return
        if as_title:
            lines = [line.strip() for line in segment.splitlines() if line.strip()]
            if lines:
                out.append("# " + " ".join(lines) + "\n")
            return
        out.append(segment)

    for match in RE_LABEL_BLOCK.finditer(text):
        emit(text[pos : match.start()], pending_title)
        pending_title = match.group(1) == "title"
        pos = match.end()
    emit(text[pos:], pending_title)
    return "".join(out)


def strip_text(
    text: str, *, aggressive: bool = False, strip_page_markers: bool = False
) -> tuple[str, dict[str, int]]:
    """剥壳单个文本,返回(结果, 删除计数)。幂等。保守模式只删无歧义闭合块。"""
    counts = {
        "closed_block": 0,
        "orphan_scaffold": 0,
        "loose_token": 0,
        "page_marker": 0,
    }
    text, counts["closed_block"] = RE_CLOSED_BLOCK.subn("", text)
    if aggressive:
        text, counts["orphan_scaffold"] = RE_ORPHAN_SCAFFOLD.subn("", text)
        text, counts["loose_token"] = RE_ANY_PROTOCOL_TOKEN.subn("", text)
    if strip_page_markers:
        text, counts["page_marker"] = RE_PAGE_MARKER.subn("", text)
    return text, counts


def residue_hits(text: str) -> list[str]:
    """扫描文本中的协议 token 残留(评分输入里必须为空)。"""
    return RE_RESIDUE_SCAN.findall(text)


def _iter_markdown_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.md"))
    if not files:
        raise SystemExit(f"no .md files under {directory}")
    return files


def run_strip(
    in_dir: Path,
    out_dir: Path,
    *,
    aggressive: bool,
    strip_page_markers: bool,
    rebuild: bool = False,
) -> int:
    files = _iter_markdown_files(in_dir)
    cleaned_files: list[tuple[Path, str]] = []
    totals = {k: 0 for k in ("closed_block", "orphan_scaffold", "loose_token", "page_marker")}
    flagged: list[tuple[str, list[str]]] = []
    for path in files:
        raw = path.read_text(encoding="utf-8")
        if rebuild:
            counts = {k: 0 for k in totals}
            cleaned = rebuild_titles(raw)
            # 产物只喂 title 指标:裸协议 token 零内容,清掉以保证自检通过
            # (正文永不删的保守原则不变;畸形标签+坐标仍由 RE_LABEL_BLOCK 统一处理)
            cleaned, counts["loose_token"] = RE_ANY_PROTOCOL_TOKEN.subn("", cleaned)
        else:
            cleaned, counts = strip_text(
                raw,
                aggressive=aggressive,
                strip_page_markers=strip_page_markers,
            )
        hits = residue_hits(cleaned)
        if hits:
            flagged.append((path.name, hits))
        for key in totals:
            totals[key] += counts[key]
        cleaned_files.append((path, cleaned))
    mode = "rebuild-titles" if rebuild else ("aggressive" if aggressive else "conservative")
    print(
        f"strip({mode}) {len(files)} files: {in_dir} -> {out_dir}; removed {totals}"
    )
    if flagged:
        print(
            f"ABORT: {len(flagged)} file(s) still contain protocol tokens after "
            f"{'aggressive' if aggressive else 'conservative'} strip — nothing written. "
            "These are malformed model outputs; review them, then re-run with "
            "--aggressive if cleaning the scaffolding is accepted:"
        )
        for name, hits in flagged:
            print(f"  {name}: {len(hits)} hit(s), e.g. {hits[:3]}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, cleaned in cleaned_files:
        (out_dir / path.name).write_text(cleaned, encoding="utf-8")
    print(f"self-check passed: 0 protocol-token residue; wrote {len(files)} files")
    return 0


def run_check(in_dir: Path) -> int:
    files = _iter_markdown_files(in_dir)
    flagged: list[tuple[str, list[str]]] = []
    for path in files:
        hits = residue_hits(path.read_text(encoding="utf-8"))
        if hits:
            flagged.append((path.name, hits))
    print(f"checked {len(files)} files in {in_dir}")
    if flagged:
        print(f"RESIDUE FOUND in {len(flagged)} files:")
        for name, hits in flagged:
            print(f"  {name}: {len(hits)} hit(s), e.g. {hits[:3]}")
        return 1
    print("clean: 0 protocol-token residue")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strip Unlimited-OCR grounding protocol tokens (conservative by default)."
    )
    parser.add_argument("--in-dir", type=Path, required=True, help="输入目录(逐 *.md)")
    parser.add_argument("--out-dir", type=Path, help="输出目录(剥壳模式必填)")
    parser.add_argument(
        "--check", action="store_true", help="只扫描不写:报告残留并以非零退出"
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="人工审核后使用:额外清理畸形脚手架(白名单标签+坐标),正文永不删",
    )
    parser.add_argument(
        "--rebuild-titles",
        action="store_true",
        help="标题重建件模式:title 标签壳+bbox 去掉、标题文本重建为 `#` 行,"
        "供 title 指标使用(输出仅喂 title 指标,不喂 OmniDocBench/其他分项)",
    )
    parser.add_argument(
        "--strip-page-markers",
        action="store_true",
        help="同时删除 <PAGE> 页标记(默认保留以对齐 0827 基线契约)",
    )
    args = parser.parse_args(argv)
    if args.check:
        return run_check(args.in_dir)
    if args.out_dir is None:
        parser.error("剥壳模式需要 --out-dir(或使用 --check)")
    if args.in_dir == args.out_dir:
        parser.error("--in-dir 与 --out-dir 必须不同(评分输入不可原地覆盖)")
    return run_strip(
        args.in_dir,
        args.out_dir,
        aggressive=args.aggressive,
        strip_page_markers=args.strip_page_markers,
        rebuild=args.rebuild_titles,
    )


if __name__ == "__main__":
    sys.exit(main())
