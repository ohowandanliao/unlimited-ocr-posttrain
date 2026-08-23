# Markdown 标题修复使用说明

这条 pipeline 只有一个输入输出关系：

```text
已有 Markdown -> OpenAI-compatible LLM API -> 标题行决策 -> 更好的 Markdown
```

它不读取 PDF、OCR、`middle.json`、Excel 或固定数据集，也不依赖 Luna/Sol。模型只返回
需要修改的行号和目标标题级别，脚本负责修改行首 ATX `#`；正文不会由模型重写。

## 文件

```text
markdown-title-repair/
├── SKILL.md
├── agents/openai.yaml
├── scripts/repair_titles.py
└── references/
    ├── config.example.json
    ├── review_rules.md
    └── usage.md
```

Python 3.10 及以上即可运行，只使用标准库。

## 配置 API

默认调用 OpenAI-compatible `chat/completions` 接口。推荐使用环境变量：

```bash
export TITLE_REPAIR_API_URL="https://your-api.example/v1/chat/completions"
export TITLE_REPAIR_API_KEY="your-key"
export TITLE_REPAIR_MODEL="your-model"
```

`references/config.example.json` 也支持直接填写 `api.url` 和 `api.model`。无鉴权服务可将
`api_key_env` 设为空字符串。特殊网关可配置 `auth_header`、`auth_scheme`、`headers` 和
`extra_body`。`api.url`、`api.model` 和额外 header 的值可写成 `${ENV_NAME}`；`*_env`
字段填写环境变量名，也兼容 `${ENV_NAME}` 写法。`extra_body` 不允许覆盖模型、消息或 JSON
输出等核心字段。

## 单文件

从 skill 目录运行，输出文件必须是尚不存在的新路径：

```bash
python3 scripts/repair_titles.py \
  --input /path/to/input.md \
  --output /path/to/output.md \
  --config references/config.example.json
```

## 目录批处理

输出会保留输入目录的相对结构：

```bash
python3 scripts/repair_titles.py \
  --input /path/to/markdown_input \
  --output /path/to/markdown_repaired \
  --config references/config.example.json
```

工具不会覆盖已有 Markdown。重跑时使用一个新的输出路径，避免把旧结果误认为本次结果。

## 模型返回契约

模型返回标准 Chat Completions 响应，其中 `choices[0].message.content` 是以下 JSON：

```json
{
  "edits": [
    {
      "line": 12,
      "source_line": "# 2.1 安装要求",
      "level": 3,
      "reason": "位于第二章的二级小节"
    }
  ],
  "unresolved": []
}
```

`level=0` 表示将误标题降为正文，`level=1..6` 表示目标标题层级。脚本会验证行号、逐字符
匹配 `source_line`、拒绝代码围栏内编辑，然后只修改标题标记。`unresolved` 非空、响应被
截断或任何编辑不合法时，该文档失败且不生成输出。

## 产物

单文件默认在输出旁生成 `<输出名>_title_repair_audit/`；目录模式在输出目录中生成
`_title_repair_audit/`：

- `summary.json`：文档成功和失败数；
- `results.jsonl`：逐文档状态、哈希、标题数量和编辑记录；
- `raw_responses/`：API 原始响应，便于复核。

API key 不会写入审计文件。

`raw_responses/` 可能包含 API 回显的原始 Markdown，`results.jsonl` 还会记录本机输入/输出绝对路径。
处理敏感数据时应按数据策略决定是否保留这些审计文件，且不要将本地配置、`.env` 或 raw responses 提交到 Git。

## 能力边界

- 每份 Markdown 调用一次 API，不做内部切块；超过 `max_input_chars` 会失败。
- 只能修复 Markdown 中已有文字的标题状态和层级，不能找回已丢失的标题文字。
- 不处理跨行标题合并、拆分或标题文案改写。
- 输出可用于标题 GT 或训练数据候选，但不应未经抽检直接视为 gold GT。
- 在本仓库中，该输出首先是 `LLM_CANDIDATE`；只有继续通过 source/target SHA、PDF 证据、split 和
  provenance 闸门后，才能单独冻结为 `SILVER_ACCEPTED`。
