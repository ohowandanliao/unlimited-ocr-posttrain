# OCR/VLM 后训练与文档解析论文检索

检索日期：2026-08-26。窗口：2024-2026。每个 API source 最多 10 条。

本页保留 `paper-search` 的全部返回结果和错误，供训练方案追溯。查询 A 用于高召回探索，查询 B
针对当前工程收窄。检索结果不等于论文质量判断；Crossref 仍返回了年份为空和 2027 的越界记录。

## 查询 A

`document OCR vision language model post-training long document Markdown structured generation repetition EOS synthetic data quality`

### Semantic Scholar（0）

请求返回 `403 Forbidden`，没有结果。

### OpenAlex（0）

请求返回 `401 Unauthorized`，没有结果。

### arXiv（10）

| # | Title | Date | Venue | Citations |
|---:|---|---:|---|---:|
| [A1](http://arxiv.org/abs/2406.15126v1) | On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey | 2024 | arXiv | 0 |
| [A2](http://arxiv.org/abs/2509.18174v1) | Baseer: A Vision-Language Model for Arabic Document-to-Markdown OCR | 2025 | arXiv | 0 |
| [A3](http://arxiv.org/abs/2403.07553v1) | The future of document indexing: GPT and Donut revolutionize table of content processing | 2024 | arXiv | 0 |
| [A4](http://arxiv.org/abs/2603.09625v2) | Grounding Synthetic Data Generation With Vision and Language Models | 2026 | arXiv | 0 |
| [A5](http://arxiv.org/abs/2601.14722v1) | Typhoon OCR: Open Vision-Language Model For Thai Document Extraction | 2026 | arXiv | 0 |
| [A6](http://arxiv.org/abs/2607.06466v1) | Verification of Dynamic Holographic Behavior in Identity Documents | 2026 | arXiv | 0 |
| [A7](http://arxiv.org/abs/2509.22283v1) | Rule-Based Reinforcement Learning for Document Image Classification with Vision Language Models | 2025 | arXiv | 0 |
| [A8](http://arxiv.org/abs/2508.14557v1) | Improving OCR using internal document redundancy | 2025 | arXiv | 0 |
| [A9](http://arxiv.org/abs/2508.18984v2) | Enhancing Document VQA Models via Retrieval-Augmented Generation | 2025 | arXiv | 0 |
| [A10](http://arxiv.org/abs/2508.05669v1) | Fine-Tuning Vision-Language Models for Markdown Conversion of Financial Tables in Malaysian Audited Financial Reports | 2025 | arXiv | 0 |

### OpenReview（0）

本机未安装 `openreview-py`；该 source 未执行成功。

### Crossref（10）

| # | Title | Date | Venue | Citations |
|---:|---|---:|---|---:|
| [A11](https://doi.org/10.1007/s10032-026-00613-6) | Benchmarking OCR and vision-language models for Turkish text recognition: a comprehensive evaluation using synthetic data | 2026 | IJDAR | 0 |
| [A12](https://doi.org/10.36227/techrxiv.175616889.90325672/v1) | Japanese-Mobile-Receipt-OCR-1.3K: A Comprehensive Dataset Analysis and Fine-tuned Vision-Language Model for Structured Receipt Data Extraction | unknown | TechRxiv | 1 |
| [A13](https://doi.org/10.1007/978-3-032-36039-7_30) | From Pixels to Structure: Lightweight Vision-Language Models for Document OCR and Structured JSON Extraction | 2027 | ICDAR 2026 proceedings | 0 |
| [A14](https://doi.org/10.21203/rs.3.rs-7357197/v1) | Japanese-Mobile-Receipt-OCR-1.3K: A Comprehensive Dataset Analysis and Fine-tuned Vision-Language Model for Structured Receipt Data Extraction | unknown | Research Square | 0 |
| [A15](https://doi.org/10.1007/s10032-025-00522-0) | Scrambled text: fine-tuning language models for OCR error correction using synthetic data | 2025 | IJDAR | 3 |
| [A16](https://doi.org/10.2139/ssrn.6919778) | Diacritic-Aware Post-OCR Correction for Vietnamese Scanned Documents: A Lightweight Baseline for Low-Quality Document Digitization | unknown | SSRN | 0 |
| [A17](https://doi.org/10.21203/rs.3.rs-10241038/v1) | Diacritic-Aware Post-OCR Correction for Vietnamese Scanned Documents: A Lightweight Baseline for Low-Quality Document Digitization | unknown | Research Square | 0 |
| [A18](https://doi.org/10.47191/etj/v9i07.09) | Advanced Retrieval Augmented Generation: Multilingual Semantic Retrieval across Document Types by Finetuning Transformer Based Language Models and OCR Integration | 2024 | Engineering and Technology Journal | 2 |
| [A19](https://doi.org/10.1109/icsece61636.2024.10729522) | Semi-Structured Tender Document Retrieval-Augmented Generation: A Framework Based on Large Language Model | 2024 | ICSECE | 3 |
| [A20](https://doi.org/10.2139/ssrn.6845342) | An OCR-free, Position-aware VLM Process for Semi-structured Business Document Structuring: Design, Evaluation, and Enterprise Deployment Evidence | unknown | SSRN | 0 |

### DBLP（0）

没有匹配结果。

## 查询 B

`OCRFlux OmniDocBench olmOCR document parsing markdown vision language model`

### Semantic Scholar（0）

请求返回 `403 Forbidden`，没有结果。

### OpenAlex（0）

请求返回 `401 Unauthorized`，没有结果。

### arXiv（10）

| # | Title | Date | Venue | Citations |
|---:|---|---:|---|---:|
| [B1](http://arxiv.org/abs/2508.05669v1) | Fine-Tuning Vision-Language Models for Markdown Conversion of Financial Tables in Malaysian Audited Financial Reports | 2025 | arXiv | 0 |
| [B2](http://arxiv.org/abs/2412.07626v2) | OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations | 2024 | arXiv | 0 |
| [B3](http://arxiv.org/abs/2603.04205v2) | Real5-OmniDocBench: A Full-Scale Physical Reconstruction Benchmark for Robust Document Parsing in the Wild | 2026 | arXiv | 0 |
| [B4](http://arxiv.org/abs/2510.19817v1) | olmOCR 2: Unit Test Rewards for Document OCR | 2025 | arXiv | 0 |
| [B5](http://arxiv.org/abs/2502.18443v3) | olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models | 2025 | arXiv | 0 |
| [B6](http://arxiv.org/abs/2512.02498v4) | dots.ocr: Multilingual Document Layout Parsing in a Single Vision-Language Model | 2025 | arXiv | 0 |
| [B7](http://arxiv.org/abs/2509.22186v2) | MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing | 2025 | arXiv | 0 |
| [B8](http://arxiv.org/abs/2604.00086v2) | Hierarchical Pre-Training of Vision Encoders with Large Language Model | 2026 | arXiv | 0 |
| [B9](http://arxiv.org/abs/2504.09480v1) | Vision-Language Model for Object Detection and Segmentation: A Review and Evaluation | 2025 | arXiv | 0 |
| [B10](http://arxiv.org/abs/2603.15206v1) | Efficient Document Parsing via Parallel Token Prediction | 2026 | arXiv | 0 |

### OpenReview（0）

本机未安装 `openreview-py`；该 source 未执行成功。

### Crossref（10）

| # | Title | Date | Venue | Citations |
|---:|---|---:|---|---:|
| [B11](https://doi.org/10.1109/cvpr52734.2025.02313) | OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations | 2025 | CVPR | 19 |
| [B12](https://doi.org/10.2139/ssrn.4915247) | Cross-Scene Visual Context Parsing with Large Vision-Language Model | unknown | SSRN | 0 |
| [B13](https://doi.org/10.1007/978-981-97-9739-4_6) | Multiformat Document Parsing and Management | 2025 | Natural Language Processing and Applications | 1 |
| [B14](https://doi.org/10.32614/cran.package.md4r) | md4r: Markdown Parser Implemented using the MD4C Library | 2024 | CRAN | 0 |
| [B15](https://doi.org/10.1109/dlcv69906.2026.11635522) | Macro-Micro Anchor-Guided Text-Aware Document Parsing | 2026 | DLCV | 0 |
| [B16](https://doi.org/10.18653/v1/2025.findings-emnlp.1088) | Intelligent Document Parsing: Towards End-to-end Document Parsing via Decoupled Content Parsing and Layout Grounding | 2025 | Findings of EMNLP | 0 |
| [B17](https://doi.org/10.1016/j.csl.2025.101809) | Transformer-based document-level discourse processing: Exploiting prior language knowledge and hierarchical parsing | 2025 | Computer Speech & Language | 0 |
| [B18](https://doi.org/10.2139/ssrn.5649150) | Graph Guided Vision Language Modeling for Complex Document Understanding | unknown | SSRN | 0 |
| [B19](https://doi.org/10.1007/s41095-024-0430-4) | CLIP-SP: Vision-language model with adaptive prompting for scene parsing | 2024 | Computational Visual Media | 8 |
| [B20](https://doi.org/10.18653/v1/2025.findings-acl.429) | STEM-POM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing | 2025 | Findings of ACL | 0 |

### DBLP（0）

没有匹配结果。

## Model Knowledge / 补充核验（5）

下列论文没有重复出现在上述 API 结果中；标题和年份又通过 arXiv API 定向核验。citation 未从统一
API 获取，因此不填写，避免伪造可比数字。

| # | Title | Date | Venue | Citations | 与本项目关系 |
|---:|---|---:|---|---:|---|
| M1 | General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model | 2024 | arXiv / AAAI 2025 | n/a | GOT-OCR2.0 的统一端到端 OCR 与格式化输出基线 |
| M2 | MonkeyOCR: Document Parsing with a Structure-Recognition-Relation Triplet Paradigm | 2025 | arXiv | n/a | 4.5M bilingual 数据和 structure/recognition/relation 拆分 |
| M3 | DeepSeek-OCR: Contexts Optical Compression | 2025 | arXiv | n/a | Unlimited-OCR 的编码器/decoder 路线来源和视觉 token 压缩边界 |
| M4 | Visual Merit or Linguistic Crutch? A Close Look at DeepSeek-OCR | 2026 | arXiv | n/a | 语义破坏后性能坍塌，直接提示语言先验和低视觉 token 的幻觉风险 |
| M5 | PaddleOCR-VL-1.6: Expanding the Frontier of Document Parsing with Under-Optimized Region Refinement and Progressive Post-Training | 2026 | arXiv | n/a | 按弱区定向清洗/渐进后训练，而不是无差别扩充语料 |

## Overview

两个查询共返回 40 条 API 记录；按标题/版本粗去重后约 37 条。高精度查询集中在文档解析 benchmark、
高分辨率视觉、结构关系和可验证 reward；高召回查询混入分类、RAG 和身份文档等低相关项。三类 API
失败已原样保留，因此这不是“六源完整覆盖”。

## Trends

- 2025 是结果最密集的年份，路线从单纯端到端转向 high-resolution/coarse-to-fine、layout grounding、
  relation prediction 和可验证 reward。
- OmniDocBench/CVPR 已成为页面级解析的共同评测锚点；Real5 开始强调扫描、倾斜、拍屏和光照等真实退化。
- 新工作普遍把表格、公式、layout/reading order 分开计分；只用一个总 edit similarity 已不够。
- 2026 的 PaddleOCR-VL-1.6 特别强调“previous-model weak regions + curated data + progressive post-training”，
  与当前应先修 badcase/data gate、而不是继续全量自然混合的结论一致。

## Key themes

1. **细分评测与 failure buckets**：文本、表格、公式、reading order、真实物理退化分别测（B2/B3/B11）。
2. **可验证训练信号**：unit-test rewards、render-and-compare 或规则 reward 取代只看 token loss（B4/M2）。
3. **高分辨率和 coarse-to-fine**：视觉分辨率是复杂表格/小字瓶颈，不能只调 decoder loss（B7/M5）。
4. **多语言和能力保持**：bilingual/multilingual 大规模混合是通用 OCR 的基础（A2/A5/B6/M2）。
5. **数据质量而非盲目规模**：合成数据需要 curation、grounding 和 previous-model weak-region 选择（A1/A4/M5）。
6. **语言先验风险**：光学压缩过强时模型更依赖语言先验，域偏移会放大重复和幻觉（M3/M4）。

## Keywords frequency

以下计数按 40 条 API 标题做不区分大小写的概念归并，不把同义词在摘要中重复扩张：

| Keyword | Count |
|---|---:|
| document | 26 |
| OCR / olmOCR | 14 |
| vision-language / VLM | 12 |
| parsing | 11 |
| synthetic / Markdown | 4 / 4 |

## Most cited by accepted paper

只使用 Crossref 返回的 citation，去掉同论文 arXiv 重复项：

| Rank | Title | Year | Citations |
|---:|---|---:|---:|
| 1 | OmniDocBench | 2025 | 19 |
| 2 | CLIP-SP | 2024 | 8 |
| 3 | Scrambled text | 2025 | 3 |
| 4 | Semi-Structured Tender Document RAG | 2024 | 3 |
| 5 | Advanced Retrieval Augmented Generation | 2024 | 2 |

## Most cited by first author

| Rank | Author | Papers in set | Total citations |
|---:|---|---:|---:|
| 1 | Linke Ouyang | 1 | 19 |
| 2 | Jiaao Li | 1 | 8 |
| 3 | Jonathan Bourne | 1 | 3 |
| 4 | Yilong Zhao | 1 | 3 |
| 5 | Ismail OUBAH | 1 | 2 |

## Recommendations for reading

1. **OmniDocBench**：先统一文本/表格/公式/reading order 的评测口径，避免继续用字符数替代质量。
2. **olmOCR 2**：学习 unit-test reward 和可验证 badcase 设计，直接对应本项目的停止/重复/协议 gate。
3. **MonkeyOCR**：理解 bilingual 数据和 relation 模块为何比“去掉 `<PAGE>`”更接近真正跨页合并。
4. **Visual Merit or Linguistic Crutch?**：用于设计语言破坏、域保持和视觉 token 压缩压力测试。
5. **PaddleOCR-VL-1.6**：参考弱区识别、定向数据修复和 progressive post-training 的实验顺序。
