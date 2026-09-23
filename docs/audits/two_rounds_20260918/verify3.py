# -*- coding: utf-8 -*-
# 独立复核脚本 3:补充验证。只读。
import json, zlib, re, statistics
from pathlib import Path

U    = Path("/home/jovyan/hyx/uocr-ms-swift-title-mask")
EVb  = U/"evaluation/readoc-view-16k_all129"
EVp  = U/"evaluation/pmc-fullce-16k_all129"
EVm  = U/"evaluation/pmc-readoc-spage-16k_all129"
GT   = Path("/home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/groundtruth")
OMNI = Path("/home/jovyan/hyx/pdf2text-auto-label-eval/OmniDocBench_v1.5/result")
PC  = {"base":EVb/"pred_clean/base","fc":EVb/"pred_clean/full-ce",
       "pmc":EVp/"pred_clean/pmc-fullce-16k","mix":EVm/"pred_clean/pmc-readoc-spage-16k"}
def jload(p):
    with open(p) as f: return json.load(f)

# ---------- 1. 未发射的表是否被计 0?(计分表池机制) ----------
print("== 1. 计分表池机制 ==")
for doc in ["4-1统计图2","3-1跨页重复表头","4-2段落中带「图x-x」","1-3单元格内换行1"]:
    row=[doc]
    for r,pfx in [("pmc","swp2_"),("mix","swp3_")]:
        f=OMNI/f"{pfx}{doc}_table_per_table_TEDS.json"
        if f.is_file():
            j=jload(f); teds=[x.get("TEDS") for x in j.values() if isinstance(x,dict)]
            row.append(f"{r}: {len(teds)}表 mean{sum(teds)/len(teds):.3f}" if teds else f"{r}: 0表")
        else:
            row.append(f"{r}: 无文件")
    print("  "+" | ".join(row))
# GT 表格总数对照:从 GT 统计 <table 标签
gtt={}
for f in GT.glob("*.md"):
    gtt[f.stem]=f.read_text(encoding="utf-8",errors="replace").count("<table")
for doc in ["4-1统计图2","3-1跨页重复表头","4-2段落中带「图x-x」"]:
    print(f"  {doc}: GT表格 {gtt.get(doc)}")

# ---------- 2. per-doc overall 缺 TEDS 时是否重归一 ----------
print("== 2. per-doc 分量结构(缺 TEDS 的篇) ==")
PDm=jload(EVm/"agentbuilder/pmc-readoc-spage-16k/per_doc_scores.json")
PDp=jload(EVp/"agentbuilder/pmc-fullce-16k/per_doc_scores.json")
for d in ["4-1统计图2","3-1跨页重复表头"]:
    for r,pd in [("pmc",PDp),("mix",PDm)]:
        e=pd.get(d,{})
        print(f"  {d} {r}: text_acc {e.get('text_accuracy')} teds {e.get('table_teds')} order_acc {e.get('reading_order_accuracy')} title {e.get('title_accuracy')} overall {e.get('overall')}")
    # 手工验算:TEDS=0 与 重归一 两种口径
    e=PDm.get(d,{})
    t=e.get("text_accuracy"); o=e.get("reading_order_accuracy"); ti=e.get("title_accuracy") or 0
    if t is not None and o is not None:
        print(f"    mix 若 TEDS 记 0: 0.3t+0.3*0+0.3o+0.1title = {0.3*t+0.3*0+0.3*o+0.1*ti:.4f} ; 若重归一/0.7 = {(0.3*t+0.3*o+0.1*ti)/0.7:.4f} ; 实测 overall {e.get('overall')}")

# ---------- 3. ATX 中位数(去循环污染) ----------
print("== 3. ATX 行/篇 中位数 ==")
def atx_counts(pdir):
    return sorted(sum(1 for l in f.read_text(encoding="utf-8",errors="replace").splitlines() if l.startswith("#")) for f in pdir.glob("*.md"))
for r,p in PC.items():
    v=atx_counts(p); n=len(v)
    print(f"  {r:4s} 中位 {v[n//2]} p90 {v[int(n*0.9)]} 总 {sum(v)}")
gtv=[sum(1 for l in (GT/(s+'.md')).read_text(encoding="utf-8",errors="replace").splitlines() if l.startswith('#')) for s in gtt]
print(f"  GT   中位 {sorted(gtv)[len(gtv)//2]} 总 {sum(gtv)}")

# ---------- 4. GT 侧四板斧(输入自重复检验) ----------
print("== 4. GT 侧 dup(3-1跨页重复表头 / 4-1统计图2) ==")
for d in ["3-1跨页重复表头","4-1统计图2","2-1横向合并"]:
    g=(GT/(d+".md")).read_text(encoding="utf-8",errors="replace")
    lines=[l.strip() for l in g.splitlines() if l.strip()]
    dup=1-len(set(lines))/len(lines) if lines else 0
    print(f"  {d}: GT {len(g)}c 行dup {dup:.2f}")

# ---------- 5. title87 逐篇 Δ 分解(非退化部分) ----------
print("== 5. title87 非退化残差 −1.41 的构成 ==")
import sys
sys.path.insert(0,str(U/"evaluation/tooling/agentbuilder_pkg"))
from agentbuilder_eval.title_metric import score_markdown_pair, extract_title_lines
set87=[p.stem for p in sorted(GT.glob("*.md")) if extract_title_lines(p.read_text(encoding="utf-8")) and p.stem!="Handwriting document"]
S_mix12={"3-1跨页重复表头","3-2跨页分页切断","3-4长表跨三页以上","4-1统计图2","4-1统计图3","4-3扫描2","5-1双栏表格2","中车株洲_方案技术_16","广东城规院_惠玩甘青_11","广东特检院_安全报告2_6","珠海机场_工作证管理细则_6","科欣环保_产业结构_5"}
T={}
for r,pdir in [("pmc",EVp/"pred_titles/pmc-fullce-16k"),("mix",EVm/"pred_titles/pmc-readoc-spage-16k")]:
    T[r]={d:(score_markdown_pair((GT/(d+'.md')).read_text(encoding="utf-8"),(pdir/(d+'.md')).read_text(encoding="utf-8"),level_weight=0.3,content_weight=0.7) or 0.0) for d in set87 if (pdir/(d+'.md')).is_file()}
nondeg=[d for d in set87 if d in T["pmc"] and d in T["mix"] and d not in S_mix12]
deltas=sorted(((T["mix"][d]-T["pmc"][d],d) for d in nondeg))
neg=[x for x in deltas if x[0]<-0.01]; pos=[x for x in deltas if x[0]>0.01]
print(f"  非退化 n{len(nondeg)}: 变差 {len(neg)} 篇合计 {sum(x[0] for x in neg):+.3f} ; 变好 {len(pos)} 篇合计 {sum(x[0] for x in pos):+.3f}")
print("  变差top6:", [(d,f"{T['pmc'][d]:.2f}->{T['mix'][d]:.2f}") for v,d in deltas[:6]])
print("  变好top4:", [(d,f"{T['pmc'][d]:.2f}->{T['mix'][d]:.2f}") for v,d in deltas[-4:]])
# 变差篇的 det-title 是否也掉(区分'没输出标题'与'输出了但内容错')
pd_=None
def detc(pdir,d):
    f=pdir/(d+".md")
    return f.read_text(encoding="utf-8",errors="replace").count("<|det|>title") if f.is_file() else -1
for v,d in deltas[:6]:
    print(f"    {d}: det-title pmc {detc(EVp/'pred/pmc-fullce-16k',d)} -> mix {detc(EVm/'pred/pmc-readoc-spage-16k',d)}")

# ---------- 6. spage 目标 vs 全文 GT(修 id 映射,按 images 路径) ----------
print("== 6. spage 片段校验(按 images 路径映射) ==")
full={}
with open(Path("/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-fullce-16k/train.jsonl"),encoding="utf-8",errors="replace") as fh:
    for line in fh:
        row=json.loads(line)
        imgs=row.get("images") or []
        if imgs:
            m=re.search(r"/pmc/(pmc\d+)/page_", imgs[0])
            if m: full[m.group(1)]=row["messages"][-1]["content"]
print(f"  full 行按 doc id 映射到 {len(full)} 篇")
ok=0; bad=0; nobase=0
with open(Path("/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-spage-16k/train.jsonl"),encoding="utf-8",errors="replace") as fh:
    for line in fh:
        row=json.loads(line)
        imgs=row.get("images") or []
        m=re.search(r"/pmc/(pmc\d+)/page_(\d+)\.png", imgs[0]) if imgs else None
        if not m: continue
        doc=m.group(1)
        tgt=row["messages"][-1]["content"]
        if doc not in full: nobase+=1; continue
        probe=tgt.strip()[50:300] if len(tgt)>300 else tgt.strip()
        if probe and probe in full[doc]: ok+=1
        else: bad+=1
print(f"  抽查: 连续片段命中 {ok} / 未命中 {bad} / 无全文基线 {nobase}")
# p0 标题硬门:spage p0 的首个 ATX 是否与全文 GT 首个 ATX 一致
same=0; diff=0; diffex=[]
with open(Path("/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-spage-16k/train.jsonl"),encoding="utf-8",errors="replace") as fh:
    for line in fh:
        row=json.loads(line)
        if not row.get("id","").endswith("_p0000"): continue
        imgs=row.get("images") or []
        m=re.search(r"/pmc/(pmc\d+)/", imgs[0]) if imgs else None
        if not m or m.group(1) not in full: continue
        def first_atx(t):
            for l in t.splitlines():
                if l.startswith("#"): return l.strip()
            return None
        a=first_atx(row["messages"][-1]["content"]); b=first_atx(full[m.group(1)])
        if a==b: same+=1
        else:
            diff+=1
            if len(diffex)<2: diffex.append((m.group(1),a,b))
print(f"  p0 标题硬门: 一致 {same} / 不一致 {diff} {diffex}")

# ---------- 7. 32k 诊断臂可用性 ----------
print("== 7. pmc-fullce-32k checkpoint-400 ==")
for ck in sorted(U.glob("output/pmc-fullce-32k/v0-*/v0-*/checkpoint-*")):
    if ck.name.startswith("checkpoint-"):
        ts=jload(ck/"trainer_state.json")
        ev=[h for h in ts["log_history"] if "eval_loss" in h]
        if ev: print(f"  {ck.name}: eval_loss {ev[-1]['eval_loss']:.4f} @s{ev[-1].get('step')} (可作'同分布更多步'诊断臂)")

# ---------- 8. mix union-healthy TEDS 损失的 Overall 当量 ----------
print("== 8. 若 mix 健康表达到 pmc 水平的 Overall 当量 ==")
# union-healthy: text_edit 0.1749 order_edit 0.0440 teds mix 0.4683 / pmc 0.6137 (脚本2实测)
for lbl,teds in [("mix 实测 0.4683",0.4683),("若达 pmc 0.6137",0.6137),("若达 base 0.5902",0.5902)]:
    ov=0.3*(1-0.1749)+0.3*teds+0.3*(1-0.0440)+0.1*0.6150
    print(f"  {lbl}: union-healthy Overall {ov:.4f}")
print("DONE")
