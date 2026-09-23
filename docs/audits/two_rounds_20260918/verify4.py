# -*- coding: utf-8 -*-
# 独立复核脚本 4:收尾。只读。
import json, re
from pathlib import Path
U   = Path("/home/jovyan/hyx/uocr-ms-swift-title-mask")
EVp = U/"evaluation/pmc-fullce-16k_all129"
EVm = U/"evaluation/pmc-readoc-spage-16k_all129"
OMNI= Path("/home/jovyan/hyx/pdf2text-auto-label-eval/OmniDocBench_v1.5/result")
GT  = Path("/home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/groundtruth")
def jload(p):
    with open(p) as f: return json.load(f)

# ---------- 1. spage 与 pmc16k 全文的 id 对齐(换方法:行内搜 id) ----------
print("== 1. spage doc 是否存在于 pmc16k 训练集 ==")
full_ids=set(); full_first={}
with open(Path("/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-fullce-16k/train.jsonl"),encoding="utf-8",errors="replace") as fh:
    for line in fh:
        row=json.loads(line)
        imgs=row.get("images") or []
        m=re.search(r"(pmc\d+)", str(imgs[0]) if imgs else (row.get("id","")+line[:200]))
        if m:
            full_ids.add(m.group(1)); full_first[m.group(1)]=row["messages"][-1]["content"]
spage_ids=[]
with open(Path("/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-spage-16k/train.jsonl"),encoding="utf-8",errors="replace") as fh:
    for line in fh:
        row=json.loads(line)
        imgs=row.get("images") or []
        m=re.search(r"(pmc\d+)", str(imgs[0]) if imgs else "")
        if m: spage_ids.append((m.group(1), row["messages"][-1]["content"], row.get("id")))
inter=set(s for s,_ ,_ in spage_ids)&full_ids
print(f"  spage {len(spage_ids)} 行, doc id 在 pmc16k 中: {len(inter)} ; spage 独有 doc: {len(set(s for s,_,_ in spage_ids)-full_ids)}")
# 命中的行:目标是否全文连续片段
ok=bad=0; ex=None
for doc,tgt,rid in spage_ids:
    if doc not in full_first: continue
    probe=tgt.strip()[50:300] if len(tgt)>300 else tgt.strip()
    if probe and probe in full_first[doc]: ok+=1
    else:
        bad+=1
        if ex is None: ex=(rid, tgt[:80], tgt[-80:])
print(f"  连续片段: 命中 {ok} / 未命中 {bad}")
if ex: print(f"  未命中例: {ex[0]} 头{ex[1]!r} 尾{ex[2]!r}")
# spage 目标形态统计
import statistics
ends_mid=0; has_atx=0; n=0; lens=[]
for doc,tgt,rid in spage_ids:
    n+=1; lens.append(len(tgt))
    s=tgt.rstrip()
    if s and s[-1] not in ".。!?:;)]”" and not s.endswith("</td>"): ends_mid+=1
    if "\n# " in tgt or tgt.startswith("#"): has_atx+=1
lens.sort()
print(f"  spage 目标: p50 {lens[n//2]} | 末尾截断(无句末标点) {ends_mid}/{n} | 含ATX {has_atx}/{n}")

# ---------- 2. per-doc TEDS 总量跌幅分解(§6.1 的 -42.0/71% 复算) ----------
print("== 2. TEDS 总量(逐表求和)跌幅分解 ==")
S_mix12={"3-1跨页重复表头","3-2跨页分页切断","3-4长表跨三页以上","4-1统计图2","4-1统计图3","4-3扫描2","5-1双栏表格2","中车株洲_方案技术_16","广东城规院_惠玩甘青_11","广东特检院_安全报告2_6","珠海机场_工作证管理细则_6","科欣环保_产业结构_5"}
def tedssum(pfx):
    d={}
    for p in OMNI.glob(pfx+"*_table_per_table_TEDS.json"):
        doc=p.name[len(pfx):-len("_table_per_table_TEDS.json")]
        j=jload(p)
        d[doc]=sum(x["TEDS"] for x in j.values() if isinstance(x,dict) and x.get("TEDS") is not None), sum(1 for x in j.values() if isinstance(x,dict) and x.get("TEDS") is not None)
    return d
tp=tedssum("swp2_"); tm=tedssum("swp3_")
tot_p=sum(v[0] for v in tp.values()); tot_m=sum(v[0] for v in tm.values())
deg_drop=sum(tp.get(d,(0,0))[0]-tm.get(d,(0,0))[0] for d in S_mix12)
print(f"  全库 TEDS 总量 pmc {tot_p:.1f} -> mix {tot_m:.1f} (Δ{tot_m-tot_p:+.1f}) ; mix退化12篇内 Δ{-deg_drop:+.1f} ({deg_drop/(tot_p-tot_m)*100:.0f}%) ; 非退化 Δ{tot_m-tot_p+deg_drop:+.1f}")
mv=sorted(((tm.get(s,(0,0))[0]-tp.get(s,(0,0))[0], s, tp.get(s,(0,0)), tm.get(s,(0,0))) for s in set(tp)|set(tm)), key=lambda x:x[0])[:8]
print("  跌幅top8 (doc, pmc(和,表数), mix(和,表数)):")
for v,s,a,b in mv: print(f"    {s}: {a} -> {b}")

# ---------- 3. mix 修复/新增退化篇的 per-doc overall ----------
print("== 3. 关键篇 per-doc overall pmc -> mix ==")
PDp=jload(EVp/"agentbuilder/pmc-fullce-16k/per_doc_scores.json")
PDm=jload(EVm/"agentbuilder/pmc-readoc-spage-16k/per_doc_scores.json")
for d in ["2-1横向合并","4-1统计图1","5-1双栏表格3","5-2表旁水印","中车株洲_方案技术_6","珠海机场_三防工作预案_23",
          "3-1跨页重复表头","4-1统计图2","3-4长表跨三页以上","4-1统计图3","4-2段落中带「图x-x」","1-3单元格内换行1","4-4示意图"]:
    a=PDp.get(d,{}).get("overall"); b=PDm.get(d,{}).get("overall")
    print(f"  {d}: {a if a is None else round(a,3)} -> {b if b is None else round(b,3)}")
print("DONE")
