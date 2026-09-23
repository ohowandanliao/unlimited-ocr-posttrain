# -*- coding: utf-8 -*-
# 独立复核脚本 2:深挖取证。只读产物,不重跑评测。
import json, zlib, sys, statistics
from pathlib import Path
import fitz

U    = Path("/home/jovyan/hyx/uocr-ms-swift-title-mask")
EVb  = U/"evaluation/readoc-view-16k_all129"
EVp  = U/"evaluation/pmc-fullce-16k_all129"
EVm  = U/"evaluation/pmc-readoc-spage-16k_all129"
GT   = Path("/home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/groundtruth")
SRC  = Path("/home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/source")
OMNI = Path("/home/jovyan/hyx/pdf2text-auto-label-eval/OmniDocBench_v1.5/result")
PC  = {"base":EVb/"pred_clean/base","fc":EVb/"pred_clean/full-ce",
       "pmc":EVp/"pred_clean/pmc-fullce-16k","mix":EVm/"pred_clean/pmc-readoc-spage-16k"}
PFX = {"pmc":"swp2_","mix":"swp3_"}
SUF = {"text":"_text_block_per_page_edit.json","order":"_reading_order_per_page_edit.json","teds":"_table_per_table_TEDS.json"}
def jload(p):
    with open(p) as f: return json.load(f)

names=[l.strip() for l in (EVm/"evaluation_names.txt").read_text().split() if l.strip()]
stems=[n[:-4] if n.endswith(".pdf") else n for n in names]
ALL=set(stems)
pages={}
for s in stems:
    try: pages[s]=fitz.open(SRC/(s+".pdf")).page_count
    except Exception: pages[s]=-1
single={s for s in stems if pages[s]==1}; multi={s for s in stems if pages[s]>=2}
gtlen={s:len((GT/(s+".md")).read_text(encoding="utf-8",errors="replace")) for s in stems if (GT/(s+".md")).is_file()}

# ---------- 0. pmc 官方值:剔除 straggler 后精确复现 ----------
STRAG="5-1双栏表格2"
def pool(r,kind,exclude=None):
    vals=[]
    if r in PFX:
        suf=SUF[kind]
        for p in sorted(OMNI.glob(PFX[r]+"*"+suf)):
            doc=p.name[len(PFX[r]):-len(suf)]
            if exclude and doc in exclude: continue
            jj=jload(p)
            if kind=="teds":
                vals+= [x["TEDS"] for x in jj.values() if isinstance(x,dict) and x.get("TEDS") is not None]
            else:
                vals+=[v for v in jj.values() if isinstance(v,(int,float))]
    else:
        g=OMNI/("readoc16k_20260827_"+("base" if r=="base" else "full_ce")+"_gtpdf"+SUF[kind])
        jj=jload(g)
        for k,v in jj.items():
            if exclude and k.split(".jpg")[0] in exclude: continue
            if kind=="teds":
                if isinstance(v,dict) and v.get("TEDS") is not None: vals.append(v["TEDS"])
            elif isinstance(v,(int,float)): vals.append(v)
    return vals
print("== 0. pmc 全量(剔 straggler) vs 官方 ==")
S0={STRAG}
te=pool("pmc","text",S0); od=pool("pmc","order",S0); td=pool("pmc","teds",S0)
print(f"  pmc excl straggler: text {1-sum(te)/len(te):.4f}(n{len(te)}) order {1-sum(od)/len(od):.4f}(n{len(od)}) TEDS {sum(td)/len(td):.4f}(n{len(td)})  [官方 0.7658/0.9065/0.4686 n258]")

# ---------- 1. 退化文档 GT 规模与类型 ----------
print("== 1. 退化文档 GT 字符数(检验'新病全为 GT<2.5k') ==")
S_pmc={"2-1横向合并","3-2跨页分页切断","4-1统计图1","4-3扫描2","5-1双栏表格2",
 "5-1双栏表格3","5-2表旁水印","中车株洲_方案技术_6","珠海机场_三防工作预案_23",
 "3-3跨页表注在不同页","科欣环保_产业结构_5"}
S_mix={"3-1跨页重复表头","3-2跨页分页切断","3-4长表跨三页以上","4-1统计图2","4-1统计图3",
 "4-3扫描2","5-1双栏表格2","中车株洲_方案技术_16","广东城规院_惠玩甘青_11",
 "广东特检院_安全报告2_6","珠海机场_工作证管理细则_6","科欣环保_产业结构_5"}
new6=S_mix-{k for k in S_pmc}&S_mix
for label,S in [("pmc退化",S_pmc),("mix退化",S_mix),("mix新病",new6)]:
    row=[]
    for d in sorted(S):
        row.append(f"{d}:{gtlen.get(d,'?')}c/{pages.get(d)}p")
    print(f"  {label}: "+"; ".join(row))

# ---------- 2. 同一文档集(union 剔严格退化)的公平对比 ----------
S_u = S_pmc|S_mix
AB={"base":(EVb,"base"),"fc":(EVb,"full-ce"),"pmc":(EVp,"pmc-fullce-16k"),"mix":(EVm,"pmc-readoc-spage-16k")}
PD={r:jload(ev/"agentbuilder"/a/"per_doc_scores.json") for r,(ev,a) in AB.items()}
def pick_ov(d):
    for k in ("corrected_overall","overall","Overall"):
        if k in d: return d[k]
    return float("nan")
print("== 2. union 剔严格退化(n=%d 剔除)同一文档集对比 ==" % len(S_u))
H=ALL-S_u
for sub,sn in [(single,"1页"),(multi,">=2页")]:
    hs=H&sub
    row=[]
    for r in ["base","pmc","mix"]:
        vs=[pick_ov(PD[r][s]) for s in hs if s in PD[r] and pick_ov(PD[r][s])==pick_ov(PD[r][s])]
        row.append(f"{r} {sum(vs)/len(vs):.4f}(n{len(vs)})")
    print(f"  per-doc overall {sn}: "+" | ".join(row))
for r in ["base","fc","pmc","mix"]:
    te=pool(r,"text",S_u); od=pool(r,"order",S_u); td=pool(r,"teds",S_u)
    print(f"  {r:4s} union-healthy: text_edit {sum(te)/len(te):.4f}(n{len(te)}) order_edit {sum(od)/len(od):.4f}(n{len(od)}) TEDS {sum(td)/len(td):.4f}(n{len(td)})")

# ---------- 3. 长度比分布(非退化篇):隐性内容缺失 ----------
print("== 3. 非退化篇 len(pred)/len(GT) 分布 ==")
for r in ["base","pmc","mix"]:
    S = S_pmc if r=="pmc" else (S_mix if r=="mix" else set())
    bands={"<=0.5":0,"0.5-0.66":0,"0.66-0.8":0,"0.8-1.25":0,"1.25-2":0,">2":0}
    ratios=[]
    for f in PC[r].glob("*.md"):
        s=f.stem
        if s in S or s not in gtlen: continue
        t=f.read_text(encoding="utf-8",errors="replace")
        rr=len(t)/gtlen[s]; ratios.append(rr)
        for b in bands:
            lo,hi={"<=0.5":(0,0.5),"0.5-0.66":(0.5,0.667),"0.66-0.8":(0.667,0.8),"0.8-1.25":(0.8,1.25),"1.25-2":(1.25,2),">2":(2,99)}[b]
            if lo<=rr<hi: bands[b]+=1; break
    q=statistics.quantiles(ratios,n=10)
    print(f"  {r:4s} p10 {q[0]:.2f} p50 {q[4]:.2f} p90 {q[8]:.2f} | {bands}")

# ---------- 4. per-doc text_edit 长尾(非退化) ----------
print("== 4. 非退化篇 per-doc text_edit 分布 ==")
for r in ["pmc","mix"]:
    S=S_pmc if r=="pmc" else S_mix
    vals=[]
    for s in ALL-S:
        if s in PD[r] and PD[r][s].get("text_edit") is not None:
            vals.append(PD[r][s]["text_edit"])
    vals.sort()
    n=len(vals)
    bad=[v for v in vals if v>0.4]
    print(f"  {r:4s} n{n} p50 {vals[n//2]:.3f} p90 {vals[int(n*0.9)]:.3f} max {vals[-1]:.3f} | >0.4 的 {len(bad)} 篇: {[round(v,2) for v in bad[:15]]}")

# ---------- 5. det-title 产量分解 ----------
print("== 5. det-title 块 502->345 分解 ==")
def dettitle(pdir):
    return {f.stem:f.read_text(encoding="utf-8",errors="replace").count("<|det|>title") for f in pdir.glob("*.md")}
dtp=dettitle(EVp/"pred/pmc-fullce-16k"); dtm=dettitle(EVm/"pred/pmc-readoc-spage-16k")
deg_hit=sum(dtm.get(d,0)-dtp.get(d,0) for d in S_mix)
all_hit=sum(dtm.get(s,0)-dtp.get(s,0) for s in set(dtp)|set(dtm))
print(f"  总Δ {all_hit} (pmc {sum(dtp.values())} -> mix {sum(dtm.values())}) ; mix退化12篇内 Δ {deg_hit} ; 非退化 Δ {all_hit-deg_hit}")
top=sorted(set(dtp)|set(dtm),key=lambda s:dtm.get(s,0)-dtp.get(s,0))[:8]
print("  跌幅top8:",[(s,f"{dtp.get(s,0)}->{dtm.get(s,0)}") for s in top])

# ---------- 6. ATX 标题行(pred_clean 正文,pred_titles 重建件) ----------
print("== 6. ATX 行为 ==")
def atx(pdir):
    d={}
    for f in pdir.glob("*.md"):
        d[f.stem]=sum(1 for l in f.read_text(encoding="utf-8",errors="replace").splitlines() if l.startswith("#"))
    return d
for r in ["pmc","mix"]:
    d=atx(PC[r])
    tot=sum(d.values())
    print(f"  {r} pred_clean ATX 总数 {tot} 篇均 {tot/len(d):.1f}")
sys.path.insert(0,str(U/"evaluation/tooling/agentbuilder_pkg"))
from agentbuilder_eval.title_metric import score_markdown_pair, extract_title_lines
set87=[p.stem for p in sorted(GT.glob("*.md")) if extract_title_lines(p.read_text(encoding="utf-8")) and p.stem!="Handwriting document"]
PT={"base":EVb/"pred_titles/base","fc":EVb/"pred_titles/full-ce","pmc":EVp/"pred_titles/pmc-fullce-16k","mix":EVm/"pred_titles/pmc-readoc-spage-16k"}
print("  extract_title_lines 口径篇均: GT %.1f | "%(sum(len(extract_title_lines((GT/(d+'.md')).read_text(encoding='utf-8'))) for d in set87)/len(set87)) +
      " | ".join(f"{r} {sum(len(extract_title_lines((PT[r]/(d+'.md')).read_text(encoding='utf-8'))) for d in set87 if (PT[r]/(d+'.md')).is_file())/len(set87):.1f}" for r in ["base","fc","pmc","mix"]))

# ---------- 7. 循环取证 ----------
print("== 7. 行内循环取证(最长重复子串) ==")
def longest_repeat(t):
    def has(L):
        seen=set()
        for i in range(len(t)-L+1):
            s=t[i:i+L]
            if s in seen: return s
            seen.add(s)
        return None
    lo,hi=8,len(t)//2; best=""
    while lo<=hi:
        mid=(lo+hi)//2
        s=has(mid)
        if s is not None: best=s; lo=mid+1
        else: hi=mid-1
    return best
TRAIN={}
for lbl,f in [("pmc",Path("/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-fullce-16k/train.jsonl")),
              ("spage",Path("/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-spage-16k/train.jsonl")),
              ("readoc",U/"data/readoc-view-16k/train.jsonl")]:
    TRAIN[lbl]=f
def train_has(unit):
    hits=[]
    u=unit[:60]
    for lbl,f in TRAIN.items():
        with open(f,encoding="utf-8",errors="replace") as fh:
            for line in fh:
                try: row=json.loads(line)
                except Exception: continue
                tgt=row.get("messages",[{}])[-1].get("content","") if row.get("messages") else ""
                if u in tgt: hits.append(lbl); break
    return hits
targets=["珠海机场_工作证管理细则_6","中车株洲_方案技术_16","广东特检院_安全报告2_6","广东城规院_惠玩甘青_11",
         "3-1跨页重复表头","4-1统计图2","中车株洲_方案技术_6","2-1横向合并"]
for d in targets:
    f=PC["mix"]/ (d+".md"); f2=PC["pmc"]/(d+".md")
    t = f.read_text(encoding="utf-8",errors="replace") if f.is_file() else ""
    t2= f2.read_text(encoding="utf-8",errors="replace") if f2.is_file() else ""
    tt = t if len(t)>len(t2) else t2
    unit=longest_repeat(tt)
    cnt=tt.count(unit) if unit else 0
    pos=tt.find(unit)/max(1,len(tt)) if unit else -1
    hits=train_has(unit) if unit else []
    print(f"  {d}: pred {len(tt)}c GT {gtlen.get(d,'?')}c unit[{len(unit)}c]x{cnt} 首现@{pos:.0%} 训练数据命中:{hits or '无'}")
    if unit: print(f"    unit头80: {unit[:80]!r}")

# ---------- 8. 广东特检院_安全报告1_5 尾部缺失 ----------
print("== 8. 广东特检院_安全报告1_5 ==")
g=(GT/"广东特检院_安全报告1_5.md").read_text(encoding="utf-8",errors="replace")
for r in ["pmc","mix"]:
    t=(PC[r]/"广东特检院_安全报告1_5.md").read_text(encoding="utf-8",errors="replace")
    print(f"  {r}: len {len(t)} ratio {len(t)/len(g):.2f} 末60: {t[-60:]!r}")
print(f"  GT末60: {g[-60:]!r}")
ov_p=PD["pmc"]["广东特检院_安全报告1_5"].get("corrected_overall",PD["pmc"]["广东特检院_安全报告1_5"].get("overall"))
ov_m=PD["mix"]["广东特检院_安全报告1_5"].get("corrected_overall",PD["mix"]["广东特检院_安全报告1_5"].get("overall"))
print(f"  overall pmc {ov_p} -> mix {ov_m}")

# ---------- 9. 训练数据统计 ----------
print("== 9. 训练数据 ==")
def data_stats(lbl,f):
    n=0; pr={}; tab=0; chars=0; lens=[]
    with open(f,encoding="utf-8",errors="replace") as fh:
        for line in fh:
            try: row=json.loads(line)
            except Exception: continue
            n+=1
            msg=row.get("messages",[])
            prompt=msg[0]["content"] if msg else ""
            tgt=msg[-1]["content"] if len(msg)>1 else ""
            pr[prompt.strip()[:30]]=pr.get(prompt.strip()[:30],0)+1
            tab+= 1 if "<table" in tgt else 0
            chars+=len(tgt); lens.append(len(tgt))
    lens.sort()
    p=lambda q: lens[int(len(lens)*q)]
    print(f"  {lbl}: n{n} prompt {pr} | 含<table行 {tab}({tab/n*100:.1f}%) | 目标chars p10 {p(0.1)} p50 {p(0.5)} p90 {p(0.9)} 总 {chars}")
    return n,tab,chars
n_p,_,c_p=data_stats("pmc16k",Path("/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-fullce-16k/train.jsonl"))
n_s,_,c_s=data_stats("spage",Path("/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-spage-16k/train.jsonl"))
n_r,_,c_r=data_stats("readoc",U/"data/readoc-view-16k/train.jsonl")
data_stats("mix合计",Path("/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-readoc-spage-16k/train.jsonl"))
print(f"  字符占比: pmc {c_p/(c_p+c_r)*100:.2f}% readoc {c_r/(c_p+c_r)*100:.2f}% (文档称 92.75/7.25)")

# ---------- 10. spage GT 构造:页拆目标是否为全文 GT 的连续片段 ----------
print("== 10. spage 抽查 ==")
full={}
with open(Path("/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-fullce-16k/train.jsonl"),encoding="utf-8",errors="replace") as fh:
    for line in fh:
        row=json.loads(line)
        if row.get("messages"): full[row.get("id","")]=row["messages"][-1]["content"]
sp=[]
with open(Path("/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-spage-16k/train.jsonl"),encoding="utf-8",errors="replace") as fh:
    for line in fh: sp.append(json.loads(line))
import re
insub=0; checked=0; weird=0
for row in sp[:400]:
    rid=row.get("id","")
    m=re.match(r"pmc_(.+)_p(\d+)$",rid)
    if not m: continue
    doc=m.group(1)
    tgt=row["messages"][-1]["content"]
    checked+=1
    # 找同文档 full 行(id 约定可能是 pmc_<doc> 或含 doc)
    key=[k for k in full if k==f"pmc_{doc}" or k.endswith(doc)][:1]
    if key and tgt and tgt.strip() and tgt.strip() in full[key[0]]:
        insub+=1
    else:
        weird+=1
        if weird<=2:
            print(f"    非连续片段例: id {rid} key {key} tgt头80 {tgt[:80]!r}")
print(f"  抽查 {checked} 行: 目标是全文 GT 连续片段的 {insub} 行, 非片段/未匹配 {weird} 行")
for row in sp[:2]:
    tgt=row["messages"][-1]["content"]
    print(f"  例 id {row.get('id')}: prompt {row['messages'][0]['content'][:40]!r} len {len(tgt)} 头60 {tgt[:60]!r} 尾60 {tgt[-60:]!r} images {str(row.get('images'))[:80]}")

# ---------- 11. 管道表/空输出/协议健康 ----------
print("== 11. 杂项 ==")
for r in ["pmc","mix"]:
    pipe=sum(1 for f in PC[r].glob("*.md") if any(l.lstrip().startswith("|") for l in f.read_text(encoding="utf-8",errors="replace").splitlines()))
    empt=sum(1 for f in PC[r].glob("*.md") if len(f.read_text(encoding="utf-8",errors="replace").strip())<100)
    unbal=0
    for f in (EVp/"pred/pmc-fullce-16k").glob("*.md") if r=="pmc" else (EVm/"pred/pmc-readoc-spage-16k").glob("*.md"):
        t=f.read_text(encoding="utf-8",errors="replace")
        if t.count("<|det|>")!=t.count("<|/det|>"): unbal+=1
    print(f"  {r}: 含管道表行文档 {pipe} | <100c 空输出 {empt} | raw det不闭合 {unbal}")
print("DONE")
