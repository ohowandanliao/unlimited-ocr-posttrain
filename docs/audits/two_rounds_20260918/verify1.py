# -*- coding: utf-8 -*-
# 独立复核脚本 1(修正版):验证文档数字。只读产物,不重跑评测。
import json, zlib, sys
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
RAW = {"pmc":EVp/"pred/pmc-fullce-16k","mix":EVm/"pred/pmc-readoc-spage-16k"}
PFX = {"pmc":"swp2_","mix":"swp3_"}
SUF = {"text":"_text_block_per_page_edit.json","order":"_reading_order_per_page_edit.json","teds":"_table_per_table_TEDS.json"}

def jload(p):
    with open(p) as f: return json.load(f)

names = [l.strip() for l in (EVm/"evaluation_names.txt").read_text().split() if l.strip()]
stems = [n[:-4] if n.endswith(".pdf") else n for n in names]
ALL = set(stems)
pages = {}
for s in stems:
    try: pages[s] = fitz.open(SRC/(s+".pdf")).page_count
    except Exception: pages[s] = -1
single = {s for s in stems if pages[s]==1}; multi = {s for s in stems if pages[s]>=2}
print(f"[0] docs 129: single {len(single)} multi {len(multi)}")

# ---------- A. metrics.json ----------
print("== A. metrics.json 官方组件 ==")
AB = {"base":(EVb,"base"),"fc":(EVb,"full-ce"),"pmc":(EVp,"pmc-fullce-16k"),"mix":(EVm,"pmc-readoc-spage-16k")}
met = {r:jload(ev/"agentbuilder"/a/"metrics.json") for r,(ev,a) in AB.items()}
print("  metrics 子字段:", json.dumps(met["pmc"].get("metrics", {}), ensure_ascii=False)[:400])

# ---------- B/C. pooled 池(支持剔除) ----------
def pool(r, kind, exclude=None):
    vals=[]
    if r in PFX:
        suf = SUF[kind]
        for p in sorted(OMNI.glob(PFX[r]+"*"+suf)):
            doc = p.name[len(PFX[r]):-len(suf)]
            if exclude and doc in exclude: continue
            j = jload(p)
            if kind=="teds":
                for x in j.values():
                    if isinstance(x,dict) and x.get("TEDS") is not None: vals.append(x["TEDS"])
            else:
                vals += [v for v in j.values() if isinstance(v,(int,float))]
    else:
        g = OMNI/("readoc16k_20260827_"+("base" if r=="base" else "full_ce")+"_gtpdf"+SUF[kind])
        j = jload(g)
        for k,v in j.items():
            if exclude and k.split(".jpg")[0] in exclude: continue
            if kind=="teds":
                if isinstance(v,dict) and v.get("TEDS") is not None: vals.append(v["TEDS"])
            elif isinstance(v,(int,float)): vals.append(v)
    return vals

print("== B. 官方 pooled 全量复现 (text/order doc加权, TEDS 表加权) ==")
pooled = {}
for r in ["base","fc","pmc","mix"]:
    te=pool(r,"text"); od=pool(r,"order"); td=pool(r,"teds")
    pooled[r]=dict(text=1-sum(te)/len(te), order=1-sum(od)/len(od), teds=sum(td)/len(td),
                   nte=len(te), nod=len(od), ntd=len(td), te=sum(te)/len(te), od=sum(od)/len(od))
    v=pooled[r]
    print(f"  {r:4s} text {v['text']:.4f}(n{v['nte']}) order {v['order']:.4f}(n{v['nod']}) TEDS {v['teds']:.4f}(n{v['ntd']})")

# ---------- C. 四板斧 ----------
def fourboard(t, gt):
    lines=[l.strip() for l in t.splitlines() if l.strip()]
    dup=1-len(set(lines))/len(lines) if lines else 0.0
    lr=len(t)/max(1,len(gt))
    zr=len(zlib.compress(t.encode(),9))/max(1,len(t.encode())) if t else 1.0
    f=[]
    if dup>=0.3: f.append(f"dup{dup:.2f}")
    if lr>=2: f.append(f"len{lr:.1f}x")
    if gt and lr<=1/3: f.append(f"collapse{lr:.2f}x")
    if zr<0.15: f.append(f"zlib{zr:.3f}")
    return f
print("== C. 四板斧退化清单 ==")
DEG={}
for r,p in PC.items():
    d={}
    for f in sorted(p.glob("*.md")):
        t=f.read_text(encoding="utf-8",errors="replace")
        gtf=GT/f.name
        gtxt=gtf.read_text(encoding="utf-8",errors="replace") if gtf.is_file() else ""
        why=fourboard(t,gtxt)
        if why: d[f.stem]=",".join(why)
    DEG[r]=d
    print(f"  {r:4s} 退化 {len(d)} 篇 (严格 {sum(1 for v in d.values() if 'dup' in v or 'len' in v)}):")
    for k in sorted(d): print(f"     {k}: {d[k]} [{pages.get(k)}p]")
S_pmc8 ={k for k,v in DEG["pmc"].items() if "dup" in v or "len" in v}
S_pmc11=set(DEG["pmc"])
S_mix9 ={k for k,v in DEG["mix"].items() if "dup" in v or "len" in v}
S_mix12=set(DEG["mix"])

# ---------- D. 反事实 ----------
print("== D. 反事实 pooled(同剔) ==")
def over(t,o,T,title): return 0.3*t+0.3*T+0.3*o+0.1*title
title87={"base":0.6862,"fc":0.6668,"pmc":0.6697,"mix":0.6150}
for name,r,S in [("pmc  excl pmc8","pmc",S_pmc8),("fc   excl pmc8","fc",S_pmc8),("base excl pmc8","base",S_pmc8),
                 ("mix  excl mix12","mix",S_mix12),("mix  excl mix9","mix",S_mix9),
                 ("pmc  excl pmc11","pmc",S_pmc11),("pmc  excl U8+9","pmc",S_pmc8|S_mix9),
                 ("mix  excl U8+9","mix",S_pmc8|S_mix9)]:
    te=pool(r,"text",S); od=pool(r,"order",S); td=pool(r,"teds",S)
    t_,o_,T_=1-sum(te)/len(te),1-sum(od)/len(od),sum(td)/len(td)
    print(f"  {name:18s} text {t_:.4f}(n{len(te)}) order {o_:.4f}(n{len(od)}) TEDS {T_:.4f}(n{len(td)}) Overall~{over(t_,o_,T_,title87[r]):.4f}")

# ---------- E. <PAGE> ----------
print("== E. <PAGE> ==")
for r,p in PC.items():
    docs=tot=sing=mult=0
    for f in p.glob("*.md"):
        c=f.read_text(encoding="utf-8",errors="replace").count("<PAGE>")
        if c:
            docs+=1; tot+=c
            if pages.get(f.stem)==1: sing+=c
            elif pages.get(f.stem,0)>=2: mult+=c
    print(f"  {r:4s} docs {docs} total {tot} (单页 {sing}/多页 {mult})")
print(f"  GT: {sum((GT/(s+'.md')).read_text(encoding='utf-8',errors='replace').count('<PAGE>') for s in stems if (GT/(s+'.md')).is_file())}")

# ---------- F. <table> 发射 ----------
print("== F. <table> 发射 ==")
TAB={}
for r,p in PC.items():
    TAB[r]={f.stem:f.read_text(encoding="utf-8",errors="replace").count("<table") for f in p.glob("*.md")}
    print(f"  {r:4s} 全库 {sum(TAB[r].values())} 单页 {sum(c for s,c in TAB[r].items() if pages.get(s)==1)} 多页 {sum(c for s,c in TAB[r].items() if pages.get(s,0)>=2)}")
mv=sorted(((TAB['mix'].get(s,0)-TAB['pmc'].get(s,0),s,TAB['pmc'].get(s,0),TAB['mix'].get(s,0)) for s in stems), key=lambda x:x[0])[:12]
print("  pmc->mix 跌幅top12:", [(s,f"{a}->{b}",f"{pages.get(s)}p") for d,s,a,b in mv])

# ---------- G. 训练曲线 ----------
print("== G. trainer_state ==")
for label,pat in [("pmc","output/pmc-fullce-16k/v0-*/v0-*/checkpoint-*"),
                  ("mix","output/pmc-readoc-spage-16k/v0-*/v0-*/checkpoint-*"),
                  ("fc","output/readoc-view-16k-full-ce/v0-*/checkpoint-*")]:
    cks=[c for c in U.glob(pat) if c.name.startswith("checkpoint-")]
    if not cks: print(f"  {label}: no ckpt"); continue
    ck=max(cks,key=lambda c:int(c.name.split("-")[-1]))
    lh=jload(ck/"trainer_state.json")["log_history"]
    tl=[h["loss"] for h in lh if "loss" in h]
    ev=[h for h in lh if "eval_loss" in h]
    evs=" | ".join(f"s{h.get('step')}: el{h['eval_loss']:.4f}/ta{h.get('eval_token_acc',float('nan')):.4f}" for h in ev[-3:])
    print(f"  {label} {ck.name}: train_loss末10 {sum(tl[-10:])/len(tl[-10:]):.4f} ; eval末3: {evs}")

# ---------- H. title87 ----------
print("== H. title87 重算 ==")
sys.path.insert(0,str(U/"evaluation/tooling/agentbuilder_pkg"))
from agentbuilder_eval.title_metric import score_markdown_pair, extract_title_lines
set87=[p.stem for p in sorted(GT.glob("*.md")) if extract_title_lines(p.read_text(encoding="utf-8")) and p.stem!="Handwriting document"]
PT={"base":EVb/"pred_titles/base","fc":EVb/"pred_titles/full-ce","pmc":EVp/"pred_titles/pmc-fullce-16k","mix":EVm/"pred_titles/pmc-readoc-spage-16k"}
T={}
def atx_lines(p):
    if not p.is_file(): return 0
    return sum(1 for l in p.read_text(encoding="utf-8",errors="replace").splitlines() if l.startswith("#"))
for r,p in PT.items():
    sc={}
    for d in set87:
        f=p/(d+".md")
        if f.is_file():
            g=(GT/(d+".md")).read_text(encoding="utf-8")
            sc[d]=score_markdown_pair(g,f.read_text(encoding="utf-8"),level_weight=0.3,content_weight=0.7) or 0.0
    T[r]=sc
    m=sum(sc.values())/len(sc) if sc else float('nan')
    print(f"  {r:4s} title87 n={len(sc)} mean {m:.4f} ATX行/篇均 {sum(atx_lines(p/(d+'.md')) for d in sc)/len(sc):.2f}")
print(f"  GT set87 标题行/篇均 {sum(len(extract_title_lines((GT/(d+'.md')).read_text(encoding='utf-8'))) for d in set87)/len(set87):.2f}")
print("  重算Overall:", " ".join(f"{r}={over(pooled[r]['text'],pooled[r]['order'],pooled[r]['teds'],sum(T[r].values())/len(T[r])):.4f}" for r in ["base","fc","pmc","mix"]))
d87=[d for d in set87 if d in T["pmc"] and d in T["mix"]]
deg87=[d for d in d87 if d in S_mix12]
dv_all=sum(T["mix"][d]-T["pmc"][d] for d in d87); dv_deg=sum(T["mix"][d]-T["pmc"][d] for d in deg87)
print(f"  mix title Δ({dv_all:+.4f}): 退化∩87 {len(deg87)}篇 {dv_deg:+.4f} ({(dv_deg/dv_all*100 if dv_all else 0):.0f}%), 非退化 {dv_all-dv_deg:+.4f}")
print("  非退化最差8:", [(d,f"{T['pmc'][d]:.2f}->{T['mix'][d]:.2f}") for d in sorted(d87,key=lambda d:T['mix'][d]-T['pmc'][d]) if d not in S_mix12][:8])
dt={r:sum(f.read_text(encoding="utf-8",errors="replace").count("<|det|>title") for f in p.glob("*.md")) for r,p in RAW.items()}
print(f"  det-title块数(raw): {dt}")

# ---------- I. 页数拆分 ----------
print("== I. 页数拆分 ==")
PD={r:jload(ev/"agentbuilder"/a/"per_doc_scores.json") for r,(ev,a) in AB.items()}
print("  per_doc keys:", sorted(next(iter(PD["mix"].values())).keys()))
def pick_ov(d):
    for k in ("corrected_overall","overall","Overall"):
        if k in d: return d[k]
    return float("nan")
for sub,sn in [(single,"1页"),(multi,">=2页")]:
    row=[]
    for r in ["base","pmc","mix"]:
        vs=[pick_ov(PD[r][s]) for s in sub if s in PD[r] and pick_ov(PD[r][s])==pick_ov(PD[r][s])]
        row.append(f"{r} {sum(vs)/len(vs):.4f}(n{len(vs)})")
    print(f"  per-doc overall {sn}: "+" | ".join(row))
for sub,sn in [(single,"1页"),(multi,">=2页")]:
    for r in ["pmc","mix"]:
        ex=ALL-sub
        te=pool(r,"text",ex); td=pool(r,"teds",ex)
        print(f"  official {r} {sn}: text_edit {sum(te)/len(te):.4f}(n{len(te)}) TEDS {sum(td)/len(td):.4f}(n{len(td)})")
print("  -- 健康(各路自剔严格退化) --")
for r in ["pmc","mix"]:
    S=S_pmc8 if r=="pmc" else S_mix9
    hm=multi-S; hs=single-S
    vs=[pick_ov(PD[r][s]) for s in hm if s in PD[r] and pick_ov(PD[r][s])==pick_ov(PD[r][s])]
    te=pool(r,"text",ALL-hm); td=pool(r,"teds",ALL-hm)
    print(f"  {r} 健康多页 n{len(hm)}: per-doc overall {sum(vs)/len(vs):.4f} | text_edit {sum(te)/len(te):.4f}(n{len(te)}) TEDS {sum(td)/len(td):.4f}(n{len(td)})")
    vs=[pick_ov(PD[r][s]) for s in hs if s in PD[r] and pick_ov(PD[r][s])==pick_ov(PD[r][s])]
    print(f"  {r} 健康单页 n{len(hs)}: per-doc overall {sum(vs)/len(vs):.4f}")
print("DONE")
