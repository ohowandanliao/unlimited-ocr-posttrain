import os, sys, json, time, hashlib, random
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import numpy as np
from collections import Counter

t0 = time.time()
def log(*a, **k):
    print(*a, flush=True)

OUT = '/tmp/uocr_audit'
os.makedirs(OUT, exist_ok=True)

DATASETS = {
    'pmc-fullce-16k': '/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-fullce-16k/train.jsonl',
    'pmc-readoc-spage-16k': '/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-readoc-spage-16k/train.jsonl',
    'readoc-view-16k': '/home/jovyan/hyx/uocr-ms-swift-title-mask/data/readoc-view-16k/train.jsonl',
}
MANIFEST = '/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-fullce-16k/length_manifest.jsonl'
TOK_PATH = '/home/jovyan/hyx/models/Unlimited-OCR'
SWIFT_REPO = '/home/jovyan/hyx/uocr-ms-swift-title-mask/repos/ms-swift-uocr'
MERGE_KEY = 'Multi page merge'

def load_jsonl(p):
    rows = []
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def user_content(r):
    for m in r.get('messages', []):
        if m.get('role') == 'user':
            return m.get('content', '') or ''
    return ''

def asst_content(r):
    for m in r.get('messages', []):
        if m.get('role') == 'assistant':
            return m.get('content', '') or ''
    return ''

log('== load jsonl ==', flush=True)
t = time.time()
ds = {k: load_jsonl(v) for k, v in DATASETS.items()}
manifest = load_jsonl(MANIFEST)
log('LOADED', {k: len(v) for k, v in ds.items()}, 'manifest', len(manifest), 'sec', round(time.time() - t, 1))

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
EOS = tok.eos_token or ''
log('TOK', type(tok).__name__, 'is_fast', tok.is_fast, 'eos_id', tok.eos_token_id, 'eos', repr(EOS))

all_prompts = set()
for rows in ds.values():
    for r in rows:
        all_prompts.add(user_content(r))
prompt_tok = {p: len(tok(p, add_special_tokens=False)['input_ids']) for p in sorted(all_prompts)}
log('PROMPT_TOK', {repr(k)[:70]: v for k, v in prompt_tok.items()}, 'sec', round(time.time() - t0, 1))

def pstats(a, extra_qs=()):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {}
    qs = [50, 90, 95, 99] + list(extra_qs)
    d = {'min': float(a.min()), 'max': float(a.max()), 'mean': round(float(a.mean()), 2)}
    for q in qs:
        d['p%d' % q] = float(np.percentile(a, q))
    return d

_DT = {}
USE_PY_GRAM = False
def gram_numpy(ids, g):
    arr = np.asarray(ids, dtype=np.int32)
    w = np.lib.stride_tricks.sliding_window_view(arr, g)
    a = np.ascontiguousarray(w)
    dt = _DT.get(g)
    if dt is None:
        dt = np.dtype([('k', np.int32, (g,))])
        _DT[g] = dt
    v = a.view(dt).reshape(-1)
    return int(np.unique(v, return_counts=True)[1].max())

def gram_py(ids, g):
    return max(Counter(tuple(ids[i:i + g]) for i in range(len(ids) - g + 1)).values())

def max_gram_repeat(ids, g):
    n = len(ids)
    if n < g:
        return 0
    if n < 2 * g:
        return 1
    if USE_PY_GRAM:
        return gram_py(ids, g)
    try:
        return gram_numpy(ids, g)
    except Exception:
        return gram_py(ids, g)

_rng = np.random.RandomState(0)
for _g in (8, 16, 32):
    _test = _rng.randint(0, 50, size=400).tolist()
    _ref = gram_py(_test, _g)
    if max_gram_repeat(_test, _g) != _ref:
        USE_PY_GRAM = True
        assert max_gram_repeat(_test, _g) == _ref, ('GRAM MISMATCH', _g)
log('GRAM_SELFTEST OK use_py=', USE_PY_GRAM)

def line_dup_rates(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return 0.0, 0.0
    c = Counter(lines)
    rate_cnt = (len(lines) - len(c)) / len(lines)
    tot = sum(len(l) for l in lines)
    dup = sum(len(l) for l, k in c.items() if k >= 2)
    rate_char = dup / tot if tot else 0.0
    return rate_cnt, rate_char

PAGE_PATS = ['<page', '<PAGE', 'PAGE>']

def buckets(ps):
    b = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6-10': 0, '>10': 0}
    for p in ps:
        if p <= 5:
            b[str(p)] += 1
        elif p <= 10:
            b['6-10'] += 1
        else:
            b['>10'] += 1
    return b

def analyze(name, rows, reuse=None):
    n = len(rows)
    prompt_counts = Counter(user_content(r) for r in rows)
    channel_counts = Counter(str(r.get('channel')) for r in rows)
    merge_flags = [MERGE_KEY in user_content(r) for r in rows]
    pages = [len(r.get('images') or []) for r in rows]
    texts = [asst_content(r) for r in rows]
    merge_idx = [i for i, f in enumerate(merge_flags) if f]
    merge_pages = [pages[i] for i in merge_idx]

    lens = [None] * n
    gram_max = {8: [None] * n, 16: [None] * n, 32: [None] * n}
    rate_cnts = [None] * n
    rate_chars = [None] * n
    eos_hits = 0
    page_in_hits = {p: 0 for p in PAGE_PATS}
    page_tgt_hits = {p: 0 for p in PAGE_PATS}
    page_ex_ids = {}
    for p in PAGE_PATS:
        page_ex_ids[('in', p)] = []
        page_ex_ids[('tgt', p)] = []

    need = [i for i in range(n) if not (reuse and i in reuse)]
    CH = 256
    for s in range(0, len(need), CH):
        idxs = need[s:s + CH]
        enc = tok([texts[i] for i in idxs], add_special_tokens=False)['input_ids']
        for i, ids in zip(idxs, enc):
            lens[i] = len(ids)
            for g in (8, 16, 32):
                gram_max[g][i] = max_gram_repeat(ids, g)
            rc, rch = line_dup_rates(texts[i])
            rate_cnts[i] = rc
            rate_chars[i] = rch
        if (s // CH) % 40 == 0:
            log('TOK', name, s, '/', len(need), 'sec', round(time.time() - t0, 1))

    for i in range(n):
        if reuse and i in reuse:
            lens[i] = reuse[i]['lens']
            for g in (8, 16, 32):
                gram_max[g][i] = reuse[i]['gm'][g]
            rate_cnts[i] = reuse[i]['rc']
            rate_chars[i] = reuse[i]['rch']
        txt = texts[i]
        if EOS and EOS in txt:
            eos_hits += 1
        uc = user_content(rows[i])
        for p in PAGE_PATS:
            if p in uc:
                page_in_hits[p] += 1
                if len(page_ex_ids[('in', p)]) < 5:
                    page_ex_ids[('in', p)].append(rows[i].get('id'))
            if p in txt:
                page_tgt_hits[p] += 1
                if len(page_ex_ids[('tgt', p)]) < 5:
                    page_ex_ids[('tgt', p)].append(rows[i].get('id'))

    est = np.asarray([273 * pages[i] + prompt_tok.get(user_content(rows[i]), 0) + lens[i] for i in range(n)],
                     dtype=np.float64)
    gt16 = int((est > 16384).sum())
    gt32 = int((est > 32768).sum())

    out = {
        'total_lines': n,
        'prompt_value_counts': {repr(k): v for k, v in prompt_counts.items()},
        'channel_counts': dict(channel_counts),
        'n_merge': len(merge_idx),
        'n_nonmerge': n - len(merge_idx),
        'merge_pages_stats': (pstats(merge_pages) if merge_pages else {}),
        'merge_pages_buckets': buckets(merge_pages),
        'all_pages_stats': pstats(pages),
        'target_tokens': pstats(lens),
        'est_total_gt_16384': {'count': gt16, 'frac': round(gt16 / n, 6) if n else 0},
        'est_total_gt_32768': {'count': gt32, 'frac': round(gt32 / n, 6) if n else 0},
        'target_eos_string_hits': eos_hits,
        'page_pattern_hits': {
            'input': dict(page_in_hits),
            'target': dict(page_tgt_hits),
            'example_ids': {('%s:%s' % k): v for k, v in page_ex_ids.items() if v},
        },
        'gram_repetition': {},
        'line_dup': {},
    }
    for g in (8, 16, 32):
        gm = np.asarray(gram_max[g], dtype=np.float64)
        d = {
            'ge2': round(float((gm >= 2).mean()), 6),
            'ge5': round(float((gm >= 5).mean()), 6),
            'ge20': round(float((gm >= 20).mean()), 6),
        }
        if g == 8:
            d.update({'p50': float(np.percentile(gm, 50)), 'p90': float(np.percentile(gm, 90)),
                      'p99': float(np.percentile(gm, 99)), 'max': float(gm.max())})
        out['gram_repetition'][g] = d
    rc = np.asarray(rate_cnts, dtype=np.float64)
    rch = np.asarray(rate_chars, dtype=np.float64)
    out['line_dup'] = {
        'count_based_ge0.2': round(float((rc >= 0.2).mean()), 6),
        'count_based_ge0.5': round(float((rc >= 0.5).mean()), 6),
        'char_weighted_ge0.2': round(float((rch >= 0.2).mean()), 6),
        'char_weighted_ge0.5': round(float((rch >= 0.5).mean()), 6),
    }
    return out, lens, gram_max, texts, rate_cnts, rate_chars

report = {'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'tokenizer_path': TOK_PATH,
          'eos_token': EOS, 'eos_token_id': tok.eos_token_id, 'datasets': {}}

log('== analyze pmc-fullce-16k ==', flush=True)
pmc_rows = ds['pmc-fullce-16k']
rep_pmc, pmc_lens, pmc_gram, pmc_texts, pmc_rc, pmc_rch = analyze('pmc-fullce-16k', pmc_rows)
report['datasets']['pmc-fullce-16k'] = rep_pmc

log('== analyze readoc-view-16k ==', flush=True)
rep_rd, rd_lens, rd_gram, rd_texts, _, _ = analyze('readoc-view-16k', ds['readoc-view-16k'])
report['datasets']['readoc-view-16k'] = rep_rd

log('== analyze mix ==', flush=True)
mix_rows = ds['pmc-readoc-spage-16k']
pmc_ids = set(r.get('id') for r in pmc_rows)
rd_ids = set(r.get('id') for r in ds['readoc-view-16k'])
mix_subset = []
for r in mix_rows:
    i = r.get('id')
    mix_subset.append('pmc' if i in pmc_ids else ('readoc' if i in rd_ids else 'spage'))
log('MIX_SUBSET_COUNTS', dict(Counter(mix_subset)))

pmc_meta = [(r.get('meta') or {}) for r in pmc_rows]
pmc_sha = {r.get('id'): (i, pmc_meta[i].get('target_sha256')) for i, r in enumerate(pmc_rows)}
reuse = {}
for j, r in enumerate(mix_rows):
    if mix_subset[j] != 'pmc':
        continue
    ent = pmc_sha.get(r.get('id'))
    if ent is None:
        continue
    pi, sha = ent
    if sha and hashlib.sha256(asst_content(r).encode()).hexdigest() == sha:
        reuse[j] = {'lens': pmc_lens[pi], 'gm': {g: pmc_gram[g][pi] for g in (8, 16, 32)},
                    'rc': pmc_rc[pi], 'rch': pmc_rch[pi]}
log('MIX_REUSE_PMC_ROWS', len(reuse), 'sec', round(time.time() - t0, 1))

rep_mix, mix_lens, mix_gram, _, _, _ = analyze('pmc-readoc-spage-16k', mix_rows, reuse=reuse)
report['datasets']['pmc-readoc-spage-16k'] = rep_mix

mix_sub_report = {}
for sub in ('pmc', 'spage', 'readoc'):
    idxs = [j for j, s in enumerate(mix_subset) if s == sub]
    mps = [len(mix_rows[j].get('images') or []) for j in idxs
           if MERGE_KEY in user_content(mix_rows[j])]
    mix_sub_report[sub] = {
        'n_rows': len(idxs),
        'n_merge': len(mps),
        'merge_pages_stats': pstats(mps) if mps else {},
        'merge_pages_buckets': buckets(mps),
        'target_tokens': pstats([mix_lens[j] for j in idxs]),
    }
report['mix_subsets'] = mix_sub_report

st = [(name, tid) for name, tid in tok.added_tokens_encoder.items()
      if any(k in name.lower() for k in ('page', 'image', 'ref', 'det'))]
report['special_tokens_page_image_ref_det'] = {'count': len(st), 'items': [[n, int(t)] for n, t in st[:300]]}

log('== reconcile ==', flush=True)
random.seed(1234)
sample_idx = random.sample(range(len(pmc_rows)), 300)
sha_mis = tok_mis = pt_mis = vt_mis = tt_mis = npg_mis = fits16_false = 0
for i in sample_idx:
    m = pmc_meta[i]
    if str(m.get('target_sha256')) != hashlib.sha256(pmc_texts[i].encode()).hexdigest():
        sha_mis += 1
    if int(m.get('target_tokens', -1)) != pmc_lens[i]:
        tok_mis += 1
    if int(m.get('prompt_tokens', -1)) != prompt_tok.get(user_content(pmc_rows[i]), -1):
        pt_mis += 1
    if int(m.get('visual_tokens', -1)) != 273 * len(pmc_rows[i].get('images') or []):
        vt_mis += 1
    calc = int(m.get('prompt_tokens', 0)) + int(m.get('target_tokens', 0)) + int(m.get('visual_tokens', 0))
    if int(m.get('total_tokens', -1)) != calc:
        tt_mis += 1
    if m.get('n_pages') is not None and int(m.get('n_pages', -1)) != len(pmc_rows[i].get('images') or []):
        npg_mis += 1
    if not m.get('fits_16k', True):
        fits16_false += 1
report['reconciliation_meta_300'] = {
    'n': 300, 'seed': 1234,
    'target_sha256_mismatch': sha_mis,
    'target_tokens_mismatch': tok_mis,
    'prompt_tokens_mismatch': pt_mis,
    'visual_tokens_vs_273xpages_mismatch': vt_mis,
    'total_tokens_sum_mismatch': tt_mis,
    'n_pages_vs_len_images_mismatch': npg_mis,
    'fits_16k_false_rows_in_sample': fits16_false,
}
id2pmc = {r.get('id'): i for i, r in enumerate(pmc_rows)}
random.seed(5678)
msample = random.sample(manifest, 300)
m_missing = m_tok_mis = m_sha_mis = 0
for mr in msample:
    i = id2pmc.get(mr.get('id'))
    if i is None:
        m_missing += 1
        continue
    if int(mr.get('target_tokens', -1)) != pmc_lens[i]:
        m_tok_mis += 1
    if str(mr.get('target_sha256')) != hashlib.sha256(pmc_texts[i].encode()).hexdigest():
        m_sha_mis += 1
report['reconciliation_manifest_300'] = {
    'n': 300, 'seed': 5678, 'manifest_lines': len(manifest),
    'ids_missing_in_train': m_missing,
    'target_tokens_mismatch': m_tok_mis,
    'target_sha256_mismatch': m_sha_mis,
}

log('== samples ==', flush=True)
def merge_indices(rows):
    return [i for i in range(len(rows)) if MERGE_KEY in user_content(rows[i])]

def sample_row(rows, lens, texts, pick):
    mi = merge_indices(rows)
    pages = [len(rows[i].get('images') or []) for i in mi]
    med = float(np.median(pages))
    if pick == 'median':
        cand = [i for i in mi if len(rows[i].get('images') or []) == int(med)]
        if not cand:
            cand = sorted(mi, key=lambda i: abs(len(rows[i].get('images') or []) - med))[:1]
        i = cand[0]
    else:
        i = max(range(len(rows)), key=lambda j: lens[j])
    txt = texts[i]
    ids = tok(txt, add_special_tokens=False)['input_ids']
    L = len(txt)
    return {
        '_which': None,
        'id': rows[i].get('id'),
        'pages': len(rows[i].get('images') or []),
        'prompt': user_content(rows[i]),
        'target_tokens_measured': len(ids),
        'target_tokens_meta': (rows[i].get('meta') or {}).get('target_tokens'),
        'target_chars': L,
        'target_first_500': txt[:500],
        'target_mid_500': txt[max(0, L // 2 - 250): L // 2 + 250],
        'target_last_500': txt[-500:],
        'last_12_token_ids': ids[-12:],
        'roundtrip_decode200_eq': bool(tok.decode(ids)[:200] == txt[:200]),
    }

s1 = sample_row(pmc_rows, pmc_lens, pmc_texts, 'median'); s1['_which'] = 'pmc-fullce-16k median-pages merge'
s2 = sample_row(pmc_rows, pmc_lens, pmc_texts, 'longest'); s2['_which'] = 'pmc-fullce-16k longest-target'
s3 = sample_row(ds['readoc-view-16k'], rd_lens, rd_texts, 'median'); s3['_which'] = 'readoc-view-16k median-pages merge'
samples = [s1, s2, s3]
report['samples'] = samples

with open(OUT + '/data_audit_samples.txt', 'w', encoding='utf-8') as f:
    for s in samples:
        f.write('=' * 80 + '\n')
        for k, v in s.items():
            if isinstance(v, str) and len(v) > 2000:
                v = v[:2000] + '...[TRUNC]'
            f.write('%s: %s\n' % (k, v))
        f.write('\n')

log('== swift suffix ==', flush=True)
try:
    if SWIFT_REPO not in sys.path:
        sys.path.insert(0, SWIFT_REPO)
    from swift.template.register import TEMPLATE_MAPPING
    tpl = TEMPLATE_MAPPING['unlimited_ocr']
    suf = list(tpl.meta.suffix) if tpl.meta.suffix is not None else None
    swift_info = {
        'suffix_repr': repr(suf),
        'suffix_len': len(suf) if suf is not None else None,
        'equals_eos_only': (suf == [EOS]),
        'eos_repr': repr(EOS),
        'ocr_template_keys': [k for k in TEMPLATE_MAPPING.keys() if 'ocr' in k.lower()][:10],
    }
except Exception as e:
    swift_info = {'error': '%s: %s' % (type(e).__name__, str(e)[:300])}
report['swift_template_suffix'] = swift_info

with open(OUT + '/data_audit_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=1, default=str)
log('WROTE', OUT + '/data_audit_report.json', OUT + '/data_audit_samples.txt')
log('TOTAL_SEC', round(time.time() - t0, 1))

for name in ('pmc-fullce-16k', 'pmc-readoc-spage-16k', 'readoc-view-16k'):
    d = report['datasets'][name]
    log('---', name)
    log('lines', d['total_lines'], 'merge', d['n_merge'], 'channels', d['channel_counts'])
    log('prompts', d['prompt_value_counts'])
    log('merge_pages', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d['merge_pages_stats'].items()},
        d['merge_pages_buckets'])
    tt = d['target_tokens']
    log('target_tok p50/p90/p95/max', round(tt['p50'], 1), round(tt['p90'], 1), round(tt['p95'], 1), round(tt['max'], 1))
    log('est>16384', d['est_total_gt_16384'], 'est>32768', d['est_total_gt_32768'])
    log('gram', d['gram_repetition'])
    log('line_dup', d['line_dup'])
    log('eos_in_target_hits', d['target_eos_string_hits'], 'page_in', d['page_pattern_hits']['input'],
        'page_tgt', d['page_pattern_hits']['target'])
for sub, v in report['mix_subsets'].items():
    log('mix_sub', sub, 'rows', v['n_rows'], 'merge', v['n_merge'], 'pages',
        {k: (round(x, 2) if isinstance(x, float) else x) for k, x in v['merge_pages_stats'].items()},
        v['merge_pages_buckets'])
log('recon_meta_300', report['reconciliation_meta_300'])
log('recon_manifest_300', report['reconciliation_manifest_300'])
log('special_tokens', report['special_tokens_page_image_ref_det']['count'],
    [n for n, _ in report['special_tokens_page_image_ref_det']['items']][:40])
for s in samples:
    log('SAMPLE', s['_which'], '| id', s['id'], '| pages', s['pages'], '| prompt', repr(s['prompt']),
        '| tok', s['target_tokens_measured'], '| roundtrip', s['roundtrip_decode200_eq'])
    log('  head:', repr(s['target_first_500'][:250]))
    log('  tail:', repr(s['target_last_500'][-250:]))
log('swift_suffix', swift_info)
log('DONE')
