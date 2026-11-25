"""Recompute all reported numbers from saved predictions."""
import json, csv, glob, statistics as st, sys
if len(sys.argv)!=7:
    sys.exit("usage: python verify.py RUN_DIR x6  (order: SmolVLM2 GIT Qwen2-VL BLIP InternVL3 Gemma)")
runs=dict(zip(['SmolVLM2','GIT','Qwen2-VL','BLIP','InternVL3','Gemma3'],sys.argv[1:]))
p=lambda a,b:round(100*a/b,2)
ref=None
for n,g in runs.items():
    d=glob.glob(g)[0]; a=glob.glob(d+'/analysis_*')[0]
    P=[json.loads(l) for l in open(d+'/predictions.jsonl')]
    I={r['image_id']:r for r in csv.DictReader(open(a+'/image_details.csv'))}
    assert len(P)==200 and len({x['image_id'] for x in P})==200 and set(I)=={x['image_id'] for x in P}
    gt={k:v['ground_truth'] for k,v in I.items()}
    if ref is None: ref=gt
    assert gt==ref
    pr={x['image_id']:x['parsed_answer'] for x in P}
    assert all(pr[k]==I[k]['prediction'] for k in I), n
    assert set(pr.values())<={'yes','no'}
    c={k:pr[k]==gt[k] for k in gt}
    conf=[sum(gt[k]==a_ and pr[k]==b for k in gt) for a_ in('yes','no') for b in('yes','no')]
    groups={}
    for k,v in I.items(): groups.setdefault(v['group_id'],{})[v['variant']]=k
    assert len(groups)==40
    sens=sp=bc=sb=inv=iacc=0
    for gr in groups.values():
        b=gr['base']; s=gr['spatial']
        assert gt[b]!=gt[s]
        sens+=pr[b]!=pr[s]; sp+=c[b] and c[s]; bc+=c[b]; sb+=c[b] and c[s]
        for v in('brightness','clutter','occlusion'):
            assert gt[gr[v]]==gt[b]
            inv+=pr[b]==pr[gr[v]]; iacc+=c[b] and c[gr[v]]
    rel={r:[c[k] for k in I if I[k]['relation']==r] for r in('left','right','above','below','inside','outside')}
    var={v:sum(c[k] for k in I if I[k]['variant']==v) for v in('base','spatial','brightness','clutter','occlusion')}
    t=[x['seconds'] for x in P]
    it=[x['input_tokens'] for x in P]
    shapes={str(x['tensor_shapes'].get('pixel_values')) for x in P}
    print(f"== {n}\n acc {p(sum(c.values()),200)} correct {sum(c.values())} conf(yy,yn,ny,nn) {conf} predyes {conf[0]+conf[2]}")
    print(f" sens {sens}/40={p(sens,40)} spair {sp}/40 | cond {sb}/{bc}={p(sb,bc)} | inv {inv}/120={p(inv,120)} invacc {iacc}/120={p(iacc,120)}")
    print(" rel", {r:(sum(v),len(v),p(sum(v),len(v))) for r,v in rel.items()})
    print(" var(/40)", var)
    print(f" time all mean {st.mean(t):.4f} med {st.median(t):.4f} | excl first mean {st.mean(t[1:]):.4f} med {st.median(t[1:]):.4f} first {t[0]:.3f} total {sum(t):.1f}")
    print(f" tokens {st.mean(it)} shapes {shapes}")
    mm={r['metric']:r for r in csv.DictReader(open(a+'/metrics.csv'))}
    for k in('image_accuracy','spatial_sensitivity','prediction_invariance','invariant_pair_accuracy','spatial_pair_accuracy','sensitivity_given_correct_base'):
        r=mm[k]; print(f"  saved {k}: {100*float(r['value']):.2f} [{100*float(r['ci_low']):.2f},{100*float(r['ci_high']):.2f}]")

print("==== per relation: predicted yes / images ; correct spatial pairs")
for n,g in runs.items():
    d=glob.glob(g)[0]; a=glob.glob(d+'/analysis_*')[0]
    I=list(csv.DictReader(open(a+'/image_details.csv')))
    by={}
    for r in I: by.setdefault(r['group_id'],{})[r['variant']]=r
    out=[]
    for rel in('left','right','above','below','inside','outside'):
        s=[r for r in I if r['relation']==rel]
        gs=[v for v in by.values() if v['base']['relation']==rel]
        sp=sum(v['base']['correct']=='True' and v['spatial']['correct']=='True' for v in gs)
        out.append(f"{rel} yes {sum(r['prediction']=='yes' for r in s)}/{len(s)} sp {sp}/{len(gs)}")
    err=[(r['relation'],r['variant'],r['ground_truth']) for r in I if r['correct']!='True']
    print(n,' | '.join(out))
    if n=='Qwen2-VL': print(err)
