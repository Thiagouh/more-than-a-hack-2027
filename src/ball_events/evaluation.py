"""Partial event reference evaluation: maximum cardinality, minimum timing error, exact boundaries."""
from bisect import bisect_left, bisect_right
from contextlib import nullcontext
from fractions import Fraction
import heapq
from pathlib import Path
import sys
import tempfile

from . import contract as c
from .artifacts import identity, write_json
from .audit import Attempt
from .events import validate_events

MATCHING_VERSION='max-cardinality-min-absolute-error-v2'


def matching(pred,refs,fps,tolerance):
    """Sparse windows + connected components + successive shortest augmenting paths.

    Fraction costs preserve decimal tolerance boundaries. Canonical IDs order all nodes/edges.
    """
    pairs=[]
    tol=Fraction(str(tolerance)); rate=Fraction(str(fps))
    for team in ('H','A'):
        ps=sorted((p for p in pred if p['team']==team),key=lambda p:p['event_id'])
        rs=sorted((r for r in refs if r['team']==team),key=lambda r:(Fraction(str(r['time_s'])),r['reference_id']))
        times=[Fraction(str(r['time_s'])) for r in rs]
        edges=[]
        parents=list(range(len(ps)+len(rs)))
        def root(a):
            while parents[a]!=a:
                parents[a]=parents[parents[a]]; a=parents[a]
            return a
        for i,p in enumerate(ps):
            time=Fraction(p['event_frame'])/rate
            for j in range(bisect_left(times,time-tol),bisect_right(times,time+tol)):
                edges.append((i,j,abs(time-times[j])))
                a,b=root(i),root(len(ps)+j)
                if a!=b: parents[b]=a
        components={}
        for edge in edges: components.setdefault(root(edge[0]),[]).append(edge)
        for group in components.values():
            pids=sorted({i for i,_,_ in group},key=lambda i:ps[i]['event_id'])
            rids=sorted({j for _,j,_ in group},key=lambda j:rs[j]['reference_id'])
            pn={p:i+1 for i,p in enumerate(pids)}
            rn={r:i+1+len(pids) for i,r in enumerate(rids)}
            sink=len(pids)+len(rids)+1
            graph=[[] for _ in range(sink+1)]
            def add(a,b,cost):
                forward=[b,len(graph[b]),1,cost]
                reverse=[a,len(graph[a]),0,-cost]
                graph[a].append(forward); graph[b].append(reverse)
                return forward
            for i in pids: add(0,pn[i],Fraction(0))
            tagged=[]
            for i,j,cost in sorted(group,key=lambda x:(ps[x[0]]['event_id'],rs[x[1]]['reference_id'])):
                tagged.append((i,j,add(pn[i],rn[j],cost)))
            for j in rids: add(rn[j],sink,Fraction(0))
            potential=[Fraction(0)]*len(graph)
            while True:
                dist=[None]*len(graph); prev=[None]*len(graph); dist[0]=Fraction(0)
                heap=[(Fraction(0),0)]
                while heap:
                    d,a=heapq.heappop(heap)
                    if d!=dist[a]: continue
                    for k,edge in enumerate(graph[a]):
                        b,_,cap,cost=edge
                        if not cap: continue
                        nd=d+cost+potential[a]-potential[b]
                        if dist[b] is None or nd<dist[b]:
                            dist[b]=nd; prev[b]=(a,k); heapq.heappush(heap,(nd,b))
                if dist[sink] is None: break
                for i,d in enumerate(dist):
                    if d is not None: potential[i]+=d
                b=sink
                while b:
                    a,k=prev[b]; edge=graph[a][k]
                    edge[2]-=1; graph[b][edge[1]][2]+=1; b=a
            for i,j,edge in tagged:
                if edge[2]==0: pairs.append((ps[i],rs[j]))
    return sorted(pairs,key=lambda pair:(pair[1]['reference_id'],pair[0]['event_id']))


def stats(values,total):
    values=sorted(values)
    def quantile(q):
        if not values: return None
        position=(len(values)-1)*q; lo=int(position); hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(position-lo)
    return dict(count=len(values),coverage=len(values)/total if total else None,
                mean=sum(values)/len(values) if values else None,median=quantile(.5),
                p05=quantile(.05),p90=quantile(.9),p95=quantile(.95))


def evaluate_events(predictions,reference,on_validated=None):
    c.fields(predictions,{'fps','events','manifest'},{'fps','events','manifest'},'predictions')
    c.fields(reference,{'metadata','events'},{'metadata','events'},'reference')
    meta=c.manifest(predictions['manifest'])
    fps=c.number(predictions['fps'],'prediction fps',True)
    if Fraction(str(fps))!=Fraction(str(meta['fps'])): raise c.InputError('Prediction fps disagrees with manifest')
    validate_events(predictions['events'],meta)
    refmeta=reference['metadata']
    c.fields(refmeta,{'provenance','independence','timing','extensions'}, {'provenance','independence','timing'},'reference metadata')
    for k in ('provenance','independence','timing'): c.text(refmeta[k],k)
    refs=c.array(reference['events'],'reference events')
    ids=set()
    for r in refs:
        c.fields(r,{'reference_id','type','team','time_s','extensions'}, {'reference_id','type','team','time_s'},'reference event')
        c.text(r['reference_id'],'reference_id')
        if r['reference_id'] in ids: raise c.InputError('Duplicate reference ID')
        ids.add(r['reference_id'])
        if r['type'] not in ('pass','shot') or r['team'] not in ('H','A'): raise c.InputError('Reference supports pass/shot with H/A')
        c.number(r['time_s'],'reference time_s')
        if r['time_s']<0: raise c.InputError('Reference time must be nonnegative')
    if on_validated is not None: on_validated()
    pred=[p for p in predictions['events'] if p['type']=='possible_touch']
    results={}
    rate=Fraction(str(fps))
    for tol in (.2,.5,1.):
        pairs=matching(pred,refs,fps,tol)
        def summarize(selected,reference_subset):
            rows=[]
            for p,r in selected:
                event_error=float(Fraction(p['event_frame'])/rate-Fraction(str(r['time_s'])))
                rows.append(dict(event_id=p['event_id'],reference_id=r['reference_id'],reference_type=r['type'],
                                 event_time_error_s=event_error,absolute_event_time_error_s=abs(event_error),
                                 confirmation_delay_s=float(Fraction(p['confirmed_at_frame']-p['event_frame'])/rate),
                                 availability_delay_s=float(Fraction(p['available_at_frame']-p['event_frame'])/rate),
                                 delivery_time_error_s=float(Fraction(p['available_at_frame'])/rate-Fraction(str(r['time_s'])))))
            metrics={k:stats([r[k] for r in rows],len(rows)) for k in
                     ('event_time_error_s','absolute_event_time_error_s','confirmation_delay_s','availability_delay_s','delivery_time_error_s')}
            covered={r['reference_id'] for _,r in selected}
            return dict(matched=len(rows),reference_events=len(reference_subset),
                        reference_recall=len(rows)/len(reference_subset) if reference_subset else None,
                        metrics=metrics,pairs=rows,uncovered_reference_ids=sorted(r['reference_id'] for r in reference_subset if r['reference_id'] not in covered))
        result=summarize(pairs,refs)
        used={p['event_id'] for p,_ in pairs}
        result['unreferenced_prediction_ids']=sorted(p['event_id'] for p in pred if p['event_id'] not in used)
        result['unreferenced_predictions']=len(result['unreferenced_prediction_ids'])
        result['by_reference_type']={kind:summarize([(p,r) for p,r in pairs if r['type']==kind],[r for r in refs if r['type']==kind]) for kind in ('pass','shot')}
        results[str(tol)]=result
    return dict(report_version='v2',matching_version=MATCHING_VERSION,reference_metadata=refmeta,tolerances_s=results,
                quantile_definition='linear interpolation at (n-1)*q',precision_defined=False,
                warning='Partial reference; unmatched predictions are not false positives. No continuous control validation.')


def report_csv(path,rows,fps,out,attempt=None):
    context=nullcontext(attempt) if attempt is not None else Attempt(out,['report-csv'])
    with context as audit:
        if rows is not None: c.integer(rows,'rows',1)
        c.number(fps,'fps',True)
        out=Path(out)
        if out.exists(): raise c.InputError('Output already exists')
        evaluation=Path(__file__).resolve().parents[2]/'evaluation'
        if str(evaluation) not in sys.path: sys.path.insert(0,str(evaluation))
        from comparar_csv import load,summarize
        provenance=identity({'csv':Path(path)}, {}, {'rows':rows,'fps':fps}, [], 'not_provided')
        audit.update(identity=provenance)
        codes=load(path,rows)
        audit.update(stage='processing')
        result=summarize(codes,fps)
        result.update(valid=True,expected_rows=rows,official_row_count_checked=rows==199319,
                      coverage={'first_frame':0,'last_frame':len(codes)-1},quality_measured=False,
                      provenance=provenance,report_version='v2')
        out.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out.parent,prefix='.report-') as tmp:
            stage=Path(tmp)/'report'; stage.mkdir(); write_json(stage/'report.json',result)
            audit.update(stage='publication'); stage.rename(out)
        audit.update(stage='published')
        return result
