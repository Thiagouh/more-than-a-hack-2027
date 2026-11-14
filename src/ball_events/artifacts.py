"""Atomic local outputs, early audit, bounded run storage and best-effort locked cache."""
from contextlib import contextmanager, nullcontext
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

from . import contract as c
from .audit import Attempt
from .detector import Detector
from .events import consume
from .storage import RunStore
from .visual import render_svg, event_table

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = {'features.csv', 'events.jsonl', 'quality.json', 'context_hints.json', 'consumer_check.json',
             'events.svg', 'events-table.csv', 'sources.csv', 'deliveries.csv'}
OFFICIAL = {
    'evaluation/possession_metrics.py': '341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d',
    'evaluation/run_evaluation.py': '1c61564fe52499ba4af3006fb5eb4147b825e92bf516651add80de49d267b19f',
    'evaluation/validate_submission.py': '07ac29499acbd337e1dc1de1832d63f6e1fa1d981e22f6ad08212529e2bbee14',
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def write_json(path, value):
    Path(path).write_text(canonical(value)+'\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def identity(inputs, meta, cfg, hints, hints_status='not_provided'):
    files = sorted(set((REPO/'src').rglob('*.py')) | set((REPO/'starter').glob('*.py'))
                   | set((REPO/'evaluation').glob('*.py')) | set((REPO/'scripts').glob('*.py')) | {REPO/'games.py'}
                   | set((REPO/'schemas').glob('*.json')))
    code = {str(p.relative_to(REPO)): sha(p) for p in files}
    deps = {}
    for name in ('numpy','pandas','kloppy'):
        try: deps[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: deps[name] = None
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError: commit = None
    return {'identity_version': 'ball-events-run-v2', 'contract_version': c.EVENT_VERSION,
            'input_sha256': {k: sha(p) for k,p in sorted(inputs.items())}, 'code_sha256': code,
            'commit': commit, 'dependencies': deps, 'python': platform.python_version(),
            'input_metadata': meta, 'configuration': cfg, 'hints': hints, 'hints_status': hints_status}


def _cache_valid(folder, key):
    try:
        folder = Path(folder)
        if folder.is_symlink() or not folder.is_dir(): return False
        mpath = folder/'manifest.json'
        if mpath.is_symlink(): return False
        data = json.loads(mpath.read_text(encoding='utf-8'))
        if not isinstance(data,dict) or data.get('cache_key') != key or data.get('contract_version') != c.EVENT_VERSION: return False
        provenance=data.get('provenance')
        if not isinstance(provenance,dict) or hashlib.sha256(canonical(provenance).encode()).hexdigest()!=key: return False
        hashes = data.get('artifact_sha256')
        if not isinstance(hashes,dict) or set(hashes) != ARTIFACTS: return False
        if any(not isinstance(v,str) or re.fullmatch('[0-9a-f]{64}',v) is None for v in hashes.values()): return False
        return all((folder/name).is_file() and not (folder/name).is_symlink() and sha(folder/name)==digest for name,digest in hashes.items())
    except (OSError, ValueError, TypeError, UnicodeError): return False


def overlap(a,b):
    a,b = Path(a).resolve(),Path(b).resolve()
    return a==b or a in b.parents or b in a.parents


def validate_destinations(out, cache_dir, inputs):
    out = Path(out).resolve()
    if out.exists(): raise c.InputError('Output already exists; choose a new run directory')
    if any(overlap(out,p) for p in inputs.values()):
        raise c.InputError('Output overlaps input')
    if cache_dir is not None:
        cache_dir = Path(cache_dir).resolve()
        if any(overlap(cache_dir,p) for p in [out,*inputs.values()]):
            raise c.InputError('Cache overlaps input or output')
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            if not cache_dir.is_dir(): raise OSError()
            with tempfile.TemporaryFile(dir=cache_dir): pass
        except OSError:
            raise c.InputError('Cache destination is not a writable directory') from None
    return out, cache_dir


@contextmanager
def cache_lock(root,key):
    # POSIX OS locks are automatically released on process exit; no stale-lock deletion.
    import fcntl
    with (root/(key+'.lock')).open('a+b') as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def cache_read(root,key,stage):
    try:
        with cache_lock(root,key):
            entry = root/key
            if not _cache_valid(entry,key): return False
            for name in sorted(ARTIFACTS): shutil.copyfile(entry/name,stage/name)
            # A content change during copy is a miss, never an accepted partial cache.
            m = json.loads((entry/'manifest.json').read_text())
            if not all(sha(stage/name)==digest for name,digest in m['artifact_sha256'].items()):
                raise OSError('Cache copy verification failed')
            return True
    except (OSError,ValueError,TypeError,KeyError):
        for name in ARTIFACTS:
            (stage/name).unlink(missing_ok=True)
        return False


def cache_write(root,key,out):
    with cache_lock(root,key):
        entry = root/key
        if _cache_valid(entry,key): return 'existing_valid'
        with tempfile.TemporaryDirectory(prefix='.cache-',dir=root) as tmp:
            target = Path(tmp)/'entry'
            shutil.copytree(out,target)
            if not _cache_valid(target,key): raise OSError('Cache verification failed')
            if entry.exists() or entry.is_symlink():
                if entry.is_symlink() or not entry.is_dir(): entry.unlink()
                else: shutil.rmtree(entry)  # invalid entry only, while holding its OS lock
            target.rename(entry)
    return 'written'


def disk_consumer(store,path,drain=False):
    """Disk queue ordered by availability. Validate predecessors on actual release."""
    db = store.db
    db.execute('CREATE TABLE delivered (id TEXT PRIMARY KEY, at_frame INTEGER)')
    received = pending = retrospective = 0
    with path.open('w',newline='',encoding='utf-8') as h:
        w=csv.writer(h,lineterminator='\n')
        w.writerow(['event_id','parent_event_id','available_at_frame','delivered_at_frame','event_frame','retrospective','status'])
        for e in store.events():
            received += 1
            at = e['available_at_frame']
            parent = e['parent_event_id']
            if parent:
                row=db.execute('SELECT at_frame FROM delivered WHERE id=?',(parent,)).fetchone()
                if row is None or row[0]>at: raise c.InputError('Queue released loss before acquisition')
            db.execute('INSERT INTO delivered VALUES (?,?)',(e['event_id'],at))
            outside = at >= store.meta['n_frames']
            pending += outside
            retrospective += at>e['event_frame']
            w.writerow([e['event_id'],parent,at,'' if outside and not drain else at,e['event_frame'],at>e['event_frame'],'pending_at_end' if outside and not drain else 'drained' if outside else 'delivered'])
    return dict(contract_valid=True,timeline_metadata_supplied=True,received_events=received,
                delivered_within_timeline=received-pending,pending_at_end_count=pending,
                pending_details='deliveries.csv',retrospective_events=retrospective,drained=drain,
                delivery_order_valid=True,produced_possession_labels=False,real_front4_integration=False)


def process(stage,tmp,meta,factory,hints,overrides,hints_status):
    store=RunStore(tmp/'run.sqlite',meta)
    try:
        detector=Detector(meta,overrides,hints,track_registry=store.registry)
        with (stage/'features.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=None
            for frame in factory():
                feature,events=detector.feed(frame)
                if writer is None:
                    writer=csv.DictWriter(handle,fieldnames=list(feature),lineterminator='\n')
                    writer.writeheader()
                writer.writerow({k:canonical(v) if isinstance(v,(list,dict)) else v for k,v in feature.items()})
                for e in events: store.add_event(e)
                store.source(feature['frame'],feature['ball_source'])
        quality=detector.finish()
        quality['hints_status']=hints_status
        with (stage/'events.jsonl').open('w',encoding='utf-8') as handle:
            for e in store.events(): handle.write(canonical(e)+'\n')
        write_json(stage/'quality.json',quality)
        write_json(stage/'context_hints.json',detector.hints)
        write_json(stage/'consumer_check.json',disk_consumer(store,stage/'deliveries.csv'))
        with (stage/'sources.csv').open('w',newline='',encoding='utf-8') as handle:
            w=csv.writer(handle,lineterminator='\n'); w.writerow(['source','start_frame','end_frame'])
            for row in store.runs(): w.writerow(row)
        with (stage/'events.svg').open('w',encoding='utf-8') as handle:
            render_svg(handle,store.events(),store.runs(),meta)
        with (stage/'events-table.csv').open('w',newline='',encoding='utf-8') as handle:
            event_table(handle,store.events(),meta)
    finally: store.close()


def run(meta, frames_factory, hints, inputs, out, overrides=None, hypothesis=None,
        hints_status='not_provided', cache_dir=None, command=None, attempt=None):
    context = nullcontext(attempt) if attempt is not None else Attempt(out,command,hypothesis)
    with context as audit:
        if not isinstance(hypothesis,str) or not hypothesis.strip(): raise c.InputError('A hypothesis is required before processing')
        c.manifest(meta)
        c.obj(inputs,'input paths')
        if not isinstance(hints_status,str) or hints_status not in ('not_provided','provided','optional_file_missing'):
            raise c.InputError('Invalid hints_status')
        cfg=c.config(overrides,meta['fps'])
        # Validate hints before hashing or opening cache.
        Detector(meta,overrides,hints)
        out,cache_root=validate_destinations(out,cache_dir,inputs)
        provenance=identity(inputs,meta,cfg,hints,hints_status)
        for p,digest in OFFICIAL.items():
            if sha(REPO/p)!=digest: raise c.InputError('Official evaluator changed; review before running')
        key=hashlib.sha256(canonical(provenance).encode()).hexdigest()
        audit.update(identity=provenance,cache_key=key,stage='processing',cache_status='miss' if cache_root else 'disabled')
        out.parent.mkdir(parents=True,exist_ok=True)
        experiment={'hypothesis':hypothesis,'command':command or sys.argv,'identity':provenance,'cache_key':key,
                    'limitations':['heuristics_not_calibrated','no_match_accuracy_claim','no_final_possession_labels']}
        with tempfile.TemporaryDirectory(prefix='.frente5-',dir=out.parent) as temp:
            tmp=Path(temp); stage=tmp/'result'; stage.mkdir()
            write_json(stage/'experiment.json',experiment)
            hit=cache_read(cache_root,key,stage) if cache_root else False
            if not hit: process(stage,tmp,meta,frames_factory,hints,overrides,hints_status)
            if identity(inputs,meta,cfg,hints,hints_status)!=provenance:
                raise c.InputError('Inputs or code changed during processing; rerun')
            hashes={name:sha(stage/name) for name in sorted(ARTIFACTS)}
            write_json(stage/'manifest.json',{'contract_version':c.EVENT_VERSION,'cache_key':key,
                                            'provenance':provenance,'artifact_sha256':hashes})
            audit.update(stage='publication')
            stage.rename(out)
        cache_status='hit' if hit else 'disabled' if cache_root is None else 'miss'
        warning=None
        if cache_root and not hit:
            try: cache_status=cache_write(cache_root,key,out)
            except OSError:
                cache_status='write_failed'; warning='Output completed; optional cache write failed'
        audit.update(stage='published',cache_status=cache_status,output_manifest_sha256=sha(out/'manifest.json'))
        return dict(cache_hit=hit,cache_status=cache_status,cache_key=key,output=str(out),warning=warning)
