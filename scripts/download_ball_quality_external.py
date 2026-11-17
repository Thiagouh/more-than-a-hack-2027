"""Download the three predeclared MIT SkillCorner matches, pinned and hash-checked."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from scripts.evaluate_ball_quality_external import MATCHES,SOURCE_COMMIT,sha


def fetch_match(mid,out):
    if mid not in MATCHES:raise ValueError('Match is outside the frozen protocol')
    folder=Path(out)/mid;folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'source.json'
    if manifest.exists():
        source=json.loads(manifest.read_text())
        if (source.get('commit')==SOURCE_COMMIT and source.get('match_id')==mid
                and all((folder/name).is_file() and sha(folder/name)==source.get(key)
                        for name,key in [('match.json','metadata_sha256'),('tracking.jsonl','tracking_sha256')])):
            return source
    base=f'https://raw.githubusercontent.com/SkillCorner/opendata/{SOURCE_COMMIT}/data/matches/{mid}/'
    with urllib.request.urlopen(base+mid+'_match.json',timeout=60) as response:metadata=response.read()
    if str(json.loads(metadata)['id'])!=mid:raise ValueError('Downloaded match metadata differs')
    with urllib.request.urlopen(base+mid+'_tracking_extrapolated.jsonl',timeout=60) as response:
        pointer=response.read().decode()
    oid=re.search(r'^oid sha256:([0-9a-f]{64})$',pointer,re.M)
    size=re.search(r'^size (\d+)$',pointer,re.M)
    if not oid or not size:raise ValueError('Expected pinned Git LFS pointer')
    url=f'https://media.githubusercontent.com/media/SkillCorner/opendata/{SOURCE_COMMIT}/data/matches/{mid}/{mid}_tracking_extrapolated.jsonl'
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=folder,delete=False) as stream:
            temporary=Path(stream.name)
            with urllib.request.urlopen(url,timeout=60) as response:
                while block:=response.read(1024*1024):stream.write(block)
        if temporary.stat().st_size!=int(size.group(1)) or sha(temporary)!=oid.group(1):
            raise ValueError('Downloaded tracking differs from the source object')
        temporary.replace(folder/'tracking.jsonl')
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
    (folder/'match.json').write_bytes(metadata)
    source=dict(match_id=mid,repository='https://github.com/SkillCorner/opendata',commit=SOURCE_COMMIT,
        license='MIT',credit='SkillCorner',tracking_sha256=oid.group(1),tracking_bytes=int(size.group(1)),
        metadata_sha256=hashlib.sha256(metadata).hexdigest(),tracking_url=url)
    manifest.write_text(json.dumps(source,indent=2)+'\n')
    return source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool:
        for source in pool.map(lambda mid:fetch_match(mid,args.out_dir),MATCHES):
            print(json.dumps(source),flush=True)


if __name__=='__main__':main()
