"""Local event evidence CLI. No match data is transmitted and no possession labels are generated."""
import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

from . import contract as c
from .artifacts import run, write_json, identity, disk_consumer
from .audit import Attempt
from .demo import write_inputs
from .evaluation import evaluate_events, report_csv, MATCHING_VERSION
from .inputs import atd, read_json, standardized
from .storage import RunStore
from .visual import render_svg, event_table


def jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                try: yield json.loads(line)
                except ValueError: raise c.InputError('Malformed JSONL record') from None


def run_meta(path):
    manifest=read_json(path)
    c.obj(manifest,'run manifest')
    c.obj(manifest.get('provenance'),'run provenance')
    return c.manifest(manifest['provenance'].get('input_metadata'))


def view(args,audit):
    root=args.run
    meta=run_meta(root/'manifest.json')
    if args.out.exists(): raise c.InputError('Output already exists')
    inputs={'manifest':root/'manifest.json','events':root/'events.jsonl','sources':root/'sources.csv'}
    options={'start':args.start,'end':args.end} if args.command=='render' else {'drain':args.drain}
    provenance=identity(inputs,meta,options,[])
    audit.update(identity=provenance,stage='processing')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent,prefix='.view-') as temp:
        tmp=Path(temp); stage=tmp/'result'; stage.mkdir()
        store=RunStore(tmp/'events.sqlite',meta)
        try:
            # File is in delivery order; causal predecessors necessarily precede children.
            for e in jsonl(root/'events.jsonl'): store.add_event(e)
            if args.command=='render':
                def sources():
                    with (root/'sources.csv').open(newline='') as h:
                        for row in csv.DictReader(h):
                            source=row.get('source')
                            if source not in c.SOURCES: raise c.InputError('Invalid source overview record')
                            try: a,b=int(row['start_frame']),int(row['end_frame'])
                            except (ValueError,KeyError): raise c.InputError('Invalid source overview interval') from None
                            if not 0<=a<b<=meta['n_frames']: raise c.InputError('Source interval outside timeline')
                            yield source,a,b
                with (stage/'events.svg').open('w',encoding='utf-8') as h:
                    render_svg(h,store.events(),sources(),meta,args.start,args.end)
                end=meta['n_frames'] if args.end is None else args.end
                with (stage/'events-table.csv').open('w',newline='',encoding='utf-8') as h:
                    event_table(h,(e for e in store.events() if args.start<=e['event_frame']<end),meta)
            else:
                write_json(stage/'consumer_check.json',disk_consumer(store,stage/'deliveries.csv',drain=args.drain))
            write_json(stage/'provenance.json',provenance)
            if identity(inputs,meta,options,[])!=provenance: raise c.InputError('Inputs changed during processing')
            audit.update(stage='publication'); stage.rename(args.out)
            audit.update(stage='published')
        finally: store.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo',help='Generate and process synthetic inputs')
    demo.add_argument('--out',type=Path,required=True)
    detector=sub.add_parser('detect',help='Detect local candidate events')
    source=detector.add_mutually_exclusive_group(required=True)
    source.add_argument('--input',type=Path); source.add_argument('--source',choices=['atd'])
    detector.add_argument('--data-dir',type=Path); detector.add_argument('--config',type=Path)
    detector.add_argument('--hypothesis',required=True); detector.add_argument('--cache-dir',type=Path)
    detector.add_argument('--out',type=Path,required=True)
    report=sub.add_parser('report-csv',help='Official CSV validation without ground truth')
    report.add_argument('csv',type=Path); report.add_argument('--rows',type=int,default=199319)
    report.add_argument('--fps',type=float,default=25); report.add_argument('--out',type=Path,required=True)
    evaluate=sub.add_parser('evaluate-events',help='Evaluate v2 touches against a partial reference')
    evaluate.add_argument('--events',type=Path,required=True); evaluate.add_argument('--manifest',type=Path)
    evaluate.add_argument('--fps',type=float,required=True); evaluate.add_argument('--reference',type=Path,required=True)
    evaluate.add_argument('--out',type=Path,required=True)
    for command in ('render','replay'):
        p=sub.add_parser(command)
        p.add_argument('--run',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
        if command=='render': p.add_argument('--start',type=int,default=0); p.add_argument('--end',type=int)
        else: p.add_argument('--drain',action='store_true')
    args=parser.parse_args(argv)
    command=['python','-m','src.ball_events',*(argv if argv is not None else sys.argv[1:])]
    try:
        with Attempt(args.out,command,getattr(args,'hypothesis',None)) as audit:
            if args.command=='report-csv':
                report_csv(args.csv,args.rows,args.fps,args.out,attempt=audit)
                print('CSV validado. Relatório descritivo; qualidade não medida.')
            elif args.command=='evaluate-events':
                if args.out.exists(): raise c.InputError('Output already exists')
                manifest_path=args.manifest or args.events.parent/'manifest.json'
                meta=run_meta(manifest_path)
                inputs={'events':args.events,'reference':args.reference,'manifest':manifest_path}
                options={'matching_version':MATCHING_VERSION,'fps':args.fps}
                provenance=identity(inputs,meta,options,[])
                audit.update(identity=provenance)
                predictions={'fps':args.fps,'manifest':meta,'events':list(jsonl(args.events))}
                reference=read_json(args.reference)
                # Function validates all collection boundaries before matching.
                result=evaluate_events(predictions,reference,on_validated=lambda:audit.update(stage='processing'))
                audit.update(stage='processing')
                result['provenance']=provenance
                if identity(inputs,meta,options,[])!=provenance: raise c.InputError('Inputs changed during evaluation')
                args.out.parent.mkdir(parents=True,exist_ok=True)
                with tempfile.TemporaryDirectory(dir=args.out.parent,prefix='.evaluation-') as tmp:
                    file=Path(tmp)/'report.json'; write_json(file,result)
                    audit.update(stage='publication'); file.rename(args.out)
                audit.update(stage='published')
                print('Referência parcial avaliada; controle contínuo não foi medido.')
            elif args.command in ('render','replay'):
                view(args,audit)
                print('Artefatos de inspeção local gerados.')
            else:
                if args.command=='demo':
                    if args.out.exists(): raise c.InputError('Output already exists')
                    path=write_inputs(args.out/'inputs',scenario='causal')
                    values=standardized(path)
                    cfg,hypothesis,cache,out=None,'Verify synthetic causal contract; no match accuracy claim.',None,args.out/'run'
                else:
                    values=standardized(args.input) if args.input else atd(args.data_dir)
                    cfg=read_json(args.config) if args.config else None
                    hypothesis,cache,out=args.hypothesis,args.cache_dir,args.out
                meta,factory,hints,paths,status=values
                result=run(meta,factory,hints,paths,out,cfg,hypothesis,status,cache,command,attempt=audit)
                print(f'Eventos processados localmente. Cache: {result["cache_status"]}. Saídas em {result["output"]}')
                if result['warning']: print(result['warning'],file=sys.stderr)
        return 0
    except (ValueError,OSError,TypeError,KeyError,UnicodeError,OverflowError,ET.ParseError) as error:
        reason=str(error) if isinstance(error,c.InputError) else 'Confira formato, arquivos e destinos; nenhum registro bruto será exibido.'
        print(f'Erro: {reason}',file=sys.stderr)
        return 1


if __name__=='__main__': sys.exit(main())
