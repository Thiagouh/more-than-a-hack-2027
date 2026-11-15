"""Score manually reviewed visual evaluation frames."""
import argparse
import json
from src.team_identification.evaluation import evaluate
from src.team_identification.manifests import read_json, write_json
from src.team_identification.pilot import sha256
from src.team_identification.protocol import load_truth, validate_result_binding, runtime_provenance
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, help='Defaults to provenance.json beside detections')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.detections.read_text().splitlines() if line.strip()]
    truth = load_truth(args.truth)
    provenance = read_json(args.provenance or args.detections.parent/'provenance.json')
    if provenance.get('status') != 'complete' or provenance.get('detections_sha256') != sha256(args.detections):
        raise ValueError('Complete execution and unchanged predictions required')
    validate_result_binding(truth,args.truth,rows,provenance)
    report = evaluate(truth, rows)
    report.update(truth_sha256=sha256(args.truth),detections_sha256=sha256(args.detections),
                  model_executed=provenance.get('model_executed'),synthetic=provenance.get('synthetic',False),
                  provenance_sha256=sha256(args.provenance or args.detections.parent/'provenance.json'),
                  environment=runtime_provenance())
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'evaluation.json', report)


if __name__ == '__main__':
    main()
