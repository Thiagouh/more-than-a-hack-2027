"""Score manually reviewed visual evaluation frames."""
import argparse
import json
from src.team_identification.evaluation import evaluate
from src.team_identification.manifests import read_json, write_json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.detections.read_text().splitlines() if line.strip()]
    report = evaluate(read_json(args.truth), rows)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'evaluation.json', report)


if __name__ == '__main__':
    main()
