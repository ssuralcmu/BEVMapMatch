#!/usr/bin/env python3
import argparse
import json

from bevmapmatch.metrics import load_errors, localization_metrics


parser = argparse.ArgumentParser()
parser.add_argument("errors_json")
args = parser.parse_args()
print(json.dumps(localization_metrics(load_errors(args.errors_json)), indent=2))

