"""Command-line interface.

    bayesian-mmm simulate  --out data
    bayesian-mmm run       --config config/config.yaml [--quick]
"""

from __future__ import annotations

import argparse
import logging

QUICK = {"sampler": {"draws": 300, "tune": 400, "chains": 2, "prior_samples": 200}}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bayesian-mmm", description="Bayesian Marketing Mix Modelling")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("simulate", help="generate a synthetic dataset with known ground truth")
    p_sim.add_argument("--out", default="data")
    p_sim.add_argument("--weeks", type=int, default=156)
    p_sim.add_argument("--seed", type=int, default=7)

    p_run = sub.add_parser("run", help="fit the model and produce all outputs")
    p_run.add_argument("--config", default="config/config.yaml")
    p_run.add_argument("--quick", action="store_true", help="few draws - for smoke tests only")
    p_run.add_argument("--output-dir", default=None)

    args = parser.parse_args(argv)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    pkg_log = logging.getLogger("bayesian_mmm")
    pkg_log.setLevel(logging.INFO)
    pkg_log.addHandler(handler)
    pkg_log.propagate = False

    if args.command == "simulate":
        from .simulate import write_synthetic

        df, *_ = write_synthetic(args.out, n_weeks=args.weeks, seed=args.seed)
        print(f"Wrote {len(df)} rows to {args.out}/mmm_synthetic.csv (+ lift_tests.csv, ground_truth.json)")
    elif args.command == "run":
        from .pipeline import run_pipeline

        overrides = dict(QUICK) if args.quick else {}
        if args.output_dir:
            overrides["output"] = {"dir": args.output_dir}
        run_pipeline(args.config, overrides)


if __name__ == "__main__":
    main()
