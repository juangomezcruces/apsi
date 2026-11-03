"""python -m ltg <stage> [options]

  catalog        crawl the six catalog pages        -> data/catalog.csv
  download       page images (+ PDFs)               -> data/pages, data/pdf
  extract        OCR / PDF text + word boxes        -> data/text, data/boxes
  text           spaCy pass over every page         -> data/results/text/*.csv.gz
  senses         'pueblo' sense classifier (BETO)   -> updates people_tokens.csv.gz
  report         text tables + report.md            -> data/results/text/report.md
  sample         hand-coding samples + APSI export  -> data/results/validation
  regions        cut out illustrations              -> data/results/visual/regions.csv
  illustration-sample  random sample per edition    -> data/results/visual/regions_sample.csv
  detect         person detector on the sample      -> data/results/visual/people_detect.csv
  zeroshot       CLIP codebook on the sample        -> data/results/visual/zeroshot.csv
  visual-sample  hand-coding sheet for images       -> data/results/validation/visual_sample.*
  visual-report  visual tables + report.md          -> data/results/visual/report.md
  validate       agreement of a hand-coded sample   --kind sense|role|visual [--dim people]
"""
import argparse
import logging
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    stage, rest = argv[0], argv[1:]
    if stage == "catalog":
        from .catalog import main as m
        return m(rest)
    if stage == "download":
        from .download import main as m
        return m(rest)
    if stage == "extract":
        from .extract import main as m
        return m(rest)
    if stage == "text":
        from .text_analysis import main as m
        return m(rest)
    if stage == "senses":
        from .senses import run
        return run()
    if stage == "report":
        from .text_analysis import report
        ap = argparse.ArgumentParser(prog="ltg report")
        ap.add_argument("--corpus", default="national", choices=["national", "state", "all"])
        return report(corpus=ap.parse_args(rest).corpus)
    if stage == "sample":
        from .text_analysis import export_for_coding
        return export_for_coding()

    from . import visual
    ap = argparse.ArgumentParser(prog=f"ltg {stage}")
    if stage == "regions":
        ap.add_argument("--editions", default="")
        ap.add_argument("--workers", type=int, default=4)
        a = ap.parse_args(rest)
        return visual.run_regions(set(filter(None, a.editions.split(","))) or None, a.workers)
    if stage == "illustration-sample":
        ap.add_argument("--n", type=int, default=300, help="illustrations per edition")
        ap.add_argument("--seed", type=int, default=0)
        a = ap.parse_args(rest)
        return visual.illustration_sample(a.n, a.seed)
    if stage == "detect":
        ap.add_argument("--all", action="store_true", help="every illustration, not the sample")
        return visual.run_detector(sample=not ap.parse_args(rest).all)
    if stage == "zeroshot":
        ap.add_argument("--model", default="openai/clip-vit-base-patch16")
        ap.add_argument("--batch", type=int, default=32)
        ap.add_argument("--all", action="store_true", help="every illustration, not the sample")
        a = ap.parse_args(rest)
        return visual.run_zeroshot(a.model, a.batch, sample=not a.all)
    if stage == "visual-sample":
        return visual.visual_sample()
    if stage == "visual-report":
        return visual.visual_report()
    if stage == "validate":
        from .config import PATHS
        from .text_analysis import validate
        ap.add_argument("--kind", choices=["sense", "role", "visual"], required=True)
        ap.add_argument("--dim", default="people", help="visual codebook dimension")
        a = ap.parse_args(rest)
        v = PATHS["results"] / "validation"
        spec = {"sense": (v / "sense_sample.csv", "predicted_sense", "gold_sense"),
                "role": (v / "role_sample.csv", "predicted_role", "gold_role"),
                "visual": (v / "visual_sample.csv", f"pred_{a.dim}", f"gold_{a.dim}")}[a.kind]
        return validate(*spec)
    print(__doc__)
    raise SystemExit(f"unknown stage {stage!r}")


if __name__ == "__main__":
    main()
