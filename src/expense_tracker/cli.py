"""Command-line interface: report, correct, train, forecast."""

import argparse
import logging

from .analysis import monthly_summary
from .categorizer import add_correction
from .config import load_config, setup_logging
from .forecast import backtest, backtest_metrics, forecast_next_month, forecast_total, next_month_label
from .pipeline import build_dataset, train_and_evaluate
from .reports import generate_report

logger = logging.getLogger(__name__)


def cmd_report(args) -> None:
    df = build_dataset(args.input)
    out_dir = args.out or load_config()["paths"]["output_dir"]
    files = generate_report(df, args.month, out_dir)
    print(monthly_summary(df).to_string(index=False))
    print("\nCreated:")
    for name, path in files.items():
        print(f"  {name:15} {path}")


def cmd_correct(args) -> None:
    path = add_correction(args.description, args.category)
    print(f"Saved: '{args.description}' -> {args.category}  ({path})")


def cmd_train(args) -> None:
    metrics = train_and_evaluate(args.labeled)
    print(f"Labeled rows: {metrics['labeled_rows']}  ({metrics['note']})\n")
    for label, key in (("Known merchants", "known_merchants"), ("Unseen merchants", "unseen_merchants")):
        m = metrics[key]
        print(f"{label} (test rows: {m['n_test']:.0f})")
        print(f"  keyword rules : {m['rules_accuracy']:.1%}")
        print(f"  model alone   : {m['model_accuracy']:.1%}")
        print(f"  rules + model : {m['hybrid_accuracy']:.1%}  (left for manual correction: {m['unresolved_pct']:.1f}%)\n")


def cmd_forecast(args) -> None:
    df = build_dataset(args.input)
    print(f"Forecast for {next_month_label(df)}: {forecast_total(df):,.0f}\n")
    print(forecast_next_month(df).to_string(index=False))
    result = backtest(df)
    print("\nBacktest (each month predicted from earlier months only):")
    print(result.round(0).to_string(index=False))
    print("\nError:", backtest_metrics(result))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="expense-tracker", description="Expense tracker and budget analyzer")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("report", help="charts, cleaned CSV and a Markdown report")
    p.add_argument("--input", help="transactions CSV (default from config.json)")
    p.add_argument("--month", help="month to focus on, YYYY-MM (default: latest)")
    p.add_argument("--out", help="output folder (default: output)")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("correct", help="save a manual category correction")
    p.add_argument("description", help='merchant, e.g. "Dominos Pizza"')
    p.add_argument("category", help="category, e.g. Food")
    p.set_defaults(func=cmd_correct)

    p = sub.add_parser("train", help="evaluate rules vs model, then train and save the model")
    p.add_argument("--labeled", help="CSV with a true_category column")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("forecast", help="next-month forecast and backtest error")
    p.add_argument("--input", help="transactions CSV")
    p.set_defaults(func=cmd_forecast)

    args = parser.parse_args(argv)
    setup_logging("DEBUG" if args.verbose else None)
    args.func(args)


if __name__ == "__main__":
    main()
