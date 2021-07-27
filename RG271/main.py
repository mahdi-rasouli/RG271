import argparse
from pathlib import Path

from scrapers.twitter import TwitterScraper


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-u",
        "--username",
        type=str,
        required=True,
        help="The user/username to query for - this is case sensitive.",
    )

    parser.add_argument(
        "--dt-max",
        type=str,
        help="ISO 8601 formatted datetime (YYYY-MM-DD) specifying to ignore any posts made on a "
        "date newer than this.",
    )

    parser.add_argument(
        "--dt-min",
        type=str,
        help="ISO 8601 formatted datetime (YYYY-MM-DD) specifying to ignore any posts made on a "
        "date older than this.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to dump any output generated by this script. Default is ./output.",
        default="./output",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    scraper = TwitterScraper(args.username, max_date=args.dt_max, min_date=args.dt_min)
    df = scraper.scrape()
    df.to_csv(output_dir.joinpath("results.csv"), index=False)


if __name__ == "__main__":
    main()
