from spatialaudiometrics import lap_challenge as lap
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Evaluate HRIRs")
    parser.add_argument("pred_dir", type=str, help="Input directory")
    parser.add_argument("ref_dir", type=str, help="Reference directory")

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    ref_dir = Path(args.ref_dir)

    df = []
    for f in pred_dir.glob("*.sofa"):
        for ref_f in ref_dir.rglob(f"*{f.name}"):
            *_, result = lap.calculate_task_two_metrics(str(ref_f), str(f))
            subject = f.name.split("_")[0]
            # add a subject column for all the rows
            result["subject"] = subject
            df.append(result)

    df = pd.concat(df, ignore_index=True)
    # Grouped by "Metric name"
    # remove subject column
    groups = df.drop(columns=["subject", "Threshold value"]).groupby("Metric name")
    print(groups.mean())
    print(groups.std().drop(columns=["Below threshold?"]))


if __name__ == "__main__":
    main()
