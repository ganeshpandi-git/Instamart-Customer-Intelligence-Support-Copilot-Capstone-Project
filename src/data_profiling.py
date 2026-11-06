
from pathlib import Path
import pandas as pd
import json

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part_a"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def profile_dataset(file_path):
    """Generate a data quality profile for a CSV file."""

    df = pd.read_csv(file_path)

    profile = {
        "file_name": file_path.name,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }

    if file_path.name == "orders.csv":
        profile["duplicate_order_id_rows"] = int(
            df.duplicated(subset=["order_id"], keep=False).sum()
        )

        delivery = pd.to_numeric(
            df["delivery_minutes"], errors="coerce"
        )

        profile["negative_delivery_times"] = int(
            (delivery < 0).sum()
        )

        amounts = pd.to_numeric(
            df["order_amount"], errors="coerce"
        )

        profile["unparseable_order_amounts"] = int(
            (df["order_amount"].notna() & amounts.isna()).sum()
        )

    if file_path.name == "customers.csv":
        age = pd.to_numeric(df["age"], errors="coerce")

        profile["invalid_ages"] = int(
            ((age < 18) | (age > 100)).sum()
        )

    return profile


def main():
    print("\n" + "=" * 55)
    print("INSTAMART - RAW DATA PROFILING")
    print("=" * 55)

    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        print("ERROR: No CSV files found in data/raw.")
        return

    all_profiles = []

    for file_path in csv_files:
        profile = profile_dataset(file_path)
        all_profiles.append(profile)

        print(f"\nDataset: {profile['file_name']}")
        print(f"Rows: {profile['total_rows']}")
        print(f"Columns: {profile['total_columns']}")
        print(f"Exact duplicate rows: {profile['exact_duplicate_rows']}")

        print("Missing values:")
        for column, count in profile["missing_values"].items():
            if count > 0:
                print(f"  {column}: {count}")

        for key, value in profile.items():
            if key not in [
                "file_name",
                "total_rows",
                "total_columns",
                "columns",
                "missing_values",
                "exact_duplicate_rows"
            ]:
                print(f"{key}: {value}")

    # Save profiling results
    report_path = OUTPUT_DIR / "data_profile.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(all_profiles, file, indent=4)

    print("\n" + "=" * 55)
    print("DATA PROFILING COMPLETED SUCCESSFULLY")
    print(f"Report saved at: {report_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
