import pandas as pd
from pathlib import Path
from ml.pipelines.utils import combine_datasets


def prepare_analysis_dataset(apartment_data_path: Path, house_data_path: Path, out_dir: Path) -> Path:
    """Build analysis dataset from raw apartment and house data."""
    df = combine_datasets(apartment_data_path, house_data_path)

    # Drop irrelevant columsns
    df.drop(columns=["URL"], inplace=True)

    # Analysis-focused cleaning: keep human-friendly fields, derive a few KPIs
    df["price_per_m2"] = df["Price"] / df["Living area"].replace(0, pd.NA)

    # Simple winsorization to reduce outliers impact on dashboard
    df["price_per_m2"] = df["price_per_m2"].clip(
        lower=df["price_per_m2"].quantile(0.01), upper=df["price_per_m2"].quantile(0.99)
    )

    # Optional: map postal_code to province here if available
    # df["province"] = df["postal_code"].map(...)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "analysis_dataset.parquet"

    df.to_parquet(out_file, index=False)

    return out_file
