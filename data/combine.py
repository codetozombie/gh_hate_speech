from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent

afrihate_label_map = {
    "Hate": 1,
    "Abuse": 1,
    "Normal": 0
}

manual_label_map = {
    1: 1, 0: 0,
    "hate": 1, "nonhate": 0,
    "Hate": 1, "Non-hate": 0, "Normal": 0
}

naijahate_label_map = {
    "HATE": 1, "ABUSIVE": 1, "OFFENSIVE": 1,
    "NON-HATE": 0, "NORMAL": 0,
    1: 1, 0: 0
}


def load_afrihate():
    df = pd.read_csv(BASE / "afrihate.csv")
    rename_map = {"tweet": "text", "label": "raw_label"}
    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        raise ValueError(
            f"afrihate missing columns: {missing}. Found: {list(df.columns)}")
    df = df.rename(columns=rename_map)
    df["label"] = df["raw_label"].map(afrihate_label_map)
    return df[["text", "label"]]


def load_manual():
    df = pd.read_csv(BASE / "manual.csv")
    if "text" not in df.columns:
        raise ValueError(
            f"manual.csv missing 'text'. Columns: {list(df.columns)}")
    possible = [c for c in df.columns if c.lower() in (
        "label", "class", "target", "polarity")]
    if not possible:
        raise ValueError(
            f"No label column found in manual.csv. Columns: {list(df.columns)}")
    label_col = possible[0]
    df = df.rename(columns={label_col: "raw_label"})
    df["label"] = df["raw_label"].map(manual_label_map)
    return df[["text", "label"]]


def load_naijahate():
    df = pd.read_excel(BASE / "naijahate.xlsx")
    rename_map = {"Tweet": "text", "Polarity": "raw_label"}
    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        raise ValueError(
            f"naijahate.xlsx missing columns: {missing}. Columns: {list(df.columns)}")
    df = df.rename(columns=rename_map)
    df["label"] = df["raw_label"].map(naijahate_label_map)
    return df[["text", "label"]]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df[df["label"].isin([0, 1])]
    if not df.empty:
        df["label"] = df["label"].astype(int)  # ensure 0/1 integers
    return df


def main():
    parts = []
    for loader in (load_afrihate, load_manual, load_naijahate):
        try:
            part = loader()
            part = clean(part)
            parts.append(part)
        except Exception as e:
            print(f"ERROR loading dataset: {e}")
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["text", "label"])
    if not combined.empty:
        combined["_lc"] = combined["text"].str.lower()
        before = len(combined)
        combined = combined.drop_duplicates(subset="_lc").drop(columns="_lc")
        removed = before - len(combined)
        if removed:
            print(f"Removed {removed} duplicate texts.")
        combined = combined.sample(
            frac=1, random_state=42).reset_index(drop=True)
        combined["label"] = combined["label"].astype(int)
    bad = sorted(set(combined["label"]) - {0, 1})
    if bad:
        raise ValueError(f"Unexpected labels present: {bad}")
    out_path = BASE / "hate.csv"  # save inside data folder
    combined.to_csv(out_path, index=False)
    print(f"Saved hate.csv with {len(combined)} rows at {out_path}")


if __name__ == "__main__":
    main()
