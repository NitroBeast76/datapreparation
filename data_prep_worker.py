"""
data_prep_worker.py
-------------------
Anime Recommendation System — Feature Matrix Preparation (Adapted for actual dataset)
Produces:
  - anime_clean_BACKUP.csv       (backup of input, untouched)
  - anime_features_scaled.csv    (encoded + scaled feature matrix, title-first)
"""

import os
import shutil
import ast
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

# ── 0. Constants ──────────────────────────────────────────────────────────────

INPUT_FILE      = "anime_clean.csv"   # adjust if needed
BACKUP_FILE     = "anime_clean_BACKUP.csv"
OUTPUT_FILE     = "anime_features_scaled.csv"

# Columns that contain string representations of lists
LIST_COLS = ["genres", "themes", "demographics", "studios"]

# Single-value categorical columns
CAT_COLS = ["type", "source", "rating"]

# Numeric columns (may contain missing values, will be imputed)
NUMERIC_COLS = ["year", "episodes"]

# ── Helper: safe list parser ─────────────────────────────────────────────────

def safe_parse_list(x):
    """Convert various string representations into a Python list."""
    if pd.isna(x):
        return []
    x_str = str(x).strip()
    if x_str == "" or x_str.lower() == "unknown":
        return []
    # Try to parse as literal Python list
    if x_str.startswith("[") and x_str.endswith("]"):
        try:
            parsed = ast.literal_eval(x_str)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    # Fallback: split by comma
    return [item.strip() for item in x_str.split(",") if item.strip()]

# ── 1. Safety check ──────────────────────────────────────────────────────────

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"'{INPUT_FILE}' not found in the current directory: {os.getcwd()}\n"
        "Please make sure the file exists before running this script."
    )

# ── 2. Backup ─────────────────────────────────────────────────────────────────

shutil.copy(INPUT_FILE, BACKUP_FILE)
print(f"[✓] Backup created: {BACKUP_FILE}")

# ── 3. Load and parse list columns ───────────────────────────────────────────

df = pd.read_csv(INPUT_FILE, encoding="utf-8")
print(f"[✓] Loaded {INPUT_FILE}: {df.shape[0]:,} rows × {df.shape[1]} columns")

for col in LIST_COLS:
    if col in df.columns:
        df[col] = df[col].apply(safe_parse_list)
        print(f"[✓] Parsed '{col}' into lists")
    else:
        print(f"[!] Column '{col}' not found, skipping.")

# ── 4. Select feature columns that exist ──────────────────────────────────────

available_list_cols   = [c for c in LIST_COLS if c in df.columns]
available_cat_cols    = [c for c in CAT_COLS if c in df.columns]
available_numeric_cols= [c for c in NUMERIC_COLS if c in df.columns]

all_feature_cols = available_list_cols + available_cat_cols + available_numeric_cols
print(f"[✓] Feature columns selected: {all_feature_cols}")

# ── 5. Multi-value encoding (genres, themes, demographics, studios) ───────────

encoded_parts = []

for col in available_list_cols:
    mlb = MultiLabelBinarizer()
    encoded = pd.DataFrame(
        mlb.fit_transform(df[col]),
        columns=[f"{col}_{cls}" for cls in mlb.classes_],
        index=df.index,
    )
    encoded_parts.append(encoded)
    print(f"[✓] '{col}' encoded → {encoded.shape[1]} columns")

# ── 6. Single-value categorical encoding ──────────────────────────────────────

for col in available_cat_cols:
    df[col] = df[col].fillna("Unknown")
    encoded = pd.get_dummies(df[col], prefix=col)
    encoded.index = df.index
    encoded_parts.append(encoded)
    print(f"[✓] '{col}' encoded → {encoded.shape[1]} columns")


# ── 7. Numeric columns ────────────────────────────────────────────────────────

if available_numeric_cols:
    numeric_df = df[available_numeric_cols].copy()
    for col in available_numeric_cols:
        # Convert to numeric, coercing errors to NaN (e.g., "Unknown" becomes NaN)
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        if numeric_df[col].isnull().any():
            median_val = numeric_df[col].median()
            # If all values are NaN, median is NaN; fallback to 0
            if pd.isna(median_val):
                median_val = 0
            numeric_df[col] = numeric_df[col].fillna(median_val)
            print(f"[✓] Filled missing '{col}' with median {median_val:.1f}")
    encoded_parts.append(numeric_df)
    print(f"[✓] Numeric columns retained for scaling: {available_numeric_cols}")
else:
    numeric_df = None
    print("[!] No numeric columns available — scaling step will be skipped.")

# ── 8. Assemble feature matrix ────────────────────────────────────────────────

if not encoded_parts:
    raise RuntimeError("No encoded parts produced. Check available feature columns.")

df_encoded = pd.concat(encoded_parts, axis=1)
print(f"[✓] Feature matrix assembled: {df_encoded.shape[0]:,} rows × {df_encoded.shape[1]} columns")

# ── 9. Scale numeric columns to [0, 1] ───────────────────────────────────────

if available_numeric_cols:
    scaler = MinMaxScaler()
    df_encoded[available_numeric_cols] = scaler.fit_transform(df_encoded[available_numeric_cols])
    print(f"[✓] Numeric columns scaled to [0, 1]: {available_numeric_cols}")

# ── 10. Insert title column first ────────────────────────────────────────────

if "title" in df.columns:
    df_encoded.insert(0, "title", df["title"].values)
    print("[✓] 'title' column inserted as first column.")
else:
    print("[!] 'title' column not found; skipping insertion.")

# ── 11. Save output ──────────────────────────────────────────────────────────

df_encoded.to_csv(OUTPUT_FILE, index=False)
print(f"\n[✓] Saved: {OUTPUT_FILE}")
print(f"    Final shape: {df_encoded.shape[0]:,} rows × {df_encoded.shape[1]} columns")
print(f"    First 5 column names: {df_encoded.columns.tolist()[:5]}")
print("\n[✓] All done. Original file has NOT been modified.")