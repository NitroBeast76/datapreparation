"""
add_images.py

Adds image URLs from MyAnimeList to the cleaned anime dataset.
This script can be stopped and restarted – it will resume where it left off.
"""

import os
import shutil
import time

import pandas as pd
import requests

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
# Replace with your own MyAnimeList Client ID
# Get one at: https://myanimelist.net/apiconfig
MAL_CLIENT_ID = "543e8907c42691ac6518c7a71f52ce64"

# Input / Output files
INPUT_FILE = "anime_clean.csv"
PARTIAL_FILE = "anime_with_images.csv"
BACKUP_FILE = "anime_clean_no_images_backup.csv"

# API rate limiting (MAL allows ~3 requests per second)
REQUEST_DELAY = 0.35  # seconds
SAVE_EVERY = 100  # save progress every N requests


# ----------------------------------------------------------------------
# HELPER FUNCTION
# ----------------------------------------------------------------------
def get_image_url(title: str) -> str | None:
    """
    Search MyAnimeList for a given anime title and return the cover image URL.
    Returns None if nothing is found or an error occurs.
    """
    url = "https://api.myanimelist.net/v2/anime"
    headers = {"X-MAL-CLIENT-ID": MAL_CLIENT_ID}
    params = {"q": title, "limit": 1, "fields": "main_picture"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code != 200:
            print(f"  API error {response.status_code} for: {title}")
            return None

        data = response.json()

        if not data.get("data"):
            print(f"  No result found for: {title}")
            return None

        picture = data["data"][0]["node"].get("main_picture", {})
        # Prefer medium size for faster website loading
        return picture.get("medium") or picture.get("large") or None

    except Exception as e:
        print(f"  Error for '{title}': {e}")
        return None


# ----------------------------------------------------------------------
# MAIN SCRIPT
# ----------------------------------------------------------------------
def main():
    print("=== Add Image URLs from MyAnimeList ===\n")

    # Step 2: Load dataset (or resume from partial)
    if os.path.exists(PARTIAL_FILE):
        df = pd.read_csv(PARTIAL_FILE)
        print(f"Resuming from previous run. Loaded {len(df)} rows.")
    else:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} anime titles.")
        print("Columns:", df.columns.tolist())
        # Add empty image_url column
        df["image_url"] = None

    # Find rows that still need an image URL
    missing = df[df["image_url"].isna()]
    total_missing = len(missing)

    if total_missing == 0:
        print("\nAll titles already have image URLs. Nothing to do.")
        return

    print(f"\n{total_missing} titles still need image URLs.")
    print(f"This will take approximately {round(total_missing / 3 / 60)} minutes.")
    print("You can leave this running in the background.\n")

    # Step 4: Loop through missing titles
    for i, (idx, row) in enumerate(missing.iterrows()):
        title = row.get("title") or row.get("Name")  # adjust column name if needed
        if title is None:
            print(f"  Skipping row {idx}: no title column found.")
            df.at[idx, "image_url"] = (
                "https://via.placeholder.com/200x300?text=No+Title"
            )
            continue

        image_url = get_image_url(title)

        if image_url:
            df.at[idx, "image_url"] = image_url
            print(f"  [{i + 1}/{total_missing}] ✓ {title}")
        else:
            df.at[idx, "image_url"] = (
                "https://via.placeholder.com/200x300?text=No+Image"
            )
            print(f"  [{i + 1}/{total_missing}] ✗ {title} — placeholder used")

        # Periodic save
        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(PARTIAL_FILE, index=False)
            print(f"  --- Progress saved ({i + 1} done) ---")

        time.sleep(REQUEST_DELAY)

    # Final save
    df.to_csv(PARTIAL_FILE, index=False)
    print(f"\n✅ All titles processed. Saved to {PARTIAL_FILE}")

    # Step 5: Statistics
    real = df[~df["image_url"].str.contains("placeholder", na=False)].shape[0]
    placeholder = df[df["image_url"].str.contains("placeholder", na=False)].shape[0]
    print(f"\nReal image URLs:  {real} ({round(real / len(df) * 100)}%)")
    print(f"Placeholders:     {placeholder} ({round(placeholder / len(df) * 100)}%)")

    # Step 6: Replace cleaned dataset
    if os.path.exists(INPUT_FILE):
        shutil.copy(INPUT_FILE, BACKUP_FILE)
        print(f"\nBackup created: {BACKUP_FILE}")

    shutil.copy(PARTIAL_FILE, INPUT_FILE)
    print(f"{INPUT_FILE} now contains image URLs.")

    # Step 7: Preview
    print("\nSample of titles and their image URLs:")
    print(df[["title", "image_url"]].head(10).to_string())


if __name__ == "__main__":
    main()
