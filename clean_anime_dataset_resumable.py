"""
clean_anime_dataset_resumable.py
--------------------------------
Resumable version of the anime dataset cleaner with API enrichment.
Continues from where the original script left off.

Usage:
    python clean_anime_dataset_resumable.py
"""

import csv
import json
import os
import time
import urllib.request
import urllib.parse
import urllib.error
from collections import defaultdict

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
INPUT_FILE = "anime_clean.csv"          # Locally cleaned file (pre-enrichment)
OUTPUT_FILE = "anime_clean.csv"         # Will be overwritten with enriched data
CHECKPOINT_FILE = "enrichment_checkpoint.json"

# Jikan API settings
JIKAN_BASE = "https://api.jikan.moe/v4"
API_DELAY = 0.4          # seconds between requests
API_MAX_RETRIES = 3
API_RETRY_WAIT = 5       # seconds after a 429

# Fields we want to fill via API (if still "Unknown")
ENRICH_FIELDS = {"year", "studios", "synopsis", "episodes", "source",
                 "rating", "genres", "duration", "english_title"}

PLACEHOLDER = "Unknown"

# ----------------------------------------------------------------------
# Jikan search function (same as original)
# ----------------------------------------------------------------------
def jikan_search(title):
    """Query Jikan for an anime by title. Returns a dict of fields or None."""
    encoded = urllib.parse.quote(title)
    url = f"{JIKAN_BASE}/anime?q={encoded}&limit=1"

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "anime-cleaner-resumable/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode())

            data = body.get("data", [])
            if not data:
                return None

            anime = data[0]

            # Extract fields
            year = str(anime.get("year") or "")
            if not year:
                prop = (anime.get("aired") or {}).get("prop", {}).get("from", {})
                year = str(prop.get("year") or "")

            studios = ", ".join(s.get("name", "") for s in (anime.get("studios") or []))
            genres = ", ".join(g.get("name", "") for g in (anime.get("genres") or []))
            synopsis = anime.get("synopsis") or ""
            episodes = str(anime.get("episodes") or "")
            rating = anime.get("rating") or ""
            source = anime.get("source") or ""
            duration = anime.get("duration") or ""
            en_title = anime.get("title_english") or ""

            return {
                "year": year.strip(),
                "studios": studios.strip(),
                "genres": genres.strip(),
                "synopsis": synopsis.strip(),
                "episodes": episodes.strip(),
                "rating": rating.strip(),
                "source": source.strip(),
                "duration": duration.strip(),
                "english_title": en_title.strip(),
            }

        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(API_RETRY_WAIT * attempt)
            else:
                return None
        except Exception:
            if attempt < API_MAX_RETRIES:
                time.sleep(2)
            else:
                return None

    return None

# ----------------------------------------------------------------------
# Helper: is a value considered empty/unknown?
# ----------------------------------------------------------------------
def is_empty(val):
    return not val or str(val).lower() in {"", "unknown", "none", "n/a", "[]", "nan"}

# ----------------------------------------------------------------------
# Main resumable enrichment
# ----------------------------------------------------------------------
def main():
    print("=" * 72)
    print("  RESUMMABLE ANIME DATASET ENRICHER")
    print("=" * 72)

    # Load the current anime_clean.csv (pre-enrichment)
    print(f"\nLoading '{INPUT_FILE}'...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    print(f"Loaded {len(rows):,} rows.")

    # Identify which fields exist in the CSV
    field_to_col = {f: f for f in ENRICH_FIELDS if f in headers}
    print(f"Fields to enrich: {list(field_to_col.keys())}")

    # Load checkpoint (set of titles already processed)
    processed_titles = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            processed_titles = set(json.load(f))
        print(f"Checkpoint found: {len(processed_titles):,} titles already processed.")
    else:
        print("No checkpoint found — starting fresh.")

    # Identify rows that still need enrichment (any field still "Unknown")
    # and whose title hasn't been processed yet.
    needs_enrich = []
    for i, row in enumerate(rows):
        title = row.get("title", "")
        if title in processed_titles:
            continue  # already attempted (success or fail)
        # Check if any enrichable field is missing
        if any(is_empty(row.get(col, "")) for col in field_to_col.values()):
            needs_enrich.append((i, row))

    total = len(needs_enrich)
    if total == 0:
        print("\nAll titles already enriched! Nothing to do.")
        return

    print(f"\n{total:,} titles still need enrichment.")
    print("Press Ctrl+C to pause; progress will be saved.\n")

    enriched_count = 0
    failed_count = 0
    checkpoint_batch = []

    try:
        for idx, (row_idx, row) in enumerate(needs_enrich):
            title = row["title"]
            print(f"  [{idx+1}/{total}] {title} ... ", end="", flush=True)

            api_data = jikan_search(title)
            time.sleep(API_DELAY)

            if api_data is None:
                failed_count += 1
                print("✗ not found")
            else:
                # Update row with API data where empty
                updated = False
                for field, col in field_to_col.items():
                    if is_empty(row.get(col, "")):
                        new_val = api_data.get(field, "")
                        if new_val and not is_empty(new_val):
                            rows[row_idx][col] = new_val
                            updated = True
                if updated:
                    enriched_count += 1
                    print("✓ updated")
                else:
                    print("○ no new fields")

            # Mark title as processed
            processed_titles.add(title)
            checkpoint_batch.append(title)

            # Save checkpoint every 100 titles
            if len(checkpoint_batch) >= 100:
                with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                    json.dump(list(processed_titles), f)
                # Also save the CSV so far (in case of crash)
                with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"  --- Progress saved ({idx+1} done) ---")
                checkpoint_batch = []

        # Final save after loop finishes
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(list(processed_titles), f)
        with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        print("\n" + "=" * 72)
        print("ENRICHMENT COMPLETE")
        print(f"  Rows updated: {enriched_count}")
        print(f"  API failures: {failed_count}")
        print(f"  Output saved to: {OUTPUT_FILE}")
        print("=" * 72)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Saving progress...")
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(list(processed_titles), f)
        with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print("Progress saved. Run this script again to resume.")
        return

if __name__ == "__main__":
    main()