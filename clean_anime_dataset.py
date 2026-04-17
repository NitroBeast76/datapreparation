"""
================================================================================
  ANIME DATASET CLEANER + ENRICHER
  Works on: Anideas_-_anime_full_dataset.csv (or any similarly structured CSV)

  Run (basic clean only):
      python clean_anime_dataset.py

  Run with API enrichment (fills Unknown fields via MyAnimeList/Jikan):
      python clean_anime_dataset.py --enrich

  Run with a custom input file:
      python clean_anime_dataset.py myfile.csv --enrich

  Outputs (always exactly two files):
      anime_cleaned.csv      ← clean, enriched dataset
      cleaning_summary.txt   ← full report of what was done
================================================================================
"""

import csv
import re
import os
import sys
import time
import json
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — edit these if your file names or column names differ
# ══════════════════════════════════════════════════════════════════════════════

INPUT_FILE = "anime_clean.csv"
OUTPUT_FILE = "anime_cleaned.csv"
SUMMARY_FILE = "cleaning_summary.txt"

# Jikan (free MyAnimeList API — no key needed)
JIKAN_BASE = "https://api.jikan.moe/v4"
API_DELAY_SECONDS = 0.4  # Jikan rate limit: ~3 req/sec; 0.4s keeps us safe
API_MAX_RETRIES = 3  # retries per title on 429 / network error
API_RETRY_WAIT = 5  # seconds to wait after a 429

# Fields we try to fill via API (only when still "Unknown" after local cleaning)
ENRICH_FIELDS = {
    "year",
    "studios",
    "synopsis",
    "episodes",
    "source",
    "rating",
    "genres",
    "duration",
    "english_title",
}

# Adult content — excluded from anime_cleaned.csv and counted in the summary
EXPLICIT_GENRES = {"hentai", "erotica"}
EXPLICIT_RATINGS = {"rx"}
ADULT_GENRES = {"ecchi"}

STUB_SYNOPSIS_LEN = 50
PLACEHOLDER = "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  FIELD AUTO-DETECTION
# ══════════════════════════════════════════════════════════════════════════════

FIELD_MAP = {
    "title": ["title", "name", "anime_title"],
    "english_title": ["english_title", "title_english", "english_name"],
    "japanese_title": ["japanese_title", "title_japanese", "japanese_name"],
    "year": ["year", "aired_year", "release_year", "start_year"],
    "episodes": ["episodes", "episode_count", "num_episodes"],
    "rating": ["rating", "age_rating", "content_rating"],
    "genres": ["genres", "genre", "genre_list"],
    "studios": ["studios", "studio", "studio_list"],
    "synopsis": ["synopsis", "description", "summary"],
    "source": ["source", "source_material"],
    "type": ["type", "media_type", "format"],
    "duration": ["duration", "episode_duration"],
}


def find_col(headers, candidates):
    lower = {h.lower(): h for h in headers}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def resolve_fields(headers):
    return {k: find_col(headers, v) for k, v in FIELD_MAP.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def get(row, col):
    if col is None:
        return ""
    return row.get(col, "").strip()


def is_empty(val):
    return not val or val.lower() in {"", "unknown", "none", "n/a", "[]", "nan"}


def extract_year_from_title(title):
    m = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", title)
    return m.group(1) if m else None


def normalise_genres(raw):
    cleaned = re.sub(r"[\[\]{}]", "", raw)
    return {g.strip().lower() for g in cleaned.split(",") if g.strip()}


def is_adult(genres_set, rating):
    if genres_set & EXPLICIT_GENRES:
        return True
    if rating.strip().lower() in EXPLICIT_RATINGS:
        return True
    return False


def progress(current, total, prefix=""):
    bar_len = 40
    filled = int(bar_len * current / total) if total else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = 100.0 * current / total if total else 0
    print(f"\r{prefix} [{bar}] {pct:5.1f}%  {current:,}/{total:,}", end="", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  JIKAN API
# ══════════════════════════════════════════════════════════════════════════════


def jikan_search(title):
    """Query Jikan for an anime by title. Returns a dict of fields or None."""
    encoded = urllib.parse.quote(title)
    url = f"{JIKAN_BASE}/anime?q={encoded}&limit=1"

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "anime-cleaner/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode())

            data = body.get("data", [])
            if not data:
                return None

            anime = data[0]

            # Year — try the top-level field first, fall back to aired.prop
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


def enrich_rows(rows, F, stats):
    """Hit the Jikan API for any row still missing fields after local cleaning."""
    field_to_col = {
        "year": F["year"],
        "studios": F["studios"],
        "genres": F["genres"],
        "synopsis": F["synopsis"],
        "episodes": F["episodes"],
        "rating": F["rating"],
        "source": F["source"],
        "duration": F["duration"],
        "english_title": F["english_title"],
    }

    needs_enrich = [
        (i, row)
        for i, row in enumerate(rows)
        if any(
            field_to_col.get(f) and is_empty(get(row, field_to_col[f]))
            for f in ENRICH_FIELDS
        )
    ]

    if not needs_enrich:
        print("  No rows need API enrichment — all fields already filled.")
        return rows, stats

    total = len(needs_enrich)
    print(f"  {total:,} rows need enrichment. Querying Jikan API …\n")

    enriched_count = 0
    api_fail_count = 0
    fields_filled = defaultdict(int)

    for idx, (row_idx, row) in enumerate(needs_enrich):
        title = get(row, F["title"])
        progress(idx + 1, total, prefix="  API")

        api_data = jikan_search(title)
        time.sleep(API_DELAY_SECONDS)

        if api_data is None:
            api_fail_count += 1
            continue

        row_improved = False
        for field, col in field_to_col.items():
            if col is None:
                continue
            if is_empty(get(row, col)):
                new_val = api_data.get(field, "").strip()
                if new_val and not is_empty(new_val):
                    rows[row_idx][col] = new_val
                    fields_filled[field] += 1
                    row_improved = True

        if row_improved:
            enriched_count += 1

    print()  # newline after progress bar

    stats["api_rows_enriched"] = enriched_count
    stats["api_rows_failed"] = api_fail_count
    stats["api_fields_filled"] = dict(fields_filled)

    return rows, stats


# ══════════════════════════════════════════════════════════════════════════════
#  LOCAL CLEANING PASS
# ══════════════════════════════════════════════════════════════════════════════


def clean_pass(rows, F, stats):
    cleaned = []
    duplicates = []
    seen_titles = {}

    for i, row in enumerate(rows):
        row_num = i + 1
        title = get(row, F["title"])
        genres_raw = get(row, F["genres"])
        rating_raw = get(row, F["rating"])
        genres_set = normalise_genres(genres_raw)

        # 1. Adult / explicit
        if is_adult(genres_set, rating_raw):
            stats["quarantined_adult"] += 1
            continue

        # 2. Duplicates
        key = title.lower().strip()
        if key and key in seen_titles:
            duplicates.append((row_num, title, seen_titles[key]))
            stats["duplicates_removed"] += 1
            continue
        if key:
            seen_titles[key] = row_num

        # 3. Year
        if F["year"] and is_empty(get(row, F["year"])):
            guessed = extract_year_from_title(title)
            row[F["year"]] = guessed if guessed else PLACEHOLDER
            if guessed:
                stats["year_filled_local"] += 1

        # 4. English title — fall back to main title so field is never blank
        if F["english_title"] and is_empty(get(row, F["english_title"])):
            row[F["english_title"]] = title
            stats["english_title_defaulted"] += 1

        # 5–12. Remaining fields → PLACEHOLDER so CSV has no empty cells
        for field, col_key in [
            ("studios", "studios"),
            ("genres", "genres"),
            ("rating", "rating"),
            ("episodes", "episodes"),
            ("source", "source"),
            ("type", "type"),
            ("duration", "duration"),
        ]:
            col = F[col_key]
            if col and is_empty(get(row, col)):
                row[col] = PLACEHOLDER

        # Synopsis — slightly different placeholder
        if F["synopsis"]:
            syn = get(row, F["synopsis"])
            if is_empty(syn):
                row[F["synopsis"]] = "No synopsis available."
                stats["synopsis_placeholder"] += 1
            elif len(syn) < STUB_SYNOPSIS_LEN:
                stats["synopsis_stub"] += 1

        cleaned.append(row)

    stats["duplicates_list"] = duplicates
    return cleaned, stats


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════


def write_summary(path, input_path, total_in, total_out, stats, do_enrich):
    sep = "=" * 72
    lines = [
        sep,
        "  ANIME DATASET — CLEANING & ENRICHMENT SUMMARY",
        f"  Run at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Source : {input_path}",
        f"  Mode   : {'Clean + API Enrichment (Jikan/MAL)' if do_enrich else 'Clean only (no API)'}",
        sep,
        "",
        "ROW COUNTS",
        f"  Original rows             : {total_in:>8,}",
        f"  Output rows (clean)       : {total_out:>8,}",
        f"  Adult content excluded    : {stats['quarantined_adult']:>8,}",
        f"  Duplicates removed        : {stats['duplicates_removed']:>8,}",
        "",
        "LOCAL AUTO-FILLS",
        f"  Year extracted from title : {stats['year_filled_local']:>8,}",
        f"  English title defaulted   : {stats['english_title_defaulted']:>8,}",
        f"  Synopsis placeholder set  : {stats['synopsis_placeholder']:>8,}",
        f"  Stub synopses found       : {stats['synopsis_stub']:>8,}  (short, kept as-is)",
        "",
    ]

    if do_enrich:
        ff = stats.get("api_fields_filled", {})
        attempted = stats.get("api_rows_enriched", 0) + stats.get("api_rows_failed", 0)
        lines += [
            "API ENRICHMENT  (Jikan — MyAnimeList, no key required)",
            f"  Rows queried              : {attempted:>8,}",
            f"  Rows successfully updated : {stats.get('api_rows_enriched', 0):>8,}",
            f"  Rows not found in MAL     : {stats.get('api_rows_failed', 0):>8,}",
            "",
            "  Fields filled by API:",
        ]
        for field in [
            "year",
            "studios",
            "genres",
            "synopsis",
            "episodes",
            "rating",
            "source",
            "duration",
            "english_title",
        ]:
            n = ff.get(field, 0)
            if n:
                lines.append(f"    {field:<20}: {n:>7,}")
        lines.append("")
    else:
        lines += [
            "API ENRICHMENT",
            "  Skipped — re-run with --enrich to fill remaining Unknown fields",
            "  via the free Jikan/MyAnimeList API (no account needed).",
            "",
        ]

    dups = stats.get("duplicates_list", [])
    if dups:
        lines.append("DUPLICATES REMOVED")
        for row_num, title, first_seen in dups[:100]:
            lines.append(
                f"  Row {row_num:<6}  '{title}'  (first seen at row {first_seen})"
            )
        if len(dups) > 100:
            lines.append(f"  … and {len(dups) - 100} more")
        lines.append("")

    lines += [
        "OUTPUT FILES",
        f"  Cleaned dataset : {OUTPUT_FILE}",
        f"  This summary    : {SUMMARY_FILE}",
        "",
        sep,
        "TIPS FOR REMAINING GAPS",
        sep,
        """
  • STILL UNKNOWN AFTER API?
    Some very old or obscure titles aren't indexed on MyAnimeList.
    Try AniList's free GraphQL API as a second source:
    https://anilist.co/graphql

  • STUB SYNOPSES
    Filter rows where len(synopsis) < 50 — these are usually short films
    or promo clips with no description anywhere online.

  • ADULT CONTENT
    Excluded rows were counted but not saved anywhere. To keep them,
    add --keep-adult flag (not yet implemented — add your own filter
    by removing the is_adult() check in clean_pass()).

  • GENRE NORMALISATION
    API genres may differ in capitalisation from the originals.
    Consider building a canonical genre list and mapping all values to it.
""",
        sep,
        "END OF SUMMARY",
        sep,
    ]

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════


def main():
    args = sys.argv[1:]
    do_enrich = "--enrich" in args
    input_path = next((a for a in args if not a.startswith("--")), INPUT_FILE)

    print("=" * 72)
    print("  ANIME DATASET CLEANER" + (" + API ENRICHER" if do_enrich else ""))
    print("=" * 72)

    if not os.path.exists(input_path):
        print(f"\n[ERROR] File not found: '{input_path}'")
        print("  • Put your CSV in the same folder as this script, or")
        print("  • Pass the path:  python clean_anime_dataset.py path/to/file.csv")
        sys.exit(1)

    print(f"\nReading '{input_path}' …")
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    F = resolve_fields(headers)
    print(f"Loaded {len(rows):,} rows, {len(headers)} columns.")
    print(f"Detected columns: { {k: v for k, v in F.items() if v} }\n")

    total_in = len(rows)
    stats = defaultdict(int)

    # ── Phase 1: local cleaning ───────────────────────────────────────────────
    print("Phase 1 — Local cleaning …")
    cleaned, stats = clean_pass(rows, F, stats)
    print(f"  {len(cleaned):,} rows remain (removed {total_in - len(cleaned):,}).")

    # ── Phase 2: API enrichment ───────────────────────────────────────────────
    if do_enrich:
        print("\nPhase 2 — API enrichment …")
        cleaned, stats = enrich_rows(cleaned, F, stats)
    else:
        print("\n  Tip: run with --enrich to auto-fill Unknown fields via Jikan API.")

    # ── Write outputs ─────────────────────────────────────────────────────────
    print(f"\nWriting '{OUTPUT_FILE}' …")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(cleaned)
    print(f"  Saved {len(cleaned):,} rows.")

    print(f"Writing '{SUMMARY_FILE}' …")
    summary_text = write_summary(
        SUMMARY_FILE, input_path, total_in, len(cleaned), stats, do_enrich
    )

    print("\n" + summary_text)
    print("\n✓ Done!")
    print(f"  → {OUTPUT_FILE}")
    print(f"  → {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
