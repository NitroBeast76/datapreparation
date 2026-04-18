# Anime Dataset Pipeline

## Overview
A Python-based data processing and machine learning pipeline for cleaning, enriching, and clustering an anime dataset. Designed for building a recommendation system.

## Tech Stack
- **Language**: Python 3.12
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn (KMeans), joblib
- **APIs**: MyAnimeList API, Jikan API (via requests/urllib)
- **Visualization**: matplotlib

## Project Structure
- `run_full_pipeline.py` — Main entry point that orchestrates all pipeline steps
- `clean_anime_dataset.py` — Cleans raw CSV and enriches via Jikan API
- `clean_anime_dataset_resumable.py.py` — Resumable version of the cleaner
- `add_images.py` — Fetches image URLs from MyAnimeList API (requires MAL Client ID)
- `data_prep_worker.py` — Feature engineering and preparation for clustering
- `clustering.py` — Trains KMeans model, generates elbow plots, saves model
- `anime_clean.csv` — Primary dataset file
- `requirements.txt` — Python dependencies

## Pipeline Steps
1. `clean_anime_dataset_resumable.py` — Clean and enrich the dataset
2. `add_images.py` — Add cover image URLs (needs MAL Client ID configured)
3. `data_prep_worker.py` — Prepare features for ML
4. `clustering.py` — Run KMeans clustering and export model

## Outputs
- `anime_clean.csv` — Cleaned dataset with images
- `anime_features_scaled.csv` — Feature matrix for ML
- `anime_clean_BACKUP.csv` — Safety backup

## Notes
- `add_images.py` requires a MAL (MyAnimeList) Client ID to be configured inside the script
- The pipeline is designed to run as a batch process, not a web server
- Run via: `python run_full_pipeline.py`
