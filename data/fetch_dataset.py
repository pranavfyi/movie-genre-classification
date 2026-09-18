"""
Fetch and process the CMU Movie Summary Corpus into a clean dataset.

Source: http://www.cs.cmu.edu/~ark/personas/
The corpus contains 42,306 movie plot summaries from Wikipedia
and metadata including genres.

Usage: python data/fetch_dataset.py
"""

import os
import sys
import csv
import json
import tarfile
import io
import re
from collections import Counter

try:
    import urllib.request
except ImportError:
    pass

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "movies.csv")
CMU_URL = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"

# Map CMU's fine-grained genres to our 7 categories
GENRE_MAP = {
    # Action
    "action": "Action", "action film": "Action", "action/adventure": "Action",
    "adventure": "Action", "adventure film": "Action", "martial arts film": "Action",
    "superhero movie": "Action", "superhero film": "Action", "spy film": "Action",
    "war film": "Action", "western": "Action", "swashbuckler film": "Action",
    "combat film": "Action", "action comedy": "Action", "disaster film": "Action",
    
    # Comedy
    "comedy": "Comedy", "comedy film": "Comedy", "romantic comedy": "Comedy",
    "slapstick": "Comedy", "satire": "Comedy", "parody": "Comedy",
    "black comedy": "Comedy", "screwball comedy": "Comedy",
    "comedy of manners": "Comedy", "gross-out film": "Comedy",
    "buddy film": "Comedy", "stoner film": "Comedy",
    
    # Drama
    "drama": "Drama", "drama film": "Drama", "family drama": "Drama",
    "biographical film": "Drama", "historical drama": "Drama",
    "coming-of-age film": "Drama", "courtroom drama": "Drama",
    "family film": "Drama", "period piece": "Drama", "political drama": "Drama",
    "docudrama": "Drama", "melodrama": "Drama", "social problem film": "Drama",
    "sports film": "Drama", "legal drama": "Drama",
    
    # Horror
    "horror": "Horror", "horror film": "Horror", "slasher film": "Horror",
    "zombie film": "Horror", "supernatural": "Horror", "monster movie": "Horror",
    "creature film": "Horror", "natural horror film": "Horror",
    "psychological horror": "Horror", "gothic film": "Horror",
    
    # Romance
    "romance film": "Romance", "romance": "Romance", "romantic drama": "Romance",
    "romantic fantasy": "Romance", "chick flick": "Romance",
    
    # Science Fiction
    "science fiction": "Science Fiction", "science fiction film": "Science Fiction",
    "sci-fi": "Science Fiction", "dystopia": "Science Fiction",
    "cyberpunk": "Science Fiction", "space western": "Science Fiction",
    "alien film": "Science Fiction", "time travel": "Science Fiction",
    
    # Thriller
    "thriller": "Thriller", "thriller film": "Thriller",
    "psychological thriller": "Thriller", "crime fiction": "Thriller",
    "crime film": "Thriller", "mystery": "Thriller", "mystery film": "Thriller",
    "film noir": "Thriller", "suspense": "Thriller", "detective fiction": "Thriller",
    "neo-noir": "Thriller", "crime thriller": "Thriller",
    "political thriller": "Thriller", "whodunit": "Thriller",
}

SAMPLES_PER_GENRE = 300  # Target samples per genre


def download_corpus():
    """Download the CMU Movie Summary Corpus."""
    print("Downloading CMU Movie Summary Corpus...")
    print(f"  URL: {CMU_URL}")
    print("  (This may take 1-2 minutes)")
    
    try:
        response = urllib.request.urlopen(CMU_URL, timeout=120)
        data = response.read()
        print(f"  Downloaded {len(data) / 1024 / 1024:.1f} MB")
        return data
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nPlease try:")
        print("  1. Check your internet connection")
        print(f"  2. Manually download from: {CMU_URL}")
        print(f"  3. Extract to: {SCRIPT_DIR}")
        sys.exit(1)


def extract_and_parse(tar_data):
    """Extract metadata and plot summaries from the tar.gz archive."""
    print("\nExtracting and parsing corpus...")
    
    metadata = {}  # movie_id -> genres list
    summaries = {}  # movie_id -> plot summary
    
    with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("movie.metadata.tsv"):
                f = tar.extractfile(member)
                if f:
                    for line in f.read().decode("utf-8", errors="ignore").strip().split("\n"):
                        parts = line.split("\t")
                        if len(parts) >= 9:
                            movie_id = parts[0]
                            try:
                                genres_json = json.loads(parts[8]) if parts[8] else {}
                                genres = list(genres_json.values())
                                metadata[movie_id] = genres
                            except (json.JSONDecodeError, IndexError):
                                pass
            
            elif member.name.endswith("plot_summaries.txt"):
                f = tar.extractfile(member)
                if f:
                    for line in f.read().decode("utf-8", errors="ignore").strip().split("\n"):
                        parts = line.split("\t", 1)
                        if len(parts) == 2:
                            movie_id, summary = parts
                            summaries[movie_id] = summary
    
    print(f"  Found {len(metadata)} movies with metadata")
    print(f"  Found {len(summaries)} movies with plot summaries")
    return metadata, summaries


def clean_summary(text, max_words=150):
    """Clean and truncate a plot summary."""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove very short summaries
    if len(text.split()) < 20:
        return None
    # Truncate to max_words
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]) + "..."
    return text


def map_genre(genres_list):
    """Map a list of CMU genres to our simplified genre. Returns the first match."""
    for genre in genres_list:
        mapped = GENRE_MAP.get(genre.lower())
        if mapped:
            return mapped
    return None


def build_dataset(metadata, summaries):
    """Build a balanced dataset from the parsed data."""
    print("\nBuilding balanced dataset...")
    
    # Collect all valid samples
    genre_samples = {g: [] for g in set(GENRE_MAP.values())}
    
    for movie_id, genres in metadata.items():
        if movie_id not in summaries:
            continue
        
        mapped_genre = map_genre(genres)
        if not mapped_genre:
            continue
        
        summary = clean_summary(summaries[movie_id])
        if not summary:
            continue
        
        genre_samples[mapped_genre].append(summary)
    
    # Print available counts
    print("\n  Available samples per genre:")
    for genre in sorted(genre_samples.keys()):
        print(f"    {genre:<20s} {len(genre_samples[genre])}")
    
    # Balance: take SAMPLES_PER_GENRE from each
    rows = []
    for genre in sorted(genre_samples.keys()):
        samples = genre_samples[genre][:SAMPLES_PER_GENRE]
        for desc in samples:
            # Escape any commas/quotes for CSV safety
            rows.append((desc, genre))
    
    print(f"\n  Total samples selected: {len(rows)}")
    return rows


def save_dataset(rows):
    """Save the processed dataset to CSV."""
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["description", "genre"])
        for desc, genre in rows:
            writer.writerow([desc, genre])
    
    print(f"\n[OK] Dataset saved to: {OUTPUT_PATH}")
    print(f"  Total rows: {len(rows)}")


def main():
    print("=" * 60)
    print("MOVIE GENRE DATASET BUILDER")
    print("=" * 60)
    print(f"Source: CMU Movie Summary Corpus")
    print(f"Target: {SAMPLES_PER_GENRE} samples x 7 genres\n")
    
    # Step 1: Download
    tar_data = download_corpus()
    
    # Step 2: Parse
    metadata, summaries = extract_and_parse(tar_data)
    
    # Step 3: Build balanced dataset
    rows = build_dataset(metadata, summaries)
    
    # Step 4: Save
    save_dataset(rows)
    
    print("\nDone! You can now run: python main.py")


if __name__ == "__main__":
    main()
