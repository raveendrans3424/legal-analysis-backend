"""
seed_provisions.py
One-time script to populate the legal_provisions table.

Usage:
    python seed_provisions.py provisions.json

Expected JSON format (a list of objects):
[
  {
    "act_name": "Bharatiya Nyaya Sanhita, 2023",
    "section_number": "103",
    "title": "Punishment for murder",
    "full_text": "Whoever commits murder shall be punished with death or...",
    "category": "murder",
    "source_url": "https://www.indiacode.nic.in/..."
  },
  ...
]

Replace the sample data below with your real dataset (scraped/compiled
from an official source such as indiacode.nic.in), or point this script
at a JSON/CSV file as shown above.
"""

import sys
import json

from db import SessionLocal, Base, engine
from models import LegalProvision

SAMPLE_PROVISIONS = [
    {
        "act_name": "Bharatiya Nyaya Sanhita, 2023",
        "section_number": "103",
        "title": "Punishment for murder",
        "full_text": (
            "Whoever commits murder shall be punished with death or "
            "imprisonment for life, and shall also be liable to fine."
        ),
        "category": "murder",
        "source_url": "https://www.indiacode.nic.in/",
    },
    {
        "act_name": "Bharatiya Nyaya Sanhita, 2023",
        "section_number": "303",
        "title": "Theft",
        "full_text": (
            "Whoever, intending to take dishonestly any movable property "
            "out of the possession of any person without that person's "
            "consent, moves that property in order to such taking, is "
            "said to commit theft."
        ),
        "category": "theft",
        "source_url": "https://www.indiacode.nic.in/",
    },
    {
        "act_name": "Bharatiya Nyaya Sanhita, 2023",
        "section_number": "318",
        "title": "Cheating",
        "full_text": (
            "Whoever, by deceiving any person, fraudulently or dishonestly "
            "induces the person so deceived to deliver any property to any "
            "person, or to consent that any person shall retain any "
            "property, commits cheating."
        ),
        "category": "cheating",
        "source_url": "https://www.indiacode.nic.in/",
    },
]


def load_provisions_from_file(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def seed(provisions: list[dict]) -> None:
    Base.metadata.create_all(bind=engine)  # ensure tables exist
    db = SessionLocal()
    try:
        for p in provisions:
            db.add(LegalProvision(**p))
        db.commit()
        print(f"Inserted {len(provisions)} provisions.")
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data = load_provisions_from_file(sys.argv[1])
    else:
        print("No file provided, seeding with SAMPLE_PROVISIONS (3 entries). "
              "Run `python seed_provisions.py your_data.json` for real data.")
        data = SAMPLE_PROVISIONS

    seed(data)
