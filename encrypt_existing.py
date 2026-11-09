"""
encrypt_existing.py - run ONCE after enabling AES-256 to protect OLD cases
that were saved before encryption existed.

  python encrypt_existing.py

For every case that has an owner (user_id): encrypts the plaintext file in
uploads/ (then deletes the plaintext copy) and encrypts extracted_text.
Cases with no owner (user_id empty, created before login existed) are
skipped and listed - delete them from the app or MySQL; they can't be
bound to a user. Safe to run again: already-encrypted cases are skipped.
"""
import os

from db import SessionLocal
from models import Case
from crypto_utils import encrypt_bytes, encrypt_text, TEXT_PREFIX, MAGIC


def main():
    db = SessionLocal()
    done = skipped = 0
    try:
        for case in db.query(Case).all():
            if case.user_id is None:
                print(f"SKIP case {case.id}: no owner (delete it)")
                skipped += 1
                continue

            changed = False
            p = case.file_path
            if p and os.path.exists(p) and not p.endswith(".enc"):
                with open(p, "rb") as f:
                    data = f.read()
                if data[:4] != MAGIC:
                    new_path = p + ".enc"
                    with open(new_path, "wb") as f:
                        f.write(encrypt_bytes(data, case.user_id))
                    case.file_path = new_path
                    changed = True
                    db.commit()          # path saved before the plaintext is removed
                    os.remove(p)

            if case.extracted_text and not case.extracted_text.startswith(TEXT_PREFIX):
                case.extracted_text = encrypt_text(case.extracted_text, case.user_id)
                changed = True

            if changed:
                db.commit()
                done += 1
                print(f"encrypted case {case.id}")
        print(f"Done. encrypted={done} skipped={skipped}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
