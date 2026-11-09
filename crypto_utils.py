"""
crypto_utils.py
AES-256-GCM encryption for uploaded case files and extracted case text.

- Key: FILE_ENCRYPTION_KEY in .env = 64 hex characters (32 bytes = 256 bits).
  Generate once on your own PC:
      python -c "import secrets; print(secrets.token_hex(32))"
  NEVER paste the key in chat, never commit it, and back it up safely.
  If the key is lost, every encrypted file is permanently unreadable.
- Mode: AES-GCM. A random 12-byte nonce is generated for every encryption.
  The 16-byte GCM tag gives integrity: any change to the stored bytes (or
  the wrong key, or the wrong AAD) makes decryption fail with DecryptionError.
- AAD ("associated data") = "user:<id>". A file or text encrypted for user 5
  cannot be decrypted as user 6, even if someone swaps database rows.
- File format:  b"LXE1" + nonce(12) + ciphertext+tag
- Text format:  "enc:v1:" + base64(file format)
"""

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from dotenv import load_dotenv

load_dotenv()

MAGIC = b"LXE1"
NONCE_LEN = 12
TEXT_PREFIX = "enc:v1:"
KEY_ENV_NAME = "FILE_ENCRYPTION_KEY"


class DecryptionError(Exception):
    """Wrong key, wrong owner, or the data was modified / corrupted."""


class KeyConfigError(RuntimeError):
    """FILE_ENCRYPTION_KEY is missing or invalid."""


def _load_key() -> bytes:
    raw = os.getenv(KEY_ENV_NAME, "").strip()
    if not raw:
        raise KeyConfigError(
            f"{KEY_ENV_NAME} is missing in .env. Generate one with:\n"
            '  python -c "import secrets; print(secrets.token_hex(32))"\n'
            f"then add this line to .env:  {KEY_ENV_NAME}=<the 64 characters>"
        )
    try:
        key = bytes.fromhex(raw)
    except ValueError:
        raise KeyConfigError(f"{KEY_ENV_NAME} must be hexadecimal (0-9, a-f) only.")
    if len(key) != 32:
        raise KeyConfigError(
            f"{KEY_ENV_NAME} must be exactly 64 hex characters (256 bits); "
            f"yours decodes to {len(key) * 8} bits."
        )
    if len(set(key)) < 8:
        raise KeyConfigError(f"{KEY_ENV_NAME} looks too weak (not random). Generate a new one.")
    return key


# Fail fast at startup: a server that silently stores plaintext is worse than
# a server that refuses to start.
_KEY = _load_key()
_AESGCM = AESGCM(_KEY)


def _aad(owner_id: int | None) -> bytes:
    return f"user:{owner_id}".encode()


def encrypt_bytes(data: bytes, owner_id: int | None = None) -> bytes:
    nonce = os.urandom(NONCE_LEN)
    return MAGIC + nonce + _AESGCM.encrypt(nonce, data, _aad(owner_id))


def decrypt_bytes(blob: bytes, owner_id: int | None = None) -> bytes:
    if len(blob) < len(MAGIC) + NONCE_LEN + 16 or blob[:4] != MAGIC:
        raise DecryptionError("Not an encrypted LEXINDIA file.")
    nonce = blob[4:4 + NONCE_LEN]
    try:
        return _AESGCM.decrypt(nonce, blob[4 + NONCE_LEN:], _aad(owner_id))
    except InvalidTag:
        raise DecryptionError("Decryption failed: wrong key/owner or data was modified.")


def encrypt_text(text: str, owner_id: int | None = None) -> str:
    blob = encrypt_bytes(text.encode("utf-8"), owner_id)
    return TEXT_PREFIX + base64.b64encode(blob).decode("ascii")


def decrypt_text(value: str | None, owner_id: int | None = None) -> str | None:
    """Old rows stored before encryption was added have no prefix -> returned as-is."""
    if value is None or not value.startswith(TEXT_PREFIX):
        return value
    try:
        blob = base64.b64decode(value[len(TEXT_PREFIX):])
    except Exception:
        raise DecryptionError("Stored text is corrupted.")
    return decrypt_bytes(blob, owner_id).decode("utf-8")


def is_encrypted_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == MAGIC
    except OSError:
        return False
