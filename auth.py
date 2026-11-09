"""
auth.py
Authentication for LEXINDIA.

- Passwords are hashed with Argon2id (argon2-cffi's PasswordHasher default).
- Login returns a signed JWT; the frontend sends it as "Authorization: Bearer <token>".
- get_current_user is the dependency that protects every endpoint touching
  case data. Authorization = every case query is filtered by the logged-in
  user's id (see main.py), so one user can never read another user's case.
- Failed logins are rate-limited per email (5 failures / 15 min).
"""

import datetime
import os
import re
import secrets
import time
from collections import defaultdict, deque

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import get_db
from models import User

load_dotenv()

# ---------- settings ----------
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "120"))
JWT_SECRET = os.getenv("JWT_SECRET", "")
if len(JWT_SECRET) < 32:
    # A missing/short secret must never silently weaken security. Use a random
    # one for this run (everyone is logged out on restart) and warn loudly.
    JWT_SECRET = secrets.token_hex(32)
    print("WARNING: JWT_SECRET is missing or shorter than 32 chars in .env. "
          "Using a temporary random key; logins will reset on every restart.")

ALLOWED_ROLES = {"advocate", "student", "researcher"}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LEN = 8
MAX_PASSWORD_LEN = 128

# ---------- password hashing (Argon2id) ----------
_ph = PasswordHasher()  # argon2id with sane defaults
# Verified against when the email doesn't exist, so response time doesn't
# reveal which emails are registered.
_DUMMY_HASH = _ph.hash("dummy-password-for-constant-time")


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _ph.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


# ---------- JWT ----------
def create_access_token(user_id: int) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + datetime.timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    unauthorized = HTTPException(
        status_code=401,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if creds is None:
        raise unauthorized
    try:
        payload = jwt.decode(
            creds.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
        user_id = int(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError):
        raise unauthorized

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise unauthorized
    return user


# ---------- login rate limiting (in-memory) ----------
MAX_FAILED_LOGINS = 5
LOCKOUT_WINDOW_SECONDS = 15 * 60
_failed_logins: dict[str, deque] = defaultdict(deque)


def _recent_failures(key: str) -> deque:
    now = time.time()
    q = _failed_logins[key]
    while q and now - q[0] > LOCKOUT_WINDOW_SECONDS:
        q.popleft()
    return q


def _check_rate_limit(key: str) -> None:
    if len(_recent_failures(key)) >= MAX_FAILED_LOGINS:
        raise HTTPException(
            status_code=429,
            detail="Too many failed attempts. Please try again in 15 minutes.",
        )


# ---------- routes ----------
router = APIRouter(prefix="/auth", tags=["auth"])


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str = "advocate"


class LoginRequest(BaseModel):
    email: str
    password: str


def _user_payload(user: User) -> dict:
    return {"id": user.id, "name": user.name, "email": user.email, "role": user.role}


def _token_response(user: User) -> dict:
    return {
        "token": create_access_token(user.id),
        "token_type": "bearer",
        "expires_in_minutes": JWT_EXPIRE_MINUTES,
        "user": _user_payload(user),
    }


@router.post("/signup")
def signup(req: SignupRequest, db: Session = Depends(get_db)):
    name = req.name.strip()
    email = req.email.strip().lower()

    if not name or len(name) > 150:
        raise HTTPException(status_code=400, detail="Please enter your name.")
    if not EMAIL_RE.match(email) or len(email) > 255:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if len(req.password) < MIN_PASSWORD_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {MIN_PASSWORD_LEN} characters.",
        )
    if len(req.password) > MAX_PASSWORD_LEN:
        raise HTTPException(status_code=400, detail="Password is too long.")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    user = User(
        name=name,
        email=email,
        password_hash=hash_password(req.password),
        role=req.role if req.role in ALLOWED_ROLES else "advocate",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _token_response(user)


@router.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    email = req.email.strip().lower()
    _check_rate_limit(email)

    user = db.query(User).filter(User.email == email).first()
    password_ok = verify_password(user.password_hash if user else _DUMMY_HASH, req.password)

    if user is None or not password_ok:
        _failed_logins[email].append(time.time())
        # Same message for "no such email" and "wrong password"
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    _failed_logins.pop(email, None)
    return _token_response(user)


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return _user_payload(current_user)
