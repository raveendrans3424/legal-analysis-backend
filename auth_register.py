"""
Drop-in /register route for LEXINDIA (FastAPI).

Usage in main.py:
    from auth_register import router as register_router
    app.include_router(register_router)

Env vars:
    DATABASE_URL  e.g. mysql+pymysql://user:pass@host:3306/lexindia   (your MySQL / RDS)
    JWT_SECRET    long random string (same one your /login uses, if you have one)

pip install fastapi sqlalchemy pymysql argon2-cffi "pyjwt>=2.8" email-validator
"""
import os
import time

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./lexindia.db")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALG = "HS256"
TOKEN_TTL = 60 * 60 * 8  # 8 hours

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
ph = PasswordHasher()  # Argon2id by default
router = APIRouter()

# Separate table name so it never clashes with an existing `users` table.
with engine.begin() as c:
    c.execute(text("""
        CREATE TABLE IF NOT EXISTS app_users (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(120) NOT NULL,
            email VARCHAR(190) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(40) NOT NULL DEFAULT 'Advocate',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""".replace("AUTO_INCREMENT", "" if DATABASE_URL.startswith("sqlite") else "AUTO_INCREMENT")))


def make_token(user_id: int, email: str) -> str:
    now = int(time.time())
    return jwt.encode({"sub": str(user_id), "email": email, "iat": now, "exp": now + TOKEN_TTL},
                      JWT_SECRET, algorithm=JWT_ALG)


class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    role: str = "Advocate"


@router.post("/register", status_code=201)
def register(body: RegisterIn):
    email = body.email.lower()
    with engine.begin() as c:
        if c.execute(text("SELECT 1 FROM app_users WHERE email=:e"), {"e": email}).first():
            raise HTTPException(409, "An account with this email already exists.")
        c.execute(text("INSERT INTO app_users (name,email,password_hash,role) VALUES (:n,:e,:p,:r)"),
                  {"n": body.name.strip(), "e": email, "p": ph.hash(body.password), "r": body.role})
        uid = c.execute(text("SELECT id FROM app_users WHERE email=:e"), {"e": email}).scalar()
    return {"access_token": make_token(uid, email), "token_type": "bearer",
            "user": {"name": body.name.strip(), "email": email, "role": body.role}}
