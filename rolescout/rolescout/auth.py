"""Password hashing (stdlib scrypt) and user helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import Profile, User, utcnow

_N, _R, _P = 2**14, 8, 1
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD = 8


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return "scrypt${}${}${}${}${}".format(
        _N, _R, _P, base64.b64encode(salt).decode(), base64.b64encode(digest).decode()
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if algo != "scrypt":
            return False
        digest = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt_b64),
                                n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(digest, base64.b64decode(hash_b64))
    except (ValueError, TypeError):
        return False


class AuthError(ValueError):
    pass


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def create_user(session: Session, email: str, password: str, name: str | None = None,
                is_admin: bool | None = None) -> User:
    email = normalize_email(email)
    if not EMAIL_RE.match(email):
        raise AuthError("Enter a valid email address.")
    if len(password or "") < MIN_PASSWORD:
        raise AuthError(f"Password must be at least {MIN_PASSWORD} characters.")
    if session.scalar(select(User).where(User.email == email)):
        raise AuthError("An account with that email already exists. Log in instead.")
    if is_admin is None:
        settings = get_settings()
        first_user = (session.scalar(select(func.count(User.id))) or 0) == 0
        is_admin = email in settings.admin_emails or first_user
    user = User(email=email, password_hash=hash_password(password), name=(name or "").strip() or None,
                is_admin=bool(is_admin))
    user.profile = Profile()
    session.add(user)
    session.flush()
    return user


def authenticate(session: Session, email: str, password: str) -> User:
    user = session.scalar(select(User).where(User.email == normalize_email(email)))
    # Hash anyway on a miss so response time doesn't reveal which emails exist.
    if user is None:
        verify_password(password, hash_password("timing-equalizer"))
        raise AuthError("Wrong email or password.")
    if not verify_password(password, user.password_hash):
        raise AuthError("Wrong email or password.")
    user.last_login_at = utcnow()
    return user
