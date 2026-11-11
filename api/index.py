"""Vercel entry point. Vercel serves the ASGI app named `app`."""

from app.main import app

__all__ = ["app"]
