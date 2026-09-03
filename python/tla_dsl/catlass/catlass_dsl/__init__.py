"""TLA device DSL class entry points (``@tla.jit`` / ``@tla.kernel``)."""

from __future__ import annotations

from .catlass import CatlassBaseDSL, TlaDSL

__all__ = ["CatlassBaseDSL", "TlaDSL"]
