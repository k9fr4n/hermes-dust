"""Dust workspace agents as a Hermes model provider.

Hermes imports this file through provider discovery (``kind: model-provider``),
never through ``register(ctx)``. The Dust profile registers here, once.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _register() -> bool:
    try:
        from providers import register_provider
    except ImportError:
        logger.warning("dust plugin: Hermes core providers module not found; skipping registration.")
        return False
    try:
        from hermes_dust.provider import dust_profile
    except ImportError as exc:
        logger.warning("dust plugin: cannot load provider profile: %s", exc)
        return False
    register_provider(dust_profile)
    return True


ACTIVE = _register() if __package__ else False
