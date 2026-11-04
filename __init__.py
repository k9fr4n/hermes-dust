"""Dust workspace agents as a Hermes model provider.

Hermes imports this file through provider discovery (``kind: model-provider``),
never through ``register(ctx)``. The Dust profile registers here, once.

The loader executes this file without putting the plugin directory on
``sys.path`` (``spec_from_file_location`` under a synthetic module name),
so the directory is added first — otherwise the sibling ``hermes_dust``
package is unimportable and the provider stays unknown.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _ensure_importable() -> None:
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)


def _register() -> bool:
    _ensure_importable()
    try:
        from providers import register_provider
    except ImportError:
        logger.warning("dust plugin: Hermes core providers module not found; skipping registration.")
        return False
    # Any other failure propagates: the loader records per-plugin import
    # errors, which is the only way to see why a provider stayed unknown.
    from hermes_dust.provider import dust_profile

    if dust_profile is None:
        logger.warning("dust plugin: provider profile unavailable; sign in with `hermes auth add dust`.")
        return False
    register_provider(dust_profile)
    return True


ACTIVE = _register()
