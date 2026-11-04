"""Regression test for provider discovery — stdlib only, no network.

Reproduces the installed layout: Hermes executes the plugin's
``__init__.py`` via ``spec_from_file_location`` under a synthetic module
name, WITHOUT the plugin directory on ``sys.path`` and without any
``agent.credential_pool`` (not signed in). Registration must still happen.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class StubProfile:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class TestBareDiscoveryLoad(unittest.TestCase):
    def test_spec_load_registers_without_login(self) -> None:
        registered: list = []
        fake_providers = types.ModuleType("providers")
        fake_providers.register_provider = registered.append  # type: ignore[attr-defined]
        fake_base = types.ModuleType("providers.base")
        fake_base.ProviderProfile = StubProfile  # type: ignore[attr-defined]

        saved = {name: sys.modules.get(name) for name in ("providers", "providers.base", "hermes_dust")}
        saved_hermes = dict(sys.modules)
        saved_path = list(sys.path)
        # Simulate the loader: plugin dir NOT on sys.path.
        sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
        sys.modules["providers"] = fake_providers
        sys.modules["providers.base"] = fake_base
        try:
            spec = importlib.util.spec_from_file_location("dust_bare_probe", ROOT / "__init__.py")
            self.assertIsNotNone(spec)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules["dust_bare_probe"] = module
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop("dust_bare_probe", None)
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod
            for name in [n for n in sys.modules if n == "hermes_dust" or n.startswith("hermes_dust.")]:
                if name not in saved_hermes:
                    del sys.modules[name]
            sys.path = saved_path

        self.assertTrue(module.ACTIVE)
        self.assertEqual(len(registered), 1)
        self.assertEqual(registered[0].name, "dust")


if __name__ == "__main__":
    unittest.main()
