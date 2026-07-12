"""Tests for language plugin discovery (code_atlas.parsing.languages)."""

from __future__ import annotations

from code_atlas.parsing import languages as languages_init


def test_discover_plugins_continues_after_one_module_fails(monkeypatch):
    """A failing built-in language module must not prevent later modules in the
    list from being imported, and must not permanently latch discovery as done
    before all modules have been attempted."""
    monkeypatch.setattr(languages_init, "_discovered", False)
    monkeypatch.setattr(
        languages_init,
        "_BUILTIN_LANGUAGE_MODULES",
        ("fake.bad.module", "fake.good.module_a", "fake.good.module_b"),
    )

    calls: list[str] = []

    def fake_import_module(name: str):
        calls.append(name)
        if name == "fake.bad.module":
            msg = "simulated import failure"
            raise ImportError(msg)

    monkeypatch.setattr(languages_init.importlib, "import_module", fake_import_module)

    languages_init.discover_plugins()

    # All three entries were attempted, including the two after the failure.
    assert calls == ["fake.bad.module", "fake.good.module_a", "fake.good.module_b"]
    # Discovery is marked complete only after every module was attempted.
    assert languages_init._discovered is True


def test_discover_plugins_is_noop_after_first_call(monkeypatch):
    """Safe to call multiple times — a second call must not re-import anything."""
    monkeypatch.setattr(languages_init, "_discovered", False)
    monkeypatch.setattr(languages_init, "_BUILTIN_LANGUAGE_MODULES", ("fake.mod_a", "fake.mod_b"))

    calls: list[str] = []

    def record_import(name: str) -> None:
        calls.append(name)

    monkeypatch.setattr(languages_init.importlib, "import_module", record_import)

    languages_init.discover_plugins()
    assert calls == ["fake.mod_a", "fake.mod_b"]

    calls.clear()
    languages_init.discover_plugins()
    assert calls == []


def test_discover_plugins_all_modules_failing_still_completes(monkeypatch):
    """Even if every built-in module fails, discover_plugins must not raise and
    must still mark discovery as complete."""
    monkeypatch.setattr(languages_init, "_discovered", False)
    monkeypatch.setattr(languages_init, "_BUILTIN_LANGUAGE_MODULES", ("fake.bad_a", "fake.bad_b"))

    def always_fail(name: str):
        msg = f"simulated failure for {name}"
        raise ImportError(msg)

    monkeypatch.setattr(languages_init.importlib, "import_module", always_fail)

    languages_init.discover_plugins()  # must not raise

    assert languages_init._discovered is True
