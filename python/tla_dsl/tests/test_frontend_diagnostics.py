import sys

from catlass import frontend_diagnostics


def test_install_excepthook_wraps_the_current_host_hook(monkeypatch) -> None:
    calls: list[tuple[object, object, object]] = []

    def host_hook(exc_type, exc_value, exc_traceback) -> None:
        calls.append((exc_type, exc_value, exc_traceback))

    monkeypatch.setattr(sys, "excepthook", host_hook)
    monkeypatch.setattr(
        frontend_diagnostics,
        "_delegate_excepthook",
        frontend_diagnostics._delegate_excepthook,
    )

    frontend_diagnostics.install_excepthook()

    assert sys.excepthook is frontend_diagnostics._catlass_excepthook
    assert frontend_diagnostics._delegate_excepthook is host_hook

    error = ValueError("host hook must be preserved")
    frontend_diagnostics._catlass_excepthook(ValueError, error, None)
    assert calls == [(ValueError, error, None)]
