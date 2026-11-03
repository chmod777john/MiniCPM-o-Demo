from pathlib import Path


def test_session_root_prefers_runtime_override(monkeypatch, tmp_path):
    import gateway

    expected = tmp_path / "configured-sessions"
    monkeypatch.setitem(gateway.GATEWAY_CONFIG, "session_dir", str(expected))
    monkeypatch.setenv("O5_SESSION_DATA_DIR", str(tmp_path / "environment-sessions"))

    assert Path(gateway._sessions_root()) == expected.resolve()


def test_session_root_uses_environment_override(monkeypatch, tmp_path):
    import gateway

    expected = tmp_path / "environment-sessions"
    monkeypatch.setitem(gateway.GATEWAY_CONFIG, "session_dir", None)
    monkeypatch.setenv("O5_SESSION_DATA_DIR", str(expected))

    assert Path(gateway._sessions_root()) == expected.resolve()
