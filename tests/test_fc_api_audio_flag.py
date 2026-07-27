"""Semantic v2 评测器的 TTS 开关投影测试。"""

from scripts.fc_api_eval.evaluator import set_session_generate_audio


def test_generate_audio_is_written_inside_session_payload() -> None:
    """服务端只读取 payload，禁止把开关错误写到事件顶层。"""

    session_init = {
        "type": "session.init",
        "payload": {"generate_audio": False},
    }

    set_session_generate_audio(session_init, enabled=True)

    assert session_init["payload"]["generate_audio"] is True
    assert "generate_audio" not in {
        key for key in session_init if key != "payload"
    }
