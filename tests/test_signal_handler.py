import logging

from signal_handler import SignalHandler


def test_signal_handler_sets_flag_and_logs(caplog):
    handler = SignalHandler()
    assert handler.shutdown_requested is False

    with caplog.at_level(logging.INFO):
        handler(15, None)

    assert "Shutdown signal 15 received" in caplog.text
    assert handler.shutdown_requested is True

