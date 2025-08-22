from signal_handler import SignalHandler


def test_signal_handler_sets_flag_and_prints(capsys):
    handler = SignalHandler()
    assert handler.shutdown_requested is False

    handler(15, None)
    captured = capsys.readouterr()

    assert "Shutdown signal 15 received" in captured.out
    assert handler.shutdown_requested is True

