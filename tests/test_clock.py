from datetime import datetime

from recorder.clock import local_epoch, set_clock


class FakePort:
    """A serial port with a scripted device behind it."""

    def __init__(self, device):
        self.device, self.sent, self.reply = device, [], b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def reset_input_buffer(self):
        self.reply = b""

    def write(self, data):
        self.sent.append(data.decode())
        self.reply = self.device(data.decode().strip()).encode() + b"\r\n"

    def readline(self):
        return self.reply


def recorder(line):
    if line == "?":
        return "RECORDER 1 unset REC REC/NOTIME/00000001.WAV dropped=0"
    return "OK 20261020T093000" if line.startswith("T") else "ERR"


def test_local_epoch_counts_wall_clock_as_utc():
    assert local_epoch(datetime(2026, 10, 20, 9, 30)) == 1792488600


def test_set_clock_sends_local_time():
    ports = {"COM5": FakePort(recorder)}
    done = set_clock(["COM5"], datetime(2026, 10, 20, 9, 30), open_port=ports.__getitem__)
    assert done == [("COM5", "20261020T093000")]
    assert ports["COM5"].sent == ["?\n", "T1792488600\n"]


def test_set_clock_leaves_other_boards_and_busy_ports_alone():
    def busy(name):
        raise OSError("access denied")

    other = FakePort(lambda line: "hello")
    assert set_clock(["COM3"], open_port=lambda name: other) == []
    assert other.sent == ["?\n"]  # asked, never told the time
    assert set_clock(["COM4"], open_port=busy) == []
