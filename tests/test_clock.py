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
    done = set_clock(["COM5"], datetime(2026, 10, 20, 9, 30), open_port=ports.__getitem__, settle=0)
    assert done == [("COM5", "20261020T093000")]
    assert ports["COM5"].sent == ["?\n", "T1792488600\n"]


def test_set_clock_leaves_other_boards_and_busy_ports_alone():
    def busy(name):
        raise OSError("access denied")

    other = FakePort(lambda line: "hello")
    said = []
    assert set_clock(["COM3"], open_port=lambda name: other, settle=0, log=said.append) == []
    assert other.sent == ["?\n"] * 3  # asked, never told the time
    assert set_clock(["COM4"], open_port=busy, settle=0, log=said.append) == []
    assert said == ["COM3: not a recorder (answered 'hello')", "COM4: can't open (access denied)"]


def test_set_clock_asks_again_when_the_first_line_is_lost():
    answers = iter(["", "RECORDER 1 unset REC - dropped=0", "OK 20261020T093000"])
    port = FakePort(lambda line: next(answers))
    assert set_clock(["COM3"], open_port=lambda name: port, settle=0) == [
        ("COM3", "20261020T093000")
    ]


def test_set_clock_checks_whether_a_silent_device_took_the_time():
    answers = iter(["RECORDER 1 unset REC - dropped=0", "", "RECORDER 1 20261020T093000 REC"])
    port = FakePort(lambda line: next(answers))
    assert set_clock(["COM3"], open_port=lambda name: port, settle=0) == [
        ("COM3", "20261020T093000")
    ]
