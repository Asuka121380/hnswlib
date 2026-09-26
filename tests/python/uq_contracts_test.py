import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.edge_estimation.contracts import (  # noqa: E402
    ContractError, EventRecord, LabelRecord, QueryRangeRecord,
    INVALID_U32, INVALID_U64, read_events, read_labels, read_query_ranges,
    validate_dataset, write_events, write_labels, write_query_ranges,
)


class ContractTest(unittest.TestCase):
    def fixture(self):
        events = [
            EventRecord(1, 0, -1, 0, 100),
            EventRecord(2, 0, -1, 1, 100, 0, INVALID_U64, 4,
                        INVALID_U32, INVALID_U32, 1, 2.0, 0.0),
            EventRecord(3, 7, 0, 2, 100, 0, 11, 4, 8, 0, 1, 2.0, 3.0),
            EventRecord(5, 0, -1, 3, 100),
        ]
        return events, [LabelRecord(2, 4.0)], [QueryRangeRecord(100, 0, 4)]

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = bytes(range(32))
            events, labels, ranges = self.fixture()
            write_events(root / "events.bin", 8, identity, events)
            write_labels(root / "labels.bin", 8, identity, labels)
            write_query_ranges(root / "ranges.bin", 8, identity, ranges)
            event_header, actual_events = read_events(root / "events.bin")
            label_header, actual_labels = read_labels(root / "labels.bin")
            range_header, actual_ranges = read_query_ranges(root / "ranges.bin")
            self.assertEqual(80, event_header.record_size)
            self.assertEqual(identity, label_header.identity)
            self.assertEqual(identity, range_header.identity)
            self.assertEqual(events, actual_events)
            validate_dataset(actual_events, actual_labels, actual_ranges)

    def test_truncation_and_state_machine_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.bin"
            events, _, _ = self.fixture()
            write_events(path, 8, bytes(32), events)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ContractError):
                read_events(path)
            broken = list(events)
            broken[2] = EventRecord(3, 7, 0, 8, 100, 0, 11, 4, 8, 0, 1, 2.0, 3.0)
            with self.assertRaises(ContractError):
                write_events(path, 8, bytes(32), broken)


if __name__ == "__main__":
    unittest.main()
