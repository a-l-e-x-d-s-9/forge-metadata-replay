import json
import threading
import time
import unittest
from unittest.mock import patch

from metadata_replay.queue import Entry, ReplayQueue, decode_document, identity


def entry(name, prompt=None):
    return Entry(identity(name.encode()), name, "source infotext", {"Prompt": prompt or name})


class QueueTests(unittest.TestCase):
    def test_portable_round_trip_and_restored_source_preview(self):
        q = ReplayQueue()
        a, b = entry("a"), entry("b")
        a.source_preview = "data:image/jpeg;base64,source"
        a.generated_preview = "data:image/jpeg;base64,generated"
        q.merge([a, b])
        q.command("move", key=b.id, index=0)
        doc = q.document()
        self.assertNotIn("source_preview", doc["entries"][1])
        restored = ReplayQueue()
        restored.merge(decode_document(json.dumps(doc).encode()))
        restored.merge([a, b])
        self.assertEqual([e.name for e in restored.entries], ["b", "a"])
        self.assertEqual(restored.entries[1].source_preview, a.source_preview)
        self.assertEqual(restored.entries[1].generated_preview, a.generated_preview)

    def test_same_name_different_files_are_not_deduplicated(self):
        a, b = entry("first bytes"), entry("second bytes")
        a.name = b.name = "image.png"
        q = ReplayQueue()
        q.merge([a, b, a])
        self.assertEqual(len(q.entries), 2)

    def test_invalid_import_is_atomic(self):
        q = ReplayQueue()
        q.merge([entry("existing")])
        doc = q.document()
        doc["entries"].append({"id": "bad", "parameters": {"Prompt": "x"}})
        with self.assertRaises(ValueError):
            q.merge(decode_document(json.dumps(doc).encode()), replace=True)
        self.assertEqual(q.entries[0].name, "existing")

    def test_batches_pause_jump_skip_and_previous(self):
        q = ReplayQueue()
        a, b, c = entry("a"), entry("b"), entry("c")
        q.merge([a, b, c])
        q.begin(2)
        self.assertEqual(q.take(lambda: False)[0].id, a.id)
        q.complete_batch(a.id)
        self.assertEqual(q.take(lambda: False)[1], 1)
        q.command("pause")
        q.command("jump", key=c.id)
        q.complete_batch(a.id)
        tasks = []
        worker = threading.Thread(target=lambda: tasks.append(q.take(lambda: False)))
        worker.start()
        time.sleep(.05)
        self.assertTrue(q.waiting)
        self.assertFalse(tasks)
        q.command("resume")
        worker.join(1)
        self.assertEqual(tasks[0][0].id, c.id)
        self.assertEqual(tasks[0][1], 0)
        q.command("previous")
        q.complete_batch(c.id)
        self.assertEqual(q.take(lambda: False)[0].id, b.id)
        q.command("skip")
        q.complete_batch(b.id)
        self.assertEqual(q.take(lambda: False)[0].id, c.id)
        q.end()

    def test_interrupt_exits_paused_wait(self):
        q = ReplayQueue()
        q.merge([entry("a")])
        q.begin(1)
        q.command("pause")
        stop = threading.Event()
        tasks = []
        worker = threading.Thread(target=lambda: tasks.append(q.take(stop.is_set)))
        worker.start()
        stop.set()
        worker.join(1)
        self.assertEqual(tasks, [None])

    def test_reordering_and_merging_during_generation(self):
        q = ReplayQueue()
        a, b, c, d = (entry(name) for name in "abcd")
        q.merge([a, b, c])
        q.begin(1)
        q.take(lambda: False)
        q.command("move", key=c.id, index=1)
        q.merge([d])
        with self.assertRaises(ValueError):
            q.merge([d], replace=True)
        with self.assertRaises(ValueError):
            q.command("remove", key=a.id)
        q.complete_batch(a.id)
        self.assertEqual(q.take(lambda: False)[0].id, c.id)

    def test_sort_reverse_shuffle_and_manual_move_after_each(self):
        q = ReplayQueue()
        q.merge([entry(name) for name in ("image10.png", "Image2.png", "image1.png")])
        q.command("order", kind="filename")
        self.assertEqual([e.name for e in q.entries], ["image1.png", "Image2.png", "image10.png"])
        first = q.entries[0].id
        q.command("move", key=first, index=2)
        self.assertEqual(q.entries[2].id, first)
        q.command("order", kind="filename_reverse")
        self.assertEqual([e.name for e in q.entries], ["image10.png", "Image2.png", "image1.png"])
        q.command("move", key=first, index=0)
        self.assertEqual(q.entries[0].id, first)
        before = [e.id for e in q.entries]
        q.command("order", kind="reverse")
        self.assertEqual([e.id for e in q.entries], list(reversed(before)))
        with patch("metadata_replay.queue.random.SystemRandom.shuffle", side_effect=lambda rows: rows.reverse()) as shuffle:
            q.command("order", kind="shuffle")
            shuffle.assert_called_once()
        self.assertEqual([e.id for e in q.entries], before)
        q.command("move", key=first, index=1)
        self.assertEqual(q.entries[1].id, first)

    def test_sort_preserves_active_batch_and_paused_state(self):
        q = ReplayQueue()
        q.merge([entry(name) for name in ("c", "a", "b")])
        q.begin(2)
        current, _ = q.take(lambda: False)
        q.command("pause")
        q.command("order", kind="filename")
        self.assertTrue(q.paused)
        self.assertEqual(q.current, current.id)
        self.assertEqual(q.cursor, current.id)
        q.complete_batch(current.id)
        q.command("resume")
        self.assertEqual(q.take(lambda: False)[0].id, current.id)


if __name__ == "__main__":
    unittest.main()
