"""Tests for the runner. They use a fake model, so no GPU or network is needed."""
import json
import tempfile
import unittest
from pathlib import Path

from contracts.check import check_response
from contracts.io import load_responses
from contracts.models import Variant
from runner.mock import make_mock_chat
from runner.run import play_variant, run_all, status

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = {"backend": "mock", "model_id": "mock", "temperature": 0, "max_tokens": 4096}


def example_variant():
    with open(ROOT / "contracts" / "examples" / "variant.json", encoding="utf-8") as f:
        return Variant.from_dict(json.load(f))


class Counting:
    """A fake model that counts calls and can fail on a chosen call."""
    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.history_sizes = []

    def __call__(self, messages):
        self.calls += 1
        self.history_sizes.append(len(messages))
        if self.calls == self.fail_on_call:
            raise RuntimeError("the server broke")
        return f"reply {self.calls}", False


class PlayVariantTests(unittest.TestCase):
    def test_each_reply_is_added_to_the_history_before_the_next_turn(self):
        chat = Counting()
        variant = example_variant()
        replies, cut_off = play_variant(variant, chat)
        self.assertEqual(len(replies), len(variant.turns))
        self.assertEqual(len(cut_off), len(variant.turns))
        # Turn 1 sends 1 message, turn 2 sends 3 (user, reply, user), turn 3 sends 5, and so on.
        self.assertEqual(chat.history_sizes, [1, 3, 5, 7, 9, 11])

    def test_saved_replies_are_not_asked_for_again(self):
        chat = Counting()
        variant = example_variant()
        play_variant(variant, chat, saved_replies=["a", "b", "c"], saved_cut_off=[False, False, False])
        self.assertEqual(chat.calls, len(variant.turns) - 3)


class RunAllTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.out = self.folder.name
        self.variants = [example_variant()]

    def tearDown(self):
        self.folder.cleanup()

    def test_mock_run_gives_valid_responses(self):
        summary = run_all(self.variants, make_mock_chat(), "mock", SETTINGS, self.out, workers=1, log=lambda m: None)
        self.assertEqual(summary["finished"], 1)
        responses = load_responses(Path(self.out) / "responses.jsonl")
        self.assertEqual(check_response(responses[0], self.variants[0]), [])

    def test_a_finished_run_is_skipped(self):
        run_all(self.variants, make_mock_chat(), "mock", SETTINGS, self.out, workers=1, log=lambda m: None)
        chat = Counting()
        summary = run_all(self.variants, chat, "mock", SETTINGS, self.out, workers=1, log=lambda m: None)
        self.assertEqual(chat.calls, 0)
        self.assertEqual(summary["already_done"], 1)

    def test_changed_settings_redo_the_run(self):
        run_all(self.variants, make_mock_chat(), "mock", SETTINGS, self.out, workers=1, log=lambda m: None)
        chat = Counting()
        run_all(self.variants, chat, "mock", {**SETTINGS, "max_tokens": 8192}, self.out, workers=1, log=lambda m: None)
        self.assertEqual(chat.calls, len(self.variants[0].turns))

    def test_a_stopped_run_continues_where_it_stopped(self):
        broken = Counting(fail_on_call=4)      # three turns work, the fourth call fails
        summary = run_all(self.variants, broken, "mock", SETTINGS, self.out, workers=1, log=lambda m: None)
        self.assertEqual(len(summary["failed"]), 1)
        self.assertEqual(status(self.variants, "mock", SETTINGS, self.out)["half_way"], 1)
        working = Counting()
        run_all(self.variants, working, "mock", SETTINGS, self.out, workers=1, log=lambda m: None)
        self.assertEqual(working.calls, len(self.variants[0].turns) - 3)   # the first 3 replies were not asked again
        self.assertEqual(status(self.variants, "mock", SETTINGS, self.out)["half_way"], 0)

    def test_a_cut_off_reply_is_asked_again_on_resume(self):
        def chat(messages):
            return "short", len(messages) == 3   # the reply to turn 2 is cut off
        run_all(self.variants, chat, "mock", SETTINGS, self.out, workers=1, log=lambda m: None)
        # Pretend the run stopped after turn 3, with the cut-off reply saved in the partial file.
        partial = {f"{self.variants[0].conv_id}|{self.variants[0].condition}|mock":
                   {"replies": ["r1", "r2", "r3"], "cut_off": [False, True, False]}}
        (Path(self.out) / "responses.jsonl").unlink()
        (Path(self.out) / "partial.json").write_text(json.dumps(partial), encoding="utf-8")
        again = Counting()
        run_all(self.variants, again, "mock", SETTINGS, self.out, workers=1, log=lambda m: None)
        # Only turn 1 is kept. Turns 2 to 6 are asked again.
        self.assertEqual(again.calls, len(self.variants[0].turns) - 1)


if __name__ == "__main__":
    unittest.main()
