"""Behavior tests using a fake Forge processor; no GPU/model downloads."""
import sys
import types
import unittest
from unittest.mock import patch

from PIL import Image

from metadata_replay.forge import run_queue, strip_loras
from metadata_replay.queue import Entry, ReplayQueue, identity


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.state = types.SimpleNamespace(interrupted=False, stopping_generation=False,
                                           skipped=False, job_no=0)
        self.calls = []
        self.callback = lambda: None
        self.processing = types.ModuleType("modules.processing")
        self.processing.process_images = self.process
        self.modules = types.ModuleType("modules")
        self.modules.processing = self.processing
        self.modules.errors = types.SimpleNamespace(report=lambda *args, **kwargs: None)
        self.modules.shared = types.SimpleNamespace(state=self.state)
        self.modules.sd_models = types.SimpleNamespace(get_closet_checkpoint_match=lambda ref: None)
        self.modules.sd_samplers = types.SimpleNamespace(samplers_map={"euler": "Euler"})
        self.patcher = patch.dict(sys.modules, {"modules": self.modules, "modules.processing": self.processing})
        self.patcher.start()
        self.base = types.SimpleNamespace(n_iter=3, batch_size=2, seed=111, subseed=-1,
            subseed_strength=0, override_settings={"original": True}, extra_generation_params={},
            comments={}, styles=["UI style"], prompt="UI prompt", negative_prompt="UI negative",
            steps=20, cfg_scale=7., width=512, height=512, enable_hr=False, close=lambda: None)
        self.q = ReplayQueue()
        self.q.merge([Entry(identity(n.encode()), n, "", {"Prompt": n, "Seed": "123", "Steps": "5"}) for n in ("a", "b")])

    def tearDown(self):
        self.patcher.stop()

    def process(self, p):
        self.calls.append(p)
        self.state.job_no += 1
        self.callback()
        return types.SimpleNamespace(images=[Image.new("RGB", (32, 32))] * p.batch_size,
            all_prompts=[p.prompt] * p.batch_size, all_negative_prompts=[p.negative_prompt] * p.batch_size,
            all_seeds=[p.seed + i for i in range(p.batch_size)], all_subseeds=[-1] * p.batch_size,
            infotexts=[f"{p.prompt}; seed={p.seed + i}" for i in range(p.batch_size)],
            comments="", info="", seed=p.seed, extra_images=[])

    def test_batch_count_and_size_per_entry_and_aggregate(self):
        result = run_queue(self.base, self.q, {"Prompt", "Steps"})
        self.assertEqual(len(self.calls), 6)
        self.assertTrue(all(p.n_iter == 1 and p.batch_size == 2 for p in self.calls))
        self.assertEqual(result.all_prompts, ["a"] * 6 + ["b"] * 6)
        self.assertEqual(len(result.images), 12)
        self.assertEqual(len(result.infotexts), 12)
        self.assertEqual(len(set(result.all_seeds)), 12)
        self.assertTrue(all(p.steps == 5 and p.negative_prompt == "UI negative" and p.styles == [] for p in self.calls))
        self.assertEqual(self.base.override_settings, {"original": True})
        self.assertEqual(self.base.styles, ["UI style"])
        self.assertTrue(all(e.generated_preview for e in self.q.entries))

    def test_original_seed_increments_across_batches(self):
        result = run_queue(self.base, self.q, {"Prompt", "Original seed"})
        self.assertEqual(result.all_seeds, list(range(123, 129)) * 2)

    def test_skip_moves_to_next_entry_at_batch_boundary(self):
        def control():
            if len(self.calls) == 1:
                self.q.command("skip")
        self.callback = control
        result = run_queue(self.base, self.q, {"Prompt"})
        self.assertEqual(result.all_prompts, ["a"] * 2 + ["b"] * 6)
        self.assertEqual(self.q.entries[0].status, "partial")

    def test_interrupt_keeps_completed_results_and_resume_cursor(self):
        def control():
            if len(self.calls) == 2:
                self.state.interrupted = True
        self.callback = control
        result = run_queue(self.base, self.q, {"Prompt"})
        self.assertEqual(len(result.images), 4)
        self.assertFalse(self.q.active)
        self.assertEqual(self.q.cursor, self.q.entries[0].id)

    def test_unavailable_model_is_visible_and_other_entries_continue(self):
        self.q.entries[0].parameters["Model"] = "not-installed"
        result = run_queue(self.base, self.q, {"Prompt", "Model"})
        self.assertEqual(self.q.entries[0].status, "error")
        self.assertEqual(result.all_prompts, ["b"] * 6)
        self.assertIn("not-installed", result.comments)

    def test_loras_removed_by_default_without_editing_source_metadata(self):
        tag = "<lora:Example_Portrait_Style_1.0:1>"
        source = self.q.entries[0].parameters
        source["Prompt"] = "portrait " + tag
        source["Negative prompt"] = "<lora:negative:0.5> blurry"
        self.base.hr_prompt = "hires <LoRA:detail:0.7>"
        self.base.hr_negative_prompt = "<lora:noise:-1> noisy"
        run_queue(self.base, self.q, {"Prompt", "Negative prompt"})
        p = self.calls[0]
        self.assertEqual(p.prompt, "portrait")
        self.assertEqual(p.negative_prompt, "blurry")
        self.assertEqual(p.hr_prompt, "hires")
        self.assertEqual(p.hr_negative_prompt, "noisy")
        self.assertEqual(source["Prompt"], "portrait " + tag)
        self.assertIn(tag, self.q.document()["entries"][0]["parameters"]["Prompt"])
        self.assertEqual(self.base.hr_prompt, "hires <LoRA:detail:0.7>")

    def test_loras_preserved_when_option_disabled(self):
        prompt = "portrait <lora:Example_Portrait_Style_1.0:1>"
        self.q.entries[0].parameters["Prompt"] = prompt
        result = run_queue(self.base, self.q, {"Prompt"}, remove_loras=False)
        self.assertEqual(result.all_prompts[0], prompt)

    def test_strip_multiple_loras_preserves_other_prompt_syntax(self):
        self.assertEqual(strip_loras("<lora:a:1><LORA:b:0.5> (portrait:1.2), <hypernet:c:1>"),
                         "(portrait:1.2), <hypernet:c:1>")
        self.assertEqual(strip_loras("line one\n<lora:a:1>line two"), "line one\nline two")
        self.assertEqual(strip_loras(["<lora:a:1>portrait", "landscape"]), ["portrait", "landscape"])
        self.assertEqual(strip_loras("  text without tags  "), "  text without tags  ")


if __name__ == "__main__":
    unittest.main()
