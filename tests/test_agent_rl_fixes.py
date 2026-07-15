import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AgentRLFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_script("build_lifeos_agent_rl_dataset")
        cls.patcher = load_script("patch_minimind_agent_rl")

    def test_lifeos_dataset_has_one_gt_per_tool_sample(self):
        rows = self.builder.build_rows(
            self.builder.load_jsonl(ROOT / "dataset" / "lifeos_sft_seed.jsonl"),
            tool_repeat=1,
            no_tool_repeat=1,
            seed=42,
        )
        self.assertEqual(len(rows), 26)
        for row in rows:
            system = row["conversations"][0]
            if row["gt"]:
                self.assertEqual(len(row["gt"]), 1)
                self.assertIn("tools", system)
            else:
                self.assertNotIn("tools", system)
            self.assertEqual(row["conversations"][-1], {"role": "assistant", "content": ""})

    def test_patch_is_idempotent_and_adds_hard_guards(self):
        source = (ROOT / "vendor" / "minimind-master" / "trainer" / "train_agent.py").read_text(encoding="utf-8")
        patched, changed = self.patcher.patch_text(source)
        self.assertTrue(changed)
        self.assertIn("expects_tool_call and not tool_calls", patched)
        self.assertIn("valid_call_count != len(tool_calls)", patched)
        self.assertIn("not calls or not valid_names", patched)
        patched_again, changed_again = self.patcher.patch_text(patched)
        self.assertFalse(changed_again)
        self.assertEqual(patched_again, patched)


if __name__ == "__main__":
    unittest.main()
