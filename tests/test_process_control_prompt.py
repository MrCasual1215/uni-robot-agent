import tempfile
import unittest
from pathlib import Path

from uni_robot_agent.context.prompt import Prompt, PromptContext, assemble_prompt


class ProcessControlPromptTests(unittest.TestCase):
    def test_cancellation_guidance_reaches_final_prompt_once(self):
        cases = (
            ("clean", "tool_calling", "immediate", ()),
            ("direct", "tool_calling", "immediate", ()),
            ("direct", "tool_calling", "background", ()),
            ("hybrid", "tool_calling", "background", ("hybrid.md",)),
            ("direct", "code_as_policy", "immediate", ()),
            ("direct", "code_as_policy", "background", ()),
            ("readonly", "tool_calling", "immediate", ()),
        )
        for profile, interaction, returns, resources in cases:
            for scheduling in ("per-resource", "global-serial"):
                for persist in (False, True):
                    with self.subTest(profile=profile, interaction=interaction,
                                      returns=returns, scheduling=scheduling, persist=persist):
                        with tempfile.TemporaryDirectory() as directory:
                            session_dir = Path(directory) if persist else None
                            context = PromptContext(
                                integration_name="fixture", task="Inspect the scene",
                                tool_profile=profile, interaction_mode=interaction,
                                tool_return_mode=returns, resource_scheduling=scheduling,
                                tool_prompt_resources=resources, session_dir=session_dir,
                            )
                            prompt = assemble_prompt(Prompt(user=context.task), context)
                            text = prompt.flatten()
                            self.assertEqual(text.count("Manage only computation processes you started."), 1)
                            self.assertIn("cancellation mechanism for its exact returned\nsession or job", text)
                            self.assertIn("specific PID recorded", text)
                            self.assertIn("Do not use broad pkill/killall matches", text)
                            self.assertIn("does not restrict which processes a kill command can affect", text)
                            self.assertIn("never replay them blindly", text)
                            self.assertEqual(sum(Path(source).name == "process_control.md"
                                                 for source in prompt.sources), 1)
                            if session_dir:
                                self.assertEqual((session_dir / "effective_prompt.md").read_text(), text)


if __name__ == "__main__":
    unittest.main()
