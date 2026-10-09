"""Affinity regressions, including unbound Slurm steps and sealed old results."""
from __future__ import annotations
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.edge_estimation.final_study import affinity, environment
from scripts.edge_estimation.final_study.common import identity, seal, write
from scripts.edge_estimation.final_study.results import rows


class AffinityTest(unittest.TestCase):
    def test_selects_only_from_allowed_nonzero_cpus(self):
        with patch.object(os, "sched_getaffinity", create=True,
                          side_effect=[{8, 12, 44}, {8}]), \
             patch.object(os, "sched_setaffinity", create=True) as setter:
            self.assertEqual(affinity.pin_current_process(), 8)
            setter.assert_called_once_with(0, {8})

    def test_ineffective_binding_is_rejected(self):
        with patch.object(os, "sched_getaffinity", create=True,
                          return_value=set(range(192))), \
             patch.object(os, "sched_setaffinity", create=True):
            with self.assertRaisesRegex(ValueError, "exactly one logical CPU"):
                affinity.pin_current_process()

    def test_binding_to_a_different_cpu_is_rejected(self):
        with patch.object(os, "sched_getaffinity", create=True,
                          side_effect=[{8, 12}, {12}]), \
             patch.object(os, "sched_setaffinity", create=True):
            with self.assertRaisesRegex(ValueError, "affinity changed"):
                affinity.pin_current_process()

    def test_empty_allowed_set_is_rejected(self):
        with patch.object(os, "sched_getaffinity", create=True, return_value=set()), \
             patch.object(os, "sched_setaffinity", create=True) as setter:
            with self.assertRaisesRegex(ValueError, "no allowed CPUs"):
                affinity.pin_current_process()
            setter.assert_not_called()

    def test_os_rejection_is_not_ignored(self):
        with patch.object(os, "sched_getaffinity", create=True, return_value={8}), \
             patch.object(os, "sched_setaffinity", create=True,
                          side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                affinity.pin_current_process()

    def test_changed_mask_before_a_runner_is_rejected(self):
        for current in ({8, 12}, {12}):
            with self.subTest(current=current), \
                 patch.object(os, "sched_getaffinity", create=True, return_value=current):
                with self.assertRaises(ValueError):
                    affinity.check_current_cpu([8])

    def test_launcher_pins_before_exec_and_preserves_arguments(self):
        calls = []
        with patch.object(sys, "platform", "linux"), \
             patch.dict(os.environ, {"SLURM_JOB_ID": "1"}), \
             patch.object(sys, "argv", ["affinity", "--cases", "case file.json"]), \
             patch.object(affinity, "pin_current_process", side_effect=lambda: calls.append("pin")), \
             patch.object(os, "execv", side_effect=lambda *a: calls.append(a)):
            affinity.main()
        self.assertEqual(calls, ["pin", (sys.executable, [
            sys.executable, "-m", "scripts.edge_estimation.final_study.run",
            "--cases", "case file.json"
        ])])

    @unittest.skipUnless(sys.platform == "linux", "requires real Linux affinity APIs")
    def test_real_child_inherits_pin(self):
        before = os.sched_getaffinity(0)
        code = (
            "import json, os, subprocess, sys; "
            "from scripts.edge_estimation.final_study.affinity import pin_current_process; "
            "cpu=pin_current_process(); "
            "child=json.loads(subprocess.check_output([sys.executable,'-c',"
            "'import json,os; print(json.dumps(sorted(os.sched_getaffinity(0))))'],text=True)); "
            "print(json.dumps({'cpu':cpu,'child':child,'parent':sorted(os.sched_getaffinity(0))}))"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                check=True, text=True, capture_output=True)
        record = json.loads(result.stdout)
        self.assertIn(record["cpu"], before)
        self.assertEqual(record["child"], [record["cpu"]])
        self.assertEqual(record["parent"], record["child"])
        self.assertEqual(os.sched_getaffinity(0), before)


class RuntimeGateTest(unittest.TestCase):
    def inspect(self, cpus, formal=True, job="OverSubscribe=NO", nodes="1"):
        env = {key: "1" for key in environment.THREAD_VARS}
        env.update(SLURM_JOB_ID="1", SLURM_JOB_NUM_NODES=nodes)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, env))
            stack.enter_context(patch.object(environment.platform, "system", return_value="Linux"))
            stack.enter_context(patch.object(os, "sched_getaffinity", create=True, return_value=set(cpus)))
            stack.enter_context(patch.object(environment, "command", return_value=job))
            stack.enter_context(patch.object(Path, "glob", return_value=[]))
            return environment.runtime(formal)

    def test_formal_gate_rejects_full_node_mask(self):
        with self.assertRaisesRegex(ValueError, "exactly one logical CPU"):
            self.inspect(range(192))

    def test_formal_gate_accepts_single_cpu(self):
        self.assertEqual(self.inspect([8])["affinity"], [8])

    def test_local_smoke_can_remain_unbound(self):
        self.assertEqual(self.inspect([8, 12], formal=False)["affinity"], [8, 12])

    def test_exclusive_and_single_node_checks_still_apply(self):
        with self.assertRaisesRegex(ValueError, "exclusive"):
            self.inspect([8], job="OverSubscribe=YES")
        with self.assertRaisesRegex(ValueError, "one node"):
            self.inspect([8], nodes="2")


class PublishedResultsTest(unittest.TestCase):
    def read_fixture(self, cpus, smoke=False):
        with tempfile.TemporaryDirectory(prefix="affinity-results-") as temporary:
            root = Path(temporary)
            cases, manifest, metrics = (root / f for f in ("cases.json", "manifest.json", "metrics.json"))
            query, case = root / "query.txt", root / "case.json"
            query.write_text("0\n", encoding="utf-8")
            seal(cases, {"cases": [{"case_config_id": "x"}]})
            seal(manifest, {
                "local_smoke": smoke, "runtime": {"affinity": cpus}, "bindings": {},
                "phase": "dev", "allocation_id": 0, "blocks": 1, "cases": identity(cases)
            })
            write(metrics, {"phase": "dev", "allocation_id": 0,
                            "block_id": 0, "case_config_id": "x"})
            seal(case, {"identity": {"run_manifest": identity(manifest), "query_order": identity(query)},
                        "outputs": [identity(metrics)]})
            seal(root / "complete.json", {"run_manifest": identity(manifest), "cases": [identity(case)]})
            return rows([root], phase="dev")

    def test_sealed_unbound_formal_results_are_rejected(self):
        for mask in (list(range(192)), None, [], [True]):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "exactly one logical CPU"):
                self.read_fixture(mask)

    def test_sealed_pinned_formal_results_are_accepted(self):
        values, _ = self.read_fixture([8])
        self.assertEqual(len(values), 1)

    def test_sealed_local_smoke_keeps_its_existing_behavior(self):
        values, _ = self.read_fixture([8, 12], smoke=True)
        self.assertEqual(len(values), 1)


if __name__ == "__main__":
    unittest.main()
