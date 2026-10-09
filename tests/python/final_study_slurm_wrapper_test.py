"""Regression checks for Slurm's copied batch scripts; no Slurm allocation needed."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(os.environ.get("EDGE_STUDY_TEST_WORKER",
    str(ROOT / "scripts/edge_estimation/final_study/worker.sh")))
SUBMIT = ROOT / "scripts/edge_estimation/submit_final_study.sh"
BASH = os.environ.get("EDGE_STUDY_TEST_BASH") or shutil.which("bash")
if not BASH and os.name == "nt":
    candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/bin/bash.exe"
    if candidate.is_file():
        BASH = str(candidate)
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


@unittest.skipUnless(BASH, "Bash is required for Slurm wrapper regression tests")
class SlurmWrapperTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="final-study-slurm-")
        self.root = Path(self.temporary.name).resolve()
        assert self.root.is_relative_to(Path(tempfile.gettempdir()).resolve())
        self.addCleanup(self.temporary.cleanup)
        self.repo = self.root / "checkout with spaces"
        self.repo.mkdir()
        subprocess.run(["git", "init", "--quiet", str(self.repo)], check=True, capture_output=True)
        self.write(self.repo / "CMakeLists.txt", "# fixture\n")
        for name in ("scripts", "scripts/edge_estimation", "scripts/edge_estimation/final_study"):
            self.write(self.repo / name / "__init__.py", "")
        self.write(self.repo / "scripts/edge_estimation/final_study/run.py",
                   "import json, os, sys\n"
                   f"print(json.dumps({{'cwd':os.getcwd(),'args':sys.argv[1:],"
                   f"'threads':{{k:os.environ.get(k) for k in {THREADS!r}}}}}))\n")
        self.outside = self.root / "outside"
        self.outside.mkdir()
        self.spooled = self.root / "var/spool/slurmd/job987654/slurm_script"
        self.spooled.parent.mkdir(parents=True)
        shutil.copyfile(WORKER, self.spooled)
        self.bin = self.root / "stub-bin"
        self.bin.mkdir()
        self.log = self.root / "stub.log"
        self.env = {**os.environ, "SLURM_JOB_ID": "987654",
                    "USER": self.root.name, "PYTHON": Path(sys.executable).as_posix(),
                    "STUB_LOG": self.log.as_posix(),
                    "PATH": str(self.bin) + os.pathsep + os.environ["PATH"]}
        for key in ("REPO", "PYTHONPATH", "PYTHONHOME", "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            self.env.pop(key, None)
        self.env["OMP_NUM_THREADS"] = "8"
        self.env["SLURM_SUBMIT_DIR"] = self.outside.as_posix()
        self.write(self.bin / "flock", "#!/usr/bin/env bash\nexit 0\n")
        self.write(self.bin / "srun",
                   '#!/usr/bin/env bash\nset -euo pipefail\n'
                   'printf "%s\\0" "$@" > "$STUB_LOG"\n'
                   'while [[ "$1" == --* ]]; do\n'
                   '  case "$1" in\n'
                   '    --chdir=*) cd -- "${1#--chdir=}" ;;\n'
                   '    --ntasks=1|--cpus-per-task=1|--cpu-bind=cores) ;;\n'
                   '    *) echo "unexpected srun option: $1" >&2; exit 9 ;;\n'
                   '  esac\n'
                   '  shift\n'
                   'done\n'
                   'exec "$@"\n')

    @staticmethod
    def write(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def execute(self, script, cwd, *args):
        return subprocess.run([BASH, "--noprofile", "--norc", str(script), *map(str, args)],
                              cwd=cwd, env=self.env, text=True, capture_output=True)

    def assert_worker(self, cwd):
        result = self.execute(self.spooled, cwd, "--cases", "case file.json", "--allocation", "0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        doc = json.loads(result.stdout)
        self.assertEqual(Path(doc["cwd"]).resolve(), self.repo.resolve())
        self.assertEqual(doc["args"], ["--cases", "case file.json", "--allocation", "0"])
        self.assertEqual(doc["threads"], {key: "1" for key in THREADS})
        args = self.log.read_bytes().decode().split("\0")
        for arg in ("--ntasks=1", "--cpus-per-task=1", "--cpu-bind=cores"):
            self.assertIn(arg, args)
        self.assertTrue(any(arg.startswith("--chdir=") for arg in args))

    def test_spooled_worker_uses_explicit_repo(self):
        self.env["REPO"] = self.repo.as_posix()
        self.assert_worker(self.outside)

    def test_spooled_worker_uses_git_checkout_when_repo_unset(self):
        self.assert_worker(self.repo / "scripts")

    def test_invalid_explicit_repo_is_rejected(self):
        self.env["REPO"] = self.outside.as_posix()
        result = self.execute(self.spooled, self.repo)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid study repository", result.stderr)
        self.assertFalse(self.log.exists())

    def test_missing_repo_and_unrelated_cwd_are_rejected(self):
        result = self.execute(self.spooled, self.outside)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cannot locate repository", result.stderr)
        self.assertFalse(self.log.exists())

    def test_submitter_exports_repo_and_sets_chdir_from_any_directory(self):
        submit = self.repo / "scripts/edge_estimation/submit_final_study.sh"
        shutil.copyfile(SUBMIT, submit)
        capture = self.root / "capture_sbatch.py"
        self.write(capture,
                   "import json, os, sys\n"
                   "with open(os.environ['STUB_LOG'], 'a', encoding='utf-8') as f:\n"
                   "    f.write(json.dumps({'repo':sys.argv[1], 'args':sys.argv[2:]})+'\\n')\n"
                   "print('987654')\n")
        self.env["TEST_CAPTURE"] = capture.as_posix()
        self.env["REPO"] = self.outside.as_posix()  # A stale caller value must not win.
        self.write(self.bin / "sbatch",
                   '#!/usr/bin/env bash\nset -euo pipefail\n'
                   'exec "$PYTHON" "$TEST_CAPTURE" "$REPO" "$@"\n')
        result = self.execute(submit, self.outside, "partition", "account", "00:01:00",
                              self.root / "results", "--protocol", "protocol.json",
                              "--dataset", "dataset.json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(len(records), 3)
        for allocation, record in enumerate(records):
            self.assertEqual(Path(record["repo"]).resolve(), self.repo.resolve())
            args = record["args"]
            chdir = next(a.split("=", 1)[1] for a in args if a.startswith("--chdir="))
            self.assertEqual(Path(chdir).resolve(), self.repo.resolve())
            self.assertIn("--exclusive", args)
            self.assertIn("--export=ALL", args)
            self.assertEqual(args[args.index("--allocation") + 1], str(allocation))


if __name__ == "__main__":
    unittest.main()
