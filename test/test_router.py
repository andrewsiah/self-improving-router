"""router.py with a fake pi: no model calls. Run: python3 -m unittest discover -s test"""
import contextlib, importlib, io, json, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Fake pi. A worker writes answer.txt (the "good" model writes the right answer) and a session file with usage.
# The judge (read-only tools) prefers whichever label's diff contains "42", or prints $FAKE_JUDGE when set.
FAKE_PI = r"""#!/usr/bin/env python3
import json, os, re, sys
from pathlib import Path
a = sys.argv[1:]
model = a[a.index("--model") + 1]
prompt = sys.stdin.read()
if a[a.index("--tools") + 1] == "read,grep,find,ls":
    if os.environ.get("FAKE_JUDGE"): print(os.environ["FAKE_JUDGE"]); sys.exit(0)
    good = next(l for l in ("A", "B") if "42" in Path(re.search(l + r": .*?diff (\S+)", prompt).group(1)).read_text())
    print("thinking...\n" + json.dumps({"preferred": good, "p1": {"A": [], "B": []}, "why": "right answer"})); sys.exit(0)
if "fail" in model: sys.exit(2)
Path("answer.txt").write_text("42\n" if "good" in model else "41\n")
s = Path(a[a.index("--session-dir") + 1]); s.mkdir(parents=True, exist_ok=True)
msg = {"role": "assistant", "usage": {"output": 100, "cost": {"total": 0.5 if "good" in model else 0.1}}}
(s / "x.jsonl").write_text(json.dumps({"type": "message", "message": msg}) + "\n")
print("done")
"""


class DualRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        pi = self.tmp / "pi"; pi.write_text(FAKE_PI); pi.chmod(0o755)
        os.environ.update(SELF_IMPROVING_ROUTER_DIR=str(self.tmp / "state"), SELF_IMPROVING_ROUTER_PI=str(pi))
        os.environ.pop("FAKE_JUDGE", None)
        sys.path.insert(0, str(ROOT))
        import router
        self.pr = importlib.reload(router)
        self.repo = self.tmp / "repo"
        git = lambda *a: subprocess.run(["git", "-C", str(self.repo), *a], check=True, capture_output=True)
        self.repo.mkdir(); git("init", "-q"); (self.repo / "README").write_text("x\n"); git("add", ".")
        git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "init")

    def task(self, models, name="20261113100000-abc123"):
        f = self.tmp / f"{name}.json"
        f.write_text(json.dumps({"prompt": "Write the answer to answer.txt", "models": models, "arm": "unsure",
                                 "route": {"tier": "standard", "type": "code", "unsure": True}}))
        return str(f)

    def run_cli(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            code = self.pr.main(list(argv))
        self.stdout = out.getvalue()
        return code

    def test_the_passing_preferred_model_wins(self):
        code = self.run_cli("dual-run", "--repo", str(self.repo), "--task", self.task(["p/bad", "p/good"]), "--check", "grep -q 42 answer.txt")
        self.assertEqual(code, 0)
        row = json.loads((self.tmp / "state/dual-runs.jsonl").read_text())
        self.assertEqual(row["winner"], "p/good"); self.assertEqual(row["preferred"], "p/good"); self.assertTrue(row["judged"])
        self.assertEqual(row["id"], "20261113100000-abc123"); self.assertEqual(row["arm"], "unsure")
        self.assertEqual(row["results"]["p/good"]["cost"], 0.5); self.assertFalse(row["results"]["p/bad"]["pass"])
        self.assertTrue((self.tmp / "state/runs/20261113100000-abc123/A.diff").exists())
        self.assertEqual(self.stdout.strip().splitlines()[-1], row["results"]["p/good"]["worktree"])
        self.assertEqual((Path(row["results"]["p/good"]["worktree"]) / "answer.txt").read_text(), "42\n")

    def test_a_failed_worker_loses_and_two_failures_skip(self):
        self.assertEqual(self.run_cli("dual-run", "--repo", str(self.repo), "--task", self.task(["p/fail", "p/bad"]), "--check", "true"), 0)
        self.assertEqual(json.loads((self.tmp / "state/dual-runs.jsonl").read_text())["winner"], "p/bad")
        self.assertEqual(self.run_cli("dual-run", "--repo", str(self.repo), "--task", self.task(["p/fail", "p/fail2"], "two"), "--check", "true"), 3)

    def test_no_verdict_falls_back_to_check_then_cost(self):
        os.environ["FAKE_JUDGE"] = "no json here"
        self.run_cli("dual-run", "--repo", str(self.repo), "--task", self.task(["p/good", "p/bad"]), "--check", "true")
        row = json.loads((self.tmp / "state/dual-runs.jsonl").read_text())
        self.assertFalse(row["judged"]); self.assertEqual(row["winner"], "p/bad")  # both pass, bad is cheaper

    def test_bad_task_file(self):
        with self.assertRaises(SystemExit):
            self.run_cli("dual-run", "--repo", str(self.repo), "--task", self.task(["p/a", "p/a"]), "--check", "true")

    def test_parse_verdict(self):
        v = self.pr.parse_verdict('text\n```\n{"preferred": "B", "p1": {"A": ["x.py:1 - bug"]}, "why": "w"}\n```')
        self.assertEqual(v, {"preferred": "B", "p1": {"A": ["x.py:1 - bug"], "B": []}, "why": "w"})
        self.assertIsNone(self.pr.parse_verdict('{"preferred": "C"}'))

    def test_outcomes_join(self):
        state = self.tmp / "state"; art = self.tmp / "art"; state.mkdir(); art.mkdir()
        (state / "routing.json").write_text(json.dumps({"routes": {"hard": ["p/big:high"]}}))
        rows = [{"ts": "2026-11-13T10:00:00.000Z", "session": "s", "call": "c1", "index": 0, "agent": "worker", "tier": "standard",
                 "minutes": 10, "tokens": 1000, "unsure": False, "model": "p/mid", "task_hash": "h", "outcome": "routed"},
                {"ts": "2026-11-13T10:30:00.000Z", "session": "s", "call": "c2", "index": 0, "agent": "worker", "tier": "hard",
                 "model": "p/big:high", "task_hash": "h", "outcome": "routed"},
                {"ts": "2026-11-13T10:31:00.000Z", "session": "s", "call": "c3", "index": 0, "agent": "worker", "outcome": "blocked-missing"}]
        (state / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (state / "runs.jsonl").write_text(json.dumps({"call": "c1", "run_id": "r1", "artifacts": str(art)}) + "\n")
        (art / "r1_worker_0_meta.json").write_text(json.dumps({"exitCode": 0, "usage": {"output": 2000}, "timestamp": 1794564000000 + 20 * 60000}))
        joined = self.pr.join_outcomes()
        self.assertEqual(len(joined), 2)
        first = joined[0]
        self.assertEqual((first["ok"], first["out_tokens"], first["wall_min"], first["retried"], first["escalated"]), (True, 2000, 20.0, True, True))
        self.assertTrue(joined[1]["top_model"])
        summary = self.pr.summarize(joined)
        self.assertIn("hard-tier model 1/2 (50%)", summary)
        self.assertIn("actual/estimate median: minutes 2.0, tokens 2.0", summary)


if __name__ == "__main__":
    unittest.main()
