"""router.py with a fake pi: no model calls. Run: python3 -m unittest discover -s test"""
import contextlib, importlib, io, json, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Fake pi. A worker writes answer.txt (a "good" model writes the right answer, and a test for it when it is "tests")
# and a session file with usage. A read-only call is a pairwise judge when its prompt says "Candidate 1", else a
# review. The judge prefers the candidate whose diff contains "42", answers "1" every time when $FAKE_FIRST is set, or
# prints $FAKE_JUDGE as is. The review lists a P1 for the side whose diff lacks "42".
FAKE_PI = r"""#!/usr/bin/env python3
import json, os, re, sys
from pathlib import Path
a = sys.argv[1:]
model = a[a.index("--model") + 1]
prompt = sys.stdin.read()
s = Path(a[a.index("--session-dir") + 1]); s.mkdir(parents=True, exist_ok=True)
def session(cost):
    msg = {"role": "assistant", "usage": {"output": 100, "cost": {"total": cost}}}
    (s / "x.jsonl").write_text(json.dumps({"type": "message", "message": msg}) + "\n")
if a[a.index("--tools") + 1] == "read,grep,find,ls":
    session(0.01)
    if os.environ.get("FAKE_JUDGE"): print(os.environ["FAKE_JUDGE"]); sys.exit(0)
    has42 = lambda p: "42" in Path(p).read_text()
    if "Candidate 1" in prompt:
        if os.environ.get("FAKE_FIRST"): answer = "1"
        else: answer = next(n for n in "12" if has42(re.search(f"Candidate {n}: .*?diff (\\S+)", prompt).group(1)))
        print("thinking...\n" + json.dumps({"preferred": answer, "why": "right answer"})); sys.exit(0)
    bad = next(l for l in ("A", "B") if not has42(re.search(l + r": .*?diff (\S+)", prompt).group(1)))
    print(json.dumps({"p1": {"A": [], "B": [], bad: ["answer.txt:1 - wrong"]}, "why": "wrong answer"})); sys.exit(0)
if "fail" in model: sys.exit(2)
Path("answer.txt").write_text("42\n" if "good" in model else "41\n")
if "tests" in model: Path("test").mkdir(); Path("test/t.sh").write_text("grep -q 42 answer.txt\n")
session(0.5 if "good" in model else 0.1)
print("done")
"""


class DualRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        pi = self.tmp / "pi"; pi.write_text(FAKE_PI); pi.chmod(0o755)
        os.environ.update(SELF_IMPROVING_ROUTER_DIR=str(self.tmp / "state"), SELF_IMPROVING_ROUTER_PI=str(pi))
        for k in ("FAKE_JUDGE", "FAKE_FIRST"): os.environ.pop(k, None)
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

    def dual(self, models, check="grep -q 42 answer.txt", name="20261113100000-abc123"):
        code = self.run_cli("dual-run", "--repo", str(self.repo), "--task", self.task(models, name), "--check", check)
        rows = (self.tmp / "state/dual-runs.jsonl").read_text().splitlines()
        return code, json.loads(rows[-1])

    def configure(self, **dual):
        (self.tmp / "state").mkdir(exist_ok=True)
        (self.tmp / "state/routing.json").write_text(json.dumps({"dual": dual}))

    def test_the_passing_preferred_model_wins(self):
        code, row = self.dual(["p/bad", "p/good"])
        self.assertEqual(code, 0)
        self.assertEqual(row["winner"], "p/good"); self.assertEqual(row["preferred"], "p/good"); self.assertTrue(row["judged"])
        self.assertEqual(row["id"], "20261113100000-abc123"); self.assertEqual(row["arm"], "unsure"); self.assertEqual(row["decidedBy"], "check")
        self.assertEqual(row["judge"]["verdict"], row["results"]["p/good"]["label"]); self.assertEqual(row["judgeCost"], 0.02)  # two judge calls
        self.assertIsNone(row["review"]); self.assertIsNone(row["audit"])
        self.assertEqual(row["results"]["p/good"]["cost"], 0.5); self.assertFalse(row["results"]["p/bad"]["pass"])
        self.assertTrue((self.tmp / "state/runs/20261113100000-abc123/A.diff").exists())
        self.assertEqual(self.stdout.strip().splitlines()[-1], row["results"]["p/good"]["worktree"])
        self.assertEqual((Path(row["results"]["p/good"]["worktree"]) / "answer.txt").read_text(), "42\n")

    def test_a_failed_worker_loses_without_judging_and_two_failures_skip(self):
        code, row = self.dual(["p/fail", "p/bad"], "true")
        self.assertEqual((code, row["winner"], row["judge"], row["judged"]), (0, "p/bad", None, False))
        self.assertEqual(self.run_cli("dual-run", "--repo", str(self.repo), "--task", self.task(["p/fail", "p/fail2"], "two"), "--check", "true"), 3)

    def test_no_verdict_falls_back_to_cost(self):
        os.environ["FAKE_JUDGE"] = "no json here"
        _, row = self.dual(["p/good", "p/bad"], "true")
        self.assertFalse(row["judged"]); self.assertEqual((row["winner"], row["decidedBy"]), ("p/bad", "cost"))  # both pass, bad is cheaper

    def test_the_other_sides_new_tests_decide(self):
        os.environ["FAKE_JUDGE"] = "no json here"
        _, row = self.dual(["p/good-tests", "p/bad"], "for t in test/*.sh; do [ -e $t ] || continue; bash $t || exit 1; done")
        self.assertEqual((row["winner"], row["decidedBy"]), ("p/good-tests", "cross"))  # bad is cheaper, but fails good's test
        good, bad = row["results"]["p/good-tests"], row["results"]["p/bad"]
        self.assertEqual((good["cross"]["files"], good["cross"]["ran"]), (0, False))
        self.assertEqual((bad["cross"]["files"], bad["cross"]["pass"]), (1, False))
        self.assertFalse((Path(bad["worktree"]) / "test/t.sh").exists())  # restored after the cross-test

    def test_disagreeing_orders_are_a_tie_that_goes_to_the_cheaper_arm(self):
        os.environ["FAKE_FIRST"] = "1"  # always answers Candidate 1, so the two orders disagree
        _, row = self.dual(["p/good", "p/bad"], "true")
        self.assertEqual((row["judge"]["verdict"], row["judged"], row["preferred"]), ("discordant", True, None))
        self.assertEqual((row["winner"], row["decidedBy"]), ("p/bad", "cost"))

    def test_review_p1s_come_before_the_judge(self):
        os.environ["FAKE_FIRST"] = "1"
        self.configure(review="p/rev")
        _, row = self.dual(["p/good", "p/bad"], "true")
        self.assertEqual((row["winner"], row["decidedBy"]), ("p/good", "p1s"))
        self.assertEqual(len(row["results"]["p/bad"]["p1s"]), 1); self.assertEqual(row["judgeCost"], 0.03)

    def test_audit_runs_at_its_rate_and_records_agreement(self):
        self.configure(audit="p/audit", auditRate=1)
        _, row = self.dual(["p/good", "p/bad"], "true")
        self.assertEqual(row["audit"]["model"], "p/audit"); self.assertTrue(row["audit"]["agree"]); self.assertEqual(row["judgeCost"], 0.04)
        self.configure(audit="p/audit", auditRate=0)
        _, row = self.dual(["p/good", "p/bad"], "true")
        self.assertIsNone(row["audit"])
        self.run_cli("report")
        self.assertIn("audit agreed 1/1", self.stdout)

    def test_cross_tests_keep_modes_and_never_follow_symlinks(self):
        mine = self.tmp / "wt"; (mine / "tests").mkdir(parents=True)
        outside = self.tmp / "outside.sh"
        rival_wt = self.tmp / "rival"; (rival_wt / "tests").mkdir(parents=True)
        for name in ("run.sh", "link.sh"): (rival_wt / "tests" / name).write_text("echo rival\n")
        git = lambda wt, *a: subprocess.run(["git", "-C", str(wt), *a], check=True, capture_output=True, text=True)
        for wt in (mine, rival_wt):
            git(wt, "init", "-q")
        (mine / "tests/run.sh").write_text("echo mine\n"); (mine / "tests/run.sh").chmod(0o755)
        (mine / "tests/link.sh").symlink_to(outside)  # dangling
        for wt in (mine, rival_wt): git(wt, "add", "-A"); git(wt, "-c", "user.name=t", "-c", "user.email=t@e.c", "commit", "-qm", "x")
        base = git(rival_wt, "rev-parse", "HEAD").stdout.strip()
        (rival_wt / "tests/run.sh").write_text("echo rival changed\n")
        (rival_wt / "tests/link.sh").write_text("echo rival changed\n")
        arm, rival = {"worktree": str(mine), "label": "A"}, {"worktree": str(rival_wt), "label": "B"}
        rec = self.pr.cross_test(arm, rival, "true", base, self.tmp, "")
        self.assertIn("skip", rec); self.assertFalse(outside.exists()); self.assertTrue((mine / "tests/link.sh").is_symlink())
        (mine / "tests/link.sh").unlink(); (mine / "tests/link.sh").write_text("echo mine\n")
        rec = self.pr.cross_test(arm, rival, "true", base, self.tmp, "")
        self.assertEqual((rec["ran"], rec["pass"]), (True, True))
        self.assertEqual(((mine / "tests/run.sh").read_text(), (mine / "tests/run.sh").stat().st_mode & 0o777), ("echo mine\n", 0o755))

    def test_a_cross_test_with_a_file_where_a_dir_is_needed_skips(self):
        # The rival adds tests/escape/z.sh but the arm has a regular file tests/escape: copying would fail mid-way,
        # so the step skips and every file the arm had is restored or left untouched.
        mine = self.tmp / "wt"; (mine / "tests").mkdir(parents=True)
        (mine / "tests/a.sh").write_text("echo mine\n"); (mine / "tests/escape").write_text("a file\n")
        rival_wt = self.tmp / "rival"; (rival_wt / "tests").mkdir(parents=True)
        git = lambda wt, *a: subprocess.run(["git", "-C", str(wt), *a], check=True, capture_output=True, text=True)
        git(rival_wt, "init", "-q")
        (rival_wt / "README").write_text("x\n"); git(rival_wt, "add", ".")
        git(rival_wt, "-c", "user.name=t", "-c", "user.email=t@e.c", "commit", "-qm", "x")
        base = git(rival_wt, "rev-parse", "HEAD").stdout.strip()
        (rival_wt / "tests/a.sh").write_text("echo rival\n"); (rival_wt / "tests/escape").mkdir()
        (rival_wt / "tests/escape/z.sh").write_text("echo rival\n"); git(rival_wt, "add", "-N", ".")
        rec = self.pr.cross_test({"worktree": str(mine), "label": "A"}, {"worktree": str(rival_wt), "label": "B"}, "true", base, self.tmp, "")
        self.assertIn("skip", rec); self.assertFalse(rec["ran"])
        self.assertEqual((mine / "tests/a.sh").read_text(), "echo mine\n")
        self.assertEqual((mine / "tests/escape").read_text(), "a file\n")

    def test_a_skipped_cross_test_counts_for_neither_side(self):
        arm = lambda label, cross: {"label": label, "exit": 0, "pass": True, "cross": cross, "p1s": [], "cost": 0.1, "minutes": 1}
        a, b = arm("A", {"ran": True, "pass": False, "void": True}), arm("B", {"ran": False, "pass": None, "skip": "x", "void": True})
        self.assertIs(self.pr.pick_winner([b, a], "A"), a); self.assertEqual(self.pr.decided_by(a, b, "A"), "judge")

    def test_report_reads_rows_from_before_the_judge_upgrade(self):
        state = self.tmp / "state"; state.mkdir()
        (state / "dual-runs.jsonl").write_text(json.dumps({"id": "old", "judge": "p/judge", "judged": True, "winner": "p/a", "results": {
            "p/a": {"pass": True, "p1s": [], "cost": 0.1, "minutes": 1.0}, "p/b": {"pass": True, "p1s": [], "cost": 0.2, "minutes": 1.0}}}) + "\n")
        self.assertEqual(self.run_cli("report"), 0); self.assertIn("0 discordant, 0/1", self.stdout)

    def test_ties_go_to_the_cheaper_then_the_faster_arm(self):
        arm = lambda label, cost, minutes: {"label": label, "exit": 0, "pass": True, "cross": {"pass": None}, "p1s": [], "cost": cost, "minutes": minutes}
        a, b = arm("A", 0.2, 5), arm("B", 0.2, 3)
        self.assertIs(self.pr.pick_winner([a, b], "discordant"), b); self.assertEqual(self.pr.decided_by(b, a, "same"), "minutes")
        a["cost"] = 0.1
        self.assertIs(self.pr.pick_winner([a, b], None), a); self.assertEqual(self.pr.decided_by(a, b, None), "cost")
        self.assertIs(self.pr.pick_winner([a, b], "B"), b); self.assertEqual(self.pr.decided_by(b, a, "B"), "judge")

    def test_max_pairs_refuses_past_the_cap(self):
        state = self.tmp / "state"; (state / "runs/busy").mkdir(parents=True)
        (state / "routing.json").write_text(json.dumps({"dual": {"maxPairs": 1}}))
        self.assertEqual(self.run_cli("dual-run", "--repo", str(self.repo), "--task", self.task(["p/good", "p/bad"]), "--check", "true"), 3)
        self.assertFalse((state / "dual-runs.jsonl").exists())
        (state / "dual-runs.jsonl").write_text(json.dumps({"id": "busy"}) + "\n")  # busy finished: the slot is free again
        self.assertEqual(self.run_cli("dual-run", "--repo", str(self.repo), "--task", self.task(["p/good", "p/bad"]), "--check", "true"), 0)

    def test_bad_task_file(self):
        with self.assertRaises(SystemExit):
            self.run_cli("dual-run", "--repo", str(self.repo), "--task", self.task(["p/a", "p/a"]), "--check", "true")

    def test_parse_verdict(self):
        v = self.pr.parse_verdict('text\n```\n{"p1": {"A": ["x.py:1 - bug"]}, "why": "w"}\n```')
        self.assertEqual(v, {"p1": {"A": ["x.py:1 - bug"], "B": []}, "why": "w"})
        self.assertIsNone(self.pr.parse_verdict('{"preferred": "A"}'))

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
