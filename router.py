#!/usr/bin/env python3
"""self-improving-router command line: dual runs and reports (Python 3.9+, standard library only).

  router.py dual-run --repo DIR --task FILE --check CMD [--timeout SECONDS]
      Runs the task on the two models in FILE ({"prompt": ..., "models": [A, B]}, written by the extension), each in
      its own git worktree from HEAD. Then, cheapest signal first: runs the check in each; runs each worker's new tests
      against the other (cross-tests); optionally has dual.review count each side's P1s blind; has dual.judge compare
      the diffs blind twice with the order swapped (disagreement is a tie, "discordant"); and at dual.auditRate has
      dual.audit judge the pair too. Appends one row to dual-runs.jsonl and prints the winning worktree as the last line.
  router.py report
      Per model: dual runs, judged runs, win %, check pass %, P1s, cost and minutes per run; judge agreement.
  router.py outcomes [--since ISO-DATE] [--jsonl]
      Route decisions joined with how each run went (needs the pi-subagents run artifacts).

Config and state live in $SELF_IMPROVING_ROUTER_DIR (default ~/.pi/agent/self-improving-router).
$SELF_IMPROVING_ROUTER_PI overrides the pi binary.
"""
import argparse, calendar, fcntl, hashlib, json, os, random, re, shutil, signal, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

AGENT_DIR = Path(os.environ.get("PI_CODING_AGENT_DIR") or Path.home() / ".pi/agent")
STATE = Path(os.environ.get("SELF_IMPROVING_ROUTER_DIR") or AGENT_DIR / "self-improving-router")
RESULTS = STATE / "dual-runs.jsonl"
PI = os.environ.get("SELF_IMPROVING_ROUTER_PI", "pi")
TIERS = ("light", "standard", "hard")
WORKER_NOTE = ("You are working in a git worktree at {cwd}, one of two independent attempts at this task. Work only "
               "inside it, and install dependencies there if the task needs them. Leave your changes uncommitted: no "
               "commits, pushes or branch changes, and nothing outside this directory. Finish the task, run the "
               "relevant tests, then reply with a short summary.")
REVIEW_PROMPT = """Blind review. Two independent attempts at the same coding task, labelled A and B, both started from \
commit {base}. Judge only the work; do not try to work out who or what wrote each one.

Task given to both:
<<<
{prompt}
>>>

Check command, run in each worktree after the attempt: {check}
{lines}

Read both diffs, and the worktrees if you need context. Do not modify anything. List each side's P1s (a bug that breaks \
behaviour, data loss, a security hole, or a missed requirement), not style nits.
End your reply with exactly one line of JSON and nothing after it:
{{"p1": {{"A": ["file:line - issue"], "B": []}}, "why": "one sentence"}}"""
JUDGE_PROMPT = """Blind pairwise review. Two independent attempts at the same coding task, labelled Candidate 1 and \
Candidate 2, both started from commit {base}. Judge only the work; do not try to work out who or what wrote each one.

Task given to both:
<<<
{prompt}
>>>

Check command, run in each worktree after the attempt: {check}
{lines}

Read both diffs, and the worktrees if you need context. Do not modify anything. Decide which candidate better does the \
task: correctness, completeness, scope, tests, simplicity.
End your reply with exactly one line of JSON and nothing after it:
{{"preferred": "1" or "2" or "same", "why": "one sentence"}}"""
# Paths that count as an arm's new tests for the cross-tests: test dirs, conftest.py and test-named files.
TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|specs?|e2e|fixtures?)/|(^|/)conftest\.py$|"
                       r"(^|/)(test_[^/]*|[^/]*(_test|\.test|\.spec|_spec))\.(py|[cm]?[jt]sx?|go|rs|rb|sh)$", re.I)
# Winner rule: the first step where the arms differ decides (lower is better); a tie there goes to the next step.
STEPS = (("finished", lambda a, verdict: a["exit"] != 0), ("check", lambda a, verdict: not a["pass"]),
         ("cross", lambda a, verdict: a["cross"]["pass"] is False and not a["cross"].get("void")), ("p1s", lambda a, verdict: len(a["p1s"])),
         ("judge", lambda a, verdict: a["label"] != verdict), ("cost", lambda a, verdict: a["cost"]),
         ("minutes", lambda a, verdict: a["minutes"]))


def sh(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, timeout=kw.pop("timeout", 120), **kw)


def read_jsonl(path):
    rows = []
    try: lines = Path(path).read_text().splitlines()
    except OSError: return rows
    for line in lines:
        try: rows.append(json.loads(line))
        except ValueError: pass
    return rows


def read_json(path):
    try: return json.loads(Path(path).read_text())
    except (OSError, ValueError): return None


def run_pi(args, prompt, cwd, timeout):
    """pi in its own process group, killed whole on timeout. Returns (exit code or "timeout", stdout)."""
    try:
        p = subprocess.Popen([PI, *args], cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True)
    except OSError as e:
        return f"cannot start pi ({e.strerror})", ""
    try:
        out, _ = p.communicate(prompt, timeout=timeout)
        return p.returncode, out
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        return "timeout", ""


def session_usage(sessions):
    """Output tokens and cost from the assistant messages of the pi session files in a directory."""
    tokens, cost = 0, 0.0
    for f in Path(sessions).rglob("*.jsonl"):
        for e in read_jsonl(f):
            m = e.get("message") or {}
            if m.get("role") == "assistant":
                u = m.get("usage") or {}
                tokens += u.get("output") or 0
                cost += (u.get("cost") or {}).get("total") or 0
    return tokens, round(cost, 4)


def run_check(check, cwd):
    """(passed, output) of the check command in cwd."""
    try:
        c = subprocess.run(["bash", "-c", check], cwd=cwd, capture_output=True, text=True, timeout=600)
        return c.returncode == 0, c.stdout + c.stderr
    except subprocess.TimeoutExpired:
        return False, "check timed out"


def run_arm(arm, prompt, check, base, timeout, rundir, sub):
    """One worker in its worktree, then the check and the diff. Fills in and returns arm."""
    wt = Path(arm["worktree"])
    cwd = wt / sub
    cwd.mkdir(parents=True, exist_ok=True)  # the launch dir may exist only untracked in the caller's tree
    t0 = time.time()
    arm["exit"], _ = run_pi(["-p", "--model", arm["model"], "--tools", "read,grep,find,ls,bash,edit,write",
                             "--session-dir", str(rundir / f"{arm['label']}-sessions"),
                             "--append-system-prompt", WORKER_NOTE.format(cwd=cwd)], prompt, cwd, timeout)
    arm["minutes"] = round((time.time() - t0) / 60, 1)
    arm["tokens"], arm["cost"] = session_usage(rundir / f"{arm['label']}-sessions")
    arm["pass"], out = run_check(check, cwd)
    (rundir / f"{arm['label']}.check.txt").write_text(out[-20000:])
    sh(["git", "-C", str(wt), "add", "-A", "-N", "--", ".", ":(exclude)node_modules"])
    diff = sh(["git", "-C", str(wt), "diff", "--binary", base, "--", ".", ":(exclude)node_modules"]).stdout
    (rundir / f"{arm['label']}.diff").write_text(diff)
    arm["diffLines"] = diff.count("\n")
    return arm


def json_lines(text):
    """The JSON objects on a reply's lines, last first."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip().strip("`")
        if line.startswith("{"):
            try: v = json.loads(line)
            except ValueError: continue
            if isinstance(v, dict): yield v


def parse_verdict(text):
    """A review reply's P1 lists: {"p1": {"A": [...], "B": [...]}, "why": str}, or None."""
    for v in json_lines(text):
        if isinstance(v.get("p1"), dict):
            return {"p1": {k: [str(x) for x in v["p1"].get(k) or []] for k in ("A", "B")}, "why": str(v.get("why", ""))[:300]}
    return None


def read_only_pi(model, text, rundir, name):
    """One read-only pi call. Returns (reply or None if the call failed, cost)."""
    sessions = rundir / f"{name}-sessions"
    code, out = run_pi(["-p", "--model", model, "--tools", "read,grep,find,ls", "--session-dir", str(sessions)], text, rundir, 1800)
    (rundir / f"{name}.txt").write_text(out)
    return (out if code == 0 else None), session_usage(sessions)[1]


def arm_lines(arms, rundir, numbered):
    """One line per arm for a judge prompt: check results and the paths of its output and diff."""
    cross = lambda a: ("not run" if a["cross"].get("void") else "FAILED" if a["cross"]["pass"] is False
                       else "PASSED" if a["cross"]["ran"] else "none added")
    return "\n".join(f"{f'Candidate {n}' if numbered else a['label']}: check {'PASSED' if a['pass'] else 'FAILED'} "
                      f"(output: {rundir / (a['label'] + '.check.txt')}); check with the other side's new tests added: {cross(a)}; "
                      f"diff {rundir / (a['label'] + '.diff')} ({a['diffLines']} lines); worktree {a['worktree']}"
                      for n, a in enumerate(arms, 1))


def review(model, arms, prompt, check, base, rundir):
    """Blind review of both diffs for P1s (A and B are the arm labels). Returns (verdict or None, cost)."""
    text = REVIEW_PROMPT.format(base=base[:12], prompt=prompt, check=check, lines=arm_lines(sorted(arms, key=lambda a: a["label"]), rundir, False))
    out, cost = read_only_pi(model, text, rundir, "review")
    return (parse_verdict(out) if out else None), cost


def judge(model, arms, prompt, check, base, rundir, name="judge"):
    """Pairwise judge asked twice with the candidate order swapped, which cancels position bias. Returns
    ({"model", "answers", "verdict"}, cost). verdict is "A" or "B" when both orders agree on it, "same" when both say
    so, "discordant" when they disagree, and None when a call failed (a failed call must not fake a tie)."""
    answers, cost = [], 0.0
    for i, order in enumerate((arms, arms[::-1]), 1):
        text = JUDGE_PROMPT.format(base=base[:12], prompt=prompt, check=check, lines=arm_lines(order, rundir, True))
        out, c = read_only_pi(model, text, rundir, f"{name}{i}")
        cost += c
        pick = next((v["preferred"] for v in json_lines(out or "") if v.get("preferred") in ("1", "2", "same")), None)
        answers.append(order[int(pick) - 1]["label"] if pick in ("1", "2") else pick)
    verdict = None if None in answers else answers[0] if answers[0] == answers[1] else "discordant"
    return {"model": model, "answers": answers, "verdict": verdict}, round(cost, 4)


def cross_test(arm, rival, check, base, rundir, sub):
    """Runs the check in arm's worktree with the test files the rival added or changed copied in, then restores the
    worktree. Returns {"files": n, "ran": bool, "pass": bool or None}. A file that cannot be copied safely (a symlink,
    a directory, a path leading outside the worktree, a file where a directory is needed) skips the step, recorded as
    "skip", so it cannot decide the pair."""
    names = sh(["git", "-C", rival["worktree"], "diff", "--name-only", "--diff-filter=AM", base, "--", ".", ":(exclude)node_modules"]).stdout
    files = [f for f in names.splitlines() if TEST_PATH.search(f)]
    rec = {"files": len(files), "ran": False, "pass": None}
    if not files: return rec
    wt, saved, added = Path(arm["worktree"]), {}, []
    try:
        for rel in files:  # check every path before touching any
            src, dst = Path(rival["worktree"]) / rel, wt / rel
            if src.is_symlink() or dst.is_symlink() or (dst.exists() and not dst.is_file()) or not dst.parent.resolve().is_relative_to(wt.resolve()):
                raise OSError(f"unsafe path {rel}")
            p = dst.parent
            while p != wt and p != p.parent:  # a file where mkdir needs a directory would abort the copy mid-way
                if p.exists() and not p.is_dir(): raise OSError(f"unsafe path {rel}")
                p = p.parent
        for rel in files:
            src, dst = Path(rival["worktree"]) / rel, wt / rel
            if dst.exists(): saved[rel] = (dst.read_bytes(), dst.stat().st_mode)
            else: added.append(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        rec["pass"], out = run_check(check, wt / sub)
        rec["ran"] = True
        (rundir / f"{arm['label']}.cross.txt").write_text(out[-20000:])
    except OSError as e:
        rec["skip"] = str(e)[:200]
    finally:
        for dst in added:  # keep restoring the rest even if one unlink or write fails
            try: dst.unlink(missing_ok=True)
            except OSError: pass
        for rel, (data, mode) in saved.items():
            try:
                (wt / rel).write_bytes(data); (wt / rel).chmod(mode)
            except OSError: pass
    return rec


def pick_winner(arms, verdict):
    """The arm that is better at the first step of STEPS where they differ; verdict is the judge's verdict."""
    return min(arms, key=lambda a: tuple(f(a, verdict) for _, f in STEPS))


def decided_by(win, other, verdict):
    """The first step of STEPS where `other` is worse than the winner, or "tie"."""
    return next((name for name, f in STEPS if f(other, verdict) > f(win, verdict)), "tie")


def max_pairs():
    return int(((read_json(STATE / "routing.json") or {}).get("dual") or {}).get("maxPairs") or 3)


def claim_slot(rundir):
    """Creates runs/<id> as this run's in-flight marker, unless dual.maxPairs runs are already in flight (a runs dir
    touched in the last 3 hours with no dual-runs.jsonl row). Counted and created under one lock."""
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "runs.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        done = {r.get("id") for r in read_jsonl(RESULTS)}
        runs = STATE / "runs"
        busy = [d for d in runs.iterdir() if d.is_dir() and d.name not in done and time.time() - d.stat().st_mtime < 3 * 3600] if runs.is_dir() else []
        if len(busy) >= max_pairs(): return False
        rundir.mkdir(parents=True)
        return True


def dual_run(args):
    task = read_json(args.task) or {}
    prompt, models = task.get("prompt"), task.get("models") or []
    if not isinstance(prompt, str) or len(models) != 2 or models[0] == models[1]:
        raise SystemExit("self-improving-router: the task file needs a prompt and two different models")
    repo = Path(sh(["git", "-C", args.repo, "rev-parse", "--show-toplevel"]).stdout.strip() or args.repo).resolve()
    base = sh(["git", "-C", str(repo), "rev-parse", "HEAD"]).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", base): raise SystemExit(f"self-improving-router: {args.repo} is not a git repo with a commit")
    here = Path(args.repo).resolve()
    sub = str(here.relative_to(repo)) if here != repo and here.is_relative_to(repo) else ""
    if sh(["git", "-C", str(repo), "status", "--porcelain"]).stdout.strip():
        print(f"self-improving-router: note: uncommitted changes in {repo} are not in the worktrees (they start from {base[:12]})", file=sys.stderr)
    rid = Path(args.task).stem if re.fullmatch(r"[\w-]+", Path(args.task).stem) else time.strftime("%Y%m%d%H%M%S")
    if (STATE / "runs" / rid).exists(): rid += "-" + os.urandom(2).hex()  # the same task file run again
    rundir = STATE / "runs" / rid
    if not claim_slot(rundir):
        print(f"self-improving-router: {max_pairs()} dual runs already in flight; run a single worker instead", file=sys.stderr)
        return 3
    labels = ["A", "B"]; random.shuffle(labels)  # blind: which model is A is random per run
    arms = []
    for model, label in zip(models, labels):
        wt = STATE / "worktrees" / f"{repo.name}-{rid}-{label}"
        r = sh(["git", "-C", str(repo), "worktree", "add", "--detach", str(wt), base])
        if r.returncode: raise SystemExit(f"self-improving-router: git worktree add failed: {r.stderr[-300:]}")
        arms.append({"model": model, "label": label, "worktree": str(wt), "p1s": []})
    print(f"self-improving-router {rid}: {models[0]} vs {models[1]}", file=sys.stderr, flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda a: run_arm(a, prompt, args.check, base, args.timeout, rundir, sub), arms))
    ran = [a for a in arms if a["exit"] == 0]
    dual = (read_json(STATE / "routing.json") or {}).get("dual") or {}
    cost, review_v, jury, audit = 0.0, None, None, None
    if len(ran) == 2:  # a worker that did not finish loses at once; no judging
        for a in arms: a["cross"] = cross_test(a, next(b for b in arms if b is not a), args.check, base, rundir, sub)
        if any("skip" in a["cross"] for a in arms):  # one side could not be cross-tested, so neither side's result counts
            for a in arms: a["cross"]["void"] = True
        if dual.get("review"):
            review_v, c = review(dual["review"], arms, prompt, args.check, base, rundir); cost += c
        jury, c = judge(dual.get("judge") or models[0], arms, prompt, args.check, base, rundir); cost += c
        if dual.get("audit") and int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < float(dual.get("auditRate", 0.1)):
            audit, c = judge(dual["audit"], arms, prompt, args.check, base, rundir, "audit"); cost += c
            audit["agree"] = None if None in (jury["verdict"], audit["verdict"]) else audit["verdict"] == jury["verdict"]
    verdict = jury and jury["verdict"]
    for a in arms:
        a.setdefault("cross", {"files": 0, "ran": False, "pass": None})
        a["p1s"] = review_v["p1"][a["label"]] if review_v else []
    win = pick_winner(arms, verdict)
    other = next(a for a in arms if a is not win)
    route = task.get("route") or {}
    row = {"at": time.strftime("%FT%TZ", time.gmtime()), "id": rid, "repo": repo.name, "base": base, "check": args.check,
           "models": models, "arm": task.get("arm"), "tier": route.get("tier"), "type": route.get("type"),
           "unsure": route.get("unsure"), "minutes": route.get("minutes"), "tokens": route.get("tokens"),
           "judge": jury, "audit": audit, "review": review_v, "judged": verdict is not None, "judgeCost": round(cost, 4),
           "preferred": next((a["model"] for a in arms if a["label"] == verdict), None),
           "winner": win["model"] if ran else None, "decidedBy": decided_by(win, other, verdict) if len(ran) == 2 else "finished",
           "results": {a["model"]: {k: a[k] for k in ("label", "exit", "pass", "cross", "p1s", "tokens", "cost", "minutes", "diffLines", "worktree")}
                       for a in arms},
           "wall": round((time.time() - t0) / 60, 1)}
    with RESULTS.open("a") as f: f.write(json.dumps(row) + "\n")
    remove = "git -C {} worktree remove --force {}".format
    if not ran:
        print(f"self-improving-router: neither worker finished; run a single worker instead. Remove both worktrees: "
              f"{remove(repo, arms[0]['worktree'])}; {remove(repo, arms[1]['worktree'])}", file=sys.stderr)
        return 3
    lose = next(a for a in arms if a is not win)
    for a in arms:
        x = a["cross"]
        print(f"  {a['label']} {a['model']}: exit={a['exit']} check={'pass' if a['pass'] else 'FAIL'} "
              f"cross={'-' if not x['ran'] or x.get('void') else 'pass' if x['pass'] else 'FAIL'} P1s={len(a['p1s'])} ${a['cost']:.3f} {a['minutes']} min", file=sys.stderr)
    audit_note = f"audit {'agreed' if audit['agree'] else 'unknown' if audit['agree'] is None else 'DISAGREED'}" if audit else "no audit"
    print(f"self-improving-router: judge {verdict} ({audit_note}), decided by {row['decidedBy']}; winner {win['model']}. Its diff is uncommitted in the worktree below; the other diff is saved at "
          f"{rundir / (lose['label'] + '.diff')}. Remove the other worktree: {remove(repo, lose['worktree'])}", file=sys.stderr)
    print(win["worktree"])
    return 0 if win["pass"] else 1


def report(_args):
    rows = [r for r in read_jsonl(RESULTS) if r.get("winner")]
    stats = {}
    for r in rows:
        for model, a in r["results"].items():
            s = stats.setdefault(model, {"runs": 0, "judged": 0, "wins": 0, "pass": 0, "p1s": 0, "cost": 0.0, "minutes": 0.0})
            s["runs"] += 1; s["pass"] += bool(a["pass"]); s["p1s"] += len(a["p1s"])
            s["cost"] += a["cost"]; s["minutes"] += a["minutes"]
            if r.get("judged"): s["judged"] += 1; s["wins"] += r["winner"] == model
    print(f"{len(rows)} dual runs; win % counts judged runs only")
    print(f"{'model':44} {'runs':>4} {'judged':>6} {'win%':>5} {'pass%':>6} {'P1/run':>6} {'$/run':>7} {'min/run':>7}")
    for model, s in sorted(stats.items()):
        n, j = s["runs"], s["judged"]
        win = f"{100 * s['wins'] / j:5.0f}" if j else f"{'-':>5}"
        print(f"{model:44} {n:4d} {j:6d} {win} {100 * s['pass'] / n:6.0f} {s['p1s'] / n:6.2f} {s['cost'] / n:7.3f} {s['minutes'] / n:7.1f}")
    judged = [r for r in rows if r.get("judged")]
    audits = [r["audit"]["agree"] for r in rows if isinstance(r.get("audit"), dict) and r["audit"].get("agree") is not None]
    discordant = sum(isinstance(r.get("judge"), dict) and r["judge"].get("verdict") == "discordant" for r in rows)  # older rows: judge is a model name
    print(f"judge: {discordant} discordant, "
          f"{sum(r.get('decidedBy') in ('cost', 'minutes') for r in judged)}/{len(judged)} judged runs decided by cost or time; "
          f"audit agreed {sum(audits)}/{len(audits)}")
    return 0


def model_id(spec):
    return re.sub(r":\w+$", "", spec or "").split("/")[-1].lower()


def join_outcomes(since=""):
    """Routed and explicit launches with what the run left behind: run id from runs.jsonl, then the pi-subagents
    `<run>_<agent>[_<index>]_meta.json` artifact (exit code, error, output tokens, end time). retried: a later launch
    from the same session with the same task; escalated: that later launch had a higher tier or a hard-tier model."""
    runs = {r.get("call"): r for r in read_jsonl(STATE / "runs.jsonl")}
    hard = {model_id(m) for m in ((read_json(STATE / "routing.json") or {}).get("routes") or {}).get("hard") or []}
    rows = [r for r in read_jsonl(STATE / "decisions.jsonl") if r.get("outcome") in ("routed", "explicit") and r.get("ts", "") >= since]
    rank = lambda r: (TIERS.index(r["tier"]) if r.get("tier") in TIERS else -1) + (10 if model_id(r.get("model")) in hard else 0)
    out = []
    for i, r in enumerate(rows):
        run = runs.get(r.get("call")) or {}
        meta = None
        if run.get("run_id") and run.get("artifacts"):
            for name in (f"{run['run_id']}_{r['agent']}_{r['index']}_meta.json", f"{run['run_id']}_{r['agent']}_meta.json"):
                meta = read_json(Path(run["artifacts"]) / name)
                if meta: break
        later = [x for x in rows[i + 1:] if r.get("task_hash") and x.get("task_hash") == r["task_hash"]
                 and x.get("session") == r.get("session") and x.get("call") != r.get("call")]
        ended = meta.get("timestamp") if meta and isinstance(meta.get("timestamp"), (int, float)) else None
        started = calendar.timegm(time.strptime(r["ts"][:19], "%Y-%m-%dT%H:%M:%S")) if ended else None
        out.append(dict(r, run_id=run.get("run_id"), ok=(meta.get("exitCode") == 0 and not meta.get("error")) if meta else None,
                        out_tokens=((meta or {}).get("usage") or {}).get("output"), top_model=model_id(r.get("model")) in hard,
                        wall_min=round((ended / 1000 - started) / 60, 1) if ended else None,
                        retried=bool(later), escalated=any(rank(x) > rank(r) for x in later)))
    return out


def summarize(rows):
    n = len(rows)
    pct = lambda a, b: f"{a}/{b} ({round(100 * a / b)}%)" if b else "0/0"
    lines = [f"launches {n}; hard-tier model {pct(sum(r['top_model'] for r in rows), n)}; "
             f"explicit {pct(sum(r['outcome'] == 'explicit' for r in rows), n)}"]
    for tier in TIERS:
        t = [r for r in rows if r.get("tier") == tier]
        done = [r for r in t if r["ok"] is not None]
        lines.append(f"{tier}: {pct(len(t), n)}; ok {pct(sum(r['ok'] for r in done), len(done))}; "
                     f"retried {sum(r['retried'] for r in t)}, escalated {sum(r['escalated'] for r in t)}")
    for u in (True, False):
        t = [r for r in rows if r.get("unsure") is u and r["ok"] is not None]
        lines.append(f"unsure {'yes' if u else 'no'}: ok {pct(sum(r['ok'] for r in t), len(t))}")

    def median_ratio(actual, estimate):
        xs = sorted(r[actual] / r[estimate] for r in rows if isinstance(r.get(actual), (int, float)) and r.get(estimate))
        return round(xs[len(xs) // 2], 2) if xs else None
    lines.append(f"actual/estimate median: minutes {median_ratio('wall_min', 'minutes')}, tokens {median_ratio('out_tokens', 'tokens')}")
    return "\n".join(lines)


def outcomes(args):
    rows = join_outcomes(args.since)
    print("\n".join(json.dumps(r) for r in rows) if args.jsonl else summarize(rows))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="router.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dual-run")
    d.add_argument("--repo", required=True); d.add_argument("--task", required=True); d.add_argument("--check", required=True)
    d.add_argument("--timeout", type=int, default=1800, help="seconds per worker (default 1800)")
    sub.add_parser("report")
    o = sub.add_parser("outcomes"); o.add_argument("--since", default=""); o.add_argument("--jsonl", action="store_true")
    args = ap.parse_args(argv)
    return {"dual-run": dual_run, "report": report, "outcomes": outcomes}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
