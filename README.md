# self-improving-router

A [pi](https://pi.dev) package that picks the model for each subagent from what the task needs. It also checks those picks against real tasks.

## The problem

An orchestrating agent that launches workers also picks their models, and it tends to pick the most expensive one. In our own runs the orchestrator chose the top model for 58% of workers, when about 26% needed it. That makes work slower and costs more. And because nobody checks whether a cheaper model would have done the job, the habit never changes.

## How it works

1. **Route questions.** Every subagent launch answers a few fixed questions in its task:

   ```
   route: tier: standard - normal bug fix | type: code | minutes: 20 | tokens: 15k | unsure: no
   ```

   `tier` is light, standard or hard, with a reason. `type` is code, computer-use, research, writing or ops. `minutes` and `tokens` are the orchestrator's estimates. `unsure: yes` means a cheaper model could honestly go either way. A launch without a tier is blocked and gets this example back.

   `routing.json` maps the answers to an ordered list of models, and the first one pi can use wins. The orchestrator never names a model. If it does name one from the `hard` list, the launch needs `tier: hard` and a reason.

2. **Decision and outcome logs.** Each launch writes one row to `decisions.jsonl`: the answers, the model, and a hash of the task, never its text. Once the launch returns, a row in `runs.jsonl` links it to its run id. `router.py outcomes` joins these rows with the run results to show the share of launches on top models, success by tier, retries and escalations, and how far the estimates were off.

3. **Dual runs.** A single code launch marked `unsure: yes` runs on two models, and other single code launches do at a small rate (`dual.rate`), until the router has seen enough (see "Evaluating less over time"). For a dual run, the launch is stopped and the orchestrator gets one command to run instead. That command:
   - makes two git worktrees from HEAD. Uncommitted changes and untracked files such as `node_modules` are not copied.
   - runs the routed model in one and the tier's partner model in the other,
   - runs the orchestrator's check command in each,
   - runs each worker's new or changed test files against the other worker's code (cross-tests), in a throwaway copy of the worktree so the worktree itself is never touched,
   - optionally has a `review` model list each side's serious issues (P1s) blind,
   - has the `judge` model compare the two diffs blind, twice with the order swapped, but only when the finish, check, cross-tests and P1s did not already decide the pair. If the two answers disagree, that counts as a tie and is logged as `discordant`,
   - at `dual.auditRate` (default 10%), has an `audit` model judge the pair the same way and logs whether it agreed. Once the last 40 audits agree at 90% or more, the audit rate drops to 2%.

   A worker that didn't finish loses at once and nothing is judged. Otherwise the first step where the two differ decides: finished, own check, cross-tests, fewer P1s, the judge's agreed preference, lower cost, shorter wall time. Ties go to the cheaper worker, then the faster one. Each run adds a row to `dual-runs.jsonl`, including which step decided it. `router.py report` shows win rate, pass rate and cost per model, how often the judge was discordant or the pair was decided on cost, and how often the audit agreed. That tells you where a cheaper model is enough and how to change the routing table.

   A keyword filter skips tasks that mention outside side effects (deploy, publish, push, migrations, sending messages) or secrets, tokens and money. It catches plain wording, not everything, so keep such work out of tasks you want paired. At most `dual.maxPairs` dual runs run at once; past that, the command says to run a single worker.

## Evaluating less over time

The router should cost less as it learns, so evaluation shrinks unless something changes:

- **Sampling decays per cell.** A cell is a tier and type, such as `standard/code`. After its last 30 dual runs all held up (the routed pick won or tied, with no failed check where the partner passed and no failed check on a launch marked `unsure: no`), the cell's rate drops to 2%, `unsure: yes` included. One miss puts the cell back at its base rate. So does any change to the cell's model list in `routing.json`, which is how a new model gets calibrated again. The state is read from `dual-runs.jsonl`.
- **Evaluation is capped by what routing saves.** Overhead is the extra arm's cost plus review, judge and audit cost. Savings are what the routed launches would have cost on the top model (the first `hard` model, priced in `prices`) minus what they cost. If overhead over the last 7 days is more than 10% of savings, or savings are zero, every rate halves (up to six times) until it is back under. Free models cost $0. Without a `prices` entry for the top model, savings read as zero and rates stay at 1/64.
- `router.py report` starts with one line: savings, overhead, net and overhead %.

## Install

You need pi, the [pi-subagents](https://github.com/nicobailon/pi-subagents) package (the router acts on its `subagent` tool), git, and Python 3.9+ for dual runs and reports.

```sh
pi install npm:pi-subagents
pi install git:github.com/andrewsiah/self-improving-router
mkdir -p ~/.pi/agent/self-improving-router
cp ~/.pi/agent/git/github.com/andrewsiah/self-improving-router/routing.example.json ~/.pi/agent/self-improving-router/routing.json
```

Then edit `routing.json`. The router does nothing until that file exists. Set `SELF_IMPROVING_ROUTER_DIR` to keep the config and logs somewhere else. If you set `PI_CODING_AGENT_DIR`, pi and the router use that directory instead of `~/.pi/agent`, so change the paths above to match.

## Config

```json
{
  "agents": ["worker", "scout"],
  "routes": {
    "light": ["anthropic/claude-haiku-5-5", "anthropic/claude-sonnet-5-5"],
    "standard": ["anthropic/claude-sonnet-5-5", "anthropic/claude-haiku-5-5"],
    "hard": ["anthropic/claude-opus-5-5:high"],
    "*/research": ["anthropic/claude-sonnet-5-5"]
  },
  "prices": {
    "claude-opus-5-5": { "input": 4, "output": 20, "cacheRead": 0.2, "cacheWrite": 5 }
  },
  "dual": {
    "rate": 0.1,
    "maxPairs": 3,
    "partner": {
      "light": "anthropic/claude-sonnet-5-5",
      "standard": "anthropic/claude-opus-5-5:high",
      "hard": "anthropic/claude-sonnet-5-5:high"
    },
    "judge": "anthropic/claude-opus-5-5:high",
    "review": "anthropic/claude-sonnet-5-5",
    "audit": "anthropic/claude-sonnet-5-5:high",
    "auditRate": 0.1
  }
}
```

These model ids are examples. Replace them with models your pi setup can run (`pi --list-models`).

- `agents`: the subagent names the router handles. List any aliases too.
- `prices`: list prices in dollars per million tokens, keyed by model id, for the first `hard` model. Only it is needed: it prices what routing saved by not using it.
- `routes`: keys are `<tier>/<type>`, `*/<type>` or `<tier>`, and the first key that matches wins. Each value is a list of models in order of preference, written `provider/id`, with an optional `:thinking` level. The file is re-read on every launch.
- `dual.partner`: the model each tier is tested against. For light and standard, pick a stronger model to find out whether the cheap one is enough. For hard, pick a cheaper model to find out whether the top one was needed.
- `dual.judge`: the model that compares the two diffs (default: the first model of the pair). Prefer a model that is neither of the pair. `dual.review` and `dual.audit` are optional and off unless set; `dual.auditRate` is the share of pairs the audit model also judges.
- `outcomes` counts every launch on a model from the `hard` list as a top-model launch, so keep cheaper fallback models out of that list.

## Commands

```sh
R=~/.pi/agent/git/github.com/andrewsiah/self-improving-router/router.py
python3 $R outcomes [--since 2026-11-01] [--jsonl]   # how routed launches went
python3 $R report                                     # savings vs eval overhead, then dual-run results per model
python3 $R rate --tier standard --type code --base 1  # the sampling rate the extension would use
python3 $R dual-run --repo DIR --task FILE --check 'npm test'   # what a blocked launch asks you to run
```

## Status

Done, with tests (`npm test`):
- the route gate,
- the decision and outcome logs, and `outcomes`,
- dual runs with cross-tests, an optional review, a pairwise judge asked in both orders only when nothing earlier decides, a sampled audit judge, and `report`,
- sampling that decays per cell and an eval budget tied to savings.

Tested end to end in a fresh pi (installed from this README, real model calls): the gate blocks a launch without `route:`, a light launch runs on the light model, the logs and `outcomes` fill in, and a dual run on a throwaway repo ends with a judge verdict and shows up in `report`.

In progress:
- Saving each finished task so it can be replayed later as an offline eval. When a new model comes out, you can test it on your own past work.
- Picking the ~100 most informative tasks for that eval: the tasks that best separate models, weighted toward the ones where routing went wrong.

A blog post with results is coming.

## License

MIT
