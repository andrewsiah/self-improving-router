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

3. **Dual runs.** Code launches marked `unsure: yes` always run on two models. Other code launches run on two models at a small rate (`dual.rate`). For a dual run, the launch is stopped and the orchestrator gets one command to run instead. That command:
   - makes two git worktrees from HEAD,
   - runs the routed model in one and the tier's partner model in the other,
   - runs the orchestrator's check command in each,
   - has a judge model compare the two diffs blind (labels A and B are assigned at random).

   The winner is the worker that finished and passed its check, with fewer serious issues. If those tie, the judge's preference decides, and then the lower cost. Each run adds a row to `dual-runs.jsonl`. `router.py report` shows win rate, pass rate and cost per model, and that tells you where a cheaper model is enough and how to change the routing table.

   A launch is never run twice when its task mentions outside side effects (deploy, publish, push, migrations, sending messages) or secrets and money. At most `dual.maxPairs` dual runs are in flight at once.

## Install

You need pi, the [pi-subagents](https://github.com/nicobailon/pi-subagents) package (the router acts on its `subagent` tool), git, and Python 3.9+ for dual runs and reports.

```sh
pi install npm:pi-subagents
pi install git:github.com/andrewsiah/self-improving-router
mkdir -p ~/.pi/agent/self-improving-router
cp ~/.pi/agent/git/github.com/andrewsiah/self-improving-router/routing.example.json ~/.pi/agent/self-improving-router/routing.json
```

Then edit `routing.json`. The router does nothing until that file exists. Set `SELF_IMPROVING_ROUTER_DIR` to keep the config and logs somewhere else.

## Config

```json
{
  "agents": ["worker", "scout"],
  "routes": {
    "light": ["anthropic/claude-haiku-5-5", "anthropic/claude-sonnet-5-5"],
    "standard": ["anthropic/claude-sonnet-5-5", "anthropic/claude-haiku-5-5"],
    "hard": ["anthropic/claude-opus-5-5:high", "anthropic/claude-sonnet-5-5:high"],
    "*/research": ["anthropic/claude-sonnet-5-5"]
  },
  "dual": {
    "rate": 0.1,
    "maxPairs": 3,
    "partner": {
      "light": "anthropic/claude-sonnet-5-5",
      "standard": "anthropic/claude-opus-5-5:high",
      "hard": "anthropic/claude-sonnet-5-5:high"
    },
    "judge": "anthropic/claude-opus-5-5:high"
  }
}
```

These model ids are examples. Replace them with models your pi setup can run (`pi --list-models`).

- `agents`: the subagent names the router handles. List any aliases too.
- `routes`: keys are `<tier>/<type>`, `*/<type>` or `<tier>`, and the first key that matches wins. Each value is a list of models in order of preference, written `provider/id`, with an optional `:thinking` level. The file is re-read on every launch.
- `dual.partner`: the model each tier is tested against. For light and standard, pick a stronger model to find out whether the cheap one is enough. For hard, pick a cheaper model to find out whether the top one was needed.

## Commands

```sh
R=~/.pi/agent/git/github.com/andrewsiah/self-improving-router/router.py
python3 $R outcomes [--since 2026-11-01] [--jsonl]   # how routed launches went
python3 $R report                                     # dual-run results per model
python3 $R dual-run --repo DIR --task FILE --check 'npm test'   # what a blocked launch asks you to run
```

## Status

The route gate, the logs, `outcomes`, and dual runs with a judge and `report` all work and have tests (`npm test`).

In progress:
- Saving each finished task so it can be replayed later as an offline eval. When a new model comes out, you can test it on your own past work.
- Picking the ~100 most informative tasks for that eval: the tasks that best separate models, weighted toward the ones where routing went wrong.
- A stronger judge for dual runs.

A blog post with results is coming.

## License

MIT
