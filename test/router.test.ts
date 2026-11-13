import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { effectiveRate, isAvailable, launchChildren, pairPlan, parseRoute, registerRouter, routeFor } from "../extensions/router.ts";

const CONFIG = {
  agents: ["worker", "coder"],
  routes: { light: ["p/small"], standard: ["p/gone", "p/mid"], hard: ["p/big:high", "p/mid"], "*/research": ["p/reader"] },
  dual: { rate: 0, maxPairs: 1, partner: { standard: "p/big:high", hard: "p/mid" } },
};
const ROUTE = "route: tier: standard - normal fix | type: code | minutes: 20 | tokens: 15k | unsure: no";
const registry = { find: (provider: string, id: string) => provider === "p" && id !== "gone" ? { id } : undefined, hasConfiguredAuth: () => true };

function setup(config: object | null = CONFIG, rand = () => 0.99, rateFor = async (_dir: string, _answers: any, base: number) => base) {
  const dir = mkdtempSync(path.join(tmpdir(), "sir-test-"));
  if (config) writeFileSync(path.join(dir, "routing.json"), JSON.stringify(config));
  const repo = path.join(dir, "repo");
  mkdirSync(repo);
  execFileSync("git", ["init", "-q", repo]);
  const handlers: Record<string, Function> = {};
  registerRouter({ on: (name: string, fn: Function) => { handlers[name] = fn; } }, { SELF_IMPROVING_ROUTER_DIR: dir }, () => Date.parse("2026-11-13T10:00:00Z"), rand, rateFor);
  const ctx = { cwd: repo, modelRegistry: registry, sessionManager: { getSessionId: () => "s1", getSessionFile: () => path.join(dir, "s", "x.jsonl") } };
  const call = (input: any, id = "c1") => handlers.tool_call({ toolName: "subagent", toolCallId: id, input }, ctx);
  const rows = (file = "decisions.jsonl") => { try { return readFileSync(path.join(dir, file), "utf8").trim().split("\n").map(l => JSON.parse(l)); } catch { return []; } };
  return { dir, repo, handlers, ctx, call, rows };
}

test("parseRoute reads the answers leniently", () => {
  assert.deepEqual(parseRoute(ROUTE), { tier: "standard", reason: "normal fix", type: "code", minutes: 20, tokens: 15000, unsure: false, unsure_why: null });
  const r = parseRoute("Fix it.\n- tier: Medium: touches two files\ntype: Computer Use\nminutes: 10-20\nunsure: yes\nunsure_why: flaky test");
  assert.equal(r?.tier, "standard"); assert.equal(r?.type, "computer-use"); assert.equal(r?.minutes, 15); assert.equal(r?.unsure, true); assert.equal(r?.unsure_why, "flaky test");
  assert.equal(parseRoute("type: code | minutes: 5"), null);
  assert.equal(parseRoute("tier: banana"), null);
});

test("launchChildren covers single, tasks, parallel and chain launches", () => {
  const agents = new Set(["worker"]);
  assert.equal(launchChildren("subagent", { agent: "worker", task: "t" }, agents).length, 1);
  assert.equal(launchChildren("subagent", { agent: "reviewer", task: "t" }, agents).length, 0);
  assert.equal(launchChildren("bash", { agent: "worker" }, agents).length, 0);
  assert.equal(launchChildren("subagent", { action: "list", agent: "worker" }, agents).length, 0);
  assert.equal(launchChildren("subagent", { workflow: "script", agent: "worker" }, agents).length, 0);
  assert.deepEqual(launchChildren("subagent", { tasks: [{ agent: "Worker ", task: "a" }, { agent: "scout", task: "b" }] }, agents).map(c => [c.agent, c.index]), [["worker", 0]]);
  assert.equal(launchChildren("subagent", { workflow: "parallel", args: { tasks: [{ agent: "worker", task: "a" }] } }, agents).length, 1);
  const chain = launchChildren("subagent", { task: "top", chain: [{ agent: "worker" }, { agent: "worker", task: "{previous} then {task}" }] }, agents);
  assert.deepEqual(chain.map(c => c.task), ["top", "{previous} then top"]);
});

test("routeFor prefers tier/type, then */type, then tier", () => {
  assert.equal(routeFor(CONFIG, parseRoute("tier: hard | type: research")!)?.key, "*/research");
  assert.equal(routeFor({ routes: { "hard/research": ["x"], "*/research": ["y"] } }, parseRoute("tier: hard | type: research")!)?.key, "hard/research");
  assert.equal(routeFor(CONFIG, parseRoute("tier: light | type: code")!)?.key, "light");
  assert.equal(routeFor({}, parseRoute("tier: light")!), null);
});

test("isAvailable needs a known model with credentials", () => {
  assert.equal(isAvailable(registry, "p/mid:high"), true);
  assert.equal(isAvailable(registry, "p/gone"), false);
  assert.equal(isAvailable(registry, "noslash"), false);
  assert.equal(isAvailable({ ...registry, hasConfiguredAuth: () => false }, "p/mid"), false);
  assert.equal(isAvailable(undefined, "anything"), true);
});

test("pairPlan: unsure always pairs, others at the rate, never non-code or side effects", () => {
  const route = (s: string) => parseRoute(s)!;
  const unsure = "route: tier: standard - x | type: code | minutes: 20 | tokens: 15k | unsure: yes";
  assert.deepEqual(pairPlan(CONFIG, route(unsure), "p/mid", unsure), { arm: "unsure", models: ["p/mid", "p/big:high"] });
  assert.equal(pairPlan(CONFIG, route("tier: standard | type: code | unsure: no"), "p/mid", "x", () => 0), null); // rate 0
  assert.deepEqual(pairPlan({ ...CONFIG, dual: { ...CONFIG.dual, rate: 0.5 } }, route("tier: standard | type: code"), "p/mid", "x", () => 0.1)?.arm, "sample");
  assert.equal(pairPlan(CONFIG, route("tier: standard | type: research | unsure: yes"), "p/mid", "x"), null);
  assert.equal(pairPlan(CONFIG, route("tier: light | type: code | unsure: yes"), "p/small", "x"), null); // no partner
  for (const words of ["then deploy it", "git -C . push origin HEAD", "send a chat message to the team", "rotate the access token", "revoke old API tokens"])
    assert.equal(pairPlan(CONFIG, route("tier: standard | type: code | unsure: yes"), "p/mid", words), null, words);
  assert.equal(pairPlan(CONFIG, route("tier: hard | type: code | unsure: yes"), "p/mid", "x"), null); // partner is the same model
});

test("pairPlan draws unsure launches at the rate it is given", () => {
  const unsure = parseRoute("tier: standard | type: code | unsure: yes")!;
  assert.equal(pairPlan(CONFIG, unsure, "p/mid", "x", () => 0.5, 0.25), null);
  assert.equal(pairPlan(CONFIG, unsure, "p/mid", "x", () => 0.1, 0.25)?.arm, "unsure");
});

test("a launch is paired at the rate router.py rate gives, and a failed lookup never pairs", async () => {
  const task = "route: tier: standard - x | type: code | unsure: yes\nFix it.";
  const asked: any[] = [];
  const low = setup(CONFIG, () => 0.5, async (_dir, answers, base) => { asked.push([answers.tier, answers.type, base]); return 0.02; });
  assert.equal(await low.call({ agent: "worker", task }), undefined); // drawn 0.5 against 0.02
  assert.deepEqual(asked, [["standard", "code", 1]]);
  assert.equal(low.rows()[0].dual, undefined);
  const full = setup(CONFIG, () => 0.5, async (_dir, _answers, base) => base);
  assert.equal((await full.call({ agent: "worker", task })).block, true);
  const none = setup(CONFIG, () => 0, async () => 0);
  assert.equal(await none.call({ agent: "worker", task }), undefined);
});

test("effectiveRate asks router.py rate: no savings data halves a base rate to 1/64", async () => {
  const { dir } = setup();
  assert.equal(await effectiveRate(dir, parseRoute("tier: standard | type: code")!, 1), 1 / 64);
});

test("without routing.json nothing is blocked or logged", async () => {
  const { call, rows } = setup(null);
  assert.equal(await call({ agent: "worker", task: "no route" }), undefined);
  assert.deepEqual(rows(), []);
});

test("a launch with no route is blocked and logged", async () => {
  const { call, rows } = setup();
  const r = await call({ agent: "coder", task: "fix the bug" });
  assert.equal(r.block, true);
  assert.match(r.reason, /route: tier: standard/);
  assert.equal(rows()[0].outcome, "blocked-missing");
});

test("a routed launch gets the first available model and logs the decision", async () => {
  const { call, rows } = setup();
  const input: any = { agent: "worker", task: `${ROUTE}\nFix the parser.` };
  assert.equal(await call(input), undefined);
  assert.equal(input.model, "p/mid");
  const row = rows()[0];
  assert.equal(row.outcome, "routed"); assert.equal(row.route, "standard"); assert.equal(row.fallback, 1);
  assert.equal(row.session, "s1"); assert.equal(row.tokens, 15000); assert.equal(row.task, undefined);
  const parallel: any = { task: ROUTE, tasks: [{ agent: "worker", task: "a" }, { agent: "worker", task: "tier: light\nb" }] };
  await call(parallel, "c2");
  assert.deepEqual(parallel.tasks.map((t: any) => t.model), ["p/mid", "p/small"]);
});

test("naming a hard-tier model needs tier hard", async () => {
  const { call, rows } = setup();
  const r = await call({ agent: "worker", model: "other/big", task: ROUTE });
  assert.equal(r.block, true);
  assert.match(r.reason, /other\/big/);
  const input: any = { agent: "worker", model: "p/big", task: "tier: hard - the user asked for it" };
  assert.equal(await call(input, "c2"), undefined);
  assert.equal(input.model, "p/big");
  assert.deepEqual(rows().map(r => r.outcome), ["blocked-hard", "explicit"]);
});

test("an unsure code launch in a git repo becomes a dual run; the re-issued call goes through", async () => {
  const { dir, repo, call, rows } = setup();
  const task = "route: tier: standard - x | type: code | unsure: yes\nunsure_why: unread code\nFix it.";
  const r = await call({ agent: "worker", task });
  assert.equal(r.block, true);
  assert.match(r.reason, /dual-run --repo/);
  const [file] = readdirSync(path.join(dir, "pairs"));
  const saved = JSON.parse(readFileSync(path.join(dir, "pairs", file), "utf8"));
  assert.deepEqual(saved.models, ["p/mid", "p/big:high"]);
  assert.equal(saved.prompt, task); assert.equal(saved.route.unsure_why, "unread code");
  assert.match(r.reason, new RegExp(repo));
  assert.deepEqual([rows()[0].outcome, rows()[0].dual], ["dual-run", "unsure"]);
  const again: any = { agent: "worker", task };
  assert.equal(await call(again, "c2"), undefined);
  assert.equal(again.model, "p/mid");
});

test("no dual run outside a git repo, past maxPairs, or for tasks, chain and workflow children", async () => {
  const { dir, call } = setup();
  const task = "tier: standard | type: code | unsure: yes";
  assert.equal(await call({ tasks: [{ agent: "worker", task }] }, "c3"), undefined);
  assert.equal(await call({ chain: [{ agent: "worker", task }, { agent: "reviewer", task: "review {previous}" }] }, "c4"), undefined);
  assert.equal(await call({ workflow: "parallel", args: { tasks: [{ agent: "worker", task }] } }, "c5"), undefined);
  assert.equal(await call({ agent: "worker", task, cwd: tmpdir() }), undefined);
  mkdirSync(path.join(dir, "runs", "busy"), { recursive: true });
  assert.equal(await call({ agent: "worker", task }, "c2"), undefined);
});

test("tool_result ties the call to its run id", async () => {
  const { dir, handlers, ctx, call, rows } = setup();
  await call({ agent: "worker", task: ROUTE });
  await handlers.tool_result({ toolName: "subagent", toolCallId: "c1", details: { runId: "r9" } }, ctx);
  await handlers.tool_result({ toolName: "subagent", toolCallId: "other", details: { runId: "r0" } }, ctx);
  assert.deepEqual(rows("runs.jsonl").map(r => [r.call, r.run_id, r.artifacts]), [["c1", "r9", path.join(dir, "s", "subagent-artifacts")]]);
});
