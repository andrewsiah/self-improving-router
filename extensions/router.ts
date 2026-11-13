// self-improving-router: subagent launches state what the task needs, and routing.json picks the model.
//
// Every routed launch carries one line of answers, e.g.
//   route: tier: standard - normal fix | type: code | minutes: 20 | tokens: 15k | unsure: no
// A launch without a tier is blocked with that example. The first model in the matching routing.json list that pi
// can use is written into the launch. Each decision goes to decisions.jsonl, and the run id from the tool result to
// runs.jsonl, so `router.py outcomes` can join them with how the run went. Some code launches are also picked for
// a dual run (two models, two worktrees, a check and a judge; see router.py): `unsure: yes` always, others at
// `dual.rate`. Without routing.json the extension does nothing. Any internal error lets the launch through.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { execFile } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import { appendFileSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const TIERS = ["light", "standard", "hard"];
export const TYPES = ["code", "computer-use", "research", "writing", "ops"];
const TIER_WORDS: Record<string, string> = { light: "light", standard: "standard", medium: "standard", hard: "hard", heavy: "hard" };
const EXAMPLE = "route: tier: standard - <why> | type: code | minutes: 20 | tokens: 15k | unsure: no";
const CLI = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../router.py");
const REISSUE_MS = 10 * 60 * 1000;
const RUN_TTL_MS = 3 * 3600 * 1000; // a dual run dir older than this with no result counts as crashed
// Work that should never run twice (outside side effects) or stays on one careful run (secrets, money).
const NO_PAIR = /secret|credential|api.?key|passw|oauth|permission|payment|billing|invoice|refund|\bdeploy|\bpublish|git push|force.push|migrat(e|ion)|\bproduction\b|\bprod\b|send (an? )?(e-?mail|message)/i;

export const stateDir = (env: Record<string, string | undefined>) =>
  env.SELF_IMPROVING_ROUTER_DIR || path.join(env.PI_CODING_AGENT_DIR || path.join(homedir(), ".pi/agent"), "self-improving-router");

// A field's value: at a line start (after an optional `route:` and bullet) or after a `|`, up to the line end or next `|`.
const field = (task: string, name: string) => {
  const re = new RegExp(`(?:^|\\|)[ \\t]*(?:route:[ \\t]*)?(?:[-*][ \\t]*)?${name}[ \\t]*[:=][ \\t]*([^|\\n]*)`, "im");
  return re.exec(task)?.[1]?.trim() || null;
};
const number = (v: string | null) => {
  const m = v && /(\d[\d,]*(?:\.\d+)?)\s*(k)?(?:\s*(?:-|–|to)\s*(\d[\d,]*(?:\.\d+)?)\s*(k)?)?/i.exec(v);
  if (!m) return null;
  const n = (s: string, k?: string) => parseFloat(s.replaceAll(",", "")) * (k ? 1000 : 1);
  const lo = n(m[1], m[2] || m[4]), hi = m[3] ? n(m[3], m[4]) : lo;
  return Math.round((lo + hi) / 2 * 10) / 10;
};

export type Route = { tier: string; reason: string | null; type: string | null; minutes: number | null; tokens: number | null; unsure: boolean | null; unsure_why: string | null };

// The route answers in a task, or null when it gives no valid tier.
export function parseRoute(task: unknown): Route | null {
  if (typeof task !== "string") return null;
  const m = /^(\w+)\b[\s\-–—:,.]*(.*)$/.exec(field(task, "tier") || "");
  const tier = m && TIER_WORDS[m[1].toLowerCase()];
  if (!tier) return null;
  const typeRaw = (field(task, "type") || "").toLowerCase().replace(/[\s_]+/g, "-");
  const unsure = (field(task, "unsure") || "").toLowerCase();
  return {
    tier, reason: m[2].trim().slice(0, 120) || null,
    type: TYPES.find(t => typeRaw === t || typeRaw.startsWith(t + "-")) || null,
    minutes: number(field(task, "minutes")), tokens: number(field(task, "tokens")),
    unsure: /^(yes|y|true)\b/.test(unsure) ? true : /^(no|n|false)\b/.test(unsure) ? false : null,
    unsure_why: field(task, "unsure_why")?.slice(0, 120) || null,
  };
}

// The children of a `subagent` call whose agent is routed: { entry, agent, index, model, task }. `entry` is the object
// whose `model` the router sets, `index` the child's place in launch order. Covers a single launch, `tasks` arrays,
// `workflow: "parallel"` and chain steps; other workflows and management actions give []. `{task}` expands to the
// launch's top-level task, as the subagent tool does.
export function launchChildren(toolName: string, input: any, agents: Set<string>) {
  if (toolName !== "subagent" || !input || typeof input !== "object" || input.action) return [];
  const opts = input.options && typeof input.options === "object" ? input.options : {};
  const list = (v: unknown) => Array.isArray(v) ? v : null;
  const workflow = input.workflow || opts.workflow;
  const parallel = typeof workflow === "string" && workflow.trim() === "parallel";
  if (workflow && !parallel) return [];
  const tasks = parallel ? list(input.args?.tasks) || list(opts.args?.tasks) : list(input.tasks) || list(opts.tasks);
  if (parallel && !tasks) return [];
  const chain = list(input.chain) || list(opts.chain);
  const entries = tasks ? tasks.map(entry => ({ entry, first: false }))
    : chain ? chain.flatMap((step, i) => list(step?.parallel) ? step.parallel.map((entry: any) => ({ entry, first: i === 0 })) : [{ entry: step, first: i === 0 }])
    : [{ entry: input, first: false }];
  const top = typeof input.task === "string" ? input.task : "";
  return entries
    .map(({ entry, first }, index) => ({ entry, first, index }))
    .filter(({ entry }) => entry && typeof entry === "object" && typeof entry.agent === "string" && agents.has(entry.agent.trim().toLowerCase()))
    .map(({ entry, first, index }) => {
      const model = [entry.model, input.model, opts.model].find(m => typeof m === "string" && m.trim());
      const task = typeof entry.task === "string" ? entry.task : chain && first ? "{task}" : "";
      return { entry, agent: entry.agent.trim().toLowerCase(), index, model: model ? model.trim() : "", task: task.replaceAll("{task}", top) };
    });
}

// routes: { "<tier>/<type>" | "*/<type>" | "<tier>": [model, fallback, ...] }; the first matching key wins.
export function routeFor(config: any, answers: Route) {
  const routes = config?.routes && typeof config.routes === "object" ? config.routes : {};
  const keys = [answers.type && `${answers.tier}/${answers.type}`, answers.type && `*/${answers.type}`, answers.tier].filter(Boolean) as string[];
  for (const key of keys) {
    const candidates = [routes[key]].flat().filter(m => typeof m === "string" && m.trim()).map((m: string) => m.trim());
    if (candidates.length) return { key, candidates };
  }
  return null;
}

// Can pi run `provider/id[:thinking]`: the registry knows it and has credentials for it. No registry: assume yes.
export function isAvailable(registry: any, spec: string) {
  if (!registry || typeof registry.find !== "function") return true;
  const base = spec.replace(/:(off|minimal|low|medium|high|xhigh|max)$/i, "");
  const slash = base.indexOf("/");
  const model = slash > 0 ? registry.find(base.slice(0, slash), base.slice(slash + 1)) : undefined;
  return !!model && (typeof registry.hasConfiguredAuth !== "function" || registry.hasConfiguredAuth(model));
}

const modelId = (spec: string) => spec.replace(/:\w+$/, "").split("/").pop()!.toLowerCase();

// An explicitly named model that appears in the `hard` route needs tier hard: naming the top model must be argued.
export const needsHard = (config: any, model: string, answers: Route) =>
  answers.tier !== "hard" && [config?.routes?.hard].flat().some(m => typeof m === "string" && modelId(m) === modelId(model));

// Should this routed code launch also run on the tier's partner model? Returns { arm, models } or null.
// Arm "unsure": `unsure: yes`, always. Arm "sample": any other code launch, at dual.rate.
export function pairPlan(config: any, answers: Route, model: string, task: string, rand = Math.random) {
  const partner = config?.dual?.partner?.[answers.tier];
  if (answers.type !== "code" || typeof partner !== "string" || !partner || modelId(partner) === modelId(model) || NO_PAIR.test(task)) return null;
  const rate = Number(config.dual.rate) || 0;
  const arm = answers.unsure === true ? "unsure" : rand() < rate ? "sample" : null;
  return arm ? { arm, models: [model, partner] } : null;
}

export const taskHash = (task: string) => createHash("sha256").update(task).digest("hex").slice(0, 16);

const readJson = (file: string) => { try { return JSON.parse(readFileSync(file, "utf8")); } catch { return null; } };
const readJsonl = (file: string): any[] => {
  try { return readFileSync(file, "utf8").split("\n").flatMap(l => { try { return l.trim() ? [JSON.parse(l)] : []; } catch { return []; } }); }
  catch { return []; }
};

// Dual runs in flight: runs/<id> dirs touched recently with no row in dual-runs.jsonl yet.
export function runsInFlight(dir: string, now = Date.now()) {
  let names: string[] = [];
  try { names = readdirSync(path.join(dir, "runs")); } catch { return 0; }
  const done = new Set(readJsonl(path.join(dir, "dual-runs.jsonl")).map(r => r?.id));
  return names.filter(n => !done.has(n)).filter(n => {
    try { return now - statSync(path.join(dir, "runs", n)).mtimeMs < RUN_TTL_MS; } catch { return false; }
  }).length;
}

const insideGitRepo = (cwd: string) => new Promise<boolean>(resolve =>
  execFile("git", ["-C", cwd, "rev-parse", "--is-inside-work-tree"], { timeout: 5000 }, (err, out) => resolve(!err && String(out).trim() === "true")));

const shq = (s: string) => `'${s.replace(/'/g, `'\\''`)}'`;

export function missingText(agents: string[]) {
  return `[self-improving-router] ${agents.join(", ")} launch has no route line, so it was not started. Don't pick a model: ` +
    `answer these and the router picks one. Re-issue with this line in the task:\n${EXAMPLE}\n` +
    `tier: light (lookup, small edit) | standard (normal build or fix) | hard (retry after a failed attempt, design, ` +
    `security, money, many parts), always with a reason. type: code|computer-use|research|writing|ops. minutes and ` +
    `tokens: your estimates of wall minutes and output tokens. unsure: yes when a cheaper model could honestly go ` +
    `either way; then add \`unsure_why: <the doubt>\`.`;
}

export function hardText(models: string[]) {
  return `[self-improving-router] ${models.join(", ")} was named with a tier below hard. Drop the model and let the route pick, ` +
    `or keep it with \`tier: hard - <why this needs the top model>\`.`;
}

export function pairText(arm: string, models: string[], cwd: string, taskFile: string) {
  return `[self-improving-router] This launch was picked for a dual run (${arm === "unsure" ? "unsure: yes" : "random sample"}: ` +
    `${models[0]} vs ${models[1]}) and was not started. Run the task on both models instead, replacing <test cmd> ` +
    `with the command that proves the work:\npython3 ${shq(CLI)} dual-run --repo ${shq(cwd)} --task ${shq(taskFile)} --check '<test cmd>'\n` +
    `It takes about as long as the task. The last line it prints is the winning worktree: continue from there. ` +
    `If it says to run a single worker instead, re-issue this same call within 10 minutes and it goes through.`;
}

export default function (pi: ExtensionAPI) { registerRouter(pi, process.env); }

export function registerRouter(pi: any, env: Record<string, string | undefined>, now = Date.now, rand = Math.random) {
  const dir = stateDir(env);
  const append = (file: string, rows: object[]) => {
    mkdirSync(dir, { recursive: true });
    appendFileSync(path.join(dir, file), rows.map(r => JSON.stringify(r) + "\n").join(""));
  };
  const pending = new Map<string, number>(); // routed tool call id -> when, until its result names the run
  const paired = new Map<string, number>();  // cwd+task hash of a launch sent to a dual run -> when
  pi.on("tool_call", async (event: any, ctx: any) => {
    try {
      const config = readJson(path.join(dir, "routing.json"));
      if (!config) return;
      const agents = new Set<string>((Array.isArray(config.agents) ? config.agents : ["worker"]).map((a: string) => String(a).toLowerCase()));
      const children = launchChildren(event.toolName, event.input, agents);
      if (!children.length) return;
      const ts = new Date(now()).toISOString();
      const session = ctx?.sessionManager?.getSessionId?.() || null;
      const topRoute = parseRoute(event.input.task);
      const plans = children.map(({ entry, agent, index, model, task }) => {
        const answers = parseRoute(task) || topRoute;
        const row: any = {
          ts, session, call: event.toolCallId || null, index, agent, ...(answers || { tier: null }),
          requested: model || null, route: null, model: null, fallback: null, task_hash: task ? taskHash(task) : null, outcome: "",
        };
        if (!answers) row.outcome = "blocked-missing";
        else if (model && needsHard(config, model, answers)) row.outcome = "blocked-hard";
        else if (model) { row.outcome = "explicit"; row.model = model; }
        else {
          row.outcome = "routed";
          const route = routeFor(config, answers);
          if (route) {
            const pick = route.candidates.findIndex(m => isAvailable(ctx?.modelRegistry, m));
            row.route = route.key; row.model = route.candidates[Math.max(pick, 0)]; row.fallback = Math.max(pick, 0);
          }
        }
        return { entry, row, answers, task };
      });
      const rows = plans.map(p => p.row);
      const missing = rows.filter(r => r.outcome === "blocked-missing");
      const hard = rows.filter(r => r.outcome === "blocked-hard");
      let block: string | null = missing.length ? missingText([...new Set(missing.map(r => r.agent as string))])
        : hard.length ? hardText([...new Set(hard.map(r => r.requested as string))]) : null;
      // A single routed launch inside a git repo may become a dual run instead.
      const only = plans.length === 1 && plans[0].row.outcome === "routed" && plans[0].row.model ? plans[0] : null;
      if (!block && only) {
        const cwd = path.resolve(ctx?.cwd || process.cwd(), typeof event.input.cwd === "string" ? event.input.cwd.replace(/^~(?=\/|$)/, homedir()) : "");
        const key = taskHash(`${cwd}\0${only.task}`);
        for (const [k, t] of paired) if (now() - t > REISSUE_MS) paired.delete(k);
        const plan = paired.delete(key) ? null : pairPlan(config, only.answers!, only.row.model, only.task, rand);
        const max = Number(config.dual?.maxPairs) || 3;
        if (plan && runsInFlight(dir, now()) < max && await insideGitRepo(cwd)) {
          const id = `${ts.replace(/\D/g, "").slice(0, 14)}-${randomBytes(3).toString("hex")}`;
          const taskFile = path.join(dir, "pairs", `${id}.json`);
          mkdirSync(path.dirname(taskFile), { recursive: true });
          writeFileSync(taskFile, JSON.stringify({ prompt: only.task, models: plan.models, arm: plan.arm, route: only.answers }, null, 1), { mode: 0o600 });
          paired.set(key, now());
          only.row.outcome = "dual-run"; only.row.dual = plan.arm;
          block = pairText(plan.arm, plan.models, cwd, taskFile);
        }
      }
      if (!block) for (const p of plans) if (p.row.outcome === "routed" && p.row.model) p.entry.model = p.row.model;
      try { append("decisions.jsonl", rows); } catch {} // an unwritable log never changes the decision
      if (block) return { block: true, reason: block };
      if (event.toolCallId) {
        pending.set(event.toolCallId, now());
        for (const [id, t] of pending) if (now() - t > 6 * 3600e3) pending.delete(id);
      }
    } catch {}
  });
  // The run id of a launch shows up only in its result: one runs.jsonl row ties the call to it.
  pi.on("tool_result", async (event: any, ctx: any) => {
    try {
      if (event.toolName !== "subagent" || !pending.delete(event.toolCallId)) return;
      const d = event.details && typeof event.details === "object" ? event.details : {};
      const sessionFile = ctx?.sessionManager?.getSessionFile?.();
      append("runs.jsonl", [{
        ts: new Date(now()).toISOString(), call: event.toolCallId, run_id: d.runId || d.asyncId || null, error: event.isError || undefined,
        artifacts: sessionFile ? path.join(path.dirname(sessionFile), "subagent-artifacts") : null,
      }]);
    } catch {}
  });
}
