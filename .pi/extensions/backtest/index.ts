/**
 * nerd-trader 回测扩展（项目级）
 *
 * 三个能力：
 *   backtest 工具   结构化参数跑 research.py；前台流式进度，后台立即返回
 *   /runs           历史记录列表、运行明细（含图表）、两次运行对比
 *   /jobs           后台任务列表、日志尾部、终止
 *
 * 前台进度与后台任务数常驻 footer；后台任务结束时通知并自动归档到 runs.json。
 * 关闭方式：环境变量 NERD_TRADER_BACKTEST_DISABLE=1，或删除本目录。
 */

import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { join } from "node:path";
import { Type } from "typebox";
import type { Static } from "typebox";
import { ChartBlock } from "./chart.js";
import { showJobs, showRuns } from "./panels.js";
import { parseOutput, progressOf } from "./parse.js";
import type { Progress } from "./parse.js";
import {
	isAlive,
	killTree,
	listJobs,
	loadRuns,
	readLog,
	readLogFrom,
	readStatus,
	saveRun,
	startJob,
} from "./proc.js";
import type { JobHandle } from "./proc.js";
import { findProjectRoot, makeRunId, MODE_FLAG, MODE_LABEL, resultPaths } from "./spec.js";
import type { JobStatus, ResultPaths, RunMode, RunRecord, RunSpec } from "./spec.js";
import { duration, pct, plainPalette, recordLines, themePalette } from "./ui.js";

const POLL_MS = 700;
const JOBS_POLL_MS = 5000;
const PROGRESS_THROTTLE_MS = 600;
const STATUS_KEY = "backtest";
const JOBS_STATUS_KEY = "backtest-jobs";
const LOG_TAIL_LINES = 12;

const PARAMS = Type.Object({
	mode: Type.Optional(
		StringEnum(["backtest", "robust", "carry"] as const, {
			description: "backtest=单次回测（默认）；robust=稳健性检验；carry=资金费率套利研究",
		}),
	),
	symbol: Type.Optional(
		Type.String({ description: "标的，如 BTC/USDT 或预设键 c1/c2；稳健性与 carry 支持逗号分隔多个" }),
	),
	strategy: Type.Optional(
		Type.String({ description: "策略：序号 1-4 或别名 dual-ma/trend-reversion/bollinger/rsrs（仅单次回测）" }),
	),
	days: Type.Optional(
		Type.Number({
			description:
				"评估窗口根数：单次默认 730，稳健性默认 2900；carry 模式下按日历天解释（默认 2900）",
		}),
	),
	segments: Type.Optional(Type.Number({ description: "稳健性分段数，默认 4" })),
	trials: Type.Optional(Type.Number({ description: "多重检验校正声明的试验次数 N（历史试过的参数越多越大）" })),
	volTarget: Type.Optional(
		Type.Union([Type.Number(), Type.Boolean()], {
			description: "波动率目标仓位：数字为目标波动率（0.4=40%），true 用默认 40%，false 关闭",
		}),
	),
	volLookback: Type.Optional(Type.Number({ description: "波动率回看根数，默认 30" })),
	volRebalance: Type.Optional(Type.Number({ description: "波动率调仓间隔（根），默认 7" })),
	background: Type.Optional(Type.Boolean({ description: "后台运行：立即返回，完成时通知，用 /jobs 查看进度" })),
});

type BacktestParams = Static<typeof PARAMS>;

interface BacktestDetails {
	id: string;
	mode: RunMode;
	phase: string;
	logLines: number;
	elapsedMs: number;
	background: boolean;
	record?: RunRecord;
	error?: string;
}

/** 把工具参数翻译成 research.py 的命令行 */
function buildSpec(root: string, params: BacktestParams): RunSpec {
	const mode: RunMode = params.mode ?? "backtest";
	const argv = ["run", "python", "research.py", ...MODE_FLAG[mode]];
	if (params.symbol) argv.push("--symbol", params.symbol);
	if (mode === "backtest" && params.strategy) argv.push("--strategy", params.strategy);
	if (params.days !== undefined) {
		// carry 的窗口是日历天，用的是另一个开关
		argv.push(params.mode === "carry" ? "--carry-days" : "--bars", String(params.days));
	}
	if (mode === "robust") {
		if (params.segments !== undefined) argv.push("--segments", String(params.segments));
		if (params.trials !== undefined) argv.push("--trials", String(params.trials));
	}
	if (params.volTarget !== undefined && params.volTarget !== false) {
		argv.push("--vol-target");
		if (typeof params.volTarget === "number") argv.push(String(params.volTarget));
		if (params.volLookback !== undefined) argv.push("--vol-lookback", String(params.volLookback));
		if (params.volRebalance !== undefined) argv.push("--vol-rebalance", String(params.volRebalance));
	}
	return {
		id: makeRunId(mode),
		mode,
		cwd: root,
		argv,
		command: ["uv", ...argv].join(" "),
		startedAt: new Date().toISOString(),
	};
}

/** 解析日志并归档；同 id 只留一条（前台与后台轮询可能都触发） */
function finalize(
	paths: ResultPaths,
	job: { id: string; mode: RunMode; command: string; startedAt: string },
	status: JobStatus,
	durationMs: number,
	background: boolean,
): RunRecord {
	const logPath = join(paths.logs, `${job.id}.log`);
	const parsed = parseOutput(job.mode, readLog(logPath));
	const record: RunRecord = {
		id: job.id,
		mode: job.mode,
		command: job.command,
		symbols: [...new Set(parsed.symbols)],
		window: parsed.window,
		createdAt: job.startedAt,
		durationMs,
		exitCode: status.exitCode,
		background,
		rows: parsed.rows,
		dsr: parsed.dsr,
		pbo: parsed.pbo,
		carry: parsed.carry,
		charts: parsed.charts,
		equity: parsed.equity,
		notes: parsed.notes,
		logPath,
	};
	saveRun(paths, record);
	return record;
}

/** 给模型看的正文：无颜色、带关键结论与后续入口 */
function modelSummary(record: RunRecord): string {
	const tail = readLog(record.logPath).split(/\r?\n/).filter((line) => line.trim().length > 0);
	const lines = [
		`${MODE_LABEL[record.mode]}完成 · ${record.symbols.join(", ")} · ${duration(record.durationMs)} · 退出码 ${record.exitCode ?? "?"}`,
		"",
		...recordLines(record, plainPalette, false),
		"",
		`最近输出：`,
		...tail.slice(-LOG_TAIL_LINES),
	];
	return lines.join("\n");
}

function delay(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

/** 后台任务轮询：结束即归档 + 通知，同时把运行数写进 footer */
function startJobsPoller(getContext: () => ExtensionContext | undefined): () => void {
	const notified = new Set<string>();
	const timer = setInterval(() => {
		const ctx = getContext();
		if (!ctx?.hasUI) return;
		const root = findProjectRoot(ctx.cwd);
		if (!root) return;
		const paths = resultPaths(root);
		const jobs = listJobs(paths);
		const archived = new Set(loadRuns(paths).map((run) => run.id));

		for (const job of jobs) {
			if (archived.has(job.id) || notified.has(job.id)) continue;
			const stillRunning = job.status.state === "running" && isAlive(job.pid);
			if (stillRunning) continue;

			notified.add(job.id);
			const status = readStatus(join(paths.jobs, `${job.id}.json`));
			if (!status || status.state === "running") {
				ctx.ui.notify(`后台回测 ${job.id} 进程已消失（可能被强制终止），未归档`, "warning");
				continue;
			}
			const record = finalize(paths, job, status, Date.now() - Date.parse(job.startedAt), true);
			const first = record.rows[0] ?? record.carry[0];
			const key =
				record.mode === "carry"
					? `净年化 ${pct((first as { netAnnual?: number } | undefined)?.netAnnual)}`
					: `总收益 ${pct((first as { totalRet?: number } | undefined)?.totalRet, 1)}`;
			ctx.ui.notify(
				`${status.state === "done" ? "✓" : "✗"} 后台回测 ${job.id} ${status.state}：` +
					`${record.symbols.join(",")} ${key}（/runs 查看）`,
				status.state === "done" ? "info" : "error",
			);
		}

		const alive = jobs.filter((job) => job.status.state === "running" && isAlive(job.pid));
		ctx.ui.setStatus(
			JOBS_STATUS_KEY,
			alive.length > 0 ? `▶ ${alive.length} 个后台回测运行中（/jobs 查看）` : undefined,
		);
		for (const id of [...notified]) {
			if (!jobs.some((job) => job.id === id)) notified.delete(id);
		}
	}, JOBS_POLL_MS);

	return () => clearInterval(timer);
}

export default function backtestExtension(pi: ExtensionAPI) {
	if (process.env.NERD_TRADER_BACKTEST_DISABLE === "1") return;

	let context: ExtensionContext | undefined;

	pi.on("session_start", (_event, ctx) => {
		context = ctx;
	});
	pi.on("session_tree", (_event, ctx) => {
		context = ctx;
	});

	const stopPoller = startJobsPoller(() => context);
	pi.on("session_shutdown", () => {
		stopPoller();
		context?.ui.setStatus(STATUS_KEY, undefined);
		context?.ui.setStatus(JOBS_STATUS_KEY, undefined);
	});

	pi.registerTool({
		name: "backtest",
		label: "回测",
		description:
			"运行 nerd-trader 的量化回测：单次回测、稳健性检验（窗口分布/分段/多重检验校正）、资金费率 carry 研究。" +
			"前台运行时实时显示阶段与耗时，结束后给出结果卡片并归档；加 background 参数可后台运行。",
		promptSnippet: "运行量化回测与稳健性检验（research.py）",
		parameters: PARAMS,
		executionMode: "sequential",

		async execute(_toolCallId, params, signal, onUpdate, ctx): Promise<{
			content: { type: "text"; text: string }[];
			details: BacktestDetails;
			isError?: boolean;
		}> {
			const root = findProjectRoot(ctx.cwd);
			if (!root) {
				throw new Error(`未找到 research.py：请在 nerd-trader 目录下运行（当前 cwd=${ctx.cwd}）`);
			}
			const paths = resultPaths(root);
			const spec = buildSpec(root, params);
			const background = params.background === true;
			const handle: JobHandle = startJob(paths, spec);
			const startedAt = Date.now();
			const details: BacktestDetails = {
				id: spec.id,
				mode: spec.mode,
				phase: "启动",
				logLines: 0,
				elapsedMs: 0,
				background,
			};

			if (background) {
				ctx.ui.setStatus(STATUS_KEY, `▶ 后台回测 ${spec.id}`);
				return {
					content: [
						{
							type: "text",
							text: `已提交后台回测 ${spec.id}\n命令：${spec.command}\n完成后会通知；进度用 /jobs 查看，日志 ${handle.logPath}`,
						},
					],
					details,
				};
			}

			let offset = 0;
			let progress: Progress = { phase: "启动", detail: "" };
			let lastEmit = 0;
			let cancelled = false;
			try {
				for (;;) {
					if (signal?.aborted) {
						cancelled = true;
						break;
					}
					const status = readStatus(handle.statusPath);
					const chunk = readLogFrom(handle.logPath, offset);
					offset = chunk.offset;
					if (chunk.text.length > 0) {
						progress = progressOf(chunk.text.split(/\r?\n/), progress);
						details.logLines += chunk.text.split("\n").length - 1;
					}
					const now = Date.now();
					const elapsed = now - startedAt;
					if (now - lastEmit >= PROGRESS_THROTTLE_MS) {
						lastEmit = now;
						details.phase = progress.detail ? `${progress.phase} ${progress.detail}` : progress.phase;
						details.elapsedMs = elapsed;
						const line = `${details.phase} · ${duration(elapsed)} · ${details.logLines} 行输出`;
						ctx.ui.setStatus(STATUS_KEY, `⏳ 回测 ${spec.id} · ${line}`);
						onUpdate?.({
							content: [{ type: "text", text: line }],
							details: { ...details },
						});
					}
					if (status && status.state !== "running") break;
					if (!status && !isAlive(handle.pid)) break;
					await delay(POLL_MS);
				}
			} finally {
				ctx.ui.setStatus(STATUS_KEY, undefined);
			}

			if (cancelled) {
				killTree(handle.pid);
				return {
					content: [{ type: "text", text: `回测已取消（${spec.id}），日志 ${handle.logPath}` }],
					details: { ...details, error: "已取消" },
					isError: true,
				};
			}

			const status = readStatus(handle.statusPath) ?? {
				state: "killed",
				exitCode: null,
				startedAt: spec.startedAt,
			};
			const record = finalize(paths, spec, status, Date.now() - startedAt, false);
			details.record = record;
			details.phase = "完成";
			details.elapsedMs = record.durationMs;

			if (status.state !== "done") {
				return {
					content: [
						{
							type: "text",
							text: `回测异常结束（${status.state}，退出码 ${status.exitCode ?? "?"}）\n${modelSummary(record)}`,
						},
					],
					details: { ...details, error: status.state },
					isError: true,
				};
			}
			return {
				content: [{ type: "text", text: modelSummary(record) }],
				details,
			};
		},

		renderCall(args, theme, _context) {
			const mode = (args.mode ?? "backtest") as RunMode;
			const parts = [MODE_LABEL[mode], args.symbol ?? (mode === "backtest" ? "c1" : "c1,c2")];
			if (args.strategy) parts.push(`策略 ${args.strategy}`);
			if (args.days !== undefined) parts.push(`${args.days} 根`);
			if (args.volTarget !== undefined && args.volTarget !== false) parts.push(`波动率目标 ${args.volTarget}`);
			if (args.background) parts.push("后台");
			return new Text(
				theme.fg("toolTitle", theme.bold("backtest ")) + theme.fg("muted", parts.join(" · ")),
				0,
				0,
			);
		},

		renderResult(result, { expanded, isPartial }, theme, _context) {
			const details = result.details;
			if (!details) {
				const first = result.content[0];
				return new Text(first?.type === "text" ? first.text : "", 0, 0);
			}
			if (isPartial) {
				return new Text(
					theme.fg("accent", "⏳ ") +
						theme.fg("muted", `${details.phase} · ${duration(details.elapsedMs)} · ${details.logLines} 行输出`),
					0,
					0,
				);
			}
			if (details.error) {
				return new Text(theme.fg("error", `回测未正常完成：${details.error}`), 0, 0);
			}
			if (details.background) {
				return new Text(
					theme.fg("accent", "▶ ") + theme.fg("muted", `后台回测 ${details.id} 已提交（/jobs 查看）`),
					0,
					0,
				);
			}
			if (!details.record) {
				const first = result.content[0];
				return new Text(first?.type === "text" ? first.text : "", 0, 0);
			}
			const palette = themePalette(theme);
			return new ChartBlock(recordLines(details.record, palette, expanded), details.record, palette);
		},
	});

	pi.registerCommand("runs", {
		description: "查看历史回测记录：明细与两次运行对比",
		handler: async (_args, ctx) => {
			if (ctx.mode !== "tui") {
				ctx.ui.notify("/runs 需要交互模式", "error");
				return;
			}
			const paths = resolvePaths(ctx);
			if (!paths) return;
			await showRuns(ctx as ExtensionCommandContext, loadRuns(paths));
		},
	});

	pi.registerCommand("jobs", {
		description: "查看后台回测任务：进度、日志尾部、终止",
		handler: async (_args, ctx) => {
			if (ctx.mode !== "tui") {
				ctx.ui.notify("/jobs 需要交互模式", "error");
				return;
			}
			const paths = resolvePaths(ctx);
			if (!paths) return;
			await showJobs(ctx as ExtensionCommandContext, () => listJobs(paths));
		},
	});
}

/** 命令入口统一解析项目根，找不到就提示并返回 undefined */
function resolvePaths(ctx: ExtensionContext): ResultPaths | undefined {
	const root = findProjectRoot(ctx.cwd);
	if (!root) {
		ctx.ui.notify(`未找到 research.py：请在 nerd-trader 目录下运行（当前 cwd=${ctx.cwd}）`, "error");
		return undefined;
	}
	return resultPaths(root);
}
