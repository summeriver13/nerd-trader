/**
 * 数据契约与路径约定（扩展与 runner 共用）
 *
 * 运行产物统一落在项目的 data/results/ 下，与 research.py 的图表同目录：
 *   runs.json          已完成的运行记录（供 /runs 展示与对比）
 *   jobs/<id>.json     任务状态（由 runner.mjs 写入）
 *   jobs/<id>.spec.json 任务定义（由扩展写入，runner 读取）
 *   logs/<id>.log      research.py 的完整标准输出
 */

import { existsSync } from "node:fs";
import { dirname, join } from "node:path";

/** 运行模式，与 research.py 的 CLI 分支一一对应 */
export type RunMode = "backtest" | "robust" | "carry";

export const MODE_LABEL: Record<RunMode, string> = {
	backtest: "单次回测",
	robust: "稳健性检验",
	carry: "资金费率 carry",
};

/** 运行模式 -> research.py 的 CLI 开关 */
export const MODE_FLAG: Record<RunMode, string[]> = {
	backtest: [],
	robust: ["--robust"],
	carry: ["--carry"],
};

/** 任务定义：扩展写盘，runner 读取后拼出 uv 命令行 */
export interface RunSpec {
	id: string;
	mode: RunMode;
	/** 项目根目录（research.py 所在目录），也是 uv 的工作目录 */
	cwd: string;
	/** 传给 `uv` 的参数，例如 ["run", "python", "research.py", "--robust", ...] */
	argv: string[];
	/** 人类可读的命令行，用于展示 */
	command: string;
	startedAt: string;
	/** runner 进程号：启动后回写，供跨会话判断任务是否还在运行 */
	pid?: number;
}

/** 任务状态：由 runner 在结束时写入 */
export interface JobStatus {
	state: "running" | "done" | "failed" | "killed";
	exitCode: number | null;
	startedAt: string;
	finishedAt?: string;
	error?: string;
}

/** 表格里的一行指标（单次回测的标的、稳健性的策略、carry 的标的共用） */
export interface MetricRow {
	name: string;
	totalRet?: number;
	annualRet?: number;
	maxDrawdown?: number;
	calmar?: number;
	sharpe?: number;
	sortino?: number;
	benchRet?: number;
	trades?: number;
	winRate?: number;
	profitFactor?: number;
	exposure?: number;
	windowCount?: number;
	windowMedian?: number;
	windowWorst?: number;
	windowPositiveShare?: number;
	windowDdMedian?: number;
	segmentsPositive?: string;
	segmentsBeaten?: string;
}

/** 多重检验校正的一行 */
export interface DsrRow {
	name: string;
	sharpe: number;
	expectedMax: number;
	dsr: number;
	passed: boolean;
}

export interface PboInfo {
	value: number;
	detail: string;
}

/** carry 逐年一行 */
export interface CarryYear {
	year: number;
	settlements?: number;
	annual?: number;
	positiveShare?: number;
	worstRolling?: number;
}

/** carry 的一行（含逐年明细） */
export interface CarryRow {
	name: string;
	span: string;
	settlements: number;
	positiveShare?: number;
	medianRate?: number;
	grossAnnual?: number;
	capitalAdjusted?: number;
	netAnnual?: number;
	rollingMedian?: number;
	rollingWorst?: number;
	years: CarryYear[];
}

/** 一次运行的完整记录，落盘到 runs.json */
export interface RunRecord {
	id: string;
	mode: RunMode;
	command: string;
	symbols: string[];
	window: string;
	createdAt: string;
	durationMs: number;
	exitCode: number | null;
	background: boolean;
	rows: MetricRow[];
	dsr: DsrRow[];
	pbo?: PboInfo;
	carry: CarryRow[];
	charts: string[];
	/** 净值序列 CSV（research.py 落盘，终端内字符曲线读它） */
	equity?: string;
	notes: string[];
	logPath: string;
}

/** 运行中的任务（runs.json 之外的临时态，从 jobs 目录重建） */
export interface JobRecord {
	id: string;
	mode: RunMode;
	command: string;
	pid: number | null;
	startedAt: string;
	status: JobStatus;
	logPath: string;
	background: boolean;
}

/** 项目内所有运行产物的路径 */
export interface ResultPaths {
	root: string;
	results: string;
	jobs: string;
	logs: string;
	runsIndex: string;
}

/** 从 cwd 向上找 research.py，定位项目根；找不到返回 undefined */
export function findProjectRoot(cwd: string): string | undefined {
	let current = cwd;
	for (let depth = 0; depth < 6; depth += 1) {
		if (existsSync(join(current, "research.py"))) return current;
		const parent = dirname(current);
		if (parent === current) break;
		current = parent;
	}
	return undefined;
}

export function resultPaths(root: string): ResultPaths {
	const results = join(root, "data", "results");
	return {
		root,
		results,
		jobs: join(results, "jobs"),
		logs: join(results, "logs"),
		runsIndex: join(results, "runs.json"),
	};
}

/** 运行 id：时间戳 + 模式，天然按时间排序 */
export function makeRunId(mode: RunMode, now: Date = new Date()): string {
	const pad = (n: number) => String(n).padStart(2, "0");
	const stamp = [
		now.getFullYear(),
		pad(now.getMonth() + 1),
		pad(now.getDate()),
		"-",
		pad(now.getHours()),
		pad(now.getMinutes()),
		pad(now.getSeconds()),
	].join("");
	return `${stamp}-${mode}`;
}
