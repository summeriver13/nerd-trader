/**
 * 任务与运行记录的落盘读写
 *
 * 一次回测的产物：
 *   data/results/jobs/<id>.spec.json   任务定义（扩展写）
 *   data/results/jobs/<id>.json        任务状态（runner 写）
 *   data/results/logs/<id>.log         完整标准输出
 *   data/results/runs.json             已归档的运行记录（/runs 数据源）
 *
 * 所有写盘都失败即忽略：回测本身不能因为记录写不进去而中断。
 */

import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { JobRecord, JobStatus, ResultPaths, RunRecord, RunSpec } from "./spec.js";

const RUNNER_PATH = join(dirname(fileURLToPath(import.meta.url)), "runner.mjs");
const MAX_JOBS = 60;
const MAX_RUNS = 300;

export interface JobHandle {
	id: string;
	pid: number | null;
	specPath: string;
	statusPath: string;
	logPath: string;
}

function ensureDir(path: string): void {
	try {
		mkdirSync(path, { recursive: true });
	} catch {
		// 目录已存在或不可写
	}
}

function readJson<T>(path: string): T | undefined {
	try {
		return JSON.parse(readFileSync(path, "utf8")) as T;
	} catch {
		return undefined;
	}
}

function writeJson(path: string, value: unknown): void {
	try {
		ensureDir(dirname(path));
		writeFileSync(path, JSON.stringify(value, null, 2), "utf8");
	} catch {
		// 忽略写盘失败
	}
}

/** 启动一个回测任务：spec 写盘后交给 runner.mjs，扩展只持有 pid */
export function startJob(paths: ResultPaths, spec: RunSpec): JobHandle {
	ensureDir(paths.jobs);
	ensureDir(paths.logs);
	const specPath = join(paths.jobs, `${spec.id}.spec.json`);
	const statusPath = join(paths.jobs, `${spec.id}.json`);
	const logPath = join(paths.logs, `${spec.id}.log`);
	writeJson(specPath, spec);
	try {
		unlinkSync(statusPath);
	} catch {
		// 同 id 的旧状态不存在属正常
	}
	writeFileSync(logPath, "", "utf8");

	const child = spawn(process.execPath, [RUNNER_PATH, specPath, logPath, statusPath], {
		detached: true,
		stdio: "ignore",
		windowsHide: true,
	});
	child.unref();

	// 回写 pid：跨会话时靠它判断任务是否还在运行
	const pid = child.pid ?? null;
	if (pid !== null) writeJson(specPath, { ...spec, pid });
	return { id: spec.id, pid, specPath, statusPath, logPath };
}

/** 结束整个进程树：Windows 用 taskkill，类 Unix 杀进程组 */
export function killTree(pid: number | null): void {
	if (pid === null) return;
	if (process.platform === "win32") {
		spawnSync("taskkill", ["/PID", String(pid), "/T", "/F"], { stdio: "ignore", windowsHide: true });
		return;
	}
	try {
		process.kill(-pid, "SIGTERM");
	} catch {
		try {
			process.kill(pid, "SIGTERM");
		} catch {
			// 进程已退出
		}
	}
}

export function isAlive(pid: number | null): boolean {
	if (pid === null) return false;
	try {
		process.kill(pid, 0);
		return true;
	} catch (error) {
		return (error as NodeJS.ErrnoException).code === "EPERM";
	}
}

export function readStatus(statusPath: string): JobStatus | undefined {
	return readJson<JobStatus>(statusPath);
}

/** 读取日志全文（回测日志量级只有几十 KB，直接读文件最简单） */
export function readLog(logPath: string): string {
	try {
		return readFileSync(logPath, "utf8");
	} catch {
		return "";
	}
}

/** 从 offset 处读取日志新增部分，offset 以字符计 */
export function readLogFrom(logPath: string, offset: number): { text: string; offset: number } {
	const text = readLog(logPath);
	return { text: text.slice(offset), offset: text.length };
}

export function readLogTail(logPath: string, maxChars: number): string {
	const text = readLog(logPath);
	return text.length > maxChars ? text.slice(-maxChars) : text;
}

/** 读日志文件大小，用于判断是否还在产出（仅用于展示） */
export function logSize(logPath: string): number {
	try {
		return statSync(logPath).size;
	} catch {
		return 0;
	}
}

export function loadRuns(paths: ResultPaths): RunRecord[] {
	const runs = readJson<RunRecord[]>(paths.runsIndex);
	if (!Array.isArray(runs)) return [];
	return runs;
}

/** 归档一次运行；同 id 只保留一条（前台与后台轮询可能都触发归档） */
export function saveRun(paths: ResultPaths, record: RunRecord): RunRecord[] {
	const runs = loadRuns(paths).filter((item) => item.id !== record.id);
	runs.unshift(record);
	const trimmed = runs.slice(0, MAX_RUNS);
	writeJson(paths.runsIndex, trimmed);
	return trimmed;
}

/** 列出任务目录下的所有任务（新的在前），并清理超出上限的旧文件 */
export function listJobs(paths: ResultPaths): JobRecord[] {
	if (!existsSync(paths.jobs)) return [];
	let files: string[] = [];
	try {
		files = readdirSync(paths.jobs).filter((name) => name.endsWith(".spec.json"));
	} catch {
		return [];
	}
	const jobs: JobRecord[] = [];
	for (const file of files) {
		const id = file.replace(/\.spec\.json$/, "");
		const spec = readJson<RunSpec>(join(paths.jobs, file));
		if (!spec) continue;
		const statusPath = join(paths.jobs, `${id}.json`);
		const logPath = join(paths.logs, `${id}.log`);
		const status =
			readStatus(statusPath) ??
			({
				state: "running",
				exitCode: null,
				startedAt: spec.startedAt,
			} satisfies JobStatus);
		jobs.push({
			id,
			mode: spec.mode,
			command: spec.command,
			pid: spec.pid ?? null,
			startedAt: spec.startedAt,
			status,
			logPath,
			background: true,
		});
	}
	jobs.sort((a, b) => b.startedAt.localeCompare(a.startedAt));
	for (const stale of jobs.slice(MAX_JOBS)) {
		for (const suffix of [".spec.json", ".json"]) {
			try {
				unlinkSync(join(paths.jobs, `${stale.id}${suffix}`));
			} catch {
				// 文件可能已被删除
			}
		}
	}
	return jobs.slice(0, MAX_JOBS);
}
