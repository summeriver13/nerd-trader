/**
 * 回测任务执行器（纯 JS，由 Node 直接启动，不经过 pi 的 TS 加载器）
 *
 * 用法：node runner.mjs <spec.json> <logPath> <statusPath>
 *
 * 职责：启动 `uv run python research.py ...`，把标准输出与错误追加到日志文件，
 *       结束时把退出状态写进 status 文件。
 * 被强制终止（taskkill /T /F）时来不及写状态文件，由扩展按「进程已消失且无状态」
 * 判定为中断。
 */

import { spawn } from "node:child_process";
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

const [specPath, logPath, statusPath] = process.argv.slice(2);
if (!specPath || !logPath || !statusPath) {
	process.stderr.write("usage: node runner.mjs <spec.json> <log> <status>\n");
	process.exit(2);
}

function writeStatus(status) {
	try {
		mkdirSync(dirname(statusPath), { recursive: true });
		writeFileSync(statusPath, JSON.stringify(status, null, 2), "utf8");
	} catch {
		// 状态写不进去也不能影响回测本身
	}
}

let spec;
try {
	spec = JSON.parse(readFileSync(specPath, "utf8"));
} catch (error) {
	process.stderr.write(`runner: 无法读取任务定义 ${specPath}: ${error.message}\n`);
	writeStatus({
		state: "failed",
		exitCode: null,
		startedAt: new Date().toISOString(),
		finishedAt: new Date().toISOString(),
		error: `无法读取任务定义: ${error.message}`,
	});
	process.exit(2);
}

mkdirSync(dirname(logPath), { recursive: true });
writeStatus({ state: "running", exitCode: null, startedAt: spec.startedAt });

const child = spawn("uv", spec.argv, {
	cwd: spec.cwd,
	env: {
		...process.env,
		PYTHONUTF8: "1",
		PYTHONUNBUFFERED: "1",
		MPLBACKEND: "Agg",
	},
	stdio: ["ignore", "pipe", "pipe"],
});

const forward = (chunk) => appendFileSync(logPath, chunk);
child.stdout.on("data", forward);
child.stderr.on("data", forward);

child.on("error", (error) => {
	appendFileSync(logPath, `\n[runner] 启动失败: ${error.message}\n`);
	writeStatus({
		state: "failed",
		exitCode: null,
		startedAt: spec.startedAt,
		finishedAt: new Date().toISOString(),
		error: error.message,
	});
});

child.on("close", (code) => {
	writeStatus({
		state: code === 0 ? "done" : "failed",
		exitCode: code,
		startedAt: spec.startedAt,
		finishedAt: new Date().toISOString(),
	});
});

for (const signal of ["SIGINT", "SIGTERM"]) {
	process.on(signal, () => {
		try {
			child.kill();
		} catch {
			// 子进程可能已经退出
		}
		writeStatus({
			state: "killed",
			exitCode: null,
			startedAt: spec.startedAt,
			finishedAt: new Date().toISOString(),
		});
		process.exit(0);
	});
}
