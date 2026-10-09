/**
 * /runs 与 /jobs 的交互面板
 *
 * 每个面板是一个状态机组件（列表 / 明细 / 对比），Esc 逐层返回。
 * 不嵌套 ctx.ui.custom：同一组件内切换状态，避免多层层叠自定义界面。
 */

import type { ExtensionCommandContext, Theme } from "@earendil-works/pi-coding-agent";
import { getSelectListTheme } from "@earendil-works/pi-coding-agent";
import { SelectList, matchesKey, truncateToWidth } from "@earendil-works/pi-tui";
import type { Component, SelectItem, TUI } from "@earendil-works/pi-tui";
import { killTree, readLogTail } from "./proc.js";
import type { JobRecord, RunRecord } from "./spec.js";
import { chartImage, compareLines, jobItem, recordItem, recordLines, supportsInlineImages, themePalette } from "./ui.js";
import { recordChart } from "./chart.js";

const MAX_DETAIL_LINES = 34;
const MAX_LOG_TAIL_CHARS = 5000;
const MAX_LIST_VISIBLE = 14;

function buildList(items: SelectItem[]): SelectList {
	return new SelectList(items, Math.min(Math.max(items.length, 3), MAX_LIST_VISIBLE), getSelectListTheme());
}

function cap(lines: string[], limit: number): string[] {
	if (lines.length <= limit) return lines;
	return [...lines.slice(0, limit), `… 另有 ${lines.length - limit} 行，按 e 展开`];
}

type RunsView = { kind: "list" } | { kind: "detail"; id: string; expanded: boolean } | { kind: "compare"; a: string; b: string };

interface RunsPanelOptions {
	runs: RunRecord[];
	theme: Theme;
	tui: TUI;
	done: () => void;
}

/** 历史记录面板：Enter 看明细、c 标记/对比、Esc 逐层返回 */
export class RunsPanel implements Component {
	private runs: RunRecord[];
	private theme: Theme;
	private tui: TUI;
	private done: () => void;
	private list: SelectList;
	private view: RunsView = { kind: "list" };
	private marked: string | undefined;

	constructor(options: RunsPanelOptions) {
		this.runs = options.runs;
		this.theme = options.theme;
		this.tui = options.tui;
		this.done = options.done;
		this.list = buildList(options.runs.map(recordItem));
	}

	render(width: number): string[] {
		if (this.view.kind === "detail") return this.renderDetail(width, this.view);
		if (this.view.kind === "compare") return this.renderCompare(width, this.view);
		return this.renderList(width);
	}

	handleInput(data: string): void {
		if (this.view.kind === "list") {
			this.handleListKey(data);
			return;
		}
		if (this.view.kind === "detail" && data === "e") {
			this.view = { ...this.view, expanded: !this.view.expanded };
			this.tui.requestRender();
			return;
		}
		if (matchesKey(data, "escape") || data === "q") {
			this.view = { kind: "list" };
			this.tui.requestRender();
		}
	}

	invalidate(): void {
		this.list.invalidate();
	}

	private renderList(width: number): string[] {
		const title = `回测记录（${this.runs.length} 条）${this.marked ? `  ·  已标记 A：${this.marked}` : ""}`;
		return [
			truncateToWidth(this.theme.fg("accent", this.theme.bold(title)), width),
			"",
			...this.list.render(width),
			"",
			truncateToWidth(
				this.theme.fg("dim", "↑↓ 选择 · Enter 明细 · c 标记并对比 · Esc 关闭"),
				width,
			),
		];
	}

	private renderDetail(width: number, view: { id: string; expanded: boolean }): string[] {
		const record = this.runs.find((item) => item.id === view.id);
		if (!record) return [this.theme.fg("error", `记录 ${view.id} 不存在`)];
		const palette = themePalette(this.theme);
		const body = recordLines(record, palette, view.expanded);
		const lines = [truncateToWidth(this.theme.fg("accent", this.theme.bold(`运行明细 ${record.id}`)), width), ""];
		for (const line of cap(body, MAX_DETAIL_LINES)) lines.push(truncateToWidth(line, width));
		const chart = recordChart(record, width, palette);
		if (chart.length > 0) lines.push("", ...chart);
		// 内联图片只在支持 kitty / iTerm2 协议的终端里画，否则会退化成一行「[Image: ...]」
		const image =
			view.expanded && supportsInlineImages() ? chartImage(record.charts.at(-1), this.theme) : undefined;
		if (image) lines.push("", ...image.render(width));
		lines.push("", truncateToWidth(this.theme.fg("dim", "e 展开/收起 · Esc 返回列表"), width));
		return lines;
	}

	private renderCompare(width: number, view: { a: string; b: string }): string[] {
		const left = this.runs.find((item) => item.id === view.a);
		const right = this.runs.find((item) => item.id === view.b);
		if (!left || !right) return [this.theme.fg("error", "对比对象不存在")];
		const lines = compareLines(left, right, themePalette(this.theme)).map((line) => truncateToWidth(line, width));
		lines.push("", truncateToWidth(this.theme.fg("dim", "Esc 返回列表"), width));
		return lines;
	}

	private handleListKey(data: string): void {
		if (matchesKey(data, "escape")) {
			if (this.marked) {
				this.marked = undefined;
				this.tui.requestRender();
				return;
			}
			this.done();
			return;
		}
		const selected = this.list.getSelectedItem()?.value;
		if (matchesKey(data, "enter") || matchesKey(data, "return")) {
			if (selected) {
				this.view = { kind: "detail", id: selected, expanded: false };
				this.tui.requestRender();
			}
			return;
		}
		if (data === "c") {
			this.toggleCompare(selected);
			return;
		}
		this.list.handleInput(data);
	}

	private toggleCompare(selected: string | undefined): void {
		if (!selected) return;
		if (!this.marked) {
			this.marked = selected;
			this.tui.requestRender();
			return;
		}
		if (this.marked === selected) {
			this.marked = undefined;
			this.tui.requestRender();
			return;
		}
		this.view = { kind: "compare", a: this.marked, b: selected };
		this.marked = undefined;
		this.tui.requestRender();
	}
}

type JobsView = { kind: "list" } | { kind: "log"; id: string };

interface JobsPanelOptions {
	reload: () => JobRecord[];
	theme: Theme;
	tui: TUI;
	done: () => void;
	notify: (message: string, type?: "info" | "warning" | "error") => void;
}

/** 任务面板：Enter 看日志尾部、x 终止、r 刷新、Esc 关闭 */
export class JobsPanel implements Component {
	private reload: () => JobRecord[];
	private theme: Theme;
	private tui: TUI;
	private done: () => void;
	private notify: (message: string, type?: "info" | "warning" | "error") => void;
	private jobs: JobRecord[];
	private list: SelectList;
	private view: JobsView = { kind: "list" };

	constructor(options: JobsPanelOptions) {
		this.reload = options.reload;
		this.theme = options.theme;
		this.tui = options.tui;
		this.done = options.done;
		this.notify = options.notify;
		this.jobs = options.reload();
		this.list = buildList(this.jobs.map((job) => jobItem(job, Date.now())));
	}

	render(width: number): string[] {
		if (this.view.kind === "log") return this.renderLog(width, this.view.id);
		const running = this.jobs.filter((job) => job.status.state === "running").length;
		const title = `回测任务（${this.jobs.length} 个，运行中 ${running}）`;
		return [
			truncateToWidth(this.theme.fg("accent", this.theme.bold(title)), width),
			"",
			...this.list.render(width),
			"",
			truncateToWidth(this.theme.fg("dim", "↑↓ 选择 · Enter 日志 · x 终止 · r 刷新 · Esc 关闭"), width),
		];
	}

	handleInput(data: string): void {
		if (this.view.kind === "log") {
			if (matchesKey(data, "escape") || data === "q") {
				this.view = { kind: "list" };
				this.tui.requestRender();
			}
			return;
		}
		if (matchesKey(data, "escape")) {
			this.done();
			return;
		}
		if (data === "r") {
			this.refresh();
			return;
		}
		const selected = this.list.getSelectedItem()?.value;
		if (matchesKey(data, "enter") || matchesKey(data, "return")) {
			if (selected) {
				this.view = { kind: "log", id: selected };
				this.tui.requestRender();
			}
			return;
		}
		if (data === "x") {
			this.kill(selected);
			return;
		}
		this.list.handleInput(data);
	}

	invalidate(): void {
		this.list.invalidate();
	}

	private renderLog(width: number, id: string): string[] {
		const job = this.jobs.find((item) => item.id === id);
		if (!job) return [this.theme.fg("error", `任务 ${id} 不存在`)];
		const tail = readLogTail(job.logPath, MAX_LOG_TAIL_CHARS).split(/\r?\n/);
		const lines = [
			truncateToWidth(this.theme.fg("accent", this.theme.bold(`${job.id}  ${job.status.state}`)), width),
			truncateToWidth(this.theme.fg("dim", job.command), width),
			"",
			...tail.map((line) => truncateToWidth(line, width)),
			"",
			truncateToWidth(this.theme.fg("dim", "Esc 返回任务列表"), width),
		];
		return lines;
	}

	private refresh(): void {
		this.jobs = this.reload();
		this.list = buildList(this.jobs.map((job) => jobItem(job, Date.now())));
		this.tui.requestRender();
	}

	private kill(id: string | undefined): void {
		const job = this.jobs.find((item) => item.id === id);
		if (!job) return;
		if (job.status.state !== "running") {
			this.notify(`${job.id} 已结束（${job.status.state}），无需终止`, "info");
			return;
		}
		killTree(job.pid);
		this.notify(`已终止 ${job.id}`, "warning");
		this.refresh();
	}
}

/** /runs 入口 */
export async function showRuns(ctx: ExtensionCommandContext, runs: RunRecord[]): Promise<void> {
	if (runs.length === 0) {
		ctx.ui.notify("还没有回测记录，先用 backtest 工具跑一次", "info");
		return;
	}
	await ctx.ui.custom<void>((tui, theme, _keybindings, done) => new RunsPanel({ runs, theme, tui, done }));
}

/** /jobs 入口 */
export async function showJobs(ctx: ExtensionCommandContext, reload: () => JobRecord[]): Promise<void> {
	const jobs = reload();
	if (jobs.length === 0) {
		ctx.ui.notify("还没有回测任务（用 backtest 工具跑一次，或加 background 参数）", "info");
		return;
	}
	await ctx.ui.custom<void>(
		(tui, theme, _keybindings, done) =>
			new JobsPanel({
				reload,
				theme,
				tui,
				done,
				notify: (message, type) => ctx.ui.notify(message, type),
			}),
	);
}
