/**
 * 终端渲染：结果卡片、进度行、/runs 与 /jobs 面板
 *
 * 表格一律用 pi-tui 的 visibleWidth/truncateToWidth 计算列宽：
 * 策略名里有中文与括号，按字符数 pad 会错位。
 */

import type { Theme } from "@earendil-works/pi-coding-agent";
import { getSelectListTheme } from "@earendil-works/pi-coding-agent";
import {
	Image,
	SelectList,
	getCapabilities,
	matchesKey,
	truncateToWidth,
	visibleWidth,
} from "@earendil-works/pi-tui";
import type { Component, SelectItem } from "@earendil-works/pi-tui";
import { readFileSync } from "node:fs";
import type { CarryRow, JobRecord, MetricRow, RunRecord } from "./spec.js";
import { MODE_LABEL } from "./spec.js";

const COLUMN_GAP = "  ";
const MAX_CHART_WIDTH_CELLS = 96;
const MAX_LIST_VISIBLE = 14;

/** 百分比显示；缺值显示为 — */
export function pct(value: number | undefined, digits = 2): string {
	if (value === undefined || Number.isNaN(value)) return "—";
	return `${(value * 100).toFixed(digits)}%`;
}

export function num(value: number | undefined, digits = 2): string {
	if (value === undefined || Number.isNaN(value)) return "—";
	return value.toFixed(digits);
}

export function duration(ms: number): string {
	const seconds = Math.round(ms / 1000);
	if (seconds < 60) return `${seconds}s`;
	const minutes = Math.floor(seconds / 60);
	if (minutes < 60) return `${minutes}m${String(seconds % 60).padStart(2, "0")}s`;
	return `${Math.floor(minutes / 60)}h${String(minutes % 60).padStart(2, "0")}m`;
}

/** 按显示宽度补空格（中文占两列） */
function padCell(text: string, width: number): string {
	const current = visibleWidth(text);
	if (current === width) return text;
	if (current > width) return truncateToWidth(text, width);
	return text + " ".repeat(width - current);
}

/** 极简表格：第一列左对齐，其余右对齐 */
function table(headers: string[], rows: string[][]): string[] {
	const widths = headers.map((header, index) =>
		Math.max(visibleWidth(header), ...rows.map((row) => visibleWidth(row[index] ?? ""))),
	);
	const line = (cells: string[]) =>
		cells
			.map((cell, index) => padCell(cell, widths[index] ?? 0))
			.join(COLUMN_GAP);
	const separator = widths.map((width) => "─".repeat(width)).join(COLUMN_GAP);
	return [line(headers), separator, ...rows.map(line)];
}

/** 上色抽象：终端用主题，写给模型的摘要用原样文本 */
export interface Palette {
	accent: (text: string) => string;
	muted: (text: string) => string;
	success: (text: string) => string;
	error: (text: string) => string;
	warning: (text: string) => string;
	dim: (text: string) => string;
	bold: (text: string) => string;
}

export function themePalette(theme: Theme): Palette {
	return {
		accent: (text) => theme.fg("accent", text),
		muted: (text) => theme.fg("muted", text),
		success: (text) => theme.fg("success", text),
		error: (text) => theme.fg("error", text),
		warning: (text) => theme.fg("warning", text),
		dim: (text) => theme.fg("dim", text),
		bold: (text) => theme.bold(text),
	};
}

const identity = (text: string) => text;

/** 无颜色输出，用于工具返回给模型的正文 */
export const plainPalette: Palette = {
	accent: identity,
	muted: identity,
	success: identity,
	error: identity,
	warning: identity,
	dim: identity,
	bold: identity,
};

/** 收益正负上色 */
function signed(palette: Palette, value: number | undefined, digits = 2): string {
	const text = pct(value, digits);
	if (value === undefined) return text;
	return value >= 0 ? palette.success(text) : palette.error(text);
}

/** 一行判词：DSR 是否过门槛 + PBO */
export function verdictOf(record: RunRecord): string {
	const best = [...record.dsr].sort((a, b) => b.dsr - a.dsr)[0];
	const parts: string[] = [];
	if (best) {
		parts.push(`最高 DSR ${num(best.dsr, 3)}（${best.name}）${best.passed ? "过门槛" : "未过 0.95"}`);
	}
	if (record.pbo) parts.push(`PBO ${pct(record.pbo.value, 1)}`);
	const passed = record.dsr.filter((row) => row.passed).length;
	if (passed > 0) parts.push(`${passed} 个候选通过`);
	return parts.join(" | ");
}

const METRIC_HEADERS = ["策略", "总收益", "年化", "回撤", "卡尔玛", "窗口中位", "正收益窗", "窗口最差", "DSR"];

function metricCells(record: RunRecord, row: MetricRow, name: string): string[] {
	const dsr = record.dsr.find((item) => item.name === row.name);
	return [
		name,
		pct(row.totalRet, 1),
		pct(row.annualRet, 1),
		pct(row.maxDrawdown, 1),
		num(row.calmar),
		pct(row.windowMedian, 1),
		pct(row.windowPositiveShare, 1),
		pct(row.windowWorst, 1),
		dsr ? num(dsr.dsr, 3) : "—",
	];
}

/** 行名去掉「标的 · 」前缀（标的信息由标题与分组行给出） */
function splitStrategyName(full: string): { symbol: string; name: string } {
	const at = full.indexOf(" · ");
	return at < 0 ? { symbol: "", name: full } : { symbol: full.slice(0, at), name: full.slice(at + 3) };
}

const COLLAPSED_ROWS = 6;

/** 按标的把策略行分组，顺带去掉行名前缀 */
function groupBySymbol(rows: MetricRow[]): Map<string, { row: MetricRow; name: string }[]> {
	const groups = new Map<string, { row: MetricRow; name: string }[]>();
	for (const row of rows) {
		const { symbol, name } = splitStrategyName(row.name);
		const list = groups.get(symbol) ?? [];
		list.push({ row, name });
		groups.set(symbol, list);
	}
	return groups;
}

/** 折叠时每个标的取等量前几行，避免出现「某个标的只剩一行」的错觉 */
function collapsedRows(rows: MetricRow[], symbolCount: number): MetricRow[] {
	const perSymbol = Math.max(1, Math.floor(COLLAPSED_ROWS / symbolCount));
	const counts = new Map<string, number>();
	const kept: MetricRow[] = [];
	for (const row of rows) {
		const symbol = splitStrategyName(row.name).symbol;
		const used = counts.get(symbol) ?? 0;
		if (used >= perSymbol) continue;
		counts.set(symbol, used + 1);
		kept.push(row);
	}
	return kept;
}

/** 稳健性：策略表（折叠时每标的取前几行）。多标的时按标的拆表，否则每行重复的前缀会把列宽挤爆 */
function robustLines(record: RunRecord, expanded: boolean): string[] {
	if (record.rows.length === 0) return [];
	const rows = expanded ? record.rows : collapsedRows(record.rows, Math.max(1, record.symbols.length));
	const groups = groupBySymbol(rows);
	const lines: string[] = [];
	for (const [symbol, group] of groups) {
		if (groups.size > 1) lines.push(symbol, "");
		lines.push(...table(METRIC_HEADERS, group.map((item) => metricCells(record, item.row, item.name))));
		if (groups.size > 1) lines.push("");
	}
	return lines;
}

/** 单次回测：评估区间 + 三行摘要 */
function backtestLines(record: RunRecord, palette: Palette): string[] {
	const row = record.rows[0];
	if (!row) return [];
	const lines = [
		`总收益 ${signed(palette, row.totalRet)}   年化 ${signed(palette, row.annualRet)}   基准 ${pct(row.benchRet)}`,
		`卡尔玛 ${num(row.calmar)}   夏普 ${num(row.sharpe)}   索提诺 ${num(row.sortino)}   最大回撤 ${pct(row.maxDrawdown)}`,
		`交易 ${row.trades ?? "—"} 回合   胜率 ${pct(row.winRate, 1)}   盈亏比 ${num(row.profitFactor)}`,
	];
	return record.window ? [`评估区间 ${record.window}`, ...lines] : lines;
}

/** carry 逐年收益：折叠时只给首尾，展开给全部（衰减是这条路的核心结论） */
function carryYearLine(row: CarryRow, expanded: boolean): string {
	if (row.years.length === 0) return "";
	const cells = row.years.map((item) => `${item.year} ${pct(item.annual, 1)}`);
	const shown = expanded || cells.length <= 3 ? cells : [cells[0], "…", cells[cells.length - 1]];
	return `  逐年: ${shown.join("  →  ")}`;
}

/** carry：毛收益到净收益的漏斗 */
function carryLines(record: RunRecord, palette: Palette, expanded: boolean): string[] {
	const lines: string[] = [];
	for (const row of record.carry) {
		lines.push(
			`${palette.bold(row.name)}  ${row.span}  ${row.settlements} 次结算  正费率 ${pct(row.positiveShare, 1)}`,
		);
		lines.push(
			`  毛年化 ${signed(palette, row.grossAnnual)}  →  按资本占用 1.5 倍 ${signed(palette, row.capitalAdjusted)}` +
				`  →  扣每月再平衡 ${signed(palette, row.netAnnual)}`,
		);
		lines.push(
			`  30 天滚动: 中位 ${pct(row.rollingMedian)}  最差 ${pct(row.rollingWorst)}  单次中位 ${pct(row.medianRate, 4)}`,
		);
		const years = carryYearLine(row, expanded);
		if (years) lines.push(years);
	}
	return lines;
}

function bodyLines(record: RunRecord, palette: Palette, expanded: boolean): string[] {
	if (record.mode === "carry") return carryLines(record, palette, expanded);
	if (record.mode === "backtest") return backtestLines(record, palette);
	return robustLines(record, expanded);
}

/** 结果卡片：折叠时只给结论，展开时给完整表格与路径 */
export function recordLines(record: RunRecord, palette: Palette, expanded: boolean): string[] {
	const head =
		`${MODE_LABEL[record.mode]} ${record.symbols.join(", ")}` +
		`  ·  ${record.window || record.createdAt}` +
		`  ·  ${duration(record.durationMs)}`;
	const lines = [palette.accent(head), "", ...bodyLines(record, palette, expanded)];

	const verdict = verdictOf(record);
	if (verdict) lines.push("", palette.muted(`判词：${verdict}`));

	const notes = expanded ? record.notes : record.notes.slice(0, 2);
	for (const note of notes) lines.push(palette.warning(note));

	const chart = record.charts[record.charts.length - 1];
	if (chart) lines.push("", palette.dim(`图表 ${chart}`));
	if (expanded) {
		lines.push(palette.dim(`日志 ${record.logPath}`), palette.dim(`命令 ${record.command}`));
	}
	return lines;
}

/** SelectList 的标签：一行内看清是什么运行、结果如何 */
function dsrMark(record: RunRecord): string {
	if (record.dsr.length === 0) return "";
	return record.dsr.some((row) => row.passed) ? " ✓DSR" : " ✗DSR";
}

/** 列表主列：只放 id 与多重检验标记。pi-tui 的 SelectList 给主列固定列宽，其他信息一律放副列 */
export function recordLabel(record: RunRecord): string {
	return `${record.id}${dsrMark(record)}`;
}

/** 列表副列：标的 · 关键指标 · 该模式最有信息量的一行 */
export function recordSubtitle(record: RunRecord): string {
	return [record.symbols.join(","), metricSummary(record), modeDetail(record)].filter(Boolean).join(" · ");
}

/** 关键指标：carry 看净年化，其余看卡尔玛与最大回撤 */
function metricSummary(record: RunRecord): string {
	if (record.mode === "carry") return `净 ${pct(record.carry[0]?.netAnnual)}`;
	return `卡尔玛 ${num(record.rows[0]?.calmar)} · 回撤 ${pct(record.rows[0]?.maxDrawdown, 1)}`;
}

/** 模式专属信息：单次看评估区间（列表里只留日期，根数在明细里看），稳健性看滚动窗口数与 PBO，carry 看费率跨度 */
function modeDetail(record: RunRecord): string {
	if (record.mode === "robust") return robustDetail(record);
	if (record.mode === "carry") return record.carry[0]?.span ?? "";
	return record.window.replace(/（[^）]*）$/, "");
}

/** 稳健性副列：滚动窗口数 + PBO */
function robustDetail(record: RunRecord): string {
	const parts: string[] = [];
	const windows = record.rows[0]?.windowCount;
	if (windows) parts.push(`${windows} 个窗口`);
	if (record.pbo) parts.push(`PBO ${pct(record.pbo.value, 1)}`);
	return parts.join(" · ");
}

export function recordItem(record: RunRecord): SelectItem {
	return { value: record.id, label: recordLabel(record), description: recordSubtitle(record) };
}

/** 终端是否支持内联图片（kitty / iTerm2 协议）；Windows Terminal 下为 false */
export function supportsInlineImages(): boolean {
	return getCapabilities().images !== null;
}

/** 图表 PNG 转 Image 组件；终端不支持图形协议时 Image 自身会退化成文本 */
export function chartImage(path: string | undefined, theme: Theme): Image | undefined {
	if (!path) return undefined;
	try {
		const data = readFileSync(path).toString("base64");
		return new Image(
			data,
			"image/png",
			{ fallbackColor: (text) => theme.fg("dim", text) },
			{ maxWidthCells: MAX_CHART_WIDTH_CELLS, filename: path },
		);
	} catch {
		return undefined;
	}
}

const JOB_MARK: Record<JobRecord["status"]["state"], string> = {
	running: "▶",
	done: "✓",
	failed: "✗",
	killed: "⛔",
};

export function jobLabel(job: JobRecord, now: number): string {
	const started = Date.parse(job.startedAt);
	const elapsed = Number.isFinite(started) ? duration(Math.max(0, now - started)) : "—";
	return `${JOB_MARK[job.status.state]} ${job.id}  ${MODE_LABEL[job.mode]}  ${elapsed}  ${job.status.state}`;
}

export function jobItem(job: JobRecord, now: number): SelectItem {
	return { value: job.id, label: jobLabel(job, now), description: job.command };
}

export interface ListPanelOptions {
	title: string;
	items: SelectItem[];
	theme: Theme;
	hint: string;
	/** 返回 true 表示该按键已处理，不再交给列表 */
	onKey: (data: string) => boolean;
}

/** 列表面板：SelectList 负责选择与过滤，额外按键由回调处理 */
export class ListPanel implements Component {
	private list: SelectList;
	private theme: Theme;
	private title: string;
	private hint: string;
	private onKey: (data: string) => boolean;

	constructor(options: ListPanelOptions) {
		this.theme = options.theme;
		this.title = options.title;
		this.hint = options.hint;
		this.onKey = options.onKey;
		this.list = this.buildList(options.items);
	}

	/** SelectList 的条目在构造时固定，刷新列表只能重建 */
	replaceItems(items: SelectItem[]): void {
		this.list = this.buildList(items);
	}

	setTitle(title: string): void {
		this.title = title;
	}

	selected(): SelectItem | null {
		return this.list.getSelectedItem();
	}

	render(width: number): string[] {
		return [
			truncateToWidth(this.theme.fg("accent", this.theme.bold(this.title)), width),
			"",
			...this.list.render(width),
			"",
			truncateToWidth(this.theme.fg("dim", this.hint), width),
		];
	}

	invalidate(): void {
		this.list.invalidate();
	}

	handleInput(data: string): void {
		if (this.onKey(data)) return;
		this.list.handleInput(data);
	}

	private buildList(items: SelectItem[]): SelectList {
		return new SelectList(items, Math.min(Math.max(items.length, 3), MAX_LIST_VISIBLE), getSelectListTheme());
	}
}

/** 两次运行的对比表：只列可比的关键指标 */
export function compareLines(a: RunRecord, b: RunRecord, palette: Palette): string[] {
	const rows: string[][] = [];
	for (let index = 0; index < Math.max(a.rows.length, b.rows.length); index += 1) {
		const left = a.rows[index];
		const right = b.rows[index];
		if (!left && !right) continue;
		rows.push([
			left?.name ?? right?.name ?? "",
			pct(left?.totalRet, 1),
			pct(right?.totalRet, 1),
			num(left?.calmar),
			num(right?.calmar),
			pct(left?.windowPositiveShare, 1),
			pct(right?.windowPositiveShare, 1),
		]);
	}
	const lines = [
		palette.accent(`对比 ${a.id}  vs  ${b.id}`),
		`${a.symbols.join(",")}  ${a.window}`,
		`${b.symbols.join(",")}  ${b.window}`,
		"",
		...table(["策略", "A 总收益", "B 总收益", "A 卡尔玛", "B 卡尔玛", "A 正收益窗", "B 正收益窗"], rows),
	];
	const verdictA = verdictOf(a);
	const verdictB = verdictOf(b);
	if (verdictA) lines.push("", palette.muted(`A：${verdictA}`));
	if (verdictB) lines.push(palette.muted(`B：${verdictB}`));
	return lines;
}

/** 明细面板：静态文本 + 可选图表，Esc 返回 */
export class DetailPanel implements Component {
	private title: string;
	private body: string[];
	private hint: string;
	private image: Image | undefined;

	constructor(title: string, body: string[], hint: string, image?: Image) {
		this.title = title;
		this.body = body;
		this.hint = hint;
		this.image = image;
	}

	setContent(title: string, body: string[], hint: string, image?: Image): void {
		this.title = title;
		this.body = body;
		this.hint = hint;
		this.image = image;
	}

	render(width: number): string[] {
		const lines = [truncateToWidth(this.title, width), ""];
		for (const line of this.body) lines.push(truncateToWidth(line, width));
		if (this.image) lines.push("", ...this.image.render(width));
		lines.push("", truncateToWidth(this.hint, width));
		return lines;
	}

	invalidate(): void {
		this.image?.invalidate();
	}
}

export { matchesKey };
