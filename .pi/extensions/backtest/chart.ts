/**
 * 终端内字符曲线
 *
 * 为什么不用 pi 的内联图片：pi-tui 只实现 kitty / iTerm2 两种图形协议，
 * Windows Terminal 下 Image 只退化成「[Image: path 1400x1000]」一行字。
 * 用 braille 字符（U+2800 区）画图则任何终端都能显示，且能按宽度自适应。
 *
 * 数据来自 research.py 落盘的 <策略>_equity.csv（净值 / 基准 / 回撤三条序列）。
 */

import type { Component } from "@earendil-works/pi-tui";
import { truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";
import { readFileSync, statSync } from "node:fs";
import type { RunRecord } from "./spec.js";
import type { Palette } from "./ui.js";
import { pct } from "./ui.js";

/** 8 点盲文的位映射：索引 = (x % 2) * 4 + (y % 4) */
const BRAILLE_BITS = [0x01, 0x02, 0x04, 0x40, 0x08, 0x10, 0x20, 0x80];
const BRAILLE_BLANK = 0x2800;
/** 空格而非 U+2800：部分字体把空白盲文渲染成方块 */
const BLANK_CELL = " ";
const BLOCKS = " ▁▂▃▄▅▆▇█";
const CHART_ROWS = 9;
const MIN_CHART_WIDTH = 28;

export interface EquitySeries {
	/** 日期标签（已截到 YYYY-MM-DD） */
	stamps: string[];
	/** 净值，已按首值归一化为 1.00 起点 */
	equity: number[];
	benchmark: number[];
	drawdown: number[];
}

interface Canvas {
	cols: number;
	bits: Uint8Array;
}

interface CacheEntry {
	mtimeMs: number;
	series: EquitySeries | undefined;
}

const cache = new Map<string, CacheEntry>();

/** 读净值 CSV；文件不存在或格式不对时返回 undefined（面板据此不画图） */
export function loadEquitySeries(path: string): EquitySeries | undefined {
	try {
		const mtimeMs = statSync(path).mtimeMs;
		const hit = cache.get(path);
		if (hit && hit.mtimeMs === mtimeMs) return hit.series;
		const series = parseEquityCsv(readFileSync(path, "utf8"));
		cache.set(path, { mtimeMs, series });
		return series;
	} catch {
		return undefined;
	}
}

function parseEquityCsv(text: string): EquitySeries | undefined {
	const stamps: string[] = [];
	const equity: number[] = [];
	const benchmark: number[] = [];
	const drawdown: number[] = [];
	for (const line of text.split("\n").slice(1)) {
		const cells = line.split(",");
		const stamp = cells[0];
		const value = Number(cells[1]);
		const bench = Number(cells[2]);
		const drop = Number(cells[3]);
		if (!stamp || Number.isNaN(value) || Number.isNaN(bench) || Number.isNaN(drop)) continue;
		stamps.push(stamp.slice(0, 10));
		equity.push(value);
		benchmark.push(bench);
		drawdown.push(drop);
	}
	if (equity.length < 2) return undefined;
	const base = equity[0] || 1;
	const benchBase = benchmark[0] || 1;
	return {
		stamps,
		equity: equity.map((item) => item / base),
		benchmark: benchmark.map((item) => item / benchBase),
		drawdown,
	};
}

function clamp(value: number, low: number, high: number): number {
	return Math.min(Math.max(value, low), high);
}

function extent(values: number[]): { low: number; high: number } {
	let low = Number.POSITIVE_INFINITY;
	let high = Number.NEGATIVE_INFINITY;
	for (const value of values) {
		if (value < low) low = value;
		if (value > high) high = value;
	}
	return { low: Number.isFinite(low) ? low : 0, high: Number.isFinite(high) ? high : 1 };
}

function setDot(canvas: Canvas, x: number, y: number): void {
	const cell = Math.floor(y / 4) * canvas.cols + Math.floor(x / 2);
	const bit = BRAILLE_BITS[(x % 2) * 4 + (y % 4)] ?? 0;
	canvas.bits[cell] = (canvas.bits[cell] ?? 0) | bit;
}

/** 逐点阵列采样并竖向填充：每列只画一个点会在陡峭处断线 */
function plotSeries(canvas: Canvas, values: number[], low: number, high: number): void {
	const dotCols = canvas.cols * 2;
	const dotRows = CHART_ROWS * 4;
	const span = high - low || 1;
	let previous = -1;
	for (let x = 0; x < dotCols; x += 1) {
		const index = Math.round((x / (dotCols - 1)) * (values.length - 1));
		const value = values[index] ?? 0;
		const y = clamp(Math.round((1 - (value - low) / span) * (dotRows - 1)), 0, dotRows - 1);
		const from = previous < 0 ? y : Math.min(previous, y);
		const to = previous < 0 ? y : Math.max(previous, y);
		for (let dot = from; dot <= to; dot += 1) setDot(canvas, x, dot);
		previous = y;
	}
}

/** 同一格同时有净值与基准时净值优先（基准退居背景色） */
function colorize(kind: string, text: string, palette: Palette): string {
	if (kind === "equity") return palette.success(text);
	if (kind === "benchmark") return palette.dim(text);
	return text;
}

function cellKind(equityBits: number, benchmarkBits: number): string {
	if (equityBits) return "equity";
	if (benchmarkBits) return "benchmark";
	return "blank";
}

function mergeRow(equity: Canvas, benchmark: Canvas, row: number, palette: Palette): string {
	let out = "";
	let run = "";
	let kind = "";
	const flush = (): void => {
		if (run.length === 0) return;
		out += colorize(kind, run, palette);
		run = "";
	};
	for (let col = 0; col < equity.cols; col += 1) {
		const index = row * equity.cols + col;
		const equityBits = equity.bits[index] ?? 0;
		const benchmarkBits = benchmark.bits[index] ?? 0;
		const next = cellKind(equityBits, benchmarkBits);
		if (next !== kind) {
			flush();
			kind = next;
		}
		const bits = equityBits || benchmarkBits;
		run += bits === 0 ? BLANK_CELL : String.fromCodePoint(BRAILLE_BLANK + bits);
	}
	flush();
	return out;
}

function padLeft(text: string, width: number): string {
	const current = visibleWidth(text);
	return current >= width ? text : " ".repeat(width - current) + text;
}

function padRight(text: string, width: number): string {
	const current = visibleWidth(text);
	return current >= width ? text : text + " ".repeat(width - current);
}

/** 三个日期标签：左、中、右对齐；列数不够时只留首尾 */
function axisLabels(series: EquitySeries, cols: number): string {
	const last = series.stamps.length - 1;
	const first = series.stamps[0] ?? "";
	const end = series.stamps[last] ?? "";
	const middle = series.stamps[Math.floor(last / 2)] ?? "";
	const labels =
		middle && visibleWidth(`${first}${middle}${end}`) + 8 <= cols ? [first, middle, end] : [first, end];
	const chars = new Array<string>(cols).fill(" ");
	const put = (text: string, start: number): void => {
		for (let index = 0; index < text.length; index += 1) {
			if (start + index < cols) chars[start + index] = text[index] ?? " ";
		}
	};
	put(first, 0);
	put(end, Math.max(0, cols - visibleWidth(end)));
	if (labels.length === 3) put(middle, Math.max(0, Math.floor((cols - visibleWidth(middle)) / 2)));
	return chars.join("");
}

/** 回撤带：一行方块高度表示回撤深度，最深值标在末尾 */
function drawdownRow(series: EquitySeries, cols: number, gutter: number, worst: number, palette: Palette): string {
	const span = Math.abs(worst) || 1;
	let band = "";
	for (let col = 0; col < cols; col += 1) {
		const index = Math.round((col / Math.max(1, cols - 1)) * (series.drawdown.length - 1));
		const level = clamp(Math.round((Math.abs(series.drawdown[index] ?? 0) / span) * (BLOCKS.length - 1)), 0, BLOCKS.length - 1);
		band += BLOCKS[level] ?? " ";
	}
	return `${palette.dim(padRight("回撤", gutter))}${palette.error(band)}  ${palette.dim(`最深 ${pct(worst, 1)}`)}`;
}

/** 纵轴刻度落在顶、中、底三行 */
function labelAt(labels: string[], row: number): string | undefined {
	if (row === 0) return labels[0];
	if (row === Math.floor(CHART_ROWS / 2)) return labels[1];
	if (row === CHART_ROWS - 1) return labels[2];
	return undefined;
}

function curveRows(equity: Canvas, benchmark: Canvas, labels: string[], gutter: number, palette: Palette): string[] {
	const rows: string[] = [];
	for (let row = 0; row < CHART_ROWS; row += 1) {
		const label = labelAt(labels, row);
		const axis = label ? `${padLeft(label, gutter - 1)}┤` : `${" ".repeat(gutter - 1)}│`;
		rows.push(palette.dim(axis) + mergeRow(equity, benchmark, row, palette));
	}
	return rows;
}

/** 把净值与基准画成 braille 曲线，附回撤带。宽度不足时返回空数组 */
export function equityChart(series: EquitySeries, width: number, palette: Palette): string[] {
	const range = extent([...series.equity, ...series.benchmark]);
	const labels = [range.high, (range.high + range.low) / 2, range.low].map((value) => `×${value.toFixed(2)}`);
	const gutter = Math.max(...labels.map((label) => visibleWidth(label))) + 2;
	const cols = width - gutter;
	if (cols < MIN_CHART_WIDTH) return [];

	const equity: Canvas = { cols, bits: new Uint8Array(cols * CHART_ROWS) };
	const benchmark: Canvas = { cols, bits: new Uint8Array(cols * CHART_ROWS) };
	plotSeries(equity, series.equity, range.low, range.high);
	plotSeries(benchmark, series.benchmark, range.low, range.high);

	const lastEquity = series.equity.at(-1) ?? 1;
	const lastBench = series.benchmark.at(-1) ?? 1;
	const worst = extent(series.drawdown).low;
	const head =
		`${palette.dim("净值")} ${palette.success(`×${lastEquity.toFixed(2)}`)}   ` +
		`${palette.dim("基准")} ${palette.muted(`×${lastBench.toFixed(2)}`)}   ` +
		`${palette.dim("最深回撤")} ${palette.error(pct(worst, 1))}`;

	return [
		truncateToWidth(head, width),
		...curveRows(equity, benchmark, labels, gutter, palette),
		palette.dim(`${" ".repeat(gutter - 1)}└${"─".repeat(cols)}`),
		`${" ".repeat(gutter)}${palette.dim(axisLabels(series, cols))}`,
		drawdownRow(series, cols, gutter, worst, palette),
	];
}

/** 某次运行的字符曲线：没有净值文件或终端太窄时返回空数组 */
export function recordChart(record: RunRecord, width: number, palette: Palette): string[] {
	if (!record.equity) return [];
	const series = loadEquitySeries(record.equity);
	if (!series) return [];
	return equityChart(series, width, palette);
}

/** 工具结果卡片用：文本块末尾按渲染宽度追加字符曲线 */
export class ChartBlock implements Component {
	private lines: string[];
	private record: RunRecord;
	private palette: Palette;

	constructor(lines: string[], record: RunRecord, palette: Palette) {
		this.lines = lines;
		this.record = record;
		this.palette = palette;
	}

	render(width: number): string[] {
		const lines = this.lines.map((line) => truncateToWidth(line, width));
		const chart = recordChart(this.record, width, this.palette);
		if (chart.length > 0) lines.push("", ...chart);
		return lines;
	}

	invalidate(): void {}
}
