/**
 * 解析 research.py 的标准输出
 *
 * 只认固定的几行格式（表格按列宽排版，因此一律从右往左切列，避免策略名里的空格被切开）：
 *   - 单次回测：💰/🛡️/🔄 三行摘要 + 图表路径
 *   - 稳健性：每个标的的「策略表 / 分段稳定性 / 多重检验校正」三张表 + PBO
 *   - carry：每个标的的年化四行、滚动统计与逐年表
 * 解析不到的细节不猜，原文仍完整落在 logs/<id>.log 里。
 */

import type { CarryRow, DsrRow, MetricRow, PboInfo, RunMode } from "./spec.js";

export interface ParsedRun {
	symbols: string[];
	window: string;
	rows: MetricRow[];
	dsr: DsrRow[];
	pbo?: PboInfo;
	carry: CarryRow[];
	charts: string[];
	equity?: string;
	notes: string[];
}

export interface Progress {
	phase: string;
	detail: string;
}

/** 表格行从右往左切出固定数量的列，剩下的都是名称 */
function splitColumns(line: string, count: number): { name: string; cells: string[] } | undefined {
	const tokens = line.trim().split(/\s+/);
	if (tokens.length <= count) return undefined;
	return { name: tokens.slice(0, tokens.length - count).join(" "), cells: tokens.slice(-count) };
}

function toNumber(cell: string | undefined): number | undefined {
	if (cell === undefined) return undefined;
	const text = cell.replace(/[,+%]/g, "");
	if (text === "n/a" || text === "—" || text === "-") return undefined;
	const value = Number(text);
	return Number.isFinite(value) ? value : undefined;
}

function toPercent(cell: string | undefined): number | undefined {
	const value = toNumber(cell);
	return value === undefined ? undefined : value / 100;
}

const METRIC_COLUMNS: readonly ((cell: string) => number | undefined)[] = [
	toPercent, // 总收益
	toPercent, // 年化
	toPercent, // 回撤
	toNumber, // 卡尔玛
	toPercent, // 持仓占比
	toNumber, // 窗口数
	toPercent, // 窗口中位
	toPercent, // 窗口P25
	toPercent, // 窗口最差
	toPercent, // 正收益窗占比
	toPercent, // 窗口回撤中位
];

/** 稳健性策略表的一行 */
function metricRow(line: string): MetricRow | undefined {
	const split = splitColumns(line, METRIC_COLUMNS.length);
	if (!split) return undefined;
	const values = split.cells.map((cell, index) => METRIC_COLUMNS[index]?.(cell));
	const [totalRet, annualRet, maxDrawdown, calmar, exposure, windowCount] = values;
	const [, , , , , , windowMedian, , windowWorst, windowPositiveShare, windowDdMedian] = values;
	return {
		name: split.name,
		totalRet,
		annualRet,
		maxDrawdown,
		calmar,
		exposure,
		windowCount,
		windowMedian,
		windowWorst,
		windowPositiveShare,
		windowDdMedian,
	};
}

/** 多重检验校正表的一行：只有以「通过 / 未通过」结尾的才算 */
function dsrRow(line: string): DsrRow | undefined {
	const split = splitColumns(line, 4);
	if (!split) return undefined;
	const verdict = split.cells[3] ?? "";
	if (!verdict.includes("通过")) return undefined;
	const dsr = toNumber(split.cells[2]);
	if (dsr === undefined) return undefined;
	return {
		name: split.name,
		sharpe: toNumber(split.cells[0]) ?? 0,
		expectedMax: toNumber(split.cells[1]) ?? 0,
		dsr,
		passed: !verdict.includes("未通过"),
	};
}

const BACKTEST_WINDOW = /📊 评估区间: (\d{4}-\d{2}-\d{2})[^→]*→ (\d{4}-\d{2}-\d{2})[^（]*（(\d+) 根/;
const BACKTEST_SYMBOL = /📊 结果摘要 \| (\S+)/;
const BACKTEST_MONEY = /💰 总收益: ([-\d.]+)% \| 年化: ([-\d.]+)% \| 基准: ([-\d.]+)%/;
const BACKTEST_RISK = /🛡️ 卡尔玛: ([-\d.]+) \| 夏普: ([-\d.]+) \| 索提诺: ([-\d.]+) \| 最大回撤: ([-\d.]+)%/;
const BACKTEST_TRADE = /🔄 交易: (\d+) 回合 \| 胜率: ([-\d.]+)% \| 盈亏比: ([-\d.]+) \| 平均持仓: ([-\d.]+) 根/;
const CHART_PATH = /📊 策略图表已保存: (.+)$/;
const EQUITY_PATH = /📈 净值序列已保存: (.+)$/;

/** 单次回测：三行摘要 + 图表路径 */
function parseBacktest(lines: string[], parsed: ParsedRun): void {
	for (const line of lines) {
		const symbol = BACKTEST_SYMBOL.exec(line);
		if (symbol?.[1]) parsed.symbols.push(symbol[1]);

		const window = BACKTEST_WINDOW.exec(line);
		if (window) parsed.window = `${window[1]} → ${window[2]}（${window[3]} 根）`;

		const money = BACKTEST_MONEY.exec(line);
		const risk = BACKTEST_RISK.exec(line);
		const trade = BACKTEST_TRADE.exec(line);
		if (money || risk || trade) {
			const row = parsed.rows[0] ?? { name: parsed.symbols[0] ?? "回测" };
			if (money) {
				row.totalRet = toPercent(money[1]);
				row.annualRet = toPercent(money[2]);
				row.benchRet = toPercent(money[3]);
			}
			if (risk) {
				row.calmar = toNumber(risk[1]);
				row.sharpe = toNumber(risk[2]);
				row.sortino = toNumber(risk[3]);
				row.maxDrawdown = toPercent(risk[4]);
			}
			if (trade) {
				row.trades = toNumber(trade[1]);
				row.winRate = toPercent(trade[2]);
				row.profitFactor = toNumber(trade[3]);
			}
			parsed.rows[0] = row;
		}

		const chart = CHART_PATH.exec(line);
		if (chart?.[1]) parsed.charts.push(chart[1].trim());

		const equity = EQUITY_PATH.exec(line);
		if (equity?.[1]) parsed.equity = equity[1].trim();
	}
}

const ROBUST_HEADER = /^(\S+\/\S+) \| 评估区间 (\S+) → (\S+) \| (\d+) 根/;
const PBO_LINE = /^PBO \(CSCV, ([^)]*)\) = ([\d.]+)% \| (.+)$/;
const METRIC_TABLE_HEAD = /^策略\s+总收益/;
const SEGMENT_TABLE_HEAD = /^策略\s+正收益段/;
const DSR_TABLE_HEAD = /^策略\s+夏普/;
const SEPARATOR = /^-{10,}$/;
const SEGMENT_ANCHOR = "\u0000";

type Table = "metric" | "segment" | "dsr" | "none";

/** 稳健性：策略表 + 分段稳定性 + 多重检验校正 + PBO */
function parseRobust(lines: string[], parsed: ParsedRun): void {
	let currentSymbol = "";
	let table: Table = "none";
	const metricRows: { symbol: string; row: MetricRow; original: string }[] = [];
	const dsrRows: { symbol: string; row: DsrRow }[] = [];
	const segments = new Map<string, { positive?: string; beaten?: string }>();

	for (const line of lines) {
		const text = line.trim();

		const header = ROBUST_HEADER.exec(text);
		if (header) {
			currentSymbol = header[1] ?? "";
			parsed.symbols.push(currentSymbol);
			const span = `${header[2]} → ${header[3]}（${header[4]} 根）`;
			parsed.window = parsed.window ? `${parsed.window}；${currentSymbol} ${span}` : `${currentSymbol} ${span}`;
			continue;
		}

		// PBO 行紧跟 DSR 表，必须先于表格分发处理，否则会被当成表格行
		const pbo = PBO_LINE.exec(text);
		if (pbo) {
			parsed.pbo = { value: toPercent(pbo[2]) ?? 0, detail: pbo[3] ?? "" };
			table = "none";
			continue;
		}

		if (text.startsWith("分段稳定性（")) {
			table = "segment";
			continue;
		}
		if (text.startsWith("多重检验校正（")) {
			table = "dsr";
			continue;
		}
		if (text.startsWith("分段收益明细")) {
			table = "none";
			continue;
		}
		if (METRIC_TABLE_HEAD.test(text)) {
			table = "metric";
			continue;
		}
		if (SEGMENT_TABLE_HEAD.test(text) || DSR_TABLE_HEAD.test(text)) continue;
		if (SEPARATOR.test(text) || text === "") continue;

		if (table === "metric") {
			const row = metricRow(line);
			if (row) metricRows.push({ symbol: currentSymbol, row, original: row.name });
			continue;
		}
		if (table === "segment") {
			const split = splitColumns(line, 4);
			if (split) {
				segments.set(`${currentSymbol}${SEGMENT_ANCHOR}${split.name}`, {
					positive: split.cells[0],
					beaten: split.cells[1],
				});
			}
			continue;
		}
		if (table === "dsr") {
			const row = dsrRow(line);
			if (row) dsrRows.push({ symbol: currentSymbol, row });
		}
	}

	// 多标的时给行名加标的前缀，避免不同标的的同名策略混在一起
	const multi = parsed.symbols.length > 1;
	const label = (symbol: string, name: string) => (multi ? `${symbol} · ${name}` : name);
	for (const item of metricRows) {
		const segment = segments.get(`${item.symbol}${SEGMENT_ANCHOR}${item.original}`);
		if (segment) {
			item.row.segmentsPositive = segment.positive;
			item.row.segmentsBeaten = segment.beaten;
		}
		item.row.name = label(item.symbol, item.original);
		parsed.rows.push(item.row);
	}
	for (const item of dsrRows) {
		item.row.name = label(item.symbol, item.row.name);
		parsed.dsr.push(item.row);
	}
}

const CARRY_HEADER =
	/^(\S+\/\S+) \| (\S+) → (\S+) \| ([\d,]+) 次结算 \| 正费率占比 ([-\d.]+)% \| 单次中位 ([-\d.]+)%/;
const CARRY_GROSS = /^毛年化[^:]*:\s*([-\d.]+)%/;
const CARRY_ADJUSTED = /^按资本占用[^:]*:\s*([-\d.]+)%/;
const CARRY_NET = /^再扣每月再平衡[^:]*:\s*([-\d.]+)%/;
const CARRY_ROLLING = /^(\d+) 天滚动年化: 中位 ([-\d.]+)% \| P5 ([-\d.]+)% \| 最差 ([-\d.]+)%/;
const CARRY_YEARS_HEAD = /^年份\s+结算次数/;

/** carry：年化四行 + 滚动统计 + 逐年表 */
function parseCarry(lines: string[], parsed: ParsedRun): void {
	let current: CarryRow | undefined;
	let inYears = false;

	for (const line of lines) {
		const text = line.trim();
		const header = CARRY_HEADER.exec(text);
		if (header) {
			current = {
				name: header[1] ?? "",
				span: `${header[2]} → ${header[3]}`,
				settlements: Number((header[4] ?? "0").replace(/,/g, "")),
				positiveShare: toPercent(header[5]),
				medianRate: toPercent(header[6]),
				years: [],
			};
			parsed.symbols.push(current.name);
			parsed.carry.push(current);
			inYears = false;
			continue;
		}
		if (!current) continue;

		if (CARRY_YEARS_HEAD.test(text)) {
			inYears = true;
			continue;
		}
		const gross = CARRY_GROSS.exec(text);
		if (gross) current.grossAnnual = toPercent(gross[1]);
		const adjusted = CARRY_ADJUSTED.exec(text);
		if (adjusted) current.capitalAdjusted = toPercent(adjusted[1]);
		const net = CARRY_NET.exec(text);
		if (net) current.netAnnual = toPercent(net[1]);
		const rolling = CARRY_ROLLING.exec(text);
		if (rolling) {
			current.rollingMedian = toPercent(rolling[2]);
			current.rollingWorst = toPercent(rolling[4]);
		}
		if (inYears) {
			const split = splitColumns(line, 4);
			const year = Number(split?.name);
			if (split && Number.isFinite(year)) {
				current.years.push({
					year,
					settlements: toNumber(split.cells[0]),
					annual: toPercent(split.cells[1]),
					positiveShare: toPercent(split.cells[2]),
					worstRolling: toPercent(split.cells[3]),
				});
			}
		}
	}
}

/** 汇总警告与失败信息（❌ 开头的行） */
function parseNotes(lines: string[]): { notes: string[]; failed: boolean } {
	const notes: string[] = [];
	let failed = false;
	for (const line of lines) {
		const text = line.trim();
		if (text.startsWith("⚠️")) notes.push(text);
		if (text.startsWith("❌")) {
			notes.push(text);
			failed = true;
		}
	}
	return { notes, failed };
}

export function parseOutput(mode: RunMode, text: string): ParsedRun & { failed: boolean } {
	const lines = text.split(/\r?\n/);
	const parsed: ParsedRun = {
		symbols: [],
		window: "",
		rows: [],
		dsr: [],
		carry: [],
		charts: [],
		notes: [],
	};
	if (mode === "backtest") parseBacktest(lines, parsed);
	else if (mode === "robust") parseRobust(lines, parsed);
	else parseCarry(lines, parsed);

	const { notes, failed } = parseNotes(lines);
	parsed.notes.push(...notes);
	return { ...parsed, failed };
}

/** 从已输出的行里推出当前阶段，用于工具行的实时进度 */
type ProgressRule = (text: string, previous: Progress) => Progress | undefined;

const PROGRESS_RULES: readonly ProgressRule[] = [
	(text) => {
		const match = /^⬇️ \[CCXT\] 获取 (\S+)/.exec(text);
		return match?.[1] ? { phase: "拉取行情", detail: match[1] } : undefined;
	},
	(text) => {
		const match = /^⬇️ \[Binance\] 获取 (\S+)/.exec(text);
		return match?.[1] ? { phase: "拉取资金费率", detail: match[1] } : undefined;
	},
	(text) => {
		const match = /^📊 数据准备完毕: 总行数 (\d+)/.exec(text);
		return match?.[1] ? { phase: "数据就绪", detail: `${match[1]} 根` } : undefined;
	},
	(text) => (text.startsWith("⚖️ 波动率目标") ? { phase: "应用波动率目标", detail: "" } : undefined),
	(text) => {
		const match = /^(\S+\/\S+) \| 评估区间 (\S+) → (\S+) \| (\d+) 根/.exec(text);
		return match?.[1] ? { phase: `回测 ${match[1]}`, detail: `${match[4]} 根` } : undefined;
	},
	(text, previous) =>
		text.startsWith("分段稳定性（") ? { phase: "分段稳定性", detail: previous.detail } : undefined,
	(text, previous) =>
		text.startsWith("多重检验校正（") ? { phase: "多重检验校正", detail: previous.detail } : undefined,
	(text, previous) => (CARRY_YEARS_HEAD.test(text) ? { phase: "逐年统计", detail: previous.detail } : undefined),
	(text) => (text.startsWith("💰 总收益") ? { phase: "汇总结果", detail: "" } : undefined),
];

export function progressOf(lines: string[], initial?: Progress): Progress {
	let progress: Progress = initial ?? { phase: "启动", detail: "" };
	for (const line of lines) {
		const text = line.trim();
		for (const rule of PROGRESS_RULES) {
			const next = rule(text, progress);
			if (next) {
				progress = next;
				break;
			}
		}
	}
	return progress;
}
