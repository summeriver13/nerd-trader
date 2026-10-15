#!/usr/bin/env bash
# 计划任务入口：cd 项目 → 跑对应数据集的采集 → 日志落 logs/scheduled/<数据集>-<日期>.log
#
#   bash scripts/scheduled_run.sh polymarket
#   bash scripts/scheduled_run.sh deribit
#   bash scripts/scheduled_run.sh bars      # 刷新 OKX 现货日线 + 重建 manifest + 验收
#   bash scripts/scheduled_run.sh paper     # 追加趋势线纸面前向账（不下单）
#
# 频度与参数在 scripts/register_tasks.sh 里（那里是采集参数的唯一出处，改参数改那里）；
# 这里只管「怎么跑」：日志按天切、退出码记进日志尾（计划任务自己的历史记录只留 0x0/0x1）。
#
# PYTHONIOENCODING 必须设：计划任务不继承交互终端的环境，不设会让中文输出
# 按本机默认代码页落进日志变成乱码（2026-10-15 实测踩到）。
set -uo pipefail
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

DATASET="${1:?用法: bash scripts/scheduled_run.sh polymarket|deribit|bars|paper}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs/scheduled
LOG="logs/scheduled/${DATASET}-$(date +%F).log"

{
  echo "=== $(date -Is) 开始 $DATASET ==="
  case "$DATASET" in
    # 用默认参数：成交从最新页往前翻、整页全重复即停，所以稳定态每小时只翻 1~2 页；
    # 不该设 --trade-pages 上限——真有大成交量时要的正是多翻几页，限页会截断新市场的历史。
    # 价格窗口保持默认 7 天重叠，防单点漏天。
    polymarket) uv run python scripts/fetch_polymarket.py ;;
    deribit) uv run python scripts/fetch_deribit.py ;;
    # 只补尾部：池子取已有缓存里的标的集合（不重新按成交额选池，否则池子每天抖动）。
    # 顺序不能换：先刷新再重建 manifest（manifest 的行数/覆盖区间取自磁盘），
    # 最后跑验收闸——验收非 0 就是这一天的数据有问题，日志会留下退出码。
    bars)
      uv run python scripts/fetch_crypto_bars.py --refresh okx/spot/1d --timeframe 1d --days 30 &&
        uv run python scripts/build_crypto_manifests.py &&
        uv run python scripts/audit_crypto_bars.py --market spot --timeframe 1d
      ;;
    # 放到 bars 之后跑：账本要的是当天最新一根 K 线，滞后会被--no-write 闸拦下。
    paper) uv run python scripts/tsmom_paper.py ;;
    *) echo "未知数据集：$DATASET"; exit 2 ;;
  esac
  STATUS=$?
  echo "=== $(date -Is) 结束（退出码 $STATUS）==="
  exit "$STATUS"
} >>"$LOG" 2>&1
