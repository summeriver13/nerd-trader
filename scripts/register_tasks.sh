#!/usr/bin/env bash
# 注册（或更新）五个 Windows 计划任务，幂等：同名任务先删后建。
#
#   nerd-trader-polymarket-hourly  每小时第 7 分：前向采集是唯一入口（历史已被实测封死），
#                                  采样分辨率由频度决定，故按小时；错开整点避开上游高峰。
#   nerd-trader-deribit-daily      每日 16:05 本机时间（UTC+8，即 08:05 UTC）：
#                                  到期结算（08:00 UTC）后 5 分钟取链，微笑不含将到期的噪声。
#   nerd-trader-bars-daily         每日 16:20：刷新 OKX 现货日线（只补尾部）→ 重建 manifest
#                                  → 跑验收闸。日线在 00:00 UTC（08:00 本机）收盘，
#                                  16:20 跑时最后一根已封口 8 小时。
#   nerd-trader-paper-daily        每日 16:45：追加趋势线纸面前向账（依赖 bars 已跑完）。
#   nerd-trader-copy-daily         每日 23:40：抓 OKX 跟单榜单快照 + 验收。榜单版本号
#                                  （dataVer）在本地 23:00 前后刷新，晚 40 分钟取当天定稿快照。
#
# 用法：
#   bash scripts/register_tasks.sh            # 注册/更新
#   bash scripts/register_tasks.sh --remove   # 移除
#   bash scripts/register_tasks.sh --query    # 查看已注册的命令与下次运行时间
#
# 两个 git-bash 专属坑（2026-10-15 实测踩到）：
#   MSYS_NO_PATHCONV=1 必设，否则 `/create` 会被当成路径改写成 `C:/Program Files/Git/create`；
#   schtasks 的输出是 GBK，直接用会在终端显示乱码，故统一经 iconv 解码。
#
# 前提：任务以当前用户身份运行（不填密码），因此要求本机登录态；
# 换机或换用户名后重跑一次本脚本即可（任务里存的是绝对路径）。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$PROJECT_DIR/scripts/scheduled_run.sh"
BASH_WIN='C:\Program Files\Git\usr\bin\bash.exe'
POLYMARKET_TASK='nerd-trader-polymarket-hourly'
DERIBIT_TASK='nerd-trader-deribit-daily'
BARS_TASK='nerd-trader-bars-daily'
PAPER_TASK='nerd-trader-paper-daily'
COPY_TASK='nerd-trader-copy-daily'
POLYMARKET_START='00:07'
DERIBIT_START='16:05'
BARS_START='16:20'
PAPER_START='16:45'
COPY_START='23:40'

command_for() {
  # 任务命令行：bash.exe 路径带空格要带引号，脚本路径无空格按原样给 bash -lc
  printf '"%s" -lc "%s %s"' "$BASH_WIN" "$RUNNER" "$1"
}

run_schtasks() {
  # 保留 schtasks 的退出码（管道会让 $? 变成 iconv 的状态）
  local status
  MSYS_NO_PATHCONV=1 schtasks "$@" 2>&1 | iconv -f GBK -t UTF-8 -c
  status=${PIPESTATUS[0]}
  return "$status"
}

register() {
  run_schtasks /create /f /tn "$POLYMARKET_TASK" /sc HOURLY /mo 1 \
    /st "$POLYMARKET_START" /tr "$(command_for polymarket)"
  run_schtasks /create /f /tn "$DERIBIT_TASK" /sc DAILY /st "$DERIBIT_START" \
    /tr "$(command_for deribit)"
  run_schtasks /create /f /tn "$BARS_TASK" /sc DAILY /st "$BARS_START" \
    /tr "$(command_for bars)"
  run_schtasks /create /f /tn "$PAPER_TASK" /sc DAILY /st "$PAPER_START" \
    /tr "$(command_for paper)"
  run_schtasks /create /f /tn "$COPY_TASK" /sc DAILY /st "$COPY_START" \
    /tr "$(command_for copy)"
  echo
  echo "已注册：$POLYMARKET_TASK（每小时第 $POLYMARKET_START 分）"
  echo "        $DERIBIT_TASK（每日 $DERIBIT_START 本机时间）"
  echo "        $BARS_TASK（每日 $BARS_START：刷新日线 + manifest + 验收）"
  echo "        $PAPER_TASK（每日 $PAPER_START：追加纸面前向账）"
  echo "        $COPY_TASK（每日 $COPY_START：抓跟单榜单快照 + 验收）"
  echo "日志：$PROJECT_DIR/logs/scheduled/<数据集>-<日期>.log"
  echo "核对：bash scripts/register_tasks.sh --query"
}

remove() {
  run_schtasks /delete /f /tn "$POLYMARKET_TASK"
  run_schtasks /delete /f /tn "$DERIBIT_TASK"
  run_schtasks /delete /f /tn "$BARS_TASK"
  run_schtasks /delete /f /tn "$PAPER_TASK"
  run_schtasks /delete /f /tn "$COPY_TASK"
}

query() {
  for name in "$POLYMARKET_TASK" "$DERIBIT_TASK" "$BARS_TASK" "$PAPER_TASK" "$COPY_TASK"; do
    echo "=== $name ==="
    run_schtasks /query /tn "$name" /fo LIST /v |
      grep -E "要运行的任务|Task To Run|下次运行时间|Next Run Time|上次运行时间|Last Run Time|上次结果|Last Result" |
      sed 's/^[[:space:]]*//'
  done
}

case "${1:-}" in
  --remove) remove ;;
  --query) query ;;
  '') register ;;
  *) echo "未知参数：$1（可用 --remove / --query）" >&2; exit 2 ;;
esac
