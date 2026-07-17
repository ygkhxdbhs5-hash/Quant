#!/usr/bin/env python3
"""Single entry point for the Q_Alpha v5 standalone backtest.

Requires local datasets under data/ (populate via downloader.update_data).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from engine.strategy import StandaloneEngine, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Q_Alpha v5 standalone backtest")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--equity-out", default="cache/equity_curve.csv")
    parser.add_argument("--chart-out", default="cache/equity_curve_v5.png")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    engine = StandaloneEngine(config=config, config_path=args.config)
    result = engine.run()

    if result.empty:
        print("No equity curve produced (check data coverage / warmup).")
        return 1

    result["Peak"] = result["Total_Equity"].cummax()
    result["Drawdown"] = (result["Total_Equity"] - result["Peak"]) / result["Peak"]
    final_return = (result["Total_Equity"].iloc[-1] / result["Total_Equity"].iloc[0] - 1) * 100
    mdd = result["Drawdown"].min() * 100

    print("\n" + "=" * 60)
    print("   STANDALONE (Pure Python) INSTITUTIONAL ENGINE (v5)   ")
    print("=" * 60)
    print(f"▶ 누적 순수익률   : {final_return:.2f}%")
    print(f"▶ 최대 낙폭       : {mdd:.2f}%")
    print(f"▶ 최종 자산가치   : {result['Total_Equity'].iloc[-1]:,.0f}")
    print("=" * 60)
    print("주의: Value(FCF/Sales Yield) 팩터는 발행주식수 데이터 부재로 이번 버전에 미포함.")
    print("      ROIC은 21% 고정세율 가정 하의 근사치. 상세 한계는 코드 주석 참고.")

    equity_out = Path(args.equity_out)
    equity_out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(equity_out)
    print(f"\nWrote equity curve -> {equity_out}")

    chart_out = Path(args.chart_out)
    chart_out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 6))
    plt.plot(result["Total_Equity"], color="darkgreen", lw=2.5, label="Standalone PIT Strategy (v5)")
    plt.title("Pure Python Standalone Engine - Equity Curve", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.savefig(chart_out, dpi=150)
    plt.close()
    print(f"차트 저장: {chart_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
