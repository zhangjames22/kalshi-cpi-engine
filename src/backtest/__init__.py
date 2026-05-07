"""Backtest engine — runs Strategies against historical orderbook + settlement
data and produces a BacktestReport.

Public surface:

    from backtest import (
        BacktestEngine, BacktestConfig,
        ExecutionModel, MidFill, CrossSpreadFill,
        BacktestReport,
    )
"""

from .engine import BacktestConfig, BacktestEngine
from .execution import CrossSpreadFill, ExecutionModel, MidFill
from .report import BacktestReport

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestReport",
    "CrossSpreadFill",
    "ExecutionModel",
    "MidFill",
]
