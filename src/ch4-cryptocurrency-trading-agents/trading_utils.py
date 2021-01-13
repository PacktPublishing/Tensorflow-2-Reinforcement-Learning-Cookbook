#!/usr/bin/env python
# Utilities to visualize agent's trade execution and portfolio performance
# Chapter 4, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib import style
from mplfinance.original_flavor import candlestick_ohlc

style.use("seaborn-whitegrid")


class TradeVisualizer(object):
    """Visualizer for stock trades"""

    def __init__(self, ticker, ticker_data_stream, title=None, skiprows=0):
        self.ticker = ticker
        # Stock/crypto market/exchange data stream. An offline file stream is used.
        # Alternatively, a web
        # API can be used to pull live data.
        self.ohlcv_df = pd.read_csv(
            ticker_data_stream, parse_dates=True, index_col="Date", skiprows=skiprows
        ).sort_values(by="Date")
        if "USD" in self.ticker:  # True for crypto-fiat currency pairs
            # Use volume of the crypto currency for volume plot.
            # A column with header="Volume" is required for default mpf plot.
            # Remove "USD" from self.ticker string and clone the crypto volume column
            self.ohlcv_df["Volume"] = self.ohlcv_df[
                "Volume " + self.ticker[:-3]  # e.g: "Volume BTC"
            ]
        self.account_balances = np.zeros(len(self.ohlcv_df.index))

        fig = plt.figure("TFRL-Cookbook", figsize=[12, 6])
        fig.suptitle(title)
        nrows, ncols = 6, 1
        gs = fig.add_gridspec(nrows, ncols)
        row, col = 0, 0
        rowspan, colspan = 2, 1

        # self.account_balance_ax = plt.subplot2grid((6, 1), (0, 0), rowspan=2, colspan=1)
        self.account_balance_ax = fig.add_subplot(
            gs[row : row + rowspan, col : col + colspan]
        )
        row, col = 2, 0
        rowspan, colspan = 8, 1

        self.price_ax = plt.subplot2grid(
            (nrows, ncols),
            (row, col),
            rowspan=rowspan,
            colspan=colspan,
            sharex=self.account_balance_ax,
        )
        self.price_ax = fig.add_subplot(gs[row : row + rowspan, col : col + colspan])

        plt.show(block=False)
        self.viz_not_initialized = True

    def _render_account_balance(self, current_step, account_balance, horizon):
        self.account_balance_ax.clear()
        date_range = self.ohlcv_df.index[current_step : current_step + len(horizon)]

        self.account_balance_ax.plot_date(
            date_range,
            self.account_balances[horizon],
            "-",
            label="Account Balance ($)",
            lw=1.0,
        )

        self.account_balance_ax.legend()
        legend = self.account_balance_ax.legend(loc=2, ncol=2)
        legend.get_frame().set_alpha(0.4)

        last_date = self.ohlcv_df.index[current_step + len(horizon)].strftime(
            "%Y-%m-%d"
        )
        last_date = matplotlib.dates.datestr2num(last_date)
        last_account_balance = self.account_balances[current_step]

        self.account_balance_ax.annotate(
            "{0:.2f}".format(account_balance),
            (last_date, last_account_balance),
            xytext=(last_date, last_account_balance),
            bbox=dict(boxstyle="round", fc="w", ec="k", lw=1),
            color="black",
        )

        self.account_balance_ax.set_ylim(
            min(self.account_balances[np.nonzero(self.account_balances)]) / 1.25,
            max(self.account_balances) * 1.25,
        )

        plt.setp(self.account_balance_ax.get_xticklabels(), visible=False)

    def render_image_observation(self, current_step, horizon):
        window_start = max(current_step - horizon, 0)
        step_range = range(window_start, current_step + 1)
        date_range = self.ohlcv_df.index[current_step : current_step + len(step_range)]
        stock_df = self.ohlcv_df[self.ohlcv_df.index.isin(date_range)]

        if self.viz_not_initialized:
            self.fig, self.axes = mpf.plot(
                stock_df,
                volume=True,
                type="candle",
                mav=2,
                block=False,
                returnfig=True,
                style="charles",
                tight_layout=True,
            )
            self.viz_not_initialized = False
        else:
            self.axes[0].clear()
            self.axes[2].clear()
            mpf.plot(
                stock_df,
                ax=self.axes[0],
                volume=self.axes[2],
                type="candle",
                mav=2,
                style="charles",
                block=False,
                tight_layout=True,
            )
        self.fig.canvas.set_window_title("TFRL-Cookbook")
        self.fig.canvas.draw()
        fig_data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        fig_data = fig_data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        self.fig.set_size_inches(12, 6, forward=True)
        self.axes[0].set_ylabel("Price ($)")
        self.axes[0].yaxis.set_label_position("left")
        self.axes[2].yaxis.set_label_position("left")  # Volume

        return fig_data

    def _render_ohlc(self, current_step, dates, horizon):
        self.price_ax.clear()

        candlesticks = zip(
            dates,
            self.ohlcv_df["Open"].values[horizon],
            self.ohlcv_df["Close"].values[horizon],
            self.ohlcv_df["High"].values[horizon],
            self.ohlcv_df["Low"].values[horizon],
        )

        candlestick_ohlc(
            self.price_ax,
            candlesticks,
            width=np.timedelta64(1, "D"),
            colorup="g",
            colordown="r",
        )
        self.price_ax.set_ylabel(f"{self.ticker} Price ($)")
        self.price_ax.tick_params(axis="y", pad=30)

        last_date = self.ohlcv_df.index[current_step].strftime("%Y-%m-%d")
        last_date = matplotlib.dates.datestr2num(last_date)
        last_close = self.ohlcv_df["Close"].values[current_step]
        last_high = self.ohlcv_df["High"].values[current_step]

        self.price_ax.annotate(
            "{0:.2f}".format(last_close),
            (last_date, last_close),
            xytext=(last_date, last_high),
            bbox=dict(boxstyle="round", fc="w", ec="k", lw=1),
            color="black",
        )

        plt.setp(self.price_ax.get_xticklabels(), visible=False)

    def _render_trades(self, trades, horizon):
        for trade in trades:
            if trade["step"] in horizon:
                date = self.ohlcv_df.index[trade["step"]].strftime("%Y-%m-%d")
                date = matplotlib.dates.datestr2num(date)
                high = self.ohlcv_df["High"].values[trade["step"]]
                low = self.ohlcv_df["Low"].values[trade["step"]]

                if trade["type"] == "buy":
                    high_low = low
                    color = "g"
                    arrow_style = "<|-"
                else:  # sell
                    high_low = high
                    color = "r"
                    arrow_style = "-|>"

                proceeds = "{0:.2f}".format(trade["proceeds"])

                self.price_ax.annotate(
                    f"{trade['type']} ${proceeds}".upper(),
                    (date, high_low),
                    xytext=(date, high_low),
                    color=color,
                    arrowprops=(
                        dict(
                            color=color,
                            arrowstyle=arrow_style,
                            connectionstyle="angle3",
                        )
                    ),
                )

    def render(self, current_step, account_balance, trades, window_size=100):
        self.account_balances[current_step] = account_balance

        window_start = max(current_step - window_size, 0)
        step_range = range(window_start, current_step + 1)

        dates = self.ohlcv_df.index[step_range]

        self._render_account_balance(current_step, account_balance, step_range)
        self._render_ohlc(current_step, dates, step_range)
        self._render_trades(trades, step_range)
        """
        self.price_ax.set_xticklabels(
            self.ohlcv_df.index[step_range], rotation=45, horizontalalignment="right",
        )
        """

        plt.grid()
        plt.pause(0.001)

    def close(self):
        plt.close()
