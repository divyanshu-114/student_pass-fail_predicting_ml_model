"""UI components module for creating interactive visualizations with Streamlit."""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional, Any

class UIBuilder:
    """Helper class to build styled UI components and charts in Streamlit.
    
    Attributes:
        palette (List[str]): List of hex color codes defining the UI theme.
    """
    def __init__(self) -> None:
        """Initializes the UI builder with predefined colors."""
        self.palette = ["#6366f1", "#a78bfa", "#10b981", "#f59e0b", "#ef4444", "#06b6d4"]
        plt.style.use("dark_background")

    def load_css(self) -> None:
        """Injects custom global CSS styles into the Streamlit app."""
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * { font-family: 'Inter', sans-serif !important; }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        h1, h2, h3 { letter-spacing: -0.02em; font-weight: 700; }
        hr { opacity: 0.10; }
        
        .metric-card {
            background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(168,85,247,0.08));
            border: 1px solid rgba(99,102,241,0.25);
            border-radius: 14px;
            padding: 18px 20px;
            text-align: center;
            margin-bottom: 8px;
        }
        .metric-card .val { font-size: 2rem; font-weight: 700; color: #a78bfa; }
        .metric-card .lbl { font-size: 0.82rem; opacity: 0.65; margin-top: 4px;
                            letter-spacing: 0.04em; text-transform: uppercase; }
        
        .result-pass {
            background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.08));
            border: 1.5px solid rgba(16,185,129,0.4);
            border-radius: 16px; padding: 22px 26px; margin-bottom: 14px;
        }
        .result-fail {
            background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
            border: 1.5px solid rgba(239,68,68,0.4);
            border-radius: 16px; padding: 22px 26px; margin-bottom: 14px;
        }
        .result-pass .headline { color: #10b981; font-size: 1.8rem; font-weight: 700; }
        .result-fail .headline { color: #ef4444; font-size: 1.8rem; font-weight: 700; }
        .prob-bar-wrap { background: rgba(255,255,255,0.07); border-radius: 99px; height:10px; margin:8px 0 4px; }
        .prob-bar-inner { height:10px; border-radius:99px; }
        
        .info-card {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 16px 18px;
            background: rgba(255,255,255,0.02);
            margin-bottom: 10px;
        }
        .small-note { font-size: 0.85rem; opacity: 0.6; }
        
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.55rem 2rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em !important;
        }
        </style>
        """, unsafe_allow_html=True)

    def render_prediction_card(self, proba: float, pred: int, cluster: int) -> None:
        """Renders the prediction result block.
        
        Args:
            proba (float): Pass probability (0 to 1).
            pred (int): Final prediction class (1 for pass, 0 for fail).
            cluster (int): Associated student cluster.
        """
        pct       = proba * 100
        bar_color = "#10b981" if pred == 1 else "#ef4444"
        label     = "PASS ✅" if pred == 1 else "FAIL ❌"
        card_cls  = "result-pass" if pred == 1 else "result-fail"
        st.markdown(f"""
        <div class="{card_cls}">
            <div class="headline">{label}</div>
            <div style="margin-top:10px; font-size:0.95rem; opacity:0.75;">Pass Probability</div>
            <div style="font-size:1.5rem; font-weight:700; color:{bar_color};">{pct:.1f}%</div>
            <div class="prob-bar-wrap">
                <div class="prob-bar-inner" style="width:{pct:.1f}%; background:{bar_color};"></div>
            </div>
            <div class="small-note" style="margin-top:6px;">Performance cluster: {cluster}</div>
        </div>
        """, unsafe_allow_html=True)

    def render_metric_card(self, value: Any, label: str) -> str:
        """Returns the HTML literal for a styled metric card.
        
        Args:
            value (Any): The primary value to show.
            label (str): Subtitle label.
            
        Returns:
            str: Evaluated HTML div.
        """
        return f"""<div class="metric-card"><div class="val">{value}</div><div class="lbl">{label}</div></div>"""

    def mpl_hist(self, ax: plt.Axes, data: pd.Series, title: str, xlabel: str, color: str = "#6366f1") -> None:
        """Plots a histogram using matplotlib into the provided axes.
        
        Args:
            ax (plt.Axes): Matplotlib axes.
            data (pd.Series): Sequence to plot.
            title (str): Chart title.
            xlabel (str): Label for the x-axis.
            color (str): Hex color to use.
        """
        ax.hist(data.dropna(), bins=25, color=color, alpha=0.85, edgecolor="none", rwidth=0.9)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel(xlabel, fontsize=8, alpha=0.7)
        ax.set_ylabel("Count", fontsize=8, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def mpl_bar(self, ax: plt.Axes, labels: List[str], values: List[float], title: str, colors: Optional[List[str]] = None) -> None:
        """Plots a bar chart using matplotlib into the provided axes.
        
        Args:
            ax (plt.Axes): Matplotlib axes.
            labels (List[str]): Sequence of string labels.
            values (List[float]): Magnitudes for each bar.
            title (str): Chart title.
            colors (Optional[List[str]]): List of parallel colors to use.
        """
        cols = colors or self.palette[:len(labels)]
        bars = ax.bar(labels, values, color=cols, alpha=0.9, edgecolor="none", width=0.55)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{int(val):,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def mpl_pie(self, ax: plt.Axes, labels: List[str], values: List[float], title: str, colors: Optional[List[str]] = None) -> None:
        """Plots a pie chart using matplotlib into the provided axes.
        
        Args:
            ax (plt.Axes): Matplotlib axes.
            labels (List[str]): Slices labels.
            values (List[float]): Quantities corresponding to labels.
            title (str): Chart title.
            colors (Optional[List[str]]): Overriding color array.
        """
        cols = colors or self.palette[:len(labels)]
        _, _, autotexts = ax.pie(
            values, labels=labels, autopct="%1.1f%%", startangle=90,
            colors=cols, pctdistance=0.82,
            wedgeprops={"linewidth": 2, "edgecolor": "#1a1a2e"},
            textprops={"fontsize": 10}
        )
        for at in autotexts:
            at.set_fontweight("bold")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

    def mpl_box(self, ax: plt.Axes, df: pd.DataFrame, col: str, title: str) -> None:
        """Plots a multiple boxplot segmented by clusters.
        
        Args:
            ax (plt.Axes): Matplotlib axes.
            df (pd.DataFrame): Input dataframe.
            col (str): Numeric column plotted on Y axis.
            title (str): Chart title.
        """
        groups, labels = [], []
        for c in sorted(df["cluster"].unique()):
            groups.append(pd.to_numeric(df[df["cluster"] == c][col], errors="coerce").dropna().values)
            labels.append(f"C{c}")
        bp = ax.boxplot(groups, labels=labels, patch_artist=True,
                        medianprops={"color": "white", "linewidth": 2})
        for patch, color in zip(bp["boxes"], self.palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
