
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # prevents plot windows from opening on the server
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import re
import os

# =============================================================
# LOAD AND PREPARE DATA
# =============================================================

def load_data():
    std_tables   = pd.read_html('pl_player_stats.html')
    wages_tables = pd.read_html('pl_wages.html')

    df_std   = std_tables[11]
    df_wages = wages_tables[10]

    # flatten columns
    df_std.columns = [col[1] if col[1] != '' else col[0]
                      for col in df_std.columns]
    df_std = df_std.loc[:, ~df_std.columns.duplicated()]

    # clean standard stats
    df_std = df_std[df_std['Player'] != 'Player'].reset_index(drop=True)
    df_std = df_std[df_std['Pos'].str.contains('FW', na=False)]
    df_std['90s'] = pd.to_numeric(df_std['90s'], errors='coerce')
    df_std['G+A'] = pd.to_numeric(df_std['G+A'], errors='coerce')
    df_std = df_std[df_std['90s'] >= 5]
    df_std['GA_per_90'] = df_std['G+A'] / df_std['90s']
    df_std = df_std[['Player', 'GA_per_90']].dropna().reset_index(drop=True)

    # clean wages
    df_wages = df_wages[df_wages['Player'] != 'Player'].reset_index(drop=True)
    df_wages = df_wages[df_wages['Pos'].str.contains('FW', na=False)]

    def extract_pounds(wage_str):
        try:
            match = re.search(r'£\s*([\d,]+)', str(wage_str))
            if match:
                return float(match.group(1).replace(',', ''))
            return np.nan
        except:
            return np.nan

    df_wages['Weekly_Wage'] = df_wages['Weekly Wages'].apply(extract_pounds)
    df_wages = df_wages[['Player', 'Weekly_Wage']].dropna()

    # merge
    df = pd.merge(df_std, df_wages, on='Player').dropna().reset_index(drop=True)

    return df


def prepare_features(df):
    mid_goals = df['GA_per_90'].median()
    mid_wage  = df['Weekly_Wage'].median()
    df['Value'] = (df['GA_per_90'] > mid_goals).astype(int)

    X_raw    = df[['GA_per_90', 'Weekly_Wage']].values.astype(float)
    T_perc   = df['Value'].values.astype(int)
    X_min_p  = X_raw.min(axis=0)
    X_max_p  = X_raw.max(axis=0)
    X_norm_p = (X_raw - X_min_p) / (X_max_p - X_min_p + 1e-8)

    return X_norm_p, T_perc, X_min_p, X_max_p, mid_goals, mid_wage


# =============================================================
# PERCEPTRON
# =============================================================

def perceptron_predict(X, weights, bias):
    return (X @ weights - bias > 0).astype(int)


def perceptron_train(X, T, epochs=200):
    weights = np.zeros(2)
    bias    = 0.0
    for epoch in range(epochs):
        I = np.arange(len(X))
        np.random.shuffle(I)
        for i in I:
            y_k     = perceptron_predict(X[i].reshape(1, -1), weights, bias)[0]
            error   = T[i] - y_k
            weights = weights + error * X[i]
            bias    = bias    - error
    return weights, bias


# =============================================================
# UTILITIES
# =============================================================

def find_player(name, df):
    return df[df['Player'].str.contains(name, case=False, na=False)]


def get_quadrant(row, mid_goals, mid_wage):
    high_goals = row['GA_per_90'] > mid_goals
    high_wage  = row['Weekly_Wage'] > mid_wage
    if high_wage and not high_goals:
        return 'Overpriced'
    elif high_wage and high_goals:
        return 'Elite'
    elif not high_wage and not high_goals:
        return 'Dead Weight'
    else:
        return 'Hidden Gem'


# =============================================================
# PLOT
# =============================================================

def generate_plot(player1_name, player2_name, df, mid_goals, mid_wage,
                  p_weights, p_bias, X_min_p, X_max_p, T_perc):

    player1 = find_player(player1_name, df).iloc[0]
    player2 = find_player(player2_name, df).iloc[0]

    fig, ax = plt.subplots(figsize=(13, 8))

    x_min = df['GA_per_90'].min()
    x_max = df['GA_per_90'].max()
    y_min = df['Weekly_Wage'].min()
    y_max = df['Weekly_Wage'].max()
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05

    # quadrant shading
    ax.fill_betweenx([mid_wage, y_max + pad_y],
                      x_min - pad_x, mid_goals, alpha=0.06, color='red')
    ax.fill_betweenx([mid_wage, y_max + pad_y],
                      mid_goals, x_max + pad_x, alpha=0.06, color='green')
    ax.fill_betweenx([y_min - pad_y, mid_wage],
                      x_min - pad_x, mid_goals, alpha=0.06, color='grey')
    ax.fill_betweenx([y_min - pad_y, mid_wage],
                      mid_goals, x_max + pad_x, alpha=0.06, color='blue')

    ax.axvline(mid_goals, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(mid_wage,  color='grey', linestyle='--', linewidth=0.8, alpha=0.5)

    # all players faded
    colors = ['red' if v == 0 else 'green' for v in T_perc]
    ax.scatter(df['GA_per_90'], df['Weekly_Wage'],
               c=colors, edgecolors='black', s=40, zorder=5, alpha=0.3)


    # highlight players
    highlight_colors = ['gold', 'cyan']
    for player, color in zip([player1, player2], highlight_colors):
        ax.scatter(player['GA_per_90'], player['Weekly_Wage'],
                   color=color, edgecolors='black', s=250, zorder=10, linewidth=2)
        ax.annotate(player['Player'],
                    (player['GA_per_90'], player['Weekly_Wage']),
                    textcoords='offset points',
                    xytext=(10, 6), fontsize=10,
                    fontweight='bold', zorder=11)

    # legend
    legend_elements = [
        Patch(facecolor='red',   alpha=0.06, label='Overpriced'),
        Patch(facecolor='green', alpha=0.06, label='Elite'),
        Patch(facecolor='grey',  alpha=0.06, label='Dead Weight'),
        Patch(facecolor='blue',  alpha=0.06, label='Hidden Gems'),
        plt.scatter([], [], color='gold', edgecolors='black',
                    s=150, label=player1['Player']),
        plt.scatter([], [], color='cyan', edgecolors='black',
                    s=150, label=player2['Player']),
    ]
    ax.legend(handles=legend_elements, fontsize=8,
              loc='upper left', framealpha=0.9)

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'£{int(y/1000)}k'))
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_xlabel('Goals + Assists per 90 Minutes', fontsize=12)
    ax.set_ylabel('Weekly Wage',                    fontsize=12)
    ax.set_title('Premier League Strikers — Player Comparison',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    # save to static folder
    plot_path = os.path.join('static', 'plots', 'comparison.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path