import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from matplotlib.patches import Patch

# =============================================================
# LOAD DATA
# =============================================================
std_tables   = pd.read_html('pl_player_stats.html')
wages_tables = pd.read_html('pl_wages.html')

df_std   = std_tables[11]
df_wages = wages_tables[10]

# flatten multi-level columns on standard stats
df_std.columns = [col[1] if col[1] != '' else col[0]
                  for col in df_std.columns]
df_std = df_std.loc[:, ~df_std.columns.duplicated()]

# =============================================================
# CLEAN STANDARD STATS
# =============================================================
df_std = df_std[df_std['Player'] != 'Player'].reset_index(drop=True)
df_std = df_std[df_std['Pos'].str.contains('FW', na=False)]
df_std['90s'] = pd.to_numeric(df_std['90s'], errors='coerce')
df_std['G+A'] = pd.to_numeric(df_std['G+A'], errors='coerce')
df_std = df_std[df_std['90s'] >= 5]
df_std['GA_per_90'] = df_std['G+A'] / df_std['90s']
df_std = df_std[['Player', 'GA_per_90']].dropna().reset_index(drop=True)

# =============================================================
# CLEAN WAGES
# =============================================================
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

# =============================================================
# MERGE
# =============================================================
df = pd.merge(df_std, df_wages, on='Player').dropna().reset_index(drop=True)

print(f"Strikers: {len(df)}")
print(df.sort_values('Weekly_Wage', ascending=False).head(10))

# =============================================================
# DEFINE VALUE LABELS
# =============================================================
mid_goals = df['GA_per_90'].median()
mid_wage  = df['Weekly_Wage'].median()

df['Value'] = (df['GA_per_90'] > mid_goals).astype(int)

print(f"\nMedian weekly wage:        £{int(mid_wage):,}")
print(f"Median goals+assists/90:   {mid_goals:.3f}")
print(f"\n0 = Bad value  ({(df['Value']==0).sum()} players)")
print(f"1 = Good value ({(df['Value']==1).sum()} players)")

# =============================================================
# PREPARE INPUTS
# =============================================================
X_raw   = df[['GA_per_90', 'Weekly_Wage']].values.astype(float)
T_perc  = df['Value'].values.astype(int)

X_min_p  = X_raw.min(axis=0)
X_max_p  = X_raw.max(axis=0)
X_norm_p = (X_raw - X_min_p) / (X_max_p - X_min_p + 1e-8)

# =============================================================
# TRAIN/TEST SPLIT
# =============================================================
np.random.seed(42)
n         = len(df)
idx       = np.random.permutation(n)
train_idx = idx[:int(0.7 * n)]
test_idx  = idx[int(0.7 * n):]

X_train, T_train = X_norm_p[train_idx], T_perc[train_idx]
X_test,  T_test  = X_norm_p[test_idx],  T_perc[test_idx]

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

p_weights, p_bias = perceptron_train(X_train, T_train)

p_train_acc = np.mean(perceptron_predict(X_train, p_weights, p_bias) == T_train) * 100
p_test_acc  = np.mean(perceptron_predict(X_test,  p_weights, p_bias) == T_test)  * 100

print(f"\nPerceptron — Train: {p_train_acc:.1f}%  Test: {p_test_acc:.1f}%")


# =============================================================
# PLOT — DECISION BOUNDARIES
# =============================================================
fig, ax = plt.subplots(figsize=(13, 8))

x_min_p = df['GA_per_90'].min()
x_max_p = df['GA_per_90'].max()
y_min_p = df['Weekly_Wage'].min()
y_max_p = df['Weekly_Wage'].max()
pad_x   = (x_max_p - x_min_p) * 0.05
pad_y   = (y_max_p - y_min_p) * 0.05

# --- quadrant shading ---
ax.fill_betweenx([mid_wage, y_max_p + pad_y],
                  x_min_p - pad_x, mid_goals,
                  alpha=0.06, color='red')
ax.fill_betweenx([mid_wage, y_max_p + pad_y],
                  mid_goals, x_max_p + pad_x,
                  alpha=0.06, color='green')
ax.fill_betweenx([y_min_p - pad_y, mid_wage],
                  x_min_p - pad_x, mid_goals,
                  alpha=0.06, color='grey')
ax.fill_betweenx([y_min_p - pad_y, mid_wage],
                  mid_goals, x_max_p + pad_x,
                  alpha=0.06, color='blue')

ax.axvline(mid_goals, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axhline(mid_wage,  color='grey', linestyle='--', linewidth=0.8, alpha=0.5)

# --- scatter coloured by value label ---
colors = ['red' if v == 0 else 'green' for v in T_perc]
ax.scatter(df['GA_per_90'], df['Weekly_Wage'],
           c=colors, edgecolors='black', s=70, zorder=5)

# --- player labels ---
for _, row in df.iterrows():
    ax.annotate(row['Player'],
                (row['GA_per_90'], row['Weekly_Wage']),
                textcoords='offset points',
                xytext=(6, 4), fontsize=7, alpha=0.85)

# --- perceptron boundary ---
x1_vals      = np.linspace(0, 1, 300)
x2_vals_norm = (p_bias - p_weights[0] * x1_vals) / (p_weights[1] + 1e-8)
x1_orig      = x1_vals      * (X_max_p[0] - X_min_p[0]) + X_min_p[0]
x2_orig      = x2_vals_norm * (X_max_p[1] - X_min_p[1]) + X_min_p[1]
mask         = (x2_orig >= y_min_p - pad_y) & (x2_orig <= y_max_p + pad_y)
ax.plot(x1_orig[mask], x2_orig[mask],
        color='blue', linewidth=2, zorder=6, label='Perceptron boundary')

# --- legend ---
legend_elements = [
    Patch(facecolor='green', alpha=0.6,  label='Good value (above avg G+A/90)'),
    Patch(facecolor='red',   alpha=0.6,  label='Bad value  (below avg G+A/90)'),
    Patch(facecolor='red',   alpha=0.06, label='Overpriced'),
    Patch(facecolor='green', alpha=0.06, label='Elite'),
    Patch(facecolor='grey',  alpha=0.06, label='Dead Weight'),
    Patch(facecolor='blue',  alpha=0.06, label='Hidden Gems'),
    plt.Line2D([0], [0], color='blue',   linewidth=2,
               label=f'Perceptron boundary (test acc: {p_test_acc:.1f}%)'),
]
ax.legend(handles=legend_elements, fontsize=8,
          loc='upper left', framealpha=0.9)

# --- formatting ---
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'£{int(y/1000)}k'))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax.set_xlim(x_min_p - pad_x, x_max_p + pad_x)
ax.set_ylim(y_min_p - pad_y, y_max_p + pad_y)
ax.set_xlabel('Goals + Assists per 90 Minutes', fontsize=12)
ax.set_ylabel('Weekly Wage',                    fontsize=12)
ax.set_title('Premier League Strikers — Perceptron vs Neural Network Decision Boundary',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('striker_value_comparison.png', dpi=150)
plt.show()

# =============================================================
# PLAYER SEARCH
# =============================================================

def find_player(name, df):
    """case insensitive partial match"""
    matches = df[df['Player'].str.contains(name, case=False, na=False)]
    return matches

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

def plot_comparison(player1, player2, df, mid_goals, mid_wage,
                    p_weights, p_bias, X_min_p, X_max_p, T_perc):

    fig, ax = plt.subplots(figsize=(13, 8))

    x_min_p = df['GA_per_90'].min()
    x_max_p = df['GA_per_90'].max()
    y_min_p = df['Weekly_Wage'].min()
    y_max_p = df['Weekly_Wage'].max()
    pad_x   = (x_max_p - x_min_p) * 0.05
    pad_y   = (y_max_p - y_min_p) * 0.05

    # --- quadrant shading ---
    ax.fill_betweenx([mid_wage, y_max_p + pad_y],
                      x_min_p - pad_x, mid_goals,
                      alpha=0.06, color='red')
    ax.fill_betweenx([mid_wage, y_max_p + pad_y],
                      mid_goals, x_max_p + pad_x,
                      alpha=0.06, color='green')
    ax.fill_betweenx([y_min_p - pad_y, mid_wage],
                      x_min_p - pad_x, mid_goals,
                      alpha=0.06, color='grey')
    ax.fill_betweenx([y_min_p - pad_y, mid_wage],
                      mid_goals, x_max_p + pad_x,
                      alpha=0.06, color='blue')

    ax.axvline(mid_goals, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(mid_wage,  color='grey', linestyle='--', linewidth=0.8, alpha=0.5)

    # --- all players as faded dots ---
    colors = ['red' if v == 0 else 'green' for v in T_perc]
    ax.scatter(df['GA_per_90'], df['Weekly_Wage'],
               c=colors, edgecolors='black',
               s=40, zorder=5, alpha=0.3)

    # --- perceptron boundary ---
    x1_vals      = np.linspace(0, 1, 300)
    x2_vals_norm = (p_bias - p_weights[0] * x1_vals) / (p_weights[1] + 1e-8)
    x1_orig      = x1_vals      * (X_max_p[0] - X_min_p[0]) + X_min_p[0]
    x2_orig      = x2_vals_norm * (X_max_p[1] - X_min_p[1]) + X_min_p[1]
    mask         = (x2_orig >= y_min_p - pad_y) & (x2_orig <= y_max_p + pad_y)
    ax.plot(x1_orig[mask], x2_orig[mask],
            color='blue', linewidth=1.5, label='Perceptron boundary')

    # --- highlight players ---
    highlight_colors = ['gold', 'cyan']
    for player, color in zip([player1, player2], highlight_colors):
        ax.scatter(player['GA_per_90'], player['Weekly_Wage'],
                   color=color, edgecolors='black',
                   s=250, zorder=10, linewidth=2)
        ax.annotate(player['Player'],
                    (player['GA_per_90'], player['Weekly_Wage']),
                    textcoords='offset points',
                    xytext=(10, 6),
                    fontsize=10,
                    fontweight='bold',
                    zorder=11)

    # --- legend ---
    legend_elements = [
        Patch(facecolor='red',   alpha=0.06, label='Overpriced'),
        Patch(facecolor='green', alpha=0.06, label='Elite'),
        Patch(facecolor='grey',  alpha=0.06, label='Dead Weight'),
        Patch(facecolor='blue',  alpha=0.06, label='Hidden Gems'),
        plt.Line2D([0], [0], color='blue', linewidth=1.5,
                   label='Perceptron boundary'),
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
    ax.set_xlim(x_min_p - pad_x, x_max_p + pad_x)
    ax.set_ylim(y_min_p - pad_y, y_max_p + pad_y)
    ax.set_xlabel('Goals + Assists per 90 Minutes', fontsize=12)
    ax.set_ylabel('Weekly Wage',                    fontsize=12)
    ax.set_title('Premier League Strikers — Player Comparison',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('player_comparison.png', dpi=150)
    plt.show()


def search_and_compare(df, mid_goals, mid_wage,
                       p_weights, p_bias, X_min_p, X_max_p, T_perc):

    print("\n" + "=" * 50)
    print("PLAYER COMPARISON TOOL")
    print("=" * 50)

    # --- search player 1 ---
    while True:
        name1   = input("\nEnter first player name: ")
        results = find_player(name1, df)
        if len(results) == 0:
            print(f"No player found matching '{name1}'. Try again.")
        elif len(results) > 1:
            print("Multiple matches found:")
            print(results['Player'].tolist())
            print("Please be more specific.")
        else:
            player1 = results.iloc[0]
            print(f"Found: {player1['Player']}")
            break

    # --- search player 2 ---
    while True:
        name2   = input("Enter second player name: ")
        results = find_player(name2, df)
        if len(results) == 0:
            print(f"No player found matching '{name2}'. Try again.")
        elif len(results) > 1:
            print("Multiple matches found:")
            print(results['Player'].tolist())
            print("Please be more specific.")
        else:
            player2 = results.iloc[0]
            print(f"Found: {player2['Player']}")
            break

    # --- print comparison table ---
    print(f"\n{'':20} {player1['Player']:>20}   {player2['Player']:>20}")
    print("-" * 65)
    print(f"{'Weekly Wage':20} "
          f"£{int(player1['Weekly_Wage']):>18,}   "
          f"£{int(player2['Weekly_Wage']):>18,}")
    print(f"{'G+A per 90':20} "
          f"{player1['GA_per_90']:>20.3f}   "
          f"{player2['GA_per_90']:>20.3f}")
    print(f"{'Quadrant':20} "
          f"{get_quadrant(player1, mid_goals, mid_wage):>20}   "
          f"{get_quadrant(player2, mid_goals, mid_wage):>20}")

    # --- plot ---
    plot_comparison(player1, player2, df, mid_goals, mid_wage,
                    p_weights, p_bias, X_min_p, X_max_p, T_perc)

    # --- search again ---
    again = input("\nCompare another pair? (y/n): ")
    if again.lower() == 'y':
        search_and_compare(df, mid_goals, mid_wage,
                           p_weights, p_bias, X_min_p, X_max_p, T_perc)


# --- run ---
search_and_compare(df, mid_goals, mid_wage,
                   p_weights, p_bias, X_min_p, X_max_p, T_perc)