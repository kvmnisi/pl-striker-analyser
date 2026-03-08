from flask import Flask, render_template, request
from striker_analysis import (
    load_data, prepare_features, perceptron_train,
    find_player, get_quadrant, generate_plot
)

app = Flask(__name__)

# =============================================================
# LOAD AND TRAIN ONCE WHEN SERVER STARTS
# =============================================================
print("Loading data and training perceptron...")
df                                          = load_data()
X_norm_p, T_perc, X_min_p, X_max_p, mid_goals, mid_wage = prepare_features(df)
p_weights, p_bias                           = perceptron_train(X_norm_p, T_perc)
print("Ready!")

# =============================================================
# ROUTES
# =============================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compare', methods=['POST'])
def compare():
    name1 = request.form.get('player1', '').strip()
    name2 = request.form.get('player2', '').strip()

    # --- find players ---
    results1 = find_player(name1, df)
    results2 = find_player(name2, df)

    # --- handle errors ---
    errors = {}
    if len(results1) == 0:
        errors['player1'] = f"No player found matching '{name1}'"
    elif len(results1) > 1:
        errors['player1'] = f"Multiple matches: {results1['Player'].tolist()} — be more specific"

    if len(results2) == 0:
        errors['player2'] = f"No player found matching '{name2}'"
    elif len(results2) > 1:
        errors['player2'] = f"Multiple matches: {results2['Player'].tolist()} — be more specific"

    if errors:
        return render_template('index.html', errors=errors,
                               name1=name1, name2=name2)

    # --- get player data ---
    player1 = results1.iloc[0]
    player2 = results2.iloc[0]

    # --- generate plot ---
    generate_plot(name1, name2, df, mid_goals, mid_wage,
                  p_weights, p_bias, X_min_p, X_max_p, T_perc)

    # --- build comparison data for template ---
    comparison = {
        'player1': {
            'name'     : player1['Player'],
            'wage'     : f"£{int(player1['Weekly_Wage']):,}",
            'ga_per_90': f"{player1['GA_per_90']:.3f}",
            'quadrant' : get_quadrant(player1, mid_goals, mid_wage),
        },
        'player2': {
            'name'     : player2['Player'],
            'wage'     : f"£{int(player2['Weekly_Wage']):,}",
            'ga_per_90': f"{player2['GA_per_90']:.3f}",
            'quadrant' : get_quadrant(player2, mid_goals, mid_wage),
        },
    }

    return render_template('index.html',
                           comparison=comparison,
                           name1=name1, name2=name2)


if __name__ == '__main__':
    app.run(debug=True)