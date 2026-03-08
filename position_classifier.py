import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================
# LOAD DATA
# =============================================================
tables = pd.read_html('pl_player_stats.html')
df     = tables[11]

# flatten multi-level columns and deduplicate
df.columns = [col[1] if col[1] != '' else col[0] for col in df.columns]

# drop duplicate columns — keep only the first occurrence
df = df.loc[:, ~df.columns.duplicated()]

# =============================================================
# CLEAN AND PREPARE DATA
# =============================================================

# remove repeated header rows
df = df[df['Player'] != 'Player'].reset_index(drop=True)

# take only the first position for players with multiple
df['Pos'] = df['Pos'].str.split(',').str[0]

# keep only the 4 main positions
df = df[df['Pos'].isin(['GK', 'DF', 'MF', 'FW'])].reset_index(drop=True)

# select input features
features = ['Gls', 'Ast', 'MP', 'Starts', 'Min', 'CrdY', 'CrdR', 'PK', 'PKatt', '90s']

# convert features to numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# drop rows with missing values
df = df.dropna(subset=features).reset_index(drop=True)

# encode positions as integers then one-hot
position_map = {'GK': 0, 'DF': 1, 'MF': 2, 'FW': 3}
df['Target']  = df['Pos'].map(position_map)
labels        = df['Target'].values.astype(int)

def one_hot_encode(labels, n_classes):
    T = np.zeros((len(labels), n_classes))
    T[np.arange(len(labels)), labels] = 1
    return T

T      = one_hot_encode(labels, n_classes=4)
X      = df[features].values.astype(float)
X_min  = X.min(axis=0)
X_max  = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min + 1e-8)

# =============================================================
# SUMMARY
# =============================================================
print("Players:  ", X_norm.shape[0])
print("Features: ", X_norm.shape[1])
print("Classes:  ", T.shape[1])
print("\nPosition breakdown:")
print(df['Pos'].value_counts())

# =============================================================
# NEURAL NETWORK
# =============================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def initialise_network(n_input, n_hidden, n_output):
    np.random.seed(42)
    W1 = np.random.randn(n_input,  n_hidden) * 0.01
    b1 = np.zeros(n_hidden)
    W2 = np.random.randn(n_hidden, n_output) * 0.01
    b2 = np.zeros(n_output)
    return W1, b1, W2, b2

def feedforward(x, W1, b1, W2, b2):
    z1 = W1.T @ x + b1
    h  = relu(z1)              # ReLU on hidden layer
    z2 = W2.T @ h + b2
    o  = sigmoid(z2)           # sigmoid on output layer
    return z1, h, o

def compute_loss(X, T, W1, b1, W2, b2):
    total = 0
    for i in range(len(X)):
        _, _, o = feedforward(X[i], W1, b1, W2, b2)
        total  += 0.5 * np.sum((o - T[i])**2)
    return total

def train(X, T, W1, b1, W2, b2, eta=0.01, epochs=300):
    loss_history = []

    for epoch in range(epochs):
        I = np.arange(len(X))
        np.random.shuffle(I)

        for i in I:
            x_k = X[i]
            t_k = T[i]

            # feedforward
            z1, h, o = feedforward(x_k, W1, b1, W2, b2)

            # output delta — sigmoid
            dn = (o - t_k) * o * (1 - o)

            # hidden delta — ReLU
            dm = (W2 @ dn) * relu_derivative(z1)

            # update weights
            W2 = W2 - eta * np.outer(h,   dn)
            b2 = b2 - eta * dn
            W1 = W1 - eta * np.outer(x_k, dm)
            b1 = b1 - eta * dm

        epoch_loss = compute_loss(X, T, W1, b1, W2, b2)
        loss_history.append(epoch_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.4f}")

    return W1, b1, W2, b2, loss_history

# =============================================================
# TRAIN
# =============================================================
W1, b1, W2, b2 = initialise_network(n_input=10, n_hidden=20, n_output=4)

print("Training...\n")
W1, b1, W2, b2, loss_history = train(
    X_norm, T, W1, b1, W2, b2, eta=0.01, epochs=300
)

# =============================================================
# ACCURACY
# =============================================================
correct = 0
for i in range(len(X_norm)):
    _, _, o = feedforward(X_norm[i], W1, b1, W2, b2)
    if np.argmax(o) == np.argmax(T[i]):
        correct += 1

position_names = ['GK', 'DF', 'MF', 'FW']
correct_per_class  = np.zeros(4)
total_per_class    = np.zeros(4)
predicted_per_class = np.zeros(4)

for i in range(len(X_norm)):
    _, _, o   = feedforward(X_norm[i], W1, b1, W2, b2)
    predicted = np.argmax(o)
    true      = np.argmax(T[i])
    total_per_class[true] += 1
    predicted_per_class[predicted] += 1
    if predicted == true:
        correct_per_class[true] += 1

print("\nPer Position Breakdown:")
print(f"{'Position':<12} {'Total':<8} {'Correct':<10} {'Accuracy'}")
print("-" * 42)
for i in range(4):
    acc = 100 * correct_per_class[i] / total_per_class[i]
    print(f"{position_names[i]:<12} {int(total_per_class[i]):<8} "
          f"{int(correct_per_class[i]):<10} {acc:.1f}%")

print("\nPrediction counts:")
for i in range(4):
    print(f"{position_names[i]}: {int(predicted_per_class[i])} predictions")