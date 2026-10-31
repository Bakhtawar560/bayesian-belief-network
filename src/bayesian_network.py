import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. THE NETWORK: the "Sprinkler" network (classic AI textbook example)
#    Each variable is 1 (True) or 0 (False).
#    STRUCTURE says: variable -> list of its parents (arrows pointing into it)
# ------------------------------------------------------------
STRUCTURE = {
    "Cloudy": [],
    "Sprinkler": ["Cloudy"],
    "Rain": ["Cloudy"],
    "WetGrass": ["Sprinkler", "Rain"],
}
ORDER = list(STRUCTURE)

# The "true" probabilities used ONLY to create the dataset.
# Each number is P(variable = 1 | parents), keyed by the parents' values.
TRUE_CPT = {
    "Cloudy":    {(): 0.5},
    "Sprinkler": {(0,): 0.5, (1,): 0.1},
    "Rain":      {(0,): 0.2, (1,): 0.8},
    "WetGrass":  {(0, 0): 0.0, (0, 1): 0.9, (1, 0): 0.9, (1, 1): 0.99},
}


def generate_data(n, seed=42):
    """Create n random days by sampling the network from top to bottom."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        day = {}
        for var in ORDER:
            parent_values = tuple(day[p] for p in STRUCTURE[var])
            day[var] = int(rng.random() < TRUE_CPT[var][parent_values])
        rows.append(day)
    return pd.DataFrame(rows, columns=ORDER)


# ------------------------------------------------------------
# 2. LEARNING: estimate every probability table (CPT) from the data
#    by counting, with Laplace smoothing (alpha = 1).
# ------------------------------------------------------------
def learn_cpts(df, alpha=1):
    cpts = {}
    for var, parents in STRUCTURE.items():
        table = {}
        for combo in itertools.product([0, 1], repeat=len(parents)):
            mask = np.ones(len(df), dtype=bool)
            for p, v in zip(parents, combo):
                mask &= (df[p].values == v)
            total = int(mask.sum())
            ones = int(df.loc[mask, var].sum())
            table[combo] = (ones + alpha) / (total + 2 * alpha)
        cpts[var] = table
    return cpts


# ------------------------------------------------------------
# 3. INFERENCE: answer questions like P(Rain | WetGrass = 1)
#    by enumeration: add up the joint probability of every
#    possible world that agrees with the evidence.
# ------------------------------------------------------------
def joint_probability(assignment, cpts):
    prob = 1.0
    for var, parents in STRUCTURE.items():
        p_one = cpts[var][tuple(assignment[p] for p in parents)]
        prob *= p_one if assignment[var] == 1 else (1 - p_one)
    return prob


def query(target, evidence, cpts):
    hidden = [v for v in ORDER if v != target and v not in evidence]
    scores = {0: 0.0, 1: 0.0}
    for target_value in (0, 1):
        for combo in itertools.product([0, 1], repeat=len(hidden)):
            world = dict(evidence)
            world[target] = target_value
            world.update(zip(hidden, combo))
            scores[target_value] += joint_probability(world, cpts)
    total = scores[0] + scores[1]
    return {0: scores[0] / total, 1: scores[1] / total}


# ------------------------------------------------------------
# 4. DRAW THE NETWORK
# ------------------------------------------------------------
def draw_network(path):
    pos = {"Cloudy": (0.5, 1.0), "Sprinkler": (0.15, 0.5),
           "Rain": (0.85, 0.5), "WetGrass": (0.5, 0.0)}
    fig, ax = plt.subplots(figsize=(7, 6))
    for child, parents in STRUCTURE.items():
        for parent in parents:
            ax.annotate("", xy=pos[child], xytext=pos[parent],
                        arrowprops=dict(arrowstyle="-|>", lw=2,
                                        shrinkA=28, shrinkB=28, color="black"))
    for name, (x, y) in pos.items():
        ax.text(x, y, name, ha="center", va="center", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.6", fc="lightblue", ec="black"))
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.2, 1.2)
    ax.axis("off")
    ax.set_title("Bayesian Belief Network: Sprinkler example")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    # Step A: create the dataset and save it
    df = generate_data(10000)
    df.to_csv("data/weather_data.csv", index=False)
    print("Dataset created: data/weather_data.csv with", len(df), "rows")
    print(df.head())

    # Step B: split 80% train / 20% test
    train = df.iloc[:8000].reset_index(drop=True)
    test = df.iloc[8000:].reset_index(drop=True)
    print("\nTrain rows:", len(train), "| Test rows:", len(test))

    # Step C: learn the probability tables from the training data
    learned = learn_cpts(train)
    print("\n" + "=" * 60)
    print("LEARNED vs TRUE probabilities  P(variable = 1 | parents)")
    print("=" * 60)
    rows = []
    for var, parents in STRUCTURE.items():
        for combo, p_learned in learned[var].items():
            cond = ", ".join(f"{p}={v}" for p, v in zip(parents, combo)) or "(no parents)"
            rows.append([var, cond, TRUE_CPT[var][combo], round(p_learned, 3)])
    print(pd.DataFrame(rows, columns=["Variable", "Given", "True", "Learned"]).to_string(index=False))

    # Step D: ask questions (inference)
    print("\n" + "=" * 60)
    print("INFERENCE (using the learned network)")
    print("=" * 60)
    questions = [
        ("Rain", {}, "P(Rain)"),
        ("Rain", {"WetGrass": 1}, "P(Rain | WetGrass=1)"),
        ("Rain", {"WetGrass": 1, "Sprinkler": 1}, "P(Rain | WetGrass=1, Sprinkler=1)"),
        ("Sprinkler", {"WetGrass": 1}, "P(Sprinkler | WetGrass=1)"),
        ("WetGrass", {"Cloudy": 1}, "P(WetGrass | Cloudy=1)"),
    ]
    for target, evidence, text in questions:
        learned_p = query(target, evidence, learned)[1]
        true_p = query(target, evidence, TRUE_CPT)[1]
        print(f"{text:38s} learned = {learned_p:.3f}   true = {true_p:.3f}")

    # Step E: test accuracy. Predict Rain from Cloudy, Sprinkler, WetGrass.
    correct = 0
    for _, row in test.iterrows():
        evidence = {"Cloudy": int(row["Cloudy"]),
                    "Sprinkler": int(row["Sprinkler"]),
                    "WetGrass": int(row["WetGrass"])}
        p = query("Rain", evidence, learned)
        prediction = 1 if p[1] >= 0.5 else 0
        correct += int(prediction == row["Rain"])
    accuracy = correct / len(test) * 100
    majority = max(test["Rain"].mean(), 1 - test["Rain"].mean()) * 100
    print("\n" + "=" * 60)
    print("TEST: predict Rain from Cloudy, Sprinkler, WetGrass")
    print("=" * 60)
    print(f"Network accuracy on {len(test)} unseen days: {accuracy:.2f} %")
    print(f"Baseline (always guess the most common answer): {majority:.2f} %")

    # Step F: draw the network
    draw_network("network_graph.png")
    print("\nSaved network_graph.png")