import numpy as np
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def run_random_forest():
    print("Loading preprocessed data...")
    X_train = load_npz("data/X_train.npz")
    X_test = load_npz("data/X_test.npz")
    y_train = np.load("data/y_train.npy")
    y_test = np.load("data/y_test.npy")
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print("\n" + "="*40)
    print(" RANDOM FOREST CLASSIFIER RESULTS ")
    print("="*40)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
    print("="*40 + "\n")
if __name__ == "__main__":
    run_random_forest()