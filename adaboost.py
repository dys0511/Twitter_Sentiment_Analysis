import numpy as np
from scipy.sparse import load_npz
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def run_adaboost():
    print("Loading preprocessed data...")
    X_train = load_npz("data/X_train.npz")
    X_test = load_npz("data/X_test.npz")
    y_train = np.load("data/y_train.npy")
    y_test = np.load("data/y_test.npy")

    print("Training AdaBoost Classifier...")
    model = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
    model.fit(X_train, y_train)
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print("\n" + "="*40)
    print(" ADABOOST CLASSIFIER RESULTS ")
    print("="*0)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_adaboost()
