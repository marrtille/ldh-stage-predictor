from src.features import preprocess_data
from src.model import get_models, train_and_evaluate, save_model
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd


def main():
    X_train, X_test, y_train, y_test = preprocess_data()

    # ⬇️ Add synthetic genetic/risk features to match app structure
    def add_extra_features(X):
        X = X.copy()
        X['Age'] = pd.Series([40 + i % 20 for i in range(len(X))])  # Simulated ages
        X['Family_History'] = pd.Series([1 if i % 3 == 0 else 0 for i in range(len(X))])
        X['Lifestyle_Score'] = pd.Series([i % 4 for i in range(len(X))])
        return X

    X_train = add_extra_features(X_train)
    X_test = add_extra_features(X_test)

    models = get_models()
    best_model = None
    best_f1 = 0

    for name, model in models.items():
        print(f"\nTraining: {name}")
        results = train_and_evaluate(model, X_train, X_test, y_train, y_test)
        for metric, score in results.items():
            print(f"{metric}: {score:.4f}")

        if results['f1_score'] > best_f1:
            best_f1 = results['f1_score']
            best_model = model

    if best_model:
        save_model(best_model, "D:/LDH_outputs/best_model.pkl")

        y_pred = best_model.predict(X_test)
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=['Early', 'Mid', 'Late'],
            cmap='Blues',
            normalize='true'
        )
        plt.title("Confusion Matrix - Normalized")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
