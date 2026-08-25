
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate(model, x_test, y_test):

    # obtain predictions from the trained model
    y_pred = model.predict(x_test)

    # obtain the classes used by the model
    classes = model.classes

    results = {}

    for class_label in classes:

        # one-vs-rest representation of this class
        y_test_binary = (y_test == class_label)
        y_pred_binary = (y_pred == class_label)

        results[class_label] = {
            "accuracy": accuracy_score(
                y_test_binary,
                y_pred_binary
            ) * 100,

            "precision": precision_score(
                y_test_binary,
                y_pred_binary,
                zero_division=0
            ) * 100,

            "recall": recall_score(
                y_test_binary,
                y_pred_binary,
                zero_division=0
            ) * 100,

            "f1": f1_score(
                y_test_binary,
                y_pred_binary,
                zero_division=0
            ) * 100,
        }

        results["overall"] = {
            "accuracy": accuracy_score(
                y_test,
                y_pred,
            ) * 100,

            "precision": precision_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0
            ) * 100,

            "recall": recall_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0
            ) * 100,

            "f1": f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0
            ) * 100,
        }

    return results


def display_results(results):

    print(
        f"{'Class':<10}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1 Score':>12}"
    )

    print("-" * 58)

    for class_label, metrics in results.items():
        if (class_label == "overall"): continue
        print(
            f"{str(class_label):<10}"
            f"{metrics['accuracy']:>11.2f}%"
            f"{metrics['precision']:>11.2f}%"
            f"{metrics['recall']:>11.2f}%"
            f"{metrics['f1']:>11.2f}%"
        )

    print(
        f"{"overall":<10}"
        f"{results["overall"]['accuracy']:>11.2f}%"
        f"{results["overall"]['precision']:>11.2f}%"
        f"{results["overall"]['recall']:>11.2f}%"
        f"{results["overall"]['f1']:>11.2f}%"
    )