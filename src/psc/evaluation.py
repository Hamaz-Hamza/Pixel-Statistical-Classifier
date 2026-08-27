import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
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

def display_normalized_confusion_matrix(model, x_test, y_test):
    y_pred = model.predict(x_test)

    classes = np.unique(np.concatenate((y_test, y_pred)))

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=classes
    )

    # Normalize each row
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return cm_normalized

def visualize_class_distributions(distribution, classes=None, title=""):
    distribution = np.asarray(distribution)

    rows, cols, n_classes = distribution.shape

    max_value = np.max(distribution)

    normalized = distribution.astype(np.float64) / max_value
    
    fig, axes = plt.subplots(
        1,
        n_classes,
        figsize=(3 * n_classes, 3.5),
        squeeze=False
    )

    axes = axes[0]

    for class_index, class_label in enumerate(classes):

        axes[class_index].imshow(
            normalized[:, :, class_index],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[class_index].set_title(
            f"Class {class_label}"
        )

        axes[class_index].axis("off")

    fig.suptitle(
        f"{title}\nMaximum value: {max_value:g}"
    )

    plt.tight_layout()
    plt.show()

def evaluate_on_dataset(dataset, model):
    from psc.helper_functions.show_image import show_image
    
    x_train = dataset.train_images()
    y_train = dataset.train_labels()
    x_test = dataset.test_images()
    y_test = dataset.test_labels()

    show_image(x_train[0])

    model.fit(x_train, y_train)
    results = evaluate(model, x_test, y_test)
    display_results(results)

    visualize_class_distributions(model.statistics, model.classes, "")