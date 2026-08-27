import numpy as np

class PixelStatisticalClassifier:
    def fit(self, x_train, y_train):

        # get the number of rows and columns
        n, rows, cols = x_train.shape

        # normalize grayscale brightness to the [0, 1] range
        normalized_images = x_train.astype(np.float64) / 255.0

        # find the number of classes and obtain a mapping for class index -> class
        self.classes = np.unique(y_train)
        n_classes = len(self.classes)
        class_indices = np.searchsorted(self.classes, y_train)

        # brightness map of training data.
        # each 2d slice represents the accumulated pixel brightness
        # for one output class.
        self.statistics = np.zeros(
            (rows, cols, n_classes),
            dtype=np.float64
        )

        # build brightness maps
        for class_index in range(n_classes):
            class_images = normalized_images[class_indices == class_index]
            self.statistics[:, :, class_index] = class_images.mean(axis=0)

        for class_index in range(n_classes):
            class_map = self.statistics[:, :, class_index]
            l2_norm = np.linalg.norm(class_map)
            
            # Prevent division by zero if a class map is empty/all zero
            if l2_norm > 0:
                self.statistics[:, :, class_index] = class_map / l2_norm

    def predict_single(self, sample):

        normalized_sample = sample.astype(np.float64) / 255.0

        sample_l2_norm = np.linalg.norm(normalized_sample)
        if sample_l2_norm > 0:
            normalized_sample = normalized_sample / sample_l2_norm

        # Cosine similarity via dot product (since both vectors are unit length)
        similarities = np.tensordot(normalized_sample, self.statistics, axes=([0, 1], [0, 1]))

        return self.classes[np.argmax(similarities)]

    def predict(self, x_test):
        return [self.predict_single(sample) for sample in x_test]

    def evaluate(self, sample):
        import matplotlib.pyplot as plt

        # normalize sample brightness to [0, 1]
        normalized_sample = sample.astype(np.float64) / 255.0

        sample_l2_norm = np.linalg.norm(normalized_sample)
        if sample_l2_norm > 0:
            normalized_sample = normalized_sample / sample_l2_norm

        # calculate the dot-product score for every class
        selected = self.statistics * normalized_sample[:, :, None]

        scores = selected.sum(axis=(0, 1))

        # visualize original brightness image
        plt.figure(figsize=(2, 2))
        plt.imshow(sample, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        plt.show()

        # print class scores
        print("Class       ", np.asarray([f" {digit}" for digit in self.classes]))
        print("Dot products", np.round(scores, 2))
        print("\nPrediction:", self.classes[np.argmax(scores)])