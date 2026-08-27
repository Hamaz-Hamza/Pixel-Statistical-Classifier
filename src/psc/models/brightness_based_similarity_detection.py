# <<< info >>>
#
# This model represents each class using a spatial brightness map rather than
# binary foreground frequency counts.
#
# During learning, each grayscale pixel is normalized to the [0, 1] range and
# the average brightness across all samples for individual classes is obtained. Brighter
# pixels therefore contribute more strongly to the learned class representation,
# while darker pixels contribute less.
#
# After the brightness values have been averaged, each class slice is
# independently normalized. this removes differences in overall magnitude
# between class slices while preserving their spatial brightness distribution.
#
# During prediction, the test image is also normalized to the [0, 1] range and
# the dot product between the normalized test image and each
# normalized class slice is then used as the similarity score.
#
# the predicted class is the class whose learned brightness map has the highest
# dot-product similarity with the test image.

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

        # normalize each class brightness map independently.
        for class_index in range(n_classes):
            class_slice = self.statistics[:, :, class_index]
            self.statistics[:, :, class_index] = class_slice / np.max(class_slice)

    def predict_single(self, sample):

        # normalize grayscale brightness to the [0, 1] range
        normalized_sample = sample.astype(np.float64) / 255.0

        # calculate the dot product between the normalized test image
        # and every normalized class brightness map
        selected = self.statistics * normalized_sample[:, :, None]

        similarities = selected.sum(axis=(0, 1))

        # return the original class label
        return self.classes[np.argmax(similarities)]

    def predict(self, x_test):
        return [self.predict_single(sample) for sample in x_test]

    def evaluate(self, sample):
        import matplotlib.pyplot as plt

        # normalize sample brightness to [0, 1]
        normalized_sample = sample.astype(np.float64) / 255.0

        # calculate the dot-product score for every class
        selected = self.statistics * normalized_sample[:, :, None]

        scores = selected.sum(axis=(0, 1))

        # visualize original brightness image
        plt.figure(figsize=(2, 2))
        plt.imshow(sample, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        plt.show()

        # print class scores
        print("Class       ", np.asarray([f"{digit}" for digit in self.classes]))
        print("Dot products", np.round(scores, 0))
        print("\nPrediction:", self.classes[np.argmax(scores)])