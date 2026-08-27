# <<< info >>>
#
# This model is a improved version of the dual-state frequency aggregation model.
# Tt normalizes sample distributions across all pixels having the same state
# to prevent either background or foreground pixels to dominate predictions due to sheer numbers

import numpy as np

class PixelStatisticalClassifier:
    def __init__(self, foreground_pixel_weight):
        self.foreground_pixel_weight = foreground_pixel_weight
    
    def fit(self, x_train, y_train):

        # get the number of rows and columns
        n, rows, cols = x_train.shape

        # make a copy of original training data to avoid modifying it
        # apply binarization
        binarized_images = (x_train > 128).astype(np.uint8)

        # find the number of classes and obtain a mapping for class index -> class
        self.classes = np.unique(y_train)
        n_classes = len(self.classes)
        class_indices = np.searchsorted(self.classes, y_train)

        # frequency map of training data i.e. 2d array representing the training data shape, 
        # but with a third channel that represents the sum of output class counts,
        # and a 4th channel distributing those counts between the two binary states (background and foreground)
        self.statistics = np.zeros((rows, cols, n_classes, 2), dtype=np.uint32)

        # build frequency map
        for class_index in range(n_classes):
            class_images = binarized_images[class_indices == class_index]

            # foreground pixels frequency map
            self.statistics[:, :, class_index, 1] = class_images.sum(axis=0)

            # background pixels frequency map
            self.statistics[:, :, class_index, 0] = (1-class_images).sum(axis=0)
        
    def predict_single(self, sample):

        # binarization
        binarized_sample = (sample > 128).astype(np.uint8)

        # select probabilities corresponding to the observed background/foreground states at each pixel.
        selected = np.take_along_axis(
            self.statistics,
            binarized_sample[:, :, None, None],
            axis=3
        ).squeeze(axis=3)

        # separate background and foreground pixels.
        background_statistics = selected[binarized_sample == 0]
        foreground_statistics = selected[binarized_sample == 1]

        # average the class statistics independently for each state
        background_average = background_statistics.mean(axis=0)
        foreground_average = foreground_statistics.mean(axis=0)

        # combine statistics from both states using the tunable weight
        # to get the final statistics
        final_statistics = (
            (1 - self.foreground_pixel_weight) * background_average
            + 
            self.foreground_pixel_weight * foreground_average
        )

        # return the original class label.
        return self.classes[np.argmax(final_statistics)]


    def predict(self, x_test):
        return [self.predict_single(sample) for sample in x_test]

    def evaluate(self, sample):
        import matplotlib.pyplot as plt

        # binarization
        binarized_sample = (sample > 128).astype(np.uint8)

        # select statistics corresponding to the observed
        # background/foreground state at each pixel
        selected = np.take_along_axis(
            self.statistics,
            binarized_sample[:, :, None, None],
            axis=3
        ).squeeze(axis=3)

        # separate background and foreground pixels
        background_statistics = selected[binarized_sample == 0]
        foreground_statistics = selected[binarized_sample == 1]

        # average statistics
        background_average = background_statistics.mean(axis=0)
        foreground_average = foreground_statistics.mean(axis=0)

        # combine using the same weighting as predict_single()
        final_statistics = (
            (1 - self.foreground_pixel_weight) * background_average
            +
            self.foreground_pixel_weight * foreground_average
        )

        # visualize image
        plt.figure(figsize=(2, 2))
        plt.imshow(sample, cmap="gray")
        plt.axis("off")
        plt.show()

        print("Class                ", np.asarray([f"  {digit}" for digit in range(10)]))
        print("Background statistics", np.round(background_average, 0))
        print("Foreground statistics", np.round(foreground_average, 0))
        print("\nFinal:",self.classes[np.argmax(final_statistics)])
