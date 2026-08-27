# <<< info >>>
#
# This model is a simple statistical machine learning model that stores information 
# about how many times a certain output class appeared over a certain pixel state (background vs foreground)
# for each pixel in the images in the input training data.
# For predictions, the sum of state-specific sample amounts for each class for each pixel is used
# and the class with the highest total sum is predicted.
# This implementation and every future version 
# assumes that the foreground is represented by bright pixels, whereas the background is represented by black pixels

import numpy as np

class PixelStatisticalClassifier:
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

        # select the class frequencies corresponding to each pixel's
        # background/foreground state.
        selected = np.take_along_axis(
            self.statistics,
            binarized_sample[:, :, None, None],
            axis=3 # last dimension (state)
        ).squeeze(axis=3)

        # accumulate frequencies for each class across the 2d image
        frequencies = selected.sum(axis=(0, 1))

        # return the original class label
        return self.classes[np.argmax(frequencies)]


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
        final_statistics = background_average + foreground_average

        # visualize image
        plt.figure(figsize=(2, 2))
        plt.imshow(sample, cmap="gray")
        plt.axis("off")
        plt.show()

        print("Class                ", np.asarray([f"  {digit}" for digit in range(10)]))
        print("Background statistics", np.round(background_average, 0))
        print("Foreground statistics", np.round(foreground_average, 0))
        print("\nFinal:",self.classes[np.argmax(final_statistics)])
