# <<< info >>>
#
# this model is a improved version of the dual-state frequency aggregation model
# it uses the calculated statistics from the previous model's methodology
# to create a probability map based on class distribution
# instead of using raw frequency totals for the highest class according to the pixel's state, 
# it uses the probability of the highest class 
# by normalizing the corresponding frequency based on total observations for an individual pixel
# this probability is then normalized according to state distribution 
# and weighted according to the state weight hyperparameter
#
# this version counterattacks the negative effects of
# not only state imbalances (background pixels dominating in numbers over foreground or viceversa)
# but also statistical imbalances (more samples recorded for background states than for foreground states or viceversa)

import numpy as np

class PixelStatisticalClassifier:
    def __init__(self, foreground_pixel_weight):
        self.foreground_pixel_weight = foreground_pixel_weight
    
    def fit(self, x_train, y_train):

        # get the number of rows and columns
        n, rows, cols = x_train.shape

        # make a copy of original training data to avoid modifying it
        # +
        # binarization
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

        # calculate total samples for all classes for a particular pixel and its state
        totals = self.statistics.sum(axis=2, keepdims=True)

        # build probability distribution by dividing the sample amounts per class by the total samples
        self.probabilities = np.divide(
            self.statistics,
            totals,
            out=np.zeros_like(self.statistics, dtype=np.float64),
            where=totals != 0
        )
        
    def predict_single(self, sample):

        # binarization
        binarized_sample = (sample > 128).astype(np.uint8)

        # select probabilities corresponding to the observed background/foreground states at each pixel.
        selected = np.take_along_axis(
            self.probabilities,
            binarized_sample[:, :, None, None],
            axis=3
        ).squeeze(axis=3)

        # separate background and foreground pixels.
        background_probabilities = selected[binarized_sample == 0]
        foreground_probabilities = selected[binarized_sample == 1]

        # average the class probabilities independently for each state
        background_average = background_probabilities.mean(axis=0)
        foreground_average = foreground_probabilities.mean(axis=0)

        # combine probabilities from both states using the tunable weight
        # to get the final probabilities
        final_probabilities = (
            (1 - self.foreground_pixel_weight) * background_average
            + 
            self.foreground_pixel_weight * foreground_average
        )

        # return the original class label.
        return self.classes[np.argmax(final_probabilities)]


    def predict(self, x_test):
        return [self.predict_single(sample) for sample in x_test]
