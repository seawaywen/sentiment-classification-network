from collections import Counter

import numpy as np
import sys
import time


def load_resource():
    with open('reviews.txt', 'r') as f:
        _reviews = list(map(lambda x: x[:-1], f.readlines()))

    with open('labels.txt', 'r') as f:
        _labels = list(map(lambda x: x[:-1].upper(), f.readlines()))

    return _reviews, _labels


def print_review_and_label(i):
    print(labels_data[i] + "\t:\t" + reviews_data[i][:80] + "...")


reviews_data, labels_data = load_resource()


positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()

for idx in range(len(reviews_data)):
    if labels_data[idx] == 'POSITIVE':
        for _word in reviews_data[idx].split(' '):
            positive_counts[_word] += 1
            total_counts[_word] += 1
    else:
        for _word in reviews_data[idx].split(' '):
            negative_counts[_word] += 1
            total_counts[_word] += 1


# positive_counts.most_common() will count the common words like 'the'
# in both positive and negative reviews, we want the words found in
# positive reviews more often than in  negative reviews, and vice versa

# calculate the ratios of word usage between positive and negative reviews
# positive_counts[word] / float(negative_counts[word] + 1)
# ratio > 1, the positive word:
#   the more skewed a word is torward positive, the farther from 1
# ratio < 1, the negative word:
#   the more skewed a word is torward negative, the closer to 0
# ratio ~ 1, the neutral word:
pos_neg_ratios = Counter()
for term, cnt in list(total_counts.most_common()):
    if cnt > 100:
        pos_neg_ratio = positive_counts[term] / float(negative_counts[term] + 1)
        pos_neg_ratios[term] = pos_neg_ratio

# center all values around natural so the absolute value from neutral of ratio
# for a word would indicate how much sentiment the word conveys
# convert ratios to logs
for _word, _ratio in pos_neg_ratios.most_common():
    pos_neg_ratios[_word] = np.log(_ratio)

# top 30 words most frequently seen in a review with 'NEGATIVE' label
top_30_negative_words = pos_neg_ratios.most_common()[:-31:-1]


class SentimentNetwork:

    def __init__(self, reviews, labels, hidden_nodes=10, learning_rate=0.1):
        """Create the SentimentNetwork with the given parameters
        Args:
            reviews(list) - list of reviews that used for training
            labels(list) - list of POSITIVE/NEGATIVE labels associated with the
                given reviews
            hidden_nodes(int) - number of nodes to create in the hidden layer
            learning_rate(float) - learning rate for training
        """
        np.random.seed(1)

        # define the parameters
        # --------------------------------------------------
        self.review_vocab = self.label_vocab = None
        self.review_vocab_size = self.label_vocab_size = 0
        self.word2index = self.label2index = {}

        self.input_nodes = self.hidden_nodes = self.output_nodes = None
        self.learning_rate = 0

        self.weights_0_1 = self.weights_1_2 = None
        self.layer_0 = None
        # --------------------------------------------------

        self.pre_process_data(reviews, labels)
        self.init_network(
            len(self.review_vocab), hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):
        review_vocab = set()
        for review in reviews:
            for word in review.split(' '):
                review_vocab.add(word)

        self.review_vocab = list(review_vocab)

        label_vocab = set()
        for label in labels:
            label_vocab.add(label)

        self.label_vocab = list(label_vocab)

        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        # create a dictionary of words in the vocabulary mapped to
        # index positions
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i

        # create a dictionary of labels mapped to index positions
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

    def init_network(
            self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate

        # initialize weights
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))

        self.weights_1_2 = np.random.normal(
            0.0, self.output_nodes**-0.5,
            (self.hidden_nodes, self.output_nodes))

        self.layer_0 = np.zeros((1, input_nodes))

    def update_input_layer(self, review):
        # reset the layer to be all 0s
        self.layer_0 *= 0

        for word in review.split(' '):
            if word in self.word2index.keys():
                self.layer_0[0][self.word2index[word]] += 1

    @staticmethod
    def get_target_for_label(label):
        """Convert a label to `0` or `1`
        Args:
            label(string): - "POSITIVE" or "NEGATIVE"
        Returns:
            `0` or `1`
        """
        return int(label == 'POSITIVE')

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_output_2_derivative(output):
        return output * (1 - output)

    def train(self, training_reviews, training_labels):
        assert(len(training_reviews) == len(training_labels))

        correct_so_far = 0

        start = time.time()

        for i in range(len(training_reviews)):
            review = training_reviews[i]
            label = training_labels[i]

            # FORWARD PASSING
            # --------------------------------------------------
            # input layer
            self.update_input_layer(review)

            # hidden layer
            layer_1 = self.layer_0.dot(self.weights_0_1)

            # output layer
            layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))
            # --------------------------------------------------

            # BACKWARD PASSING
            # --------------------------------------------------
            # output error
            layer_2_error = layer_2 - self.get_target_for_label(label)
            layer_2_delta = layer_2_error * \
                self.sigmoid_output_2_derivative(layer_2)

            # back-propagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
            # hidden layer gradients - no non-linearity
            # so it's the same as error
            layer_1_delta = layer_1_error

            # update the weights
            self.weights_1_2 -= layer_1.T.dot(layer_2_delta) * \
                self.learning_rate
            self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * \
                self.learning_rate
            # --------------------------------------------------

            # keep track of correct predictions
            if layer_2 >= 0.5 and label == 'POSITIVE':
                correct_so_far += 1
            elif layer_2 < 0.5 and label == 'NEGATIVE':
                correct_so_far += 1

            # debug info
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write(
                '\rProgress:{}% Speed(reviews/sec):{}  #Correct:{} '
                '#Trained:{}  Training Accuracy:{}%'.format(
                    str(100 * i / float(len(training_reviews)))[:4],
                    str(reviews_per_second)[0:5],
                    str(correct_so_far),
                    str(i+1),
                    str(correct_so_far * 100 / float(i+1))[:4])
            )

            if i % 2500 == 0:
                print('')

    def test(self, testing_reviews, testing_labels):
        """Attempts to predict the labels for the given testing_reviews
        and use the test_labels to calculate the accurary of those
        predictions"""

        correct = 0

        start = time.time()

        for i in range(len(testing_reviews)):
            predict = self.run(testing_reviews[i])
            if predict == testing_labels[i]:
                correct += 1

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write(
                '\rProgress:{}% Speed(reviews/sec):{}  #Correct:{} '
                '#Tested:{}  Training Accuracy:{}%'.format(
                    str(100 * i / float(len(testing_reviews)))[:4],
                    str(reviews_per_second)[0:5],
                    str(correct),
                    str(i+1),
                    str(correct * 100 / float(i+1))[:4])
            )

    def run(self, review):
        """Returns a POSITIVE or NEGATIVE prediction for the given review"""

        # input layer
        self.update_input_layer(review.lower())

        # hidden layer
        layer_1 = self.layer_0.dot(self.weights_0_1)

        # output layer
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))

        return 'POSITIVE' if layer_2[0] >= 0.5 else 'NEGATIVE'


if __name__ == '__main__':
    # print('labels.txt \t : \t reviews.txt\n')
    # pretty_print_review_and_label(0)
    # print(pos_neg_ratios.most_common())
    # print(top_30_negative_words)

    mlp = SentimentNetwork(
        reviews_data[:-1000], labels_data[:-1000], learning_rate=0.001)
    mlp.train(reviews_data[:-1000], labels_data[:-1000])

    mlp.test(reviews_data[-1000:], labels_data[-1000:])