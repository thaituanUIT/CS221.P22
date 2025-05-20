import collections
import math
import random
import numpy as np
import tensorflow as tf
from load_data import *

class SkipGram:
    def __init__(self,
                 *,
                train_path = './data/data.txt',
                model_path = './model/skipgram_wordvec.bin',
                min_count = 5,
                batch_size = 200,
                embedding_size = 200,
                window_size = 5,
                num_sampled = 100,
                num_steps = 100000):
        self.data_index = 0
        self.train_path = train_path
        self.model_path = model_path
        self.min_count = min_count
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.num_sampled = num_sampled
        self.num_steps = num_steps
        self.dataset = DataLoader().dataset
        self.words = self.read_data(self.dataset)

    def read_data(self, dataset):
        words = []
        for data in dataset:
            words.extend(data)
        return words

    def build_dataset(self, 
                    words, 
                    min_count):

        count = [['UNK', -1]]
        count.extend([item for item in collections.Counter(words).most_common() if item[1] >= min_count])
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count

        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reverse_dictionary

    def generate_batch(self, 
                    batch_size, 
                    window_size, 
                    data):

        batch = np.ndarray(shape = (batch_size), dtype = np.int32)
        labels = np.ndarray(shape = (batch_size, 1), dtype = np.int32)
        span = 2 * window_size + 1 
        buffer = collections.deque(maxlen = span) 
        for _ in range(span):
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)

        for i in range(batch_size // (window_size*2)):
            target = window_size
            targets_to_avoid = [window_size]
            for j in range(window_size*2):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * window_size*2 + j] = buffer[window_size]
                labels[i * window_size*2 + j, 0] = buffer[target]
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1)%len(data)

        return batch, labels

    def train_wordvec(self, 
                    vocabulary_size, 
                    batch_size, 
                    embedding_size, 
                    window_size, 
                    num_sampled, 
                    num_steps, 
                    data):

        graph = tf.Graph()
        with graph.as_default():

            train_inputs = tf.placeholder(tf.int32, shape = [batch_size])

            train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])

            with tf.device('/cpu:0'):

                embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev = 1.0 / math.sqrt(embedding_size)))

                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                                biases = nce_biases,
                                                labels = train_labels,
                                                inputs = embed,
                                                num_sampled = num_sampled,
                                                num_classes = vocabulary_size))

            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))
            normalized_embeddings = embeddings / norm

            init = tf.global_variables_initializer()

        with tf.Session(graph = graph) as session:

            init.run()

            average_loss = 0

            for step in range(num_steps):
                batch_inputs, batch_labels = self.generate_batch(batch_size, window_size, data)

                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                _, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)

                average_loss += loss_val
                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    print("Average loss at step ", step, ":", average_loss)
                    average_loss = 0
            final_embeddings = normalized_embeddings.eval()

        return final_embeddings

    def save_embedding(self, 
                    final_embeddings, 
                    reverse_dictionary):
        
        f = open(self.modelpath, 'w+')
        for index, item in enumerate(final_embeddings):
            f.write(reverse_dictionary[index] + '\t' + ','.join([str(vec) for vec in item]) + '\n')
        f.close()

    def train(self):
        data, count, dictionary, reverse_dictionary = self.build_dataset(self.words, self.min_count)
        vocabulary_size = len(count)
        final_embeddings = self.train_wordvec(vocabulary_size, self.batch_size, self.embedding_size, self.window_size, self.num_sampled, self.num_steps, data)
        self.save_embedding(final_embeddings, reverse_dictionary)