import re 
import numpy as np
import tensorflow as tf
import collections
import math
from load_data import *

class CBOW:
    def __init__(
        self,
        *,
        min_count: int = 5,
        batch_size: int = 200,
        embedding_size: int = 200,
        window_size: int = 1,
        num_steps: int = 100000,
        num_sampled: int = 100,
        train_path: str = './data/data.txt',
        model_path: str = './model/cbow.bin'
    ):
        self.data_index = 0
        self.min_count = min_count
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.num_steps = num_steps
        self.num_sampled = num_sampled
        self.train_path = train_path
        self.model_path = model_path
        self.dataset = DataLoader().dataset
        self.words = self.read_data(self.dataset)
        
    def read_data(self, 
                dataset):
        
        words = []
        for line in dataset:
            words.extend(line)
        return words
    
    def build_dataset(self, 
                    words, 
                    min_count):
        
        count = [['UNK', -1]]
        reserved_words = [item for item in collections.Counter(words).most_common() if item[1] >= min_count]
        count.extend(reserved_words)
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count = unk_count + 1
            data.append(index)
        count[0][1] = unk_count
        print(len(count))
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reverse_dictionary
    
    def generate_batch(self, 
                    data, 
                    batch_size, 
                    skip_window):
        
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        buffer = collections.deque(maxlen=span)

        for _ in range(span):
            buffer.append(data[self.data_index])
            self.data_indexdata_index = (self.data_index + 1) % len(data)

        for i in range(batch_size):
            target = skip_window
            target_to_avoid = [skip_window]
            col_idx = 0
            for j in range(span):
                if j == span // 2:
                    continue
                batch[i, col_idx] = buffer[j]
                col_idx += 1
            labels[i, 0] = buffer[target]

            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)

        assert batch.shape[0] == batch_size and batch.shape[1] == span - 1

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
        with graph.as_default(), tf.device('cuda:0'):
            train_dataset = tf.placeholder(tf.int32, shape=[batch_size, 2 * window_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            
            softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
            softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

            context_embeddings = []
            for i in range(2 * window_size):
                context_embeddings.append(tf.nn.embedding_lookup(embeddings, train_dataset[:, i]))
                
            avg_embed = tf.reduce_mean(tf.stack(axis=0, values=context_embeddings), 
                                    0, 
                                    keep_dims=False)
            
            loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(weights=softmax_weights, 
                                        biases=softmax_biases, 
                                        inputs=avg_embed,
                                        labels=train_labels, 
                                        num_sampled=num_sampled,
                                        num_classes=vocabulary_size)
                )
            
            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
            
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 
                                        1, 
                                        keep_dims=True)
                        )
            
            normalized_embeddings = embeddings / norm

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            average_loss = 0
            for step in range(num_steps):
                batch_data, batch_labels = self.generate_batch(batch_size, window_size, data)
                feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
                _, l = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += l
                if step % 2000 == 0:
                    if step > 0:
                        average_loss = average_loss / 2000
                    print('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0
            final_embeddings = normalized_embeddings.eval()
        return final_embeddings

    def save_embedding(self,
                    final_embeddings, 
                    model_path, 
                    reverse_dictionary):
        f = open(model_path,'w+')
        for index, item in enumerate(final_embeddings):
            f.write(reverse_dictionary[index] + '\t' + ','.join([str(vec) for vec in item]) + '\n')
        f.close()

    def train(self):
        data, count, dictionary, reverse_dictionary = self.build_dataset(self.words, self.min_count)
        vocabulary_size = len(count)
        final_embeddings = self.train_wordvec(vocabulary_size, self.batch_size, self.embedding_size, self.window_size, self.num_sampled, self.num_steps, data)
        self.save_embedding(final_embeddings, self.model_path, reverse_dictionary)