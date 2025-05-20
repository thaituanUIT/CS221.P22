import numpy as np

class WordCluster:
    def __init__(self, embedding_path=None):
        self.embedding_path = embedding_path
        self.word_embedding_dict, self.word_dict, self.word_embeddings = self.load_model(self.embedding_path)
        self.similar_num = 10

    def load_model(self, embedding_path):
        print('loading models....')
        word_embedding_dict = {}
        word_embeddings = []
        word_dict = {}
        index = 0
        for line in open(embedding_path):
            line = line.strip().split('\t')
            word = line[0]
            word_embedding = np.array([float(item) for item in line[1].split(',') if item])
            word_embedding_dict[word] = word_embedding
            word_embeddings.append(word_embedding)
            word_dict[index] = word
            index += 1
        return word_embedding_dict, word_dict, np.array(word_embeddings)

    def similarity_cosine(self, word):
        A = self.word_embedding_dict[word]
        B = (self.word_embeddings).T
        dot_num = np.dot(A, B)
        denom = np.linalg.norm(A) * np.linalg.norm(B)
        cos = dot_num / denom
        sims = 0.5 + 0.5 * cos
        sim_dict = {self.word_dict[index]: sim for index, sim in enumerate(sims.tolist()) if word != self.word_dict[index]}
        sim_words = sorted(sim_dict.items(), key=lambda asd: asd[1], reverse=True)[:self.similar_num]
        return sim_words

    def get_similar_words(self, word):
        if word in self.word_embedding_dict:
            return self.similarity_cosine(word)
        else:
            return []