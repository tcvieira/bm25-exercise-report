import math

# Implementation from https://en.wikipedia.org/wiki/Okapi_BM25


class BM25Simple(object):
    PARAM_K1 = 1.2
    PARAM_B = 0.75
    EPSILON = 0.25

    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.dl = [float(len(d)) for d in corpus]
        self.avgdl = sum(self.dl) / self.corpus_size
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.average_idf = 0
        self._initialize()

    def _initialize(self):
        for document in self.corpus:
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)

            for word, freq in frequencies.items():
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in self.df.items():
            self.idf[word] = math.log(
                self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

        self.average_idf = sum(
            map(lambda k: float(self.idf[k]), self.idf.keys())) / len(self.idf.keys())

    def _get_score(self, document, index):
        score = 0
        for word in document:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else self.EPSILON * \
                self.average_idf
            score += (idf * self.f[index][word] * (self.PARAM_K1 + 1)
                      / (self.f[index][word] + self.PARAM_K1 * (1 - self.PARAM_B + self.PARAM_B * self.dl[index] / self.avgdl)))
        return score

    def _get_scores(self, document):
        scores = []
        for index in range(self.corpus_size):
            score = self._get_score(document, index)
            scores.append(score)
        return scores

    def get_scores(self, query, k=None):
        """Returns the `scores` of most relevant documents according to `query`"""
        result = [(index, score)
                  for index, score in enumerate(self._get_scores(query))]
        result.sort(key=lambda x: x[1], reverse=True)
        _, scores = zip(*result)
        return scores

    def get_top_n(self, query, corpus, n=20):
        """Returns the `indexes` most relevant documents according to `query`"""
        result = [(index, score)
                  for index, score in enumerate(self._get_scores(query))]
        result.sort(key=lambda x: x[1], reverse=True)
        indexes, _ = zip(*result)
        return [corpus[i] for i in indexes[:n]]
