# Hierarchical Clustering Dendrogram

import numpy as np
import string
import json
from collections import Counter
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering


class SciKit:
    def __init__(self, filePath):
        self.filePath = filePath
        self.charCount = None
        self.columnCount = None
        self.nGrams = [0] * 4

    def plot_dendrogram(self, model, **kwargs):
        # create linkage matrix and then plot the dendrogram
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    def main1(self):
        iris = load_iris()
        X = iris.data

        # setting distance_threshold=0 ensures we compute the full tree.
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

        model = model.fit(X)
        plt.title('Hierarchical Clustering Dendrogram')
        # plot the top three levels of the dendrogram
        self.plot_dendrogram(model, truncate_mode='level', p=3)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()

    def main2(self):
        X = np.array([[ord('A'), 2.0], [ord('A'), 5.4], [ord('A'), 2.2],
                      [ord('B'), 5.0], [ord('B'), 5.1], [ord('B'), 5.2]])
        clustering = AgglomerativeClustering().fit(X)
        print(clustering.labels_)
        print(clustering.n_clusters_)

    def charCounters(self, filePath):
        """ total character count histogram """
        result = Counter()
        with open(filePath, "r") as f:
            for line in f:
                line = line.strip()
                result.update(line)
        return dict(result)

    def columnCounters(self, filePath):
        """ total character count for position on line histogram """
        column_counters = [{c: 0 for c in string.printable} for i in range(0, 80)]
        with open(filePath, "r") as f:
            for line in f:
                line = line.strip()
                i = 0
                for c in line:
                    if c not in column_counters[i]:
                        column_counters[i][c] = 0
                    column_counters[i][c] += 1
                    i += 1
        return column_counters

    def nGramsCounters(self, filePath, ngram=2):
        """ bi-tri-4-5 grams on lines """
        result = Counter()
        with open(filePath, "r") as f:
            for line in f:
                line = line.strip()
                result.update(line[x:x + ngram] for x in range(len(line) - ngram + 1))
        return dict(result)

    def initCounters(self):
        with open("pickles/charCounters.json", "w") as f:
            self.charCount = self.charCounters(self.filePath)
            json.dump(self.charCount, f, indent=1)
        with open("pickles/columnCounters.json", "w") as f:
            self.charCount = self.columnCounters(self.filePath)
            json.dump(self.charCount, f, indent=1)
        for i in range(2, 6):
            with open(f"pickles/nGramsCounters{i}.json", "w") as f:
                self.nGrams[i - 2] = self.nGramsCounters(self.filePath, ngram=i)
                json.dump(self.nGrams[i - 2], f, indent=1)

    def loadCounters(self):
        with open("pickles/charCounters.json", "r") as f:
            self.charCount = json.load(f)
        with open("pickles/columnCounters.json", "r") as f:
            self.columnCount = json.load(f)
        for i in range(2, 6):
            with open(f"pickles/nGramsCounters{i}.json", "r") as f:
                self.nGrams[i - 2] = json.load(f)


if __name__ == '__main__':
    sk = SciKit("target/m10rom.lst")
    # sk.initCounters()
    sk.loadCounters()
    sk.main2()
