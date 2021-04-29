# Hierarchical Clustering Dendrogram

import numpy as np
import string
from collections import Counter
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

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


def main1():
    iris = load_iris()
    X = iris.data

    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    model = model.fit(X)
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


def main2():
    X = np.array([[ord('A'), 2.0], [ord('A'), 5.4], [ord('A'), 2.2],
                  [ord('B'), 5.0], [ord('B'), 5.1], [ord('B'), 5.2]])
    clustering = AgglomerativeClustering().fit(X)
    print(clustering.labels_)
    print(clustering.n_clusters_)


def charCounters():
    """ total character count histogram """
    result = Counter()
    with open("target/m10rom.lst", "r") as f:
        for line in f:
            line = line.strip()
            result.update(line)
    return dict(result)


def columnCounters():
    """ total character count for position on line histogram """
    column_counters = np.array([{c: 0 for c in string.printable} for i in range(0, 80)])
    with open("target/m10rom.lst", "r") as f:
        for line in f:
            line = line.strip()
            i = 0
            for c in line:
                if c not in column_counters[i]:
                    column_counters[i][c] = 0
                column_counters[i][c] += 1
                i += 1
    return column_counters


def nGramsCounters(ngram=2):
    result = Counter()
    with open("target/m10rom.lst", "r") as f:
        for line in f:
            line = line.strip()
            result.update(line[x:x + ngram] for x in range(len(line) - ngram + 1))
    return dict(result)


if __name__ == '__main__':
    # print(charCounters())
    # print("-" * 50)
    # print(columnCounters())
    # print("-" * 50)
    print(nGramsCounters(ngram=2))
    print("-" * 50)
    # print(nGramsCounters(ngram=3))
    # print("-" * 50)
    # print(nGramsCounters(ngram=4))
    # print("-" * 50)
    # print(nGramsCounters(ngram=5))
