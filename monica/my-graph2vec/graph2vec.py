"""Graph2Vec module."""

import json
import glob
import hashlib
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
#from param_parser import parameter_parser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()

def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path.strip(".json").split("/")[-1]
    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"])

    if "features" in data.keys():
        features = data["features"]
    else:
        features = nx.degree(graph)

    features = {int(k): v for k, v in features.items()}
    return graph, features, name

def feature_extractor(path, rounds):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc

def save_embedding(output_path, model, files, dimensions):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    for f in files:
        identifier = f.split("/")[-1].strip(".json")
        out.append([int(identifier)] + list(model.docvecs["g_"+identifier]))
    column_names = ["type"]+["x_"+str(dim) for dim in range(dimensions)]
    out = pd.DataFrame(out, columns=column_names)
    out = out.sort_values(["type"])
    out.to_csv(output_path, index=None)

if __name__ == "__main__":
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """

    graphs = glob.glob("graph2vec/dataset/" + "*.json")
    print("\nFeature extraction started.\n")
    document_collections = Parallel(n_jobs=1)(delayed(feature_extractor)(g, 2) for g in tqdm(graphs))

    print("\nOptimization started.\n")
    model = Doc2Vec(document_collections, vector_size = 128, window = 2, min_count = 5, dm = 0,
                sample = 0.0001, workers = 10, epochs = 20, alpha =0.025)

    # Retrieve node embeddings and corresponding subjects
    node_ids = model.docvecs.index_to_key  # list of node IDs
    print(node_ids)
    node_embeddings = (
        model.docvecs.vectors
    )  # numpy.ndarray of size number of nodes times embeddings dimensionality

    # Everything has label 0.
    labels = np.zeros((len(node_ids),), dtype=int)
    node_subjects = pd.Series(dict(zip(node_ids, labels)))
    node_targets = node_subjects.loc[[node_id for node_id in node_ids]]

    transform = TSNE  # PCA

    trans = transform(n_components=2)
    node_embeddings_2d = trans.fit_transform(node_embeddings)

    # draw the embedding points, coloring them by the target label (paper subject)
    alpha = 0.7
    label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
    node_colours = [label_map[target] for target in node_targets]

    plt.figure(figsize=(7, 7))
    plt.axes().set(aspect="equal")
    plt.scatter(
        node_embeddings_2d[:, 0],
        node_embeddings_2d[:, 1],
        c=node_colours,
        cmap="jet",
        alpha=alpha,
    )
    plt.title("{} visualization of graph embeddings".format(transform.__name__))
    plt.savefig("tsne_graph_embeddings.png", format='png')

    #save_embedding(args.output_path, model, graphs, args.dimensions)

