import sys
import random
import numpy as np
import pandas as pd
import os
from scipy.sparse import lil_matrix, csr_matrix

# DSD tools
import glidetools.algorithm.dsd as dsdtool
from scipy.spatial.distance import squareform, pdist

category_table = {'Molecular Function (Gene Ontology)': "GO:0003674",
                   'Biological Process (Gene Ontology)': "GO:0008150",
                   'Cellular Component (Gene Ontology)': "GO:0005575"}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class Protein:
    def __init__(self, pID : str):
        self.id = pID
        self.links = dict()
        self.totalWeight = 0

    def addLink(self, pOther: str, score : int):
        self.links[pOther] = score
        self.totalWeight += score

    def chooseRandomLink(self):
        r = random.randrange(0,self.totalWeight)
        for pID,score in self.links.items():
            if r < score:
                return pID
            else:
                r -= score
        return ""


class ProteinNetwork:
    def __init__(self, link_file : str, go_file : str):
        self.nodes = dict() # str -> protein
        self.go = dict() # str -> set of GO terms

        with open(link_file, 'r') as file:
            next(file)
            for line in file:
                self._read_plink_file_line(line)

        with open(go_file, 'r') as file:
            for line in file:
                self._read_go_file_line(line)

    def _read_plink_file_line(self, line : str):
        p1, p2, score = line.split()

        # Initialize nodes if they don't already exist
        if self.nodes.get(p1) is None:
            self.nodes[p1] = Protein(p1)
        if self.nodes.get(p2) is None:
            self.nodes[p2] = Protein(p2)
        
        self.nodes[p1].addLink(p2, int(score))
        self.nodes[p2].addLink(p1, int(score))
    
    def _read_go_file_line(self, line : str):
        pid, category, go_term, _ = line.split("\t")
        if not "Gene Ontology" in category:
            return
        if self.go.get(pid) is None:
            self.go[pid] = set()
        self.go[pid].add(go_term)
        if category_table[category] not in self.go[pid]:
            self.go[pid].add(category_table[category])

    # def printProteinLinks(self, pID : str):
    #     print(self.nodes[pID].id)
    #     print(len(self.nodes[pID].links))
    #     print(self.nodes[pID].chooseRandomLink())

    def getOrderedPIDs(self):
        return sorted(self.nodes.keys())
    
    def getOrderedGoTerms(self):
        return sorted(set(term for terms in self.go.values() for term in terms))

    def toAdjacencyMatrix(self):
        proteins = self.getOrderedPIDs()
        n = len(proteins)
        idx = {p: i for i, p in enumerate(proteins)}
        
        sparse = lil_matrix((n, n), dtype=np.float32)

        for p1, protein in self.nodes.items():
            i = idx[p1]
            for p2, score in protein.links.items():
                sparse[i, idx[p2]] = score

        return csr_matrix(sparse)
    
    def getDSDMatrix(self, t:int=5, normalized:bool=True):
        adj = self.toAdjacencyMatrix()
        # temporarily convert to dense array for dsd
        if hasattr(adj, 'toarray'):
            adj = adj.toarray()
        dsdEmb = dsdtool.compute_dsd_embedding(self.toAdjacencyMatrix(), t, 1, normalized)
        dsdDist = squareform(pdist(dsdEmb))
        del adj
        return dsdDist

    def getGoMatrix(self):
        proteins = self.getOrderedPIDs()
        go_terms = self.getOrderedGoTerms()
        go_idx = {t: j for j, t in enumerate(go_terms)}
        
        sparse = lil_matrix((len(proteins), len(go_terms)), dtype=np.uint8)  # binary, so uint8 is fine

        for i, pid in enumerate(proteins):
            for term in self.go.get(pid, set()):
                if term in go_idx:
                    sparse[i, go_idx[term]] = 1

        return csr_matrix(sparse)

    def getFullTrainingData(self):
        return self.toAdjacencyMatrix(), self.getGoMatrix()

def export_matrix_chunked(matrix, row_labels, col_labels, filepath, chunk_size=500):
    # write to csv row by row
    with open(filepath, 'w') as f:
        f.write(',' + ','.join(col_labels) + '\n')
        for start in range(0, len(row_labels), chunk_size):
            end = min(start + chunk_size, len(row_labels))
            chunk = matrix[start:end]
            # handle sparse slices
            if hasattr(chunk, 'toarray'):
                chunk = chunk.toarray()
            df_chunk = pd.DataFrame(chunk, index=row_labels[start:end], columns=col_labels)
            df_chunk.to_csv(f, header=False)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} LINK_FILE GO_FILE")
    link_file = sys.argv[1]
    go_file = sys.argv[2]
    net = ProteinNetwork(link_file, go_file)
    adj, go = net.getFullTrainingData()
    dsd = net.getDSDMatrix()
    
    # Export to CSV for use elsewhere
    # Include protein IDs as row and column labels
    os.makedirs(RESULTS_DIR, exist_ok=True)
    proteins = net.getOrderedPIDs()
    go_terms = net.getOrderedGoTerms()

    # adj_df = pd.DataFrame(adj, index=proteins, columns=proteins)
    # go_df = pd.DataFrame(go, index=proteins, columns=go_terms)
    # dsd_df = pd.DataFrame(dsd, index=proteins, columns=proteins)
    
    # adj_df.to_csv(os.path.join(RESULTS_DIR, "adjacency_matrix.csv"))
    # go_df.to_csv(os.path.join(RESULTS_DIR, "go_matrix.csv"))
    # dsd_df.to_csv(os.path.join(RESULTS_DIR, "dsd_matrix.csv"))

    export_matrix_chunked(adj, proteins, proteins, os.path.join(RESULTS_DIR, "adjacency_matrix.csv"))
    export_matrix_chunked(adj, proteins, proteins, os.path.join(RESULTS_DIR, "go_matrix.csv"))
    export_matrix_chunked(adj, proteins, proteins, os.path.join(RESULTS_DIR, "dsd_matrix.csv"))
