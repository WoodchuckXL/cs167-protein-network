import sys
import random
import numpy as np
import pandas as pd
import os

# DSD tools
import glidetools.algorithm.dsd as dsdtool
from scipy.spatial.distance import squareform, pdist

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
        array = np.empty((n, n))

        for i in range(n):
            protein1 = self.nodes.get(proteins[i])
            if protein1 is None:
                continue
            for j in range(n):
                score = protein1.links.get(proteins[j], 0)
                array[i][j] = score
        
        #normalize values
        # array /= np.max(array)
        return array 
    
    def getDSDMatrix(self, t:int=5):
        dsdEmb = dsdtool.compute_dsd_embedding(self.toAdjacencyMatrix(), t, 1, True)
        dsdDist = squareform(pdist(dsdEmb))

        return dsdDist

    def getGoMatrix(self):
        proteins = self.getOrderedPIDs()
        go_terms = self.getOrderedGoTerms()
        n = len(proteins)
        m = len(go_terms)

        array = np.empty((n, m))

        for i in range(n):
            # protein1 = self.nodes.get(proteins[i])
            for j in range(m):
                go_term = go_terms[j]
                if go_term in self.go.get(proteins[i], set()):
                    array[i][j] = 1
                else:
                    array[i][j] = 0
        return array

    def getFullTrainingData(self):
        return self.toAdjacencyMatrix(), self.getGoMatrix()

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
    proteins = net.getOrderedPIDs()
    go_terms = net.getOrderedGoTerms()
    adj_df = pd.DataFrame(adj, index=proteins, columns=proteins)
    go_df = pd.DataFrame(go, index=proteins, columns=go_terms)
    dsd_df = pd.DataFrame(dsd, index=proteins, columns=proteins)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    adj_df.to_csv(os.path.join(RESULTS_DIR, "adjacency_matrix.csv"))
    go_df.to_csv(os.path.join(RESULTS_DIR, "go_matrix.csv"))
    dsd_df.to_csv(os.path.join(RESULTS_DIR, "dsd_matrix.csv"))