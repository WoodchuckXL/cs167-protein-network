import sys
import random

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
    def __init__(self, infile : str):
        self.nodes = dict() # str -> protein

        with open(infile, 'r') as file:
            next(file)
            for line in file:
                self._read_plink_file_line(line)
                pass

    def _read_plink_file_line(self, line : str):
        p1, p2, score = line.split()

        # Initialize nodes if they don't already exist
        if self.nodes.get(p1) is None:
            self.nodes[p1] = Protein(p1)
        if self.nodes.get(p2) is None:
            self.nodes[p2] = Protein(p2)
        
        self.nodes[p1].addLink(p2, int(score))
        self.nodes[p2].addLink(p1, int(score))
        pass

    def printProteinLinks(self, pID : str):
        print(self.nodes[pID].id)
        print(len(self.nodes[pID].links))
        print(self.nodes[pID].chooseRandomLink())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} FILE_NAME")
    net = ProteinNetwork(sys.argv[1])
    net.printProteinLinks("4932.Q0010")
    pass