import sys
import os
import pandas as pd
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class KNNClassification:
    def __init__(self, dsd: pd.DataFrame, go: pd.DataFrame) -> None:
        self.dsd = dsd
        self.go = go

        self.neighbor_weights = {'uniform': lambda n,k,d:1,
                                 'linear':lambda n,k,d: 1-(n/k), 
                                 'power':lambda n,k,d: 1/(n+1),
                                 'inverse_distance':lambda n,k,d: 1/d}
        pass

    def random_2fold(self, k):
        full_go = self.go
        f1 = get_random_fold(.5, self.dsd.index)
        f1_go = go_df.loc[go_df.index.isin(test_proteins)]
        f2_go = go_df.loc[~go_df.index.isin(test_proteins)]
        f2 = f2_go.index

        for wt in ['uniform', 'linear', 'inverse_distance']:
            self.go = f2_go
            f1_pred = self.classify_k_nearest(k, f1, weight_type=wt)
            self.go = f1_go
            f2_pred = self.classify_k_nearest(k, f2, weight_type=wt)
            pred = pd.concat([f1_pred, f2_pred])
            self.predictions_to_tsv(pred, f"pred_{wt}")

            # Get accuracy and recall
            self.go = full_go
            print(f"Stats for {wt}:")
            print(self.compute_fmax(pred))

        pass

    def compute_fmax(self, pred:pd.DataFrame):
        f_max = 0
        for t in [.8,.7,.6,.5,.4,.3,.2,.1]:
            # Get the mean F1 across all proteins for some t
            p_av, r_av = 0, 0
            for protein in pred.index:
                p, r = self.get_f1(pred.loc[protein], self.go.loc[protein], t)
                p_av += p
                r_av += r
            p_av /= pred.index.size
            r_av /= pred.index.size
            f1 = (2*p_av*r_av)/(p_av+r_av) if (p_av + r_av) > 0 else 0.0

            # Check f_max
            if f1 > f_max:
                f_max = f1
                t_max = t
                p_max = p_av
                r_max = r_av

        return f_max, t_max, p_max, r_max

    def get_f1(self, pred_row:pd.Series, true_val:pd.Series, t: float):
        y_pred = (pred_row.to_numpy() >= t)

        TP = ((y_pred == 1) & (true_val == 1)).sum()
        FP = ((y_pred == 1) & (true_val == 0)).sum()
        FN = ((y_pred == 0) & (true_val == 1)).sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        return precision, recall
    

    def predictions_to_tsv(self, pred: pd.DataFrame, filename: str = "predictions.tsv"):
        # Create a predictions.tsv output file
        with open(os.path.join(RESULTS_DIR, filename), 'w') as tsv:
            for protein in pred.index:
                for go in pred.columns:
                    score = pred.at[protein, go]
                    if type(score) == np.float64 and score > 0.0:
                        tsv.write(f"{protein}\t{go}\t{score}\n")
        pass

    def classify_k_nearest(self, k: int, test_proteins: pd.Index, weight_type: str = 'uniform') -> pd.DataFrame:
        get_mult = self.neighbor_weights[weight_type]

        prot_p = pd.DataFrame(data=0.0, index=test_proteins, columns=self.go.columns)
        for protein in prot_p.index:
            if protein not in self.dsd.index:
                # We have no network data for this protein, just return NaN
                continue
            if protein in self.go.index:
                # We have the go labels for this exact protein
                prot_p.loc[protein] = self.go.loc[protein]
                continue

            # Get list of k closest neighbors and sum their contributions
            sorted_dsd_distance = self.dsd[protein].sort_values(ascending=True, na_position='last')
            n = 0
            norm_const = 0
            row_pred = prot_p.loc[protein].to_numpy(dtype=float).copy()
            for neighbor in sorted_dsd_distance.index:
                if neighbor in self.go.index:
                    # Calculate multiplier for the nth element
                    mult = get_mult(n,k,sorted_dsd_distance[neighbor])
                    norm_const += mult
                    # Add related go terms to sum
                    row_pred += mult*self.go.loc[neighbor].to_numpy(dtype=float)
                    n += 1
                if n >= k:
                    break
            if n != k: 
                print(f"Could not find {k} closest neighbors with go labels.")
            row_pred /= norm_const
            prot_p.loc[protein] = row_pred

        return prot_p

def get_random_fold(p: float, protein_index: pd.Index, seed = 3759798) -> pd.Index:
    num_prot = protein_index.size
    num_elements = int(num_prot * p)
    # print(num_elements)
    np.random.seed(seed) # Just to fix the folds for now
    indices = np.random.choice(range(0, num_prot), size=num_elements, replace=False)
    # print(indices)
    return dsd_df.index[indices]

# Calling conventions are
# % python3 knn_baseline {DSD_FILE} {GO_FILE}
if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Usage: knn_baseline {{DSD_FILE}} {{GO_FILE}}")
        exit() 
    dsd_df = pd.read_csv(sys.argv[1], index_col=0)
    go_df = pd.read_csv(sys.argv[2], index_col=0)

    # Fold creation
    # test_proteins = get_random_fold(.5, dsd_df.index)
    # xv_go = go_df.loc[~go_df.index.isin(test_proteins)]

    # Test set
    with open("data/testsuperset.ps", "r") as f:
        test_proteins = pd.Index([line.strip() for line in f if line.strip()])

    knn = KNNClassification(dsd_df, go_df)

    # knn.random_2fold(10)
    pred = knn.classify_k_nearest(10, test_proteins, 'linear')
    knn.predictions_to_tsv(pred, filename="predictions.tsv")
    pass