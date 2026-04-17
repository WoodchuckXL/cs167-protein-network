# =============================================================================
# DEPENDENCIES
# =============================================================================
# Core:
#   pandas      - data loading and manipulation
#   numpy       - numerical operations
#   scipy       - statistical models and ML utilities
# Visualization:
#   matplotlib  - base plotting library
#   seaborn     - heatmap and statistical visualizations
# =============================================================================

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import os

import warnings
warnings.filterwarnings('ignore')

def load_data(go_path, adjacency_path, min_positive=1000, max_positive=10000, binarize=True, binarize_threshold=0.30):
    """
    Loads and aligns GO label matrix and adjacency matrix for protein function prediction.

    Args:
        go_path (str): Path to the GO matrix CSV file
        adjacency_path (str): Path to the adjacency matrix CSV file
        min_positive (int): Minimum number of positive examples required to keep a GO term (default 30)
        max_positive (int): Maximum number of positive examples required to keep a GO term (default 30)
        binarize (bool): Whether to convert GO confidence scores to binary labels (default True)
        binarize_threshold (float): Threshold above which a confidence score is considered positive (default 0.30)

    Returns:
        X (pd.DataFrame): Feature matrix, rows are genes, columns are gene similarity scores
        Y (pd.DataFrame): Label matrix, rows are genes, columns are filtered GO terms
    """

    # Load CSVs with gene IDs as index
    go_matrix = pd.read_csv(go_path, index_col=0)
    adjacency_matrix = pd.read_csv(adjacency_path, index_col=0)

    # Binarize GO confidence scores if requested
    if binarize:
        go_matrix = (go_matrix > binarize_threshold).astype(int)

    # Align the two matrices on shared gene IDs
    shared_genes = go_matrix.index.intersection(adjacency_matrix.index)
    X = adjacency_matrix.loc[shared_genes]
    Y = go_matrix.loc[shared_genes]

    # Filter GO terms with fewer than min_positive positive examples
    positive_counts = Y.sum(axis=0)
    Y = Y.loc[:, positive_counts >= min_positive]

    # Filter GO terms with more than max_positive positive examples
    positive_counts = Y.sum(axis=0)
    Y = Y.loc[:, positive_counts <= max_positive]

    # Summary
    print(f"Genes after alignment: {len(shared_genes)}")
    print(f"GO terms before filtering: {len(positive_counts)}")
    print(f"GO terms after filtering (min_positive={min_positive}) (max_positive={max_positive}): {Y.shape[1]}")
    print(f"Feature vector length: {X.shape[1]}")

    return X, Y


def descriptive_stats(X, Y, output_dir, top_n=20):
    """
    Generates descriptive statistics and visualizations for the protein function prediction dataset.

    Args:
        X (pd.DataFrame): Feature matrix from load_data
        Y (pd.DataFrame): Binary label matrix from load_data
        output_dir (str): Directory to save plots
        top_n (int): Number of most common GO terms to include in correlation heatmap (default 20)

    Returns:
        None (saves plots to output_dir)
    """

    os.makedirs(output_dir, exist_ok=True)

    # --- Fig 1: Bar chart of positive example counts across all surviving GO terms ---
    positive_counts = Y.sum(axis=0).sort_values(ascending=False)

    plt.figure(figsize=(max(10, len(positive_counts) // 4), 6))
    plt.bar(range(len(positive_counts)), positive_counts.values)
    plt.xlabel("GO Term")
    plt.ylabel("Number of Positive Examples")
    plt.title("Positive Example Counts per GO Term")
    plt.xticks(range(len(positive_counts)), positive_counts.index, rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "go_term_counts.png"))
    plt.close()
    print(f"Saved GO term counts plot to {output_dir}/go_term_counts.png")

    # --- Fig 4: Correlation heatmap of top N most common GO terms ---
    top_terms = positive_counts.head(top_n).index
    Y_top = Y[top_terms]
    corr_matrix = Y_top.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        annot_kws={"size": 8}
    )
    plt.title(f"Correlation Matrix of Top {top_n} Most Common GO Terms")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "go_term_correlation.png"))
    plt.close()
    print(f"Saved correlation heatmap to {output_dir}/go_term_correlation.png")

def _balance_single_label(X, y, random_seed=None):
    """Balances a single GO term dataset by downsampling negatives."""
    rng = np.random.default_rng(random_seed)
    positive_idx = y[y == 1].index
    negative_idx = y[y == 0].index
    n_positive = len(positive_idx)
    n_negative = len(negative_idx)

    if n_positive >= n_negative:
        sampled_positive_idx = rng.choice(positive_idx, size=n_negative, replace=False)
        selected_idx = negative_idx.tolist() + sampled_positive_idx.tolist()
    else:
        sampled_negative_idx = rng.choice(negative_idx, size=n_positive, replace=False)
        selected_idx = positive_idx.tolist() + sampled_negative_idx.tolist()

    return X.loc[selected_idx], y.loc[selected_idx]


def make_model(X, y, model_type, params):
    """
    Trains a single model on the given data.

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Binary label series
        model_type (str): One of 'logistic', 'svm', 'random_forest'
        params (dict): Hyperparameters for the model

    Returns:
        model: Trained sklearn model
    """
    if model_type == 'logistic':
        model = LogisticRegression(**params, max_iter=1000)
    elif model_type == 'linear_svm':
        # LinearSVC doesn't support predict_proba natively so we wrap it
        base = LinearSVC(C=1.0, max_iter=2000)
        model = CalibratedClassifierCV(base, cv=3)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X, y)
    return model


def test_model(model, X, y, go_term, save_path):
    """
    Evaluates a trained model and saves the AUROC curve to file.

    Args:
        model: Trained sklearn model
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Binary label series
        go_term (str): GO term ID for labeling the plot
        save_path (str): Path to save the AUROC curve plot

    Returns:
        auroc (float): Area under the ROC curve
    """
    y_prob = model.predict_proba(X)[:, 1]
    auroc = roc_auc_score(y, y_prob)
    fpr, tpr, _ = roc_curve(y, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {go_term}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return auroc


def choose_model(X, Y, go_terms, output_dir, n_folds=3, random_seed=None):
    """
    For each GO term, balances the dataset, selects the best model type via 
    cross validation, retrains on full balanced data, and saves the AUROC curve 
    for the winning model.

    Args:
        X (pd.DataFrame): Full feature matrix from load_data
        Y (pd.DataFrame): Full binary label matrix from load_data
        go_terms (list of str): GO term IDs to model
        output_dir (str): Directory to save AUROC curve plots
        n_folds (int): Number of cross validation folds (default 3)
        random_seed (int, optional): Random seed for reproducibility (default None)

    Returns:
        models (list): Trained models, one per GO term
        results (dict): Dictionary mapping go_term -> best model type and AUROC score
    """

    os.makedirs(output_dir, exist_ok=True)

    model_configs = {
        'logistic': {'C': 1.0, 'solver': 'lbfgs', 'n_jobs': -1},
        'linear_svm': {},  # handled separately below
        'random_forest': {'n_estimators': 100, 'max_depth': 5, 'n_jobs': -1}
    }

    models = []
    results = {}
    n_selected = 0

    for i, go_term in enumerate(go_terms):
        y = Y[go_term]

        # Balance dataset for this GO term on the fly
        X_bal, y_bal = _balance_single_label(X, y, random_seed=random_seed)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        mean_aurocs = {}

        for model_type, params in model_configs.items():
            fold_aurocs = []

            for train_idx, val_idx in skf.split(X_bal, y_bal):
                X_train, X_val = X_bal.iloc[train_idx], X_bal.iloc[val_idx]
                y_train, y_val = y_bal.iloc[train_idx], y_bal.iloc[val_idx]

                model = make_model(X_train, y_train, model_type, params)
                y_prob = model.predict_proba(X_val)[:, 1]
                fold_aurocs.append(roc_auc_score(y_val, y_prob))

            mean_aurocs[model_type] = np.mean(fold_aurocs)

        best_model_type = max(mean_aurocs, key=mean_aurocs.get)
        best_auroc = mean_aurocs[best_model_type]

        final_model = make_model(X_bal, y_bal, best_model_type, model_configs[best_model_type])

        save_path = os.path.join(output_dir, f"{go_term.replace(':', '_')}_auroc.png")
        test_model(final_model, X_bal, y_bal, go_term, save_path)

        models.append(final_model)
        results[go_term] = {
            'best_model_type': best_model_type,
            'cv_auroc': best_auroc,
            'all_aurocs': mean_aurocs
        }
        n_selected += 1

        print(f"{n_selected}/{len(go_terms)} models selected")

    print(f"{n_selected}/{len(go_terms)} models selected")
    print(f"Model type breakdown:")
    type_counts = {}
    for v in results.values():
        t = v['best_model_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    for model_type, count in type_counts.items():
        print(f"  {model_type}: {count}")

    return models, results


def make_final_model(X, Y, go_terms, output_dir, random_seed=None):
    """
    Trains a model for each GO term via choose_model and wraps them into
    a single prediction function.

    Args:
        X (pd.DataFrame): Full feature matrix from load_data
        Y (pd.DataFrame): Full binary label matrix from load_data
        go_terms (list of str): GO term IDs to model
        output_dir (str): Directory to save AUROC curve plots
        random_seed (int, optional): Random seed for reproducibility (default None)

    Returns:
        predict_fn (callable): Function that takes a single feature vector and
                               returns a dict of {go_term: prediction} for all GO terms
        models (list): Trained models, one per GO term
        results (dict): Dictionary mapping go_term -> best model type and AUROC score
    """

    models, results = choose_model(X, Y, go_terms, output_dir, random_seed=random_seed)

    def predict_fn(x):
        x = np.array(x).reshape(1, -1)
        predictions = {}
        for model, go_term in zip(models, go_terms):
            prob = model.predict_proba(x)[0, 1]
            predictions[go_term] = 1 if prob >= 0.5 else 0
        return predictions

    return predict_fn, models, results


def test_final_model(predict_fn, X, Y, go_terms, output_dir):
    """
    Evaluates the full multi-label prediction pipeline using hamming distance,
    mean AUROC, and micro precision and recall. Saves a summary report and
    box plot of per-gene accuracy to output_dir.

    Args:
        predict_fn (callable): Prediction function from make_final_model
        X (pd.DataFrame): Full feature matrix from load_data
        Y (pd.DataFrame): Full binary label matrix from load_data
        go_terms (list of str): GO term IDs that were modeled
        output_dir (str): Directory to save summary report and box plot

    Returns:
        summary (dict): Dictionary of evaluation metrics
    """

    os.makedirs(output_dir, exist_ok=True)

    # Filter Y to only the GO terms that were modeled
    Y_filtered = Y[go_terms]

    # Get predictions for every gene in X
    all_predictions = []
    for gene in X.index:
        pred = predict_fn(X.loc[gene])
        all_predictions.append([pred[go_term] for go_term in go_terms])

    Y_pred = np.array(all_predictions)
    Y_true = Y_filtered.values

    # --- Hamming distance per gene ---
    # Hamming distance is the fraction of labels that differ
    hamming_per_gene = np.mean(Y_pred != Y_true, axis=1)
    mean_hamming = np.mean(hamming_per_gene)

    # Accuracy per gene (1 - hamming)
    accuracy_per_gene = 1 - hamming_per_gene
    mean_accuracy = np.mean(accuracy_per_gene)

    # --- Mean AUROC across all GO terms ---
    auroc_per_term = []
    for j, go_term in enumerate(go_terms):
        try:
            auroc = roc_auc_score(Y_true[:, j], Y_pred[:, j])
            auroc_per_term.append(auroc)
        except ValueError:
            # Skip if only one class present in Y_true for this term
            pass
    mean_auroc = np.mean(auroc_per_term)

    # --- Micro precision and recall ---
    micro_precision = precision_score(Y_true, Y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(Y_true, Y_pred, average='micro', zero_division=0)

    summary = {
        'mean_hamming_distance': mean_hamming,
        'mean_accuracy': mean_accuracy,
        'mean_auroc': mean_auroc,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'n_genes': len(X),
        'n_go_terms': len(go_terms)
    }

    # --- Box plot of accuracy across all genes ---
    plt.figure(figsize=(8, 6))
    plt.boxplot(accuracy_per_gene, vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', color='navy'),
                medianprops=dict(color='white', linewidth=2))
    plt.ylabel("Per-Gene Accuracy (1 - Hamming Distance)")
    plt.title("Distribution of Prediction Accuracy Across All Genes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_boxplot.png"))
    plt.close()

    # --- Save summary report ---
    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("FINAL MODEL EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Genes evaluated:       {summary['n_genes']}\n")
        f.write(f"GO terms modeled:      {summary['n_go_terms']}\n\n")
        f.write(f"Mean Hamming Distance: {summary['mean_hamming_distance']:.4f}\n")
        f.write(f"Mean Accuracy:         {summary['mean_accuracy']:.4f}\n")
        f.write(f"Mean AUROC:            {summary['mean_auroc']:.4f}\n")
        f.write(f"Micro Precision:       {summary['micro_precision']:.4f}\n")
        f.write(f"Micro Recall:          {summary['micro_recall']:.4f}\n")

    print(f"Summary report saved to {report_path}")
    print(f"Mean Accuracy: {mean_accuracy:.4f} | Mean AUROC: {mean_auroc:.4f} | "
          f"Precision: {micro_precision:.4f} | Recall: {micro_recall:.4f}")

    return summary

def main():
    # =========================================================================
    # PARAMETERS
    # =========================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--adj-path',
                        default="./results/adjacency_matrix_no_dsd.csv",
                        help='Path to adjacency matrix CSV file')
    parser.add_argument('--go-path',
                        default="./results/go_matrix.csv",
                        help='Path to GO matrix CSV file')
    args = parser.parse_args()

    GO_PATH = args.go_path
    ADJACENCY_PATH = args.adj_path
    OUTPUT_DIR = "results"
    MIN_POSITIVE = 500
    MAX_POSITIVE = 10000
    BINARIZE = True # DO NOT MAKE THIS FALSE
    BINARIZE_THRESHOLD = 0.30
    RANDOM_SEED = 42
    TOP_N_CORR = 50
    N_FOLDS = 3

    # =========================================================================
    # STEP 1: Load data
    # =========================================================================
    print("=" * 50)
    print("STEP 1: Loading data")
    print("=" * 50)
    X, Y = load_data(
        go_path=GO_PATH,
        adjacency_path=ADJACENCY_PATH,
        min_positive=MIN_POSITIVE,
        max_positive=MAX_POSITIVE,
        binarize=BINARIZE,
        binarize_threshold=BINARIZE_THRESHOLD
    )

    # =========================================================================
    # STEP 2: Descriptive statistics
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 2: Descriptive statistics")
    print("=" * 50)
    descriptive_stats(X, Y, output_dir=OUTPUT_DIR, top_n=TOP_N_CORR)

    #train/test split
    #80-20 split? i think this is fine for now idk
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"Training on {X_train.shape[0]} proteins.")
    print(f"Training on {X_test.shape[0]} unseen proteins.")

    # =========================================================================
    # STEP 3: Train and select best model per GO term
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 3: Training and selecting models")
    print("=" * 50)
    go_terms = Y.columns.tolist()
    predict_fn, models, results = make_final_model(
        X=X_train,
        Y=Y_train,
        go_terms=go_terms,
        output_dir=OUTPUT_DIR,
        random_seed=RANDOM_SEED
    )

    #print which model we chose to be best
    print("\n"+"="*50)
    print("SELECTED MODELS PER GO TERM")
    print("="*50)
    results_df=pd.DataFrame.from_dict(results, orient='index')
    results_df=results_df.sort_values(by='cv_auroc',ascending=False)
    for go_term, row in results_df.iterrows():
        best_model=row['best_model_type']
        score=row['cv_auroc']
        print(f"{go_term:<15} | Selected: {best_model:<15} | CV AUROC: {score:.3f}")

    results_df.to_csv(os.path.join(OUTPUT_DIR, "selected_models.csv"))
    # =========================================================================
    # STEP 4: Evaluate final model
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 4: Evaluating final model")
    print("=" * 50)
    summary = test_final_model(
        predict_fn=predict_fn,
        X=X_test,
        Y=Y_test,
        go_terms=go_terms,
        output_dir=OUTPUT_DIR
    )

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)


if __name__ == "__main__":
    main()
