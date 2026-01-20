import pandas as pd
import pickle
import os
from collections import Counter
def find_top_n_snps_from_selected_algorithms(snp_importance_df, top_n, algorithm_cols_to_use, min_occurrence_threshold=1):
    """
    Identifies top N SNPs that appear frequently across a specified list of algorithms.

    Args:
        snp_importance_df (pd.DataFrame): DataFrame where rows are SNPs, columns are algorithms
                                          (or algorithm-derived metrics like PC loadings),
                                          and values are importance scores (higher is better).
        top_n (int): The number of top SNPs to consider from each specified algorithm.
        algorithm_cols_to_use (list): A list of column names from snp_importance_df
                                      to be included in this analysis.
        min_occurrence_threshold (int): The minimum number of specified algorithms a SNP
                                        must appear in (within their respective top N lists)
                                        to be included in the final output.

    Returns:
        dict: A dictionary where keys are SNP names that meet the threshold,
              and values are the counts of how many *specified* algorithms
              selected them in their top N. Sorted by count (desc) then SNP name (asc).
    """
    if snp_importance_df.empty or not algorithm_cols_to_use:
        return {}

    selected_snps_by_algo = {}
    for algo_col in algorithm_cols_to_use:
        if algo_col not in snp_importance_df.columns:
            print(f"Warning: Algorithm column '{algo_col}' not found in importance DataFrame. Skipping.")
            continue
        # Sort by the specific algorithm's importance score (descending) and get top N
        # Ensure we handle NaNs by dropping them before sorting or filling with a very low value
        top_snps_for_algo = snp_importance_df[algo_col].dropna().sort_values(ascending=False).head(top_n).index.tolist()
        selected_snps_by_algo[algo_col] = top_snps_for_algo

    all_top_snps_flat_list = []
    for snp_list in selected_snps_by_algo.values():
        all_top_snps_flat_list.extend(snp_list)

    snp_counts = Counter(all_top_snps_flat_list)

    frequent_snps = {snp: count for snp, count in snp_counts.items() if count >= min_occurrence_threshold}

    sorted_frequent_snps = dict(sorted(frequent_snps.items(), key=lambda item: (-item[1], item[0])))

    return sorted_frequent_snps

def load_raw(traits = None, top_x = 0, data_dir=None):
    current_dir = os.getcwd()
    if data_dir is not None:
        if data_dir.startswith('~/'):
            data_dir = os.path.expanduser(data_dir)
        elif data_dir.startswith('./'):
            data_dir = os.path.abspath(data_dir)
        print(f"Loading raw data from {data_dir}")
        os.chdir(data_dir)

    rf = {}
    svm = {}
    for trait in traits:
        rf_file = f"{trait}_importance.csv"
        svm_file = f"processed_{trait}_svm_importance.csv"
        rf[trait] = pd.read_csv(rf_file, index_col=0)
        svm[trait] = pd.read_csv(svm_file, index_col=0)
        print(f"{trait} RF Importance Shape: {rf[trait].shape}")
        print(f"{trait} SVM Importance Shape: {svm[trait].shape}")
        # Load importance scores across all traits from RF
    rf['all'] = pd.read_csv("traits_all_importance.csv", index_col=0)

    # Load genotype data (SNPs)
    geno_file = "geno.csv"  # Replace with your dataset file
    geno_data = pd.read_csv(geno_file, index_col=0)  # Individual IDs as rows, SNPs as columns
    print(f"Genotype Data Shape: {geno_data.shape}")

    # Load phenotype data (trait information like stem_color, stem_strength, etc.)
    pheno_file = "pheno.csv"  # Replace with your dataset file
    pheno_data = pd.read_csv(pheno_file, index_col=0)  # Individuals x Traits
    print(f"Phenotype Data Shape: {pheno_data.shape}")

    # Align genotype and phenotype datasets (ensure they are aligned by individuals)
    common_samples = geno_data.index.intersection(pheno_data.index)
    geno_data = geno_data.loc[common_samples]
    pheno_data = pheno_data.loc[common_samples]
    print(f"Aligned Genotype Data: {geno_data.shape}")
    print(f"Aligned Phenotype Data: {pheno_data.shape}")

    if top_x:
        # Select the top SNPs based on importance
        selected_snps = set({})
        for trait in traits:
            top_rf = set(rf[trait].max(axis=1).nlargest(top_x).index)
            # top_svm[trait] = set(svm[trait].max(axis=1).nlargest(top_x).index)
            print(f'top_rf for {trait}', len(top_rf))
            # Combine all selected SNPs into a single set
            selected_snps |= top_rf
        print(f"Number of SNPs selected for pruning: {len(selected_snps)}")
        # Prune genotype matrix to keep only the selected SNPs
    else:
        importance_file = traits[0] + "_snp_importance_scores.csv"
        all_snp_importances_df = pd.read_csv(importance_file, index_col=0)
        # Define which algorithm scores to use for this analysis
        # These names must match columns in all_snp_importances_df
        algorithms_to_consider = []
        if 'SelectKBest_Score' in all_snp_importances_df.columns: algorithms_to_consider.append('SelectKBest_Score')
        if 'RFE_Score' in all_snp_importances_df.columns: algorithms_to_consider.append('RFE_Score')
        if 'TreeImportance_Score' in all_snp_importances_df.columns: algorithms_to_consider.append('TreeImportance_Score')
        if 'PCA_PC1_Loading_Abs' in all_snp_importances_df.columns: algorithms_to_consider.append('PCA_PC1_Loading_Abs')
        if 'PCA_PC2_Loading_Abs' in all_snp_importances_df.columns: algorithms_to_consider.append('PCA_PC2_Loading_Abs')
        if 'SVD_Comp1_Abs' in all_snp_importances_df.columns: algorithms_to_consider.append('SVD_Comp1_Abs')

        # Add more (e.g. PCA_PC2_Loading_Abs, SVD_Comp1_Abs) if desired

        if algorithms_to_consider:
            top_n_per_algo = 1100 # Look at the top 20 SNPs from each specified method
            min_agreements = 5    # SNP must be in top N for at least this many methods

            print(f"Considering algorithms: {algorithms_to_consider}")
            print(f"Finding top {top_n_per_algo} SNPs from each, requiring agreement in at least {min_agreements} methods.")

            selected_snps = find_top_n_snps_from_selected_algorithms(
                all_snp_importances_df,
                top_n=top_n_per_algo,
                algorithm_cols_to_use=algorithms_to_consider,
                min_occurrence_threshold=min_agreements
            )
            print(f"\nSNPs appearing in the top {top_n_per_algo} of at least {min_agreements} specified algorithms ({len(selected_snps)} SNPs):")
                    
    pruned_geno_data = geno_data[list(selected_snps)]
    print(f"Pruned Genotype Data Shape: {pruned_geno_data.shape}")

    raw_data = pd.concat([pruned_geno_data, pheno_data[trait]], axis=1)

    
    print(f"Combined Raw Data Shape: {raw_data.shape}")
    os.chdir(current_dir)
    return raw_data

def load_cg(f_name, data_dir=None):
    current_dir = os.getcwd()
    if data_dir is not None:
        if data_dir.startswith('~/'):
            data_dir = os.path.expanduser(data_dir)
        elif data_dir.startswith('./'):
            data_dir = os.path.abspath(data_dir)
        print(f"Loading CG from {data_dir}")
        os.chdir(data_dir)

    # Load the graph object from the pickle file
    with open(f_name, "rb") as f:
        cg = pickle.load(f)

    # Check the type of the loaded object
    print(type(cg))  # Likely <class ''>
    print(cg.labels)
    os.chdir(current_dir)
    return cg
