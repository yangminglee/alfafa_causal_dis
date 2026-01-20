import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression, mutual_info_regression, RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from collections import Counter
import os # For saving CSV

# --- Pre-filtering Functions ---
def filter_low_variance_snps(snp_df, threshold=0.04):
    """
    Removes SNPs with variance below a certain threshold.
    """
    if snp_df.empty:
        return pd.DataFrame(), VarianceThreshold(threshold=threshold)
    selector = VarianceThreshold(threshold=threshold)
    snp_values_filtered = selector.fit_transform(snp_df)
    snp_df_filtered = pd.DataFrame(snp_values_filtered, columns=snp_df.columns[selector.get_support()], index=snp_df.index)
    return snp_df_filtered, selector

def filter_collinear_snps(snp_df, threshold=0.95):
    """
    Removes highly correlated (collinear) SNPs. Keeps the first SNP in a highly correlated pair.
    """
    if snp_df.empty or snp_df.shape[1] < 2:
        return snp_df.copy()

    corr_matrix = snp_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for column in upper.columns:
        if column not in to_drop:
            correlated_features = upper[upper[column] > threshold].index
            for feature in correlated_features:
                if feature not in to_drop:
                    to_drop.add(feature)

    snp_df_filtered = snp_df.drop(columns=list(to_drop))
    if to_drop:
        print(f"    Dropped {len(to_drop)} collinear SNPs. Examples: {list(to_drop)[:5]}...")
    return snp_df_filtered

# --- Dimensionality Reduction & Feature Selection Functions (condensed for brevity) ---
def reduce_with_pca(df, n_components=2, standardize=True, random_state=42):
    X = df.copy(); pca_instance = PCA(n_components=n_components, random_state=random_state)
    if X.empty: return pd.DataFrame(), pca_instance
    if standardize: X_scaled = StandardScaler().fit_transform(X)
    else: X_scaled = X.values
    actual_n_components = n_components
    if isinstance(n_components, int) and n_components > X_scaled.shape[1]: actual_n_components = X_scaled.shape[1]
    elif isinstance(n_components, float) and n_components > 1.0: actual_n_components = min(int(n_components), X_scaled.shape[1])
    if actual_n_components <= 0 and X_scaled.shape[1] > 0 : actual_n_components = 1
    elif X_scaled.shape[1] == 0: return pd.DataFrame(index=df.index), PCA(n_components=1, random_state=random_state)

    pca_instance.set_params(n_components=actual_n_components)
    try:
        principal_components = pca_instance.fit_transform(X_scaled)
        pc_cols = [f'PC{i+1}' for i in range(principal_components.shape[1])]
        return pd.DataFrame(data=principal_components, columns=pc_cols, index=df.index), pca_instance
    except ValueError:
        if X_scaled.shape[0] < actual_n_components and X_scaled.shape[0] > 0:
            actual_n_components = X_scaled.shape[0]
            pca_instance.set_params(n_components=actual_n_components)
            principal_components = pca_instance.fit_transform(X_scaled)
            pc_cols = [f'PC{i+1}' for i in range(principal_components.shape[1])]
            return pd.DataFrame(data=principal_components, columns=pc_cols, index=df.index), pca_instance
        return pd.DataFrame(index=df.index), pca_instance

def reduce_with_truncated_svd(df, n_components=2, random_state=42):
    X = df.values; svd_instance = TruncatedSVD(n_components=n_components, random_state=random_state)
    if X.shape[1] == 0: return pd.DataFrame(index=df.index), svd_instance
    actual_n_components = n_components
    if actual_n_components >= X.shape[1]: actual_n_components = max(0, X.shape[1] - 1)
    if actual_n_components <= 0 and X.shape[1] > 0: actual_n_components = 1
    elif X.shape[1] == 0: return pd.DataFrame(index=df.index), TruncatedSVD(n_components=1, random_state=random_state)

    svd_instance.set_params(n_components=actual_n_components)
    transformed_components = svd_instance.fit_transform(X)
    svd_cols = [f'SVD_Component{i+1}' for i in range(transformed_components.shape[1])]
    return pd.DataFrame(data=transformed_components, columns=svd_cols, index=df.index), svd_instance

def select_k_best_features(df, target_series, k=10, problem_type='regression'):
    selector_instance = SelectKBest(k=k)
    if df.empty or df.shape[1] == 0: return pd.DataFrame(index=df.index), selector_instance, pd.Series(dtype=float)
    score_func = f_regression if problem_type == 'regression' else f_classif
    X, y = df.values, target_series.values
    actual_k = k if isinstance(k, str) and k == 'all' else min(k, X.shape[1])
    if actual_k <= 0: actual_k = 1 if X.shape[1] > 0 else 0
    if actual_k == 0: return pd.DataFrame(index=df.index), SelectKBest(score_func=score_func, k=1), pd.Series(dtype=float)

    selector_instance.set_params(score_func=score_func, k=actual_k)
    X_new = selector_instance.fit_transform(X, y)
    selected_feature_names = df.columns[selector_instance.get_support()]
    df_selected = pd.DataFrame(X_new, columns=selected_feature_names, index=df.index)
    feature_scores = pd.Series(selector_instance.scores_, index=df.columns).sort_values(ascending=False) if hasattr(selector_instance, 'scores_') else pd.Series(dtype=float)
    return df_selected, selector_instance, feature_scores

def select_with_rfe(df, target_series, n_features_to_select=10, problem_type='regression', random_state=42):
    estimator = LinearRegression() if problem_type == 'regression' else LogisticRegression(solver='liblinear', random_state=random_state, max_iter=200)
    rfe_instance = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
    if df.empty or df.shape[1] == 0: return pd.DataFrame(index=df.index), rfe_instance, pd.Series(dtype=float)
    X, y = df.values, target_series.values
    actual_n_features = min(n_features_to_select, X.shape[1])
    if actual_n_features <= 0: actual_n_features = 1 if X.shape[1] > 0 else 0
    if actual_n_features == 0: return pd.DataFrame(index=df.index), RFE(estimator=estimator, n_features_to_select=1), pd.Series(dtype=float)

    rfe_instance.set_params(n_features_to_select=actual_n_features)
    rfe_instance.fit(X, y)
    selected_feature_names = df.columns[rfe_instance.support_]
    df_rfe_selected = pd.DataFrame(X[:, rfe_instance.support_], columns=selected_feature_names, index=df.index)
    feature_ranking = pd.Series(rfe_instance.ranking_, index=df.columns).sort_values()
    return df_rfe_selected, rfe_instance, feature_ranking

def select_with_tree_importance(df, target_series, n_features_to_select=10, problem_type='regression', random_state=42):
    model_instance = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1) if problem_type == 'regression' \
        else RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    if df.empty or df.shape[1] == 0: return pd.DataFrame(index=df.index), pd.Series(dtype=float), model_instance
    X, y = df, target_series
    actual_n_features = min(n_features_to_select, X.shape[1])
    if actual_n_features <= 0: actual_n_features = 1 if X.shape[1] > 0 else 0
    if actual_n_features == 0: return pd.DataFrame(index=df.index), pd.Series(dtype=float), model_instance

    model_instance.fit(X, y)
    importances = model_instance.feature_importances_
    feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    top_features = feature_importances.head(actual_n_features).index.tolist()
    df_selected = df[top_features]
    return df_selected, feature_importances, model_instance

# --- New Function to Find Top N SNPs from Selected Algorithms ---
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

def filter_low_variance_snps(snp_df, threshold=0.04):
    """
    Removes SNPs with variance below a certain threshold.
    A common threshold for MAF > 0.01 (variance ~2*0.01*0.99 = 0.0198)
    A common threshold for MAF > 0.05 (variance ~2*0.05*0.95 = 0.095)
    Let's use a threshold that might correspond to MAF around 0.02-0.03.
    Variance for MAF=m is 2*m*(1-m) assuming HWE for biallelic SNPs coded 0,1,2.
    threshold=0.04 -> 2m(1-m)=0.04 -> m-m^2=0.02 -> m^2-m+0.02=0 -> m ~ 0.02 or 0.98

    Args:
        snp_df (pd.DataFrame): DataFrame of SNPs.
        threshold (float): Variance threshold. Features with variance below this will be removed.

    Returns:
        pd.DataFrame: Filtered SNP DataFrame.
        VarianceThreshold: Fitted VarianceThreshold object.
    """
    selector = VarianceThreshold(threshold=threshold)
    snp_values_filtered = selector.fit_transform(snp_df)
    snp_df_filtered = pd.DataFrame(snp_values_filtered, columns=snp_df.columns[selector.get_support()], index=snp_df.index)
    return snp_df_filtered, selector

def filter_collinear_snps(snp_df, threshold=0.95):
    """
    Removes highly correlated (collinear) SNPs.
    Keeps the first SNP in a highly correlated pair.

    Args:
        snp_df (pd.DataFrame): DataFrame of SNPs.
        threshold (float): Correlation threshold. If abs(correlation) > threshold, one SNP is removed.

    Returns:
        pd.DataFrame: Filtered SNP DataFrame.
    """
    corr_matrix = snp_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for column in upper.columns:
        if column not in to_drop: # Don't check columns already marked for removal
            correlated_features = upper[upper[column] > threshold].index
            for feature in correlated_features:
                if feature not in to_drop: # Ensure we only add if not already processed
                    to_drop.add(feature)

    snp_df_filtered = snp_df.drop(columns=list(to_drop))
    print(f"    Dropped {len(to_drop)} collinear SNPs: {list(to_drop)[:10]}...") # Print first 10 dropped
    return snp_df_filtered

def find_frequently_selected_snps(selected_features_by_method, min_occurrence_threshold=2):
    """
    Identifies SNPs that are selected by at least a minimum number of feature selection methods.

    Args:
        selected_features_by_method (dict): A dictionary where keys are method names (str)
                                            and values are lists or sets of selected SNP names.
                                            Example: {'KBest': ['SNP1', 'SNP2'], 'RFE': ['SNP2', 'SNP3']}
        min_occurrence_threshold (int): The minimum number of methods a SNP must be selected by
                                        to be included in the result.

    Returns:
        dict: A dictionary where keys are SNP names that meet the threshold,
              and values are the counts of how many methods selected them.
              Example: {'SNP2': 2}
    """
    if not selected_features_by_method:
        return {}

    all_selected_snps_flat_list = []
    for method_name, snp_list in selected_features_by_method.items():
        if isinstance(snp_list, (list, set, pd.Series)):
            all_selected_snps_flat_list.extend(list(snp_list))
        else:
            print(f"Warning: SNP list for method '{method_name}' is not a list or set. Skipping.")

    snp_counts = Counter(all_selected_snps_flat_list)

    frequent_snps = {snp: count for snp, count in snp_counts.items() if count >= min_occurrence_threshold}

    # Sort by count (descending) then by SNP name (ascending) for consistent output
    sorted_frequent_snps = dict(sorted(frequent_snps.items(), key=lambda item: (-item[1], item[0])))

    return sorted_frequent_snps

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Parse two input arguments: -n and -t.")

    # Define arguments with default values
    parser.add_argument("-t", type=int, default=0, help="An integer from [0,1,2,3])")
    parser.add_argument("--data_dir", type=str, default="~/data/crop/alfafa/", help="dir to data")

    # Parse arguments
    args = parser.parse_args()

    geno_file = 'geno.csv'
    geno_data = pd.read_csv(geno_file, header=0, index_col=0)
    print(f"Genotype Data Loaded: {geno_data.shape}")

    # Load the phenotype data
    pheno_file = "pheno.csv"
    pheno_data = pd.read_csv(pheno_file, header=0, index_col=0)
    print(f"Phenotype Data Loaded: {pheno_data.shape}")

    # ----------------------------
    # STEP 2: Align Genotype and Phenotype Data
    # ----------------------------
    # Filter genotype and phenotype data to include only common samples
    common_samples = geno_data.index.intersection(pheno_data.index)
    raw_snp_df = geno_data.loc[common_samples]
    pheno_data = pheno_data.loc[common_samples]
    TRAITS = ["stem_color", "stem_fill", "stem_strength", "winter_injury"]
    trait = TRAITS[args.t] # Example trait
    trait_series = pheno_data[trait]
    causal_snps= None
    print(f"Aligned Genotype Data: {raw_snp_df.shape}")
    print(f"Aligned Phenotype Data for trait {trait}: {trait_series.shape}")

    # --- Pre-filtering ---
    print(f"\n--- Pre-filtering ---")
    variance_thresh = 0.04
    snps_var_filtered, _ = filter_low_variance_snps(raw_snp_df, threshold=variance_thresh)
    print(f"Shape after low variance filter (threshold={variance_thresh}): {snps_var_filtered.shape}")

    correlation_thresh = 0.90
    features_df_filtered = filter_collinear_snps(snps_var_filtered, threshold=correlation_thresh)
    print(f"Shape after co-linearity filter (threshold={correlation_thresh}): {features_df_filtered.shape}")

    # --- Initialize DataFrame for SNP Importances ---
    # Use SNPs that passed filtering as the index
    if not features_df_filtered.empty:
        all_snp_importances_df = pd.DataFrame(index=features_df_filtered.columns)
    else:
        all_snp_importances_df = pd.DataFrame() # Will remain empty if no features

    if features_df_filtered.empty or features_df_filtered.shape[1] == 0:
        print("\nNo features remaining after pre-filtering. Halting detailed analysis.")
    else:
        print(f"\n--- Starting Dimensionality Reduction & Feature Selection with {features_df_filtered.shape[1]} features ---")

        # --- PCA ---
        n_pcs_to_analyze = min(5, features_df_filtered.shape[1] -1 if features_df_filtered.shape[1]>1 else 1)
        if n_pcs_to_analyze <=0 and features_df_filtered.shape[1] > 0: n_pcs_to_analyze = 1

        if n_pcs_to_analyze > 0:
            _, pca_model = reduce_with_pca(features_df_filtered, n_components=n_pcs_to_analyze)
            if hasattr(pca_model, 'components_'):
                print(f"\n1. PCA performed with {pca_model.n_components_} components.")
                for i in range(min(2, pca_model.n_components_)): # Save loadings for PC1, PC2
                    pc_loadings = pca_model.components_[i, :]
                    all_snp_importances_df[f'PCA_PC{i+1}_Loading_Abs'] = np.abs(pc_loadings)
            else:
                print("\n1. PCA model did not produce components.")

        # --- Truncated SVD ---
        n_svd_comps_to_analyze = min(5, features_df_filtered.shape[1] -1 if features_df_filtered.shape[1]>1 else 1)
        if n_svd_comps_to_analyze <=0 and features_df_filtered.shape[1] > 0: n_svd_comps_to_analyze = 1

        if n_svd_comps_to_analyze > 0:
            _, svd_model = reduce_with_truncated_svd(features_df_filtered, n_components=n_svd_comps_to_analyze)
            if hasattr(svd_model, 'components_'):
                print(f"\n2. TruncatedSVD performed with {svd_model.n_components} components.")
                for i in range(min(2, svd_model.n_components)): # Save for SVD_Comp1, SVD_Comp2
                    svd_comp_values = svd_model.components_[i, :]
                    all_snp_importances_df[f'SVD_Comp{i+1}_Abs'] = np.abs(svd_comp_values)
            else:
                print("\n2. SVD model did not produce components.")

        # --- Feature Selection Methods ---
        k_select = min(50, features_df_filtered.shape[1]) # Select more for better overlap analysis
        if k_select <=0 : k_select = 1 if features_df_filtered.shape[1] > 0 else 0

        if features_df_filtered.shape[1] > 0 and k_select > 0:
            print(f"\n--- Feature Selection (k={k_select}) ---")

            # SelectKBest
            _, _, kbest_scores = select_k_best_features(features_df_filtered, trait_series, k=k_select, problem_type='regression')
            if not kbest_scores.empty:
                all_snp_importances_df['SelectKBest_Score'] = kbest_scores
                print(f"\n3. SelectKBest (Regression) scores collected.")

            # RFE
            _, rfe_model, rfe_ranking = select_with_rfe(features_df_filtered, trait_series, n_features_to_select=k_select, problem_type='regression')
            if not rfe_ranking.empty:
                max_rank = rfe_ranking.max()
                all_snp_importances_df['RFE_Score'] = max_rank + 1 - rfe_ranking # Higher is better
                print(f"\n4. RFE (Regression) scores (derived from ranking) collected.")

            # Tree-based Importance
            _, tree_importances, _ = select_with_tree_importance(features_df_filtered, trait_series, n_features_to_select=k_select, problem_type='regression')
            if not tree_importances.empty:
                all_snp_importances_df['TreeImportance_Score'] = tree_importances
                print(f"\n5. Tree-based (Regression) importances collected.")

        # --- Save SNP Importances to CSV ---
        if not all_snp_importances_df.empty:
            csv_filename =trait + "_snp_importance_scores.csv"
            try:
                all_snp_importances_df.to_csv(csv_filename)
                print(f"\n--- SNP Importance Scores saved to {csv_filename} ---")
                print(all_snp_importances_df.head())
            except Exception as e:
                print(f"Error saving SNP importance scores to CSV: {e}")
