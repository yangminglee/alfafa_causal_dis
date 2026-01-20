import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

import pickle
from sklearn.linear_model import LinearRegression

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.cit import chisq, fisherz
from causallearn.graph.Endpoint import Endpoint 


def get_selected_snps(results_file, P_VALUE_THRESHOLD_SUGGESTIVE, MAX_SNPS = 250, MIN_SNPS = 5):
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"找不到结果文件: {results_file}")
    else:
        print(f'load features from {results_file}')
    df = pd.read_csv(results_file)
    
    # 兼容不同的列名
    p_col = 'P_value' if 'P_value' in df.columns else df.columns[3]
    coef_col = 'Coef' if 'Coef' in df.columns else df.columns[1]
    
    df = df.sort_values(p_col)
    
    # 获取显著 SNP
    significant_df = df[df[p_col] < P_VALUE_THRESHOLD_SUGGESTIVE]
    n_sig = len(significant_df)
    
    print(f"--- SNP 筛选报告 ---")
    print(f"   - 阈值筛选 (P < {P_VALUE_THRESHOLD_SUGGESTIVE}): {n_sig} 个 SNP")
    
    if n_sig > MAX_SNPS:
        print(f"   - ⚠️ 数量过多 (> {MAX_SNPS})，截取 Top {MAX_SNPS}。")
        selected_df = df.head(MAX_SNPS)
    elif n_sig < MIN_SNPS:
        print(f"   - ⚠️ 数量过少 (< {MIN_SNPS})，强制选取 Top {MIN_SNPS}。")
        selected_df = df.head(MIN_SNPS)
    else:
        print(f"   - ✅ 数量适中，全部纳入。")
        selected_df = significant_df
        
    # 返回字典: {SNP: {'Coef': val, 'P_value': val}}
    return selected_df.set_index('SNP')[[coef_col, p_col]].to_dict('index')

def load_data(selected_snps, GENO_FILE, PHENO_FILE, TARGET_TRAIT):
    print("\n>>> 1. 加载并对齐原始数据...")
    if not os.path.exists(GENO_FILE) or not os.path.exists(PHENO_FILE):
        raise FileNotFoundError("找不到 geno.csv 或 pheno.csv")

    geno = pd.read_csv(GENO_FILE)
    pheno = pd.read_csv(PHENO_FILE)
    
    # 标准化索引
    for df in [geno, pheno]:
        if 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'Sample_ID'}, inplace=True)
            df.set_index('Sample_ID', inplace=True)
            
    # 对齐
    common = geno.index.intersection(pheno.index)
    if len(common) == 0:
        raise ValueError("没有找到重叠样本，请检查 Sample_ID")
        
    geno = geno.loc[common]
    pheno = pheno.loc[common]
    
    if TARGET_TRAIT not in pheno.columns:
        raise ValueError(f"在表型文件中找不到性状: {TARGET_TRAIT}")

    mask = ~pheno[TARGET_TRAIT].isna()
    
    # 构建分析矩阵 [SNP1, SNP2, ..., Trait]
    # 确保 Trait 在最后一列
    valid_snps = [s for s in selected_snps if s in geno.columns]
    if len(valid_snps) < len(selected_snps):
        print(f"⚠️ 警告: 有 {len(selected_snps) - len(valid_snps)} 个 SNP 在基因型文件中未找到，将被忽略。")
    
    data_df = pd.concat([geno.loc[mask, valid_snps], pheno.loc[mask, [TARGET_TRAIT]]], axis=1)
    
    return data_df

def causal_dis_pc(data_matrix, labels, pheno, bk = None, alpha=0.05):
    snp_pattern = '^chr.*\d$'
    bk = BackgroundKnowledge()
    assert(pheno in labels), f"Pattern {p} not found in labels"
    bk.add_forbidden_by_pattern(pheno, snp_pattern)
    cg = pc(data_matrix, node_names=labels, alpha=0.05, background_knowledge=bk)  # Causal graph search with 95% confidence
    print(f'number of nodes {(cg.G.get_num_nodes())}')
    print(f'number of edges {(cg.G.get_num_edges())}')
    print(f'number of parents of {cg.G.nodes[len(labels)-1].name} is {len(cg.G.get_parents(cg.G.nodes[len(labels)-1]))}')
    return cg


def parse_args():
    parser = argparse.ArgumentParser(description="Causal Learning with PC")
    parser.add_argument("--trait", type=str, default = 'stem_color', help="Target phenotype column name")
    parser.add_argument("--output", type=str, default = None, help="Output filename")
    parser.add_argument("--p_value", type=float, default=0.05)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    TARGET_TRAIT = args.trait
    
    DATA_DIR = os.path.expanduser('~/data/plant/alfalfa')
    GENO_FILE = os.path.join(DATA_DIR, 'geno.csv')
    PHENO_FILE = os.path.join(DATA_DIR, 'pheno.csv')
    N_PCS = 10
    PCA_FILE = os.path.join(DATA_DIR, 'pca_scores_'+str(N_PCS)+'.csv')
    TOP_RESULTS_FILE = os.path.join(DATA_DIR, 'feature_'+TARGET_TRAIT+'_100it.csv')

    # 筛选策略参数
    # 严格阈值约为 0.05 / 2434 ≈ 2e-5
    P_VALUE_THRESHOLD_SUGGESTIVE = args.p_value
    MIN_SNPS = 5              # 最少分析多少个 (保底)
    MAX_SNPS = 250             # 最多分析多少个 (防止图太乱/跑不动)

    try:
        # 1. 动态筛选
        snps_dict = get_selected_snps(TOP_RESULTS_FILE, P_VALUE_THRESHOLD_SUGGESTIVE)
        selected_snps = list(snps_dict.keys())
        # 2. 加载数据
        df = load_data(selected_snps, GENO_FILE, PHENO_FILE, TARGET_TRAIT)
        # 3. 运行 PC 算法
        labels = df.columns.tolist()
        data_matrix = df.values  # Combined SNPs and phenotypes
        cg = causal_dis_pc(data_matrix, labels, TARGET_TRAIT, bk = None, alpha=0.05)
    except Exception as e:
        print(f"❌ Error: {e}")

    if args.output is None:
        f_name = 'cg_'+str(TARGET_TRAIT)+'_'+str(len(selected_snps))
    else:
        f_name = args.output
    if False:
        with open(f_name+'.pkl', "rb") as f:
            cg =  pickle.load(f)
            print(f'load cg from {f_name}')
    if True:
        from util_cg import cg2nxgraph, save_nxgraph, cg_trim_ancester, cg_quantify_linear
        nx_graph = cg2nxgraph(cg)
        #load_causal_graph(f_name+'.graphml')
        save_nxgraph(nx_graph, f_name+'.graphml')
        nx_graph = cg_trim_ancester(nx_graph, TARGET_TRAIT)
        nx_graph = cg_quantify_linear(nx_graph, df, TARGET_TRAIT)
        save_nxgraph(nx_graph, f_name+'_sub.graphml')
        #build_and_save_graph(cg, labels, TARGET_TRAIT, snps_dict,
        #                     os.path.join(DATA_DIR, f_name))
    else:
        with open(f_name+'.pkl', "wb") as f:
            pickle.dump(cg, f)
            print(f'cg saved to {f_name}')
    if False:
        cg.draw_pydot_graph(labels)  # Display graphical representation
        plt.title(f"Causal Graph for {str(TARGET_TRAIT)}")
        plt.show()
        plt.savefig(f_name+".png")  # Saves the graph as a PNG file

    



        
