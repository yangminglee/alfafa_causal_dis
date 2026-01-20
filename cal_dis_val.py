import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import pickle

# 机器学习库 (用于验证)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error

# Causal Learn
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

# ==========================================
# 1. 你的原始工具函数 (保持不变)
# ==========================================

def get_selected_snps(results_file, P_VALUE_THRESHOLD_SUGGESTIVE, MAX_SNPS=250, MIN_SNPS=5):
    """
    加载 GWAS 统计结果，筛选 Top SNP，同时返回 Rejected SNP 用于做 Negative Control
    """
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"找不到结果文件: {results_file}")
    
    df = pd.read_csv(results_file)
    p_col = 'P_value' if 'P_value' in df.columns else df.columns[3]
    coef_col = 'Coef' if 'Coef' in df.columns else df.columns[1]
    
    df = df.sort_values(p_col)
    
    # 获取显著 SNP
    significant_df = df[df[p_col] < P_VALUE_THRESHOLD_SUGGESTIVE]
    n_sig = len(significant_df)
    
    print(f"--- SNP 筛选报告 ---")
    print(f"   - 阈值筛选 (P < {P_VALUE_THRESHOLD_SUGGESTIVE}): {n_sig} 个 SNP")
    
    rejected_snps = [] # 用于负对照实验

    if n_sig > MAX_SNPS:
        print(f"   - ⚠️ 数量过多 (> {MAX_SNPS})，截取 Top {MAX_SNPS}。")
        selected_df = df.head(MAX_SNPS)
        rejected_df = df.iloc[MAX_SNPS:] # 其余的作为 rejected
    elif n_sig < MIN_SNPS:
        print(f"   - ⚠️ 数量过少 (< {MIN_SNPS})，强制选取 Top {MIN_SNPS}。")
        selected_df = df.head(MIN_SNPS)
        rejected_df = df.iloc[MIN_SNPS:]
    else:
        print(f"   - ✅ 数量适中，全部纳入。")
        selected_df = significant_df
        rejected_df = df[~df['SNP'].isin(selected_df['SNP'])] # 非显著的作为 rejected

    # 获取 Rejected SNPs 列表 (取 P 值最大的那些作为纯噪音)
    rejected_snps = rejected_df.sort_values(p_col, ascending=False).head(50)['SNP'].tolist()

    return selected_df.set_index('SNP')[[coef_col, p_col]].to_dict('index'), rejected_snps

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
    
    valid_snps = [s for s in selected_snps if s in geno.columns]
    if len(valid_snps) < len(selected_snps):
        print(f"⚠️ 警告: 有 {len(selected_snps) - len(valid_snps)} 个 SNP 在基因型文件中未找到。")
    
    # 确保 Trait 在最后一列
    data_df = pd.concat([geno.loc[mask, valid_snps], pheno.loc[mask, [TARGET_TRAIT]]], axis=1)
    
    return data_df, geno # 返回 geno 是为了后续方便提取 rejected snps 的数据

def causal_dis_pc(data_matrix, labels, target_trait, alpha=0.05, verbose=True):
    snp_pattern = '^chr.*\d$'
    bk = BackgroundKnowledge()
    # 约束：Trait 不能指向 SNP (Forbidden: Trait -> SNP)
    bk.add_forbidden_by_pattern(target_trait, snp_pattern)
    
    # 运行 PC
    cg = pc(data_matrix, node_names=labels, alpha=alpha, background_knowledge=bk, verbose=verbose)
    
    if verbose:
        print(f'Nodes: {(cg.G.get_num_nodes())}, Edges: {(cg.G.get_num_edges())}')
    return cg

# ==========================================
# 2. 新增：验证实验模块
# ==========================================

def experiment_1_stability_heatmap(data_df, labels, target_trait, n_bootstraps=50):
    """
    [鲁棒性验证] 通过 Bootstrap 重采样，绘制边的稳定性热图
    """
    print(f"\n>>> [Exp 1] 正在运行稳定性分析 (N={n_bootstraps})...")
    stability_matrix = pd.DataFrame(0, index=labels, columns=labels)
    n_samples = len(data_df)
    
    for i in range(n_bootstraps):
        # 1. 重采样
        sample_df = resample(data_df, n_samples=n_samples, random_state=i)
        
        # 2. 运行 PC (静默模式)
        try:
            cg = causal_dis_pc(sample_df.values, labels, target_trait, alpha=0.05, verbose=False)
            
            # 3. 提取边 (转化为邻接矩阵计数)
            # causal-learn 的 graph 对象比较复杂，这里简化提取
            nodes = cg.G.get_nodes()
            for edge in cg.G.get_graph_edges():
                # node1 -> node2 or node1 -- node2 or node1 <-> node2
                n1 = edge.get_node1().get_name()
                n2 = edge.get_node2().get_name()
                
                # 记录边的存在 (不区分方向，先只看骨架稳定性，或者根据 Endpoint 区分)
                # 这里为了简单展示“关联的稳定性”，我们双向都加1，或者只记录 Directed
                # 如果要严格区分方向，需要检查 edge.get_endpoint1()
                
                stability_matrix.loc[n1, n2] += 1
                # 暂时视为无向骨架稳定性
                if n1 != n2:
                    stability_matrix.loc[n2, n1] += 1
        except Exception as e:
            pass # 忽略某次计算失败

    # 归一化
    stability_prob = stability_matrix / n_bootstraps
    
    # 绘图: 只看与 Target Trait 相关的 Top 特征
    plt.figure(figsize=(10, 8))
    # 为了图表清晰，只画 Trait 和它的 Top 15 个邻居
    trait_neighbors = stability_prob[target_trait].sort_values(ascending=False).head(15).index
    subset_matrix = stability_prob.loc[trait_neighbors, trait_neighbors]
    
    sns.heatmap(subset_matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.title(f'Causal Edge Stability (Target: {target_trait})')
    plt.tight_layout()
    plt.savefig(f"exp1_stability_{target_trait}.png")
    print(f"✅ 稳定性热图已保存至 exp1_stability_{target_trait}.png")
    return stability_prob

def experiment_2_predictive_validation(data_df, target_trait, causal_parents, full_geno_df, rejected_snps):
    """
    [有效性验证] 对比 Causal Feature vs All Features vs Rejected Features 的预测能力
    """
    print(f"\n>>> [Exp 2 & 3] 正在运行预测能力对比与负对照实验...")
    
    y = data_df[target_trait]
    
    # 1. Causal Set (来自图的 Parents/Neighbors)
    # 如果没找到 Parents，就用整个 Selected Set (降级方案)
    if len(causal_parents) == 0:
        print("   ⚠️ 警告: 因果图中未发现指向 Target 的 Parent，使用所有 Selected SNPs 代替。")
        X_causal = data_df.drop(columns=[target_trait])
    else:
        valid_parents = [p for p in causal_parents if p in data_df.columns]
        X_causal = data_df[valid_parents]

    # 2. Random/Rejected Set (负对照)
    # 从原始 geno 文件中提取 rejected snps 的数据
    # 需要对齐索引
    common_idx = data_df.index
    valid_rejected = [s for s in rejected_snps if s in full_geno_df.columns]
    # 确保数量和 Causal Set 差不多，公平比较
    n_feats = X_causal.shape[1]
    if len(valid_rejected) > n_feats:
        valid_rejected = valid_rejected[:n_feats]
    
    X_rejected = full_geno_df.loc[common_idx, valid_rejected]

    # 3. Full Model (Baseline) - 避免维度灾难，这里可能只用 Selected Set 作为 Baseline
    X_full = data_df.drop(columns=[target_trait])

    # 划分数据集
    # 假设是回归问题 (Phenotype通常是连续的)，如果是分类请改为 Classifier
    model = RandomForestRegressor(n_estimators=100, random_state=42) 
    
    results = {}
    
    for name, X_data in [("Causal Graph", X_causal), ("Negative Control", X_rejected), ("Full Selected", X_full)]:
        if X_data.shape[1] == 0:
            results[name] = 0
            continue
            
        X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred) # 或者 accuracy_score
        results[name] = score
        print(f"   - {name} (N_feats={X_data.shape[1]}): R2 = {score:.4f}")

    # 绘图
    plt.figure(figsize=(6, 5))
    plt.bar(results.keys(), results.values(), color=['#d62728', '#7f7f7f', '#1f77b4'])
    plt.ylabel('Predictive R2 Score')
    plt.title('Validation: Causal Graph vs Baselines')
    plt.savefig(f"exp2_prediction_{target_trait}.png")
    print(f"✅ 预测对比图已保存至 exp2_prediction_{target_trait}.png")

# ==========================================
# 3. 主程序
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Causal Learning with PC & Validation")
    parser.add_argument("--trait", type=str, default='stem_color', help="Target phenotype")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--p_value", type=float, default=2e-5) # 默认调严一点
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    TARGET_TRAIT = args.trait
    
    # 请修改为你的实际路径
    DATA_DIR = os.path.expanduser('~/data/plant/alfalfa')
    GENO_FILE = os.path.join(DATA_DIR, 'geno.csv')
    PHENO_FILE = os.path.join(DATA_DIR, 'pheno.csv')
    TOP_RESULTS_FILE = os.path.join(DATA_DIR, 'feature_'+TARGET_TRAIT+'_100it.csv') # 假设存在

    P_VAL = args.p_value

    try:
        # --- Step 1: 特征选择与数据加载 ---
        # 这一步返回 selected_snps 用于构建图，rejected_snps 用于负对照
        snps_dict, rejected_snps = get_selected_snps(TOP_RESULTS_FILE, P_VAL)
        selected_snps = list(snps_dict.keys())
        
        # 加载数据 (data_df 用于因果图, full_geno 用于提取 rejected 数据)
        data_df, full_geno_df = load_data(selected_snps, GENO_FILE, PHENO_FILE, TARGET_TRAIT)
        
        # --- Step 2: 构建主因果图 (Original PC) ---
        print("\n>>> 2. 构建因果图 (PC Algorithm)...")
        labels = data_df.columns.tolist()
        cg = causal_dis_pc(data_df.values, labels, TARGET_TRAIT, alpha=0.05)
        
        # 寻找 Target 的 Parents (直接因果)
        # 注意: causallearn 的 index 是从 0 开始的
        target_idx = labels.index(TARGET_TRAIT)
        target_node = cg.G.get_nodes()[target_idx]
        parents = cg.G.get_parents(target_node)
        parent_names = [p.get_name() for p in parents]
        print(f"   因果父节点 (Direct Causes of {TARGET_TRAIT}): {parent_names}")
        
        # 保存主结果 (兼容你的 util_cg)
        f_name = args.output if args.output else f'cg_{TARGET_TRAIT}_{len(selected_snps)}'
        with open(f_name+'.pkl', "wb") as f:
            pickle.dump(cg, f)
            
        # --- Step 3: 运行验证实验 (Validation) ---
        # 实验 1: 稳定性热图
        experiment_1_stability_heatmap(data_df, labels, TARGET_TRAIT, n_bootstraps=50)
        
        # 实验 2 & 3: 预测能力与负对照
        experiment_2_predictive_validation(data_df, TARGET_TRAIT, parent_names, full_geno_df, rejected_snps)

        print("\n🎉 全部流程结束！结果已保存。")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()