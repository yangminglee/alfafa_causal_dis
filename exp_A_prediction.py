import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

# ============================
# 1. 基础工具 (复用)
# ============================

def load_nxgraph(filename):
    """加载 GraphML 因果图"""
    if not os.path.exists(filename):
        print(f"⚠️ [Graph Error] 文件未找到: {filename}")
        return None
    try:
        graph = nx.read_graphml(filename)
        # 类型清洗
        for u, v, data in graph.edges(data=True):
            if 'weight' in data:
                try: data['weight'] = float(data['weight'])
                except ValueError: pass
        if not isinstance(graph, nx.DiGraph):
            graph = nx.DiGraph(graph)
        return graph
    except Exception as e:
        print(f"⚠️ [Graph Error] 读取失败 {filename}: {e}")
        return None

def load_data_custom(required_snps, GENO_FILE, PHENO_FILE, target_traits):
    """加载数据 (支持多性状)"""
    print(">>> [Data] 正在加载原始数据...")
    if not os.path.exists(GENO_FILE) or not os.path.exists(PHENO_FILE):
        raise FileNotFoundError("找不到 geno.csv 或 pheno.csv")

    geno = pd.read_csv(GENO_FILE)
    pheno = pd.read_csv(PHENO_FILE)
    
    # 标准化索引
    for df in [geno, pheno]:
        if 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'Sample_ID'}, inplace=True)
            df.set_index('Sample_ID', inplace=True)
        elif 'Sample_ID' in df.columns:
            df.set_index('Sample_ID', inplace=True)
            
    common = geno.index.intersection(pheno.index)
    if len(common) == 0:
        raise ValueError("没有找到重叠样本，请检查 Sample_ID")
    
    geno = geno.loc[common]
    pheno = pheno.loc[common]
    
    # 构建最终矩阵
    # 只需要加载存在的性状
    valid_traits = [t for t in target_traits if t in pheno.columns]
    
    # 检查 SNP
    valid_snps = [s for s in required_snps if s in geno.columns]
    
    # 使用第一个性状做 mask (假设性状缺失情况类似，或者取交集)
    if valid_traits:
        mask = ~pheno[valid_traits[0]].isna()
        data_df = pd.concat([geno.loc[mask, valid_snps], pheno.loc[mask, valid_traits]], axis=1)
    else:
        raise ValueError("Pheno 文件中没有找到任何目标性状！")
        
    return data_df

def identify_key_players(graph, target_trait, top_k_hubs=5):
    """从图中识别 Parents 和 Hubs"""
    # 模糊匹配 Target Node Name
    node_list = list(graph.nodes())
    target_node = target_trait
    if target_trait not in node_list:
        # 尝试寻找包含 trait 名字的节点
        matches = [n for n in node_list if target_trait in n]
        if matches:
            target_node = matches[0]
        else:
            print(f"   ⚠️ 图中未找到节点 '{target_trait}'，跳过。")
            return [], []

    # 1. Parents
    parents = list(graph.predecessors(target_node))
    parents = [n for n in parents if 'chr' in n or 'SNP' in n]

    # 2. Hubs
    try:
        ancestors = nx.ancestors(graph, target_node)
    except:
        ancestors = []
    
    candidate_hubs = [n for n in ancestors if n not in parents and ('chr' in n or 'SNP' in n)]
    out_degrees = {n: graph.out_degree(n) for n in candidate_hubs}
    hubs = sorted(out_degrees, key=out_degrees.get, reverse=True)[:top_k_hubs]
    
    return parents, hubs

# ============================
# 2. 实验 A: 核心逻辑
# ============================

def run_single_trait_experiment(df, target_trait, parents, hubs, random_snps):
    """为一个性状运行预测竞赛"""
    results = {}
    model = LinearRegression()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    teams = {
        'Parents': parents,
        'Hubs': hubs,
        'Random': random_snps
    }
    
    y = df[target_trait]
    
    for team, feats in teams.items():
        if not feats:
            results[team] = 0
            continue
        valid = [f for f in feats if f in df.columns]
        if not valid:
            results[team] = 0
            continue
            
        X = df[valid]
        scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        results[team] = np.mean(scores)
        
    return results

def plot_side_by_side(all_results):
    """
    画图: 2x2 网格，展示 4 个性状的结果
    all_results: dict { 'trait_name': {'Parents': 0.15, ...}, ... }
    """
    traits = list(all_results.keys())
    n_traits = len(traits)
    
    # 动态布局：如果是 4 个就 2x2，如果是 3 个就 1x3 等
    cols = 2 if n_traits > 1 else 1
    rows = (n_traits + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows), sharey=True)
    axes = axes.flatten() if n_traits > 1 else [axes]
    
    colors = ['#d62728', '#1f77b4', 'grey'] # Red, Blue, Grey
    
    print("\n>>> [Plotting] 生成对比图...")
    
    for i, trait in enumerate(traits):
        ax = axes[i]
        res = all_results[trait]
        
        names = list(res.keys())
        values = list(res.values())
        
        bars = ax.bar(names, values, color=colors, alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title(f"Target: {trait}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Predictive R2" if i % cols == 0 else "")
        
        # 在柱子上标数值
        for bar in bars:
            height = bar.get_height()
            label = f"{height:.3f}" if height != 0 else "0"
            ax.text(bar.get_x() + bar.get_width()/2., height, label,
                    ha='center', va='bottom' if height>0 else 'top', fontsize=9)
                    
        # 调整 Y 轴范围，让图好看点 (至少从 -0.05 到 max)
        current_ylim = ax.get_ylim()
        ax.set_ylim(min(-0.05, current_ylim[0]), max(0.2, current_ylim[1] * 1.1))

    # 隐藏多余的子图 (如果有的话)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Experiment A: Causal Hierarchy Validation Across Traits", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig("ExpA_SideBySide_Result.png", dpi=300)
    print("✅ 最终图表已保存: ExpA_SideBySide_Result.png")

def save_individual_plots(all_results, output_dir="."):
    """
    Saves separate figures for each trait in multiple formats (eps, pdf, pgf, png).
    """
    formats = ['eps', 'pdf', 'pgf', 'png']
    colors = ['#d62728', '#1f77b4', 'grey'] # Red (Parents), Blue (Hubs), Grey (Random)
    
    # Ensure output directory exists
    if output_dir != "." and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n>>> [Plotting] Saving individual figures...")

    for trait, res in all_results.items():
        # Create a new figure for each trait
        plt.figure(figsize=(4, 3)) # Standard single-column size for papers
        
        names = list(res.keys())
        values = list(res.values())
        
        # Plot bars
        bars = plt.bar(names, values, color=colors, alpha=0.8, width=0.6)
        plt.axhline(0, color='black', linewidth=0.8)
        
        # Styling
        plt.title(f"Trait: {trait}", fontsize=12)
        plt.ylabel("Predictive $R^2$ Score", fontsize=10)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            # Position text slightly above or below bar depending on value
            xy_pos = (bar.get_x() + bar.get_width() / 2, height)
            xy_text = (0, 3) if height >= 0 else (0, -12)
            
            label = f"{height:.3f}" if abs(height) > 0.001 else "0"
            
            plt.annotate(label, xy=xy_pos, xytext=xy_text,
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=9)

        # Adjust layout to prevent cutting off labels
        plt.tight_layout()

        # Save loop for multiple formats
        for fmt in formats:
            filename = f"ExpA_Result_{trait}.{fmt}"
            filepath = os.path.join(output_dir, filename)
            try:
                plt.savefig(filepath, dpi=300, format=fmt, bbox_inches='tight')
            except Exception as e:
                print(f"   ⚠️ Failed to save {fmt} for {trait}: {e}")
        
        print(f"   ✅ Saved plots for {trait} (EPS, PDF, PGF, PNG)")
        
        # Close the figure to free memory
        plt.close()

if __name__ == "__main__":
    # --- 配置区域 ---
    DATA_DIR = os.path.expanduser('~/data/plant/alfalfa')
    GENO_FILE = os.path.join(DATA_DIR, 'geno.csv')
    PHENO_FILE = os.path.join(DATA_DIR, 'pheno.csv')
    
    # 定义 4 个性状及其对应的 GraphML 文件路径
    # 确保这些文件真实存在
    TRAIT_CONFIG = {
        "Stem Color": "cg_stem_color_141_sub.graphml",
        "Stem Fill":  "cg_stem_fill_155_sub.graphml",
        "Stem Strength":      "cg_stem_strength_151_sub.graphml",
        "Winter Injury":     "cg_winter_injury_214_sub.graphml"
    }
    
    # --- 执行 ---
    try:
        # 1. 预扫描所有需要的 SNP (避免反复加载大文件)
        # 我们需要先读图，找出所有的 parents 和 hubs
        all_required_snps = set()
        trait_key_players = {} # 缓存: {trait: (parents, hubs)}
        
        print(">>> [Pre-scan] 扫描所有图结构...")
        for trait, graph_file in TRAIT_CONFIG.items():
            full_path = os.path.join(DATA_DIR, graph_file) if not os.path.isabs(graph_file) else graph_file
            # 如果文件不在 DATA_DIR，尝试当前目录
            if not os.path.exists(full_path):
                full_path = graph_file 
            
            G = load_nxgraph(full_path)
            if G:
                parents, hubs = identify_key_players(G, trait)
                if parents:
                    trait_key_players[trait] = (parents, hubs)
                    all_required_snps.update(parents)
                    all_required_snps.update(hubs)
                else:
                    print(f"⚠️ 警告: {trait} 的图中没有找到 Parents。")
            else:
                print(f"⚠️ 跳过 {trait} (加载失败)")

        # 2. 准备随机 SNP (作为通用的 Random Control)
        # 简单起见，我们为所有性状使用同一个 Random Pool，数量取平均 Parent 数
        if trait_key_players:
            avg_n_parents = int(np.mean([len(p) for p, h in trait_key_players.values()]))
            geno_header = pd.read_csv(GENO_FILE, nrows=0)
            all_snps_in_geno = [c for c in geno_header.columns if 'chr' in c or 'SNP' in c]
            
            # 排除掉所有关键 SNP
            pool = list(set(all_snps_in_geno) - all_required_snps)
            random_snps = np.random.choice(pool, size=avg_n_parents, replace=False).tolist()
            all_required_snps.update(random_snps)
            
            print(f"   [Control] 生成了 {len(random_snps)} 个随机对照 SNP")

            # 3. 一次性加载数据
            target_traits_list = list(TRAIT_CONFIG.keys())
            full_df = load_data_custom(list(all_required_snps), GENO_FILE, PHENO_FILE, target_traits_list)
            
            # 4. 循环运行实验
            all_results = {}
            print("\n>>> [Experiment] 开始多性状预测竞赛...")
            
            for trait, (parents, hubs) in trait_key_players.items():
                print(f"   Processing: {trait} (Parents={len(parents)}, Hubs={len(hubs)})")
                
                # 为该性状运行回归
                res = run_single_trait_experiment(full_df, trait, parents, hubs, random_snps)
                all_results[trait] = res
                
                # 打印单行结果
                print(f"      -> R2: Parents={res['Parents']:.4f} | Hubs={res['Hubs']:.4f} | Random={res['Random']:.4f}")

            # 5. 画图
            if all_results:
                # plot_side_by_side(all_results)
                save_individual_plots(all_results)
            else:
                print("❌ 没有产生任何结果，无法画图。")
        else:
            print("❌ 没有成功解析任何图结构。")

    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()