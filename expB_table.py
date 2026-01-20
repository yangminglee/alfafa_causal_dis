import pandas as pd
import networkx as nx
import numpy as np
import os
import re

def to_snake_case(name):
    """将 'Stem Color' 转换为 'stem_color' 以匹配文件名"""
    return name.lower().replace(" ", "_")

def load_graph(filename):
    """加载 GraphML 并清洗"""
    if not os.path.exists(filename):
        print(f"⚠️ 警告: 找不到图文件 {filename}")
        return None
    G = nx.read_graphml(filename)
    # 类型转换
    for u, v, data in G.edges(data=True):
        if 'weight' in data:
            try: data['weight'] = float(data['weight'])
            except: pass
    return G

def load_stats(trait_snake, data_dir):
    """加载对应的 GWAS 统计文件 (feature_trait_100it.csv)"""
    # 假设文件名格式为 feature_stem_color_100it.csv
    filename = os.path.join(data_dir, f"feature_{trait_snake}_100it.csv")
    
    if not os.path.exists(filename):
        # 尝试备用命名
        filename = os.path.join(data_dir, f"feature_{trait_snake}.csv")
        if not os.path.exists(filename):
            print(f"⚠️ 警告: 找不到统计文件 {filename}，表中相关列将为空。")
            return pd.DataFrame()
            
    df = pd.read_csv(filename)
    # 统一列名
    cols = df.columns
    try:
        p_col = next(c for c in cols if 'P_value' in c or 'p_val' in c)
        coef_col = next(c for c in cols if 'Coef' in c or 'beta' in c)
        return df.set_index('SNP')[[p_col, coef_col]]
    except:
        print(f"⚠️ 统计文件 {filename} 列名识别失败。")
        return pd.DataFrame()

def generate_latex_rows(trait_display, graph_file, data_dir):
    trait_snake = to_snake_case(trait_display)
    
    # 1. 加载数据
    G = load_graph(os.path.join(data_dir, graph_file))
    if G is None: return ""
    
    stats_df = load_stats(trait_snake, data_dir)
    
    # 2. 识别 Target Node (在图中找到对应的性状节点)
    # 有时图中的名字是 'stem_color' 而不是 'Stem Color'
    target_node = trait_snake 
    if target_node not in G.nodes():
        # 尝试模糊匹配
        matches = [n for n in G.nodes() if trait_snake in str(n).lower()]
        if matches: target_node = matches[0]
        else: return ""

    # 3. 提取 Parents 和 Hubs
    parents = [n for n in G.predecessors(target_node) if 'chr' in str(n)]
    
    try: ancestors = list(nx.ancestors(G, target_node))
    except: ancestors = []
    hubs = [n for n in ancestors if n not in parents and n != target_node and 'chr' in str(n)]
    
    rows_data = []

    # --- 处理 Parents (按权重排序) ---
    for node in parents:
        weight = G[node][target_node].get('weight', 0)
        pval = stats_df.loc[node].iloc[0] if node in stats_df.index and not stats_df.empty else np.nan
        coef = stats_df.loc[node].iloc[1] if node in stats_df.index and not stats_df.empty else np.nan
        
        rows_data.append({
            "Role": "Direct Parent",
            "SNP": node.replace("_", "\\_"),
            "P": pval,
            "Coef": coef,
            "Weight": abs(float(weight)),
            "Degree": G.out_degree(node)
        })
    
    # 排序 Parents
    parents_sorted = sorted([r for r in rows_data if r['Role']=="Direct Parent"], key=lambda x: x['Weight'], reverse=True)

    # --- 处理 Hubs (按出度排序) ---
    hubs_data = []
    for node in hubs:
        pval = stats_df.loc[node].iloc[0] if node in stats_df.index and not stats_df.empty else np.nan
        coef = stats_df.loc[node].iloc[1] if node in stats_df.index and not stats_df.empty else np.nan
        
        hubs_data.append({
            "Role": "Upstream Hub",
            "SNP": node.replace("_", "\\_"),
            "P": pval,
            "Coef": coef,
            "Weight": np.nan, # Hub 没有直接连接
            "Degree": G.out_degree(node)
        })
    
    # 排序 Hubs (Top 5)
    hubs_sorted = sorted(hubs_data, key=lambda x: x['Degree'], reverse=True)[:5]
    
    final_rows = parents_sorted + hubs_sorted
    
    # --- 生成 LaTeX 片段 ---
    latex_chunk = f"\\multicolumn{{6}}{{l}}{{\\textbf{{{trait_display}}}}} \\\\\n"
    latex_chunk += "\\midrule\n"
    
    for row in final_rows:
        p_str = f"{row['P']:.1e}" if not pd.isna(row['P']) else "-"
        coef_str = f"{row['Coef']:.3f}" if not pd.isna(row['Coef']) else "-"
        weight_str = f"{row['Weight']:.3f}" if not pd.isna(row['Weight']) else "-"
        
        role_display = "\\textit{Parent}" if row['Role'] == "Direct Parent" else "\\textit{Hub}"
        if row == parents_sorted[0] or (hubs_sorted and row == hubs_sorted[0]):
            pass # 保持 role_display
        else:
            role_display = "" # 同组不重复显示 Role，保持整洁 (可选)
            # 或者每次都显示:
            role_display = "\\textit{Parent}" if row['Role'] == "Direct Parent" else "\\textit{Hub}"

        latex_chunk += f"{role_display} & {row['SNP']} & {p_str} & {coef_str} & {weight_str} & {row['Degree']} \\\\\n"
    
    latex_chunk += "\\midrule\n"
    return latex_chunk

def main():
    DATA_DIR = os.path.expanduser('~/data/plant/alfalfa')
    
    TRAIT_CONFIG = {
        "Stem Color": "cg_stem_color_141_sub.graphml",
        "Stem Fill":  "cg_stem_fill_155_sub.graphml",
        "Stem Strength": "cg_stem_strength_151_sub.graphml",
        "Winter Injury": "cg_winter_injury_214_sub.graphml"
    }
    
    full_latex = r"""
\begin{table}[ht]
\centering
\caption{\textbf{Causal Drivers Across Four Traits.} Identifies Direct Parents (sorted by edge strength) and Top Upstream Hubs (sorted by centrality) for each phenotype.}
\label{tab:multi_trait_causal}
\resizebox{\textwidth}{!}{
\begin{tabular}{llcccc}
\toprule
\textbf{Role} & \textbf{SNP ID} & \textbf{GWAS $P$} & \textbf{Effect} & \textbf{Edge W.} & \textbf{Out-Deg.} \\
\midrule
"""
    
    print(">>> 开始生成多性状表格...")
    for trait_name, graph_file in TRAIT_CONFIG.items():
        print(f"   Processing {trait_name}...")
        full_latex += generate_latex_rows(trait_name, graph_file, DATA_DIR)
        
    full_latex += r"""\bottomrule
\end{tabular}
}
\end{table}
"""
    
    with open("multi_trait_table.tex", "w") as f:
        f.write(full_latex)
    
    print("\n✅ 表格已保存至 multi_trait_table.tex")
    print("请将该文件内容复制到你的论文 LaTeX 项目中。")

if __name__ == "__main__":
    main()