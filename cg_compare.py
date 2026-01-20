import pandas as pd
import numpy as np
import networkx as nx
import os
import re

# ==========================================
# 0. 基础组件 (PathRelaxer)
# ==========================================
class PathRelaxer:
    """处理 SNP 位置映射的辅助类"""
    def __init__(self, snp_nodes, window_size=5000):
        self.window_size = window_size
        self.snp_map = {} 
        self._build_locus_map(snp_nodes)
        
    def _parse_position(self, snp_name):
        try:
            parts = re.split(r'[_:\-]', str(snp_name))
            if len(parts) >= 2:
                pos = int(parts[-1])
                chrom = "_".join(parts[:-1])
                return chrom, pos
        except:
            pass
        return None, None

    def _build_locus_map(self, snp_nodes):
        snp_positions = []
        for snp in snp_nodes:
            chrom, pos = self._parse_position(snp)
            if chrom is not None and pos is not None:
                snp_positions.append({'snp': snp, 'chrom': chrom, 'pos': pos})
            else:
                self.snp_map[snp] = snp # 无法解析则映射回自身

        if not snp_positions:
            return

        df = pd.DataFrame(snp_positions)
        df = df.sort_values(by=['chrom', 'pos'])
        
        current_locus_id = 0
        for chrom, group in df.groupby('chrom'):
            last_pos = -999999
            for idx, row in group.iterrows():
                if row['pos'] - last_pos <= self.window_size:
                    df.at[idx, 'locus'] = f"{chrom}_Locus_{current_locus_id}"
                else:
                    current_locus_id += 1
                    df.at[idx, 'locus'] = f"{chrom}_Locus_{current_locus_id}"
                    last_pos = row['pos']
        
        for _, row in df.iterrows():
            self.snp_map[row['snp']] = row['locus']

    def get_locus(self, snp_name):
        return self.snp_map.get(snp_name, snp_name)

# ==========================================
# 1. 核心路径提取函数 (保持原样，不修改)
# ==========================================
def get_raw_paths_and_effects(G, target_node):
    paths_data = []
    try:
        ancestors = nx.ancestors(G, target_node)
    except Exception as e:
        return pd.DataFrame()

    if not ancestors:
        return pd.DataFrame()

    for node in ancestors:
        try:
            paths = list(nx.all_simple_paths(G, source=node, target=target_node))
        except:
            continue
            
        for path in paths:
            cumulative_effect = 1.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                w = G[u][v].get('weight', 1.0)
                cumulative_effect *= w
            
            genetic_path = path[:-1]
            path_str = " -> ".join(genetic_path)
            
            paths_data.append({
                'Path_String': path_str,
                'Source_Node': genetic_path[0],
                'Genetic_Path_Length': len(genetic_path),
                'Total_Effect': cumulative_effect,
                'Full_Path': path # 包含 Trait
            })
            
    return pd.DataFrame(paths_data)

# ==========================================
# 2. 辅助函数：保存单条路径为 GraphML
# ==========================================
def save_path_to_graphml(G_original, path_nodes, filename, description=""):
    """
    从原图中提取特定路径的节点和边，保存为新的 GraphML。
    """
    # 创建子图
    H = G_original.subgraph(path_nodes).copy()
    
    # 添加描述作为图的属性 (可选，GraphML支持)
    H.graph['description'] = description
    
    # 确保目录存在
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    try:
        nx.write_graphml(H, filename)
        # print(f"已保存路径图: {filename}")
    except Exception as e:
        print(f"保存路径图失败 {filename}: {e}")

# ==========================================
# 3. 主逻辑：综合对比函数
# ==========================================
def compare_two_causal_graphs(G1, trait1, G2, trait2, window_size=5000, output_dir="./comparison_results"):
    """
    综合对比两个因果图。
    满足需求：
    1. 统计重叠 SNP (允许位置误差) 及其对 Trait 的总影响。
    2. 提取并保存 4 条关键路径 (最长/最短/最重要/最不重要)。
    3. 统计包含相同 SNP 的所有路径对。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"=== 开始对比 {trait1} vs {trait2} (Window={window_size}) ===")
    
    # --- 步骤 0: 基础数据准备 ---
    # 获取原始路径数据
    df1 = get_raw_paths_and_effects(G1, trait1)
    df2 = get_raw_paths_and_effects(G2, trait2)
    
    if df1.empty or df2.empty:
        print("错误：其中一个图没有提取到有效路径。")
        return None, None
    else:
        print(f"提取到 {len(df1)} 条 {trait1} 路径，{len(df2)} 条 {trait2} 路径。")
    
    # 初始化 Relaxer
    all_snps = (set(G1.nodes()) | set(G2.nodes())) - {trait1, trait2}
    relaxer = PathRelaxer(list(all_snps), window_size=window_size)
    
    # 计算每个 SNP 对 Trait 的总效应 (Sum of effects of all paths starting from this SNP)
    # 注意：这里我们只统计作为 Source 的效应，或者你可以统计出现在路径中任何位置的效应
    # 为了简化且符合直觉，我们统计 "Source Node Total Effect"
    snp_effect_1 = df1.groupby('Source_Node')['Total_Effect'].sum().reset_index()
    snp_effect_2 = df2.groupby('Source_Node')['Total_Effect'].sum().reset_index()
    
    # 添加 Locus 列
    snp_effect_1['Locus'] = snp_effect_1['Source_Node'].apply(relaxer.get_locus)
    snp_effect_2['Locus'] = snp_effect_2['Source_Node'].apply(relaxer.get_locus)

    # =========================================================
    # 需求 1: 保留两个图中重叠的 SNP 名称及属性
    # =========================================================
    print("\n[分析 1: SNP 重叠分析]")
    
    # 按 Locus 合并
    merged_snps = pd.merge(
        snp_effect_1, snp_effect_2, 
        on='Locus', 
        how='inner', 
        suffixes=(f'_{trait1}', f'_{trait2}')
    )
    
    # 整理输出列
    cols = ['Source_Node_' + trait1, 'Source_Node_' + trait2, 'Locus', 'Total_Effect_' + trait1, 'Total_Effect_' + trait2]
    overlap_result = merged_snps[cols].copy()
    
    # 计算相关性类型
    overlap_result['Type'] = np.where(
        overlap_result[f'Total_Effect_{trait1}'] * overlap_result[f'Total_Effect_{trait2}'] > 0,
        'Synergy', 'Trade-off'
    )
    
    print(f"   - 发现 {len(overlap_result)} 个重叠 SNP (Locus级别)")
    overlap_csv = os.path.join(output_dir, "1_overlapping_snps.csv")
    overlap_result.to_csv(overlap_csv, index=False)
    print(f"   - 结果已保存: {overlap_csv}")

    # =========================================================
    # 需求 2: 提取并保存 4 条关键路径 (GraphML)
    # =========================================================
    print("\n[分析 2: 关键路径提取与保存]")
    
    def extract_and_save_key_paths(df, G_origin, trait_name):
        key_paths = {}
        
        # 1. 最长路径 (Length 最大)
        max_len = df['Genetic_Path_Length'].max()
        key_paths['Longest'] = df[df['Genetic_Path_Length'] == max_len].iloc[0]
        
        # 2. 最短路径 (Length 最小)
        min_len = df['Genetic_Path_Length'].min()
        key_paths['Shortest'] = df[df['Genetic_Path_Length'] == min_len].iloc[0]
        
        # 3. 最重要路径 (Abs Effect 最大)
        df['Abs_Effect'] = df['Total_Effect'].abs()
        key_paths['Most_Important'] = df.sort_values(by='Abs_Effect', ascending=False).iloc[0]
        
        # 4. 最不重要路径 (Abs Effect 最小但 > 0)
        # 排除 0 效应路径(如果有)
        df_nonzero = df[df['Abs_Effect'] > 1e-9]
        if not df_nonzero.empty:
            key_paths['Least_Important'] = df_nonzero.sort_values(by='Abs_Effect', ascending=True).iloc[0]
        else:
            key_paths['Least_Important'] = df.sort_values(by='Abs_Effect', ascending=True).iloc[0]
            
        # 保存
        saved_files = []
        for p_type, row in key_paths.items():
            fname = os.path.join(output_dir, f"path_{trait_name}_{p_type}.graphml")
            desc = f"{p_type} Path for {trait_name}: {row['Path_String']} (Effect={row['Total_Effect']:.4f})"
            save_path_to_graphml(G_origin, row['Full_Path'], fname, description=desc)
            saved_files.append(fname)
            
        return saved_files

    files1 = extract_and_save_key_paths(df1, G1, trait1)
    files2 = extract_and_save_key_paths(df2, G2, trait2)
    print(f"   - 已保存 {len(files1) + len(files2)} 个关键路径文件 (.graphml)")

    # =========================================================
    # 需求 3: 统计所有包含相同 SNP 的路径对
    # =========================================================
    print("\n[分析 3: 共享 SNP 的路径关联分析]")
    
    # 3.1 为每条路径建立 SNP Locus 集合
    # 我们不修改 get_raw_paths_and_effects，所以在这里后处理
    def get_path_locus_set(full_path_list):
        # 去掉 trait
        snps = full_path_list[:-1]
        # 转换为 locus 集合
        return set(relaxer.get_locus(s) for s in snps)

    df1['Locus_Set'] = df1['Full_Path'].apply(get_path_locus_set)
    df2['Locus_Set'] = df2['Full_Path'].apply(get_path_locus_set)
    
    # 3.2 暴力匹配? 不，效率太低。使用倒排索引 (Inverted Index)
    # Locus -> [Path_Index_List]
    locus_to_paths_1 = {}
    for idx, row in df1.iterrows():
        for locus in row['Locus_Set']:
            if locus not in locus_to_paths_1: locus_to_paths_1[locus] = []
            locus_to_paths_1[locus].append(idx)
            
    # 3.3 查找匹配
    shared_path_pairs = []
    
    # 遍历 df2 的每一条路径
    for idx2, row2 in df2.iterrows():
        matched_indices_1 = set()
        
        # 检查该路径中的每个 Locus 是否出现在 df1 中
        for locus in row2['Locus_Set']:
            if locus in locus_to_paths_1:
                # 记录所有包含该 locus 的 df1 路径索引
                matched_indices_1.update(locus_to_paths_1[locus])
        
        # 如果有匹配，记录下来
        for idx1 in matched_indices_1:
            row1 = df1.loc[idx1]
            
            # 找到具体共享了哪些 Locus (用于解释)
            common_loci = row1['Locus_Set'].intersection(row2['Locus_Set'])
            
            shared_path_pairs.append({
                f'Path_{trait1}': row1['Path_String'],
                f'Path_{trait2}': row2['Path_String'],
                f'Effect_{trait1}': row1['Total_Effect'],
                f'Effect_{trait2}': row2['Total_Effect'],
                'Shared_Loci_Count': len(common_loci),
                'Shared_Loci': ";".join(list(common_loci))
            })
            
    # 转换为 DataFrame
    df_shared = pd.DataFrame(shared_path_pairs)
    
    if not df_shared.empty:
        # 按共享程度和效应排序
        df_shared['Total_Strength'] = df_shared[f'Effect_{trait1}'].abs() + df_shared[f'Effect_{trait2}'].abs()
        df_shared = df_shared.sort_values(by=['Shared_Loci_Count', 'Total_Strength'], ascending=False)
        
        shared_csv = os.path.join(output_dir, "3_shared_snp_paths.csv")
        df_shared.to_csv(shared_csv, index=False)
        print(f"   - 发现 {len(df_shared)} 对包含共同 SNP 的路径")
        print(f"   - 结果已保存: {shared_csv}")
    else:
        print("   - 未发现任何包含共同 SNP 的路径对。")

    return overlap_result, df_shared

# ==========================================
# 4. 使用示例 (Copy paste to run)
# ==========================================
# 假设 G_color 和 G_fill 是你的 NetworkX 图
# compare_two_causal_graphs(G_color, 'Stem_Color', G_fill, 'Stem_Fill', window_size=5000)
def calculate_network_impact_no_overlap(G_origin, selected_paths_list, trait_node):
    """
    核心算法：计算一组路径对 Trait 的净影响，去除重叠。
    
    逻辑：
    1. 构建这些路径组成的子网 (Subgraph)。
    2. 找到子网中的“源头节点” (在子网内入度为0)。
    3. 计算这些源头节点在子网内对 Trait 的总效应并求和。
    """
    if not selected_paths_list:
        return 0.0
    
    # 1. 构建路径联合子图
    # 我们需要子图包含路径上的所有边和点
    nodes_in_subgraph = set()
    edges_in_subgraph = set()
    
    for path in selected_paths_list:
        # path 是完整路径 list: ['S1', 'S2', 'Trait']
        nodes_in_subgraph.update(path)
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edges_in_subgraph.add((u, v))
            
    # 从原图中提取这些边构建子图
    # 使用 DiGraph 而不是 subgraph，以确保我们只包含选定的边
    H = nx.DiGraph()
    for u, v in edges_in_subgraph:
        # 复制权重
        w = G_origin[u][v].get('weight', 0.0)
        H.add_edge(u, v, weight=w)
        
    # 2. 寻找局部源头 (Local Sources)
    # 即：在 H 中入度为 0 的节点 (Trait 除外)
    local_sources = [n for n in H.nodes() if H.in_degree(n) == 0 and n != trait_node]
    
    # 3. 计算净效应
    net_impact = 0.0
    
    for source in local_sources:
        # 计算该 source 在子图 H 中到 Trait 的总效应
        # 注意：必须在 H 中算，不能在 G 中算，否则会引入非共享路径
        try:
            paths = list(nx.all_simple_paths(H, source=source, target=trait_node))
            source_total_effect = 0.0
            for p in paths:
                eff = 1.0
                for i in range(len(p) - 1):
                    eff *= H[p[i]][p[i+1]]['weight']
                source_total_effect += eff
            
            net_impact += source_total_effect
            
        except:
            continue
            
    return net_impact

def analyze_shared_mechanism_impact(G1, trait1, G2, trait2, window_size=5000, output_dir="./comparison_results"):
    """
    修改后的需求 3 实现：
    1. 找到各自包含共享 SNP 的路径。
    2. 统计数量。
    3. 计算去重后的总影响 (Net Impact)。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"=== 分析共享机制的总影响 (De-overlapping) ===")
    
    # 1. 准备基础数据
    df1 = get_raw_paths_and_effects(G1, trait1)
    df2 = get_raw_paths_and_effects(G2, trait2)
    
    # 初始化映射器
    all_snps = (set(G1.nodes()) | set(G2.nodes())) - {trait1, trait2}
    relaxer = PathRelaxer(list(all_snps), window_size=window_size)
    
    # 2. 识别共享 SNP (Locus 级别)
    # 获取各自涉及的 Locus 集合
    def get_nodes_locus_set(graph, trait):
        nodes = set(graph.nodes()) - {trait}
        return {relaxer.get_locus(n) for n in nodes}
        
    loci_1 = get_nodes_locus_set(G1, trait1)
    loci_2 = get_nodes_locus_set(G2, trait2)
    
    # 交集
    shared_loci = loci_1.intersection(loci_2)
    print(f"   - 发现 {len(shared_loci)} 个共享 SNP Locus")
    
    if len(shared_loci) == 0:
        print("   - 无共享 SNP，分析结束。")
        return

    # 3. 筛选路径
    # 函数：判断一条路径是否包含共享 Locus
    def is_path_shared(full_path_list):
        # path nodes (exclude trait)
        snps = full_path_list[:-1]
        path_loci = {relaxer.get_locus(s) for s in snps}
        # 只要交集不为空，即视为“共享机制路径”
        return not path_loci.isdisjoint(shared_loci)

    # 筛选 G1
    df1['Is_Shared'] = df1['Full_Path'].apply(is_path_shared)
    df1_shared = df1[df1['Is_Shared']].copy()
    
    # 筛选 G2
    df2['Is_Shared'] = df2['Full_Path'].apply(is_path_shared)
    df2_shared = df2[df2['Is_Shared']].copy()
    
    # 4. 统计与计算
    print(f"\n[结果统计]")
    
    # --- Trait 1 分析 ---
    count1 = len(df1_shared)
    # 计算去重影响
    paths_list_1 = df1_shared['Full_Path'].tolist()
    impact1 = calculate_network_impact_no_overlap(G1, paths_list_1, trait1)
    
    print(f"1. {trait1} (Stem Color):")
    print(f"   - 包含共享 SNP 的路径数: {count1} (占总路径 {len(df1)})")
    print(f"   - 共享机制的净影响 (Net Impact): {impact1:.4f}")
    
    # --- Trait 2 分析 ---
    count2 = len(df2_shared)
    paths_list_2 = df2_shared['Full_Path'].tolist()
    impact2 = calculate_network_impact_no_overlap(G2, paths_list_2, trait2)
    
    print(f"2. {trait2} (Stem Fill):")
    print(f"   - 包含共享 SNP 的路径数: {count2} (占总路径 {len(df2)})")
    print(f"   - 共享机制的净影响 (Net Impact): {impact2:.4f}")
    
    # --- 结论判断 ---
    print("\n[多效性结论]")
    if impact1 * impact2 > 0:
        print(f"   >> 协同作用 (Synergy): 共享的遗传模块同时{'促进' if impact1>0 else '抑制'}两个性状。")
    else:
        print(f"   >> 拮抗/权衡 (Trade-off): 共享的遗传模块对两个性状产生相反的影响。")
        
    # 保存筛选出的路径供查阅
    df1_shared[['Path_String', 'Total_Effect']].to_csv(os.path.join(output_dir, f"3_shared_paths_{trait1}.csv"), index=False)
    df2_shared[['Path_String', 'Total_Effect']].to_csv(os.path.join(output_dir, f"3_shared_paths_{trait2}.csv"), index=False)

if __name__ == "__main__":
    cg_stem_color_file = '~/data/plant/alfalfa/cg_stem_color_141.graphml'
    cg_stem_fill_file = '~/data/plant/alfalfa/cg_stem_strength_151.graphml'
    cg_stem_color_file = os.path.expanduser(cg_stem_color_file)
    cg_stem_fill_file = os.path.expanduser(cg_stem_fill_file)
    from util_cg import load_nxgraph
    G_color = load_nxgraph(cg_stem_color_file)
    G_fill = load_nxgraph(cg_stem_fill_file)
    compare_two_causal_graphs(G_color, 'stem_color', G_fill, 'stem_strength', window_size=260000)
    analyze_shared_mechanism_impact(G_color, 'stem_color', G_fill, 'stem_strength', window_size=260000)

