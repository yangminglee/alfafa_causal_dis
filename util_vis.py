import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import networkx as nx
from matplotlib.patches import Patch
import re

color_label_map = {
    "red": "Trait",
    "blue": "Chr 1",
    "green": "Chr 2",
    "orange": "Chr 3",
    "purple": "Chr 4",
    "pink": "Chr 5",
    "yellow": "Chr 6",
    "cyan": "Chr 7",
    "magenta": "Chr 8",
    "gray": "Unmapped Nodes"
}

layout_funcs = {
    "spring": nx.spring_layout,
    "kamada_kawai": nx.kamada_kawai_layout,
    "circular": nx.circular_layout,
    "shell": nx.shell_layout,
    "spectral": nx.spectral_layout,
    "random": nx.random_layout,
    "spiral": nx.spiral_layout,
    "planar": lambda G: nx.planar_layout(G)  # wrapped to handle errors later
}

def assign_node_color(node):
    if node.startswith("stem_") or node.startswith("winter_"):
        return "red"
    elif node.startswith("chr1"):
        return "blue"
    elif node.startswith("chr2"):
        return "green"
    elif node.startswith("chr3"):
        return "orange"
    elif node.startswith("chr4"):
        return "purple"
    elif node.startswith("chr5"):
        return "pink"
    elif node.startswith("chr6"):
        return "yellow"
    elif node.startswith("chr7"):
        return "cyan"
    elif node.startswith("chr8"):
        return "magenta"
    else:
        return "gray"  # Default color for unmatched nodes

def venn_compare(rf_color, svm_color, cg_color, threshold = 0.0001, top_x=500, trait="", result_dir="./"):
    plt.rcParams.update({
        "text.usetex": True,         # Use mathtext (not actual LaTeX)
        "pgf.texsystem": "pdflatex",  # Required if you switch back to usetex=True
        "font.family": "serif",
        "font.size": 10
    })
    # Get top SNPs by importance threshold from RF and SVM for a specific trait
    top_rf_color = set(rf_color[rf_color.max(axis=1) > threshold].index)
    top_svm_color = set(svm_color[svm_color.max(axis=1) > threshold/1000].index)
    #top_rf_all_traits = set(rf_all_traits[rf_all_traits.max(axis=1) > threshold].index)
    top_cg_color = set(cg_color['Name1'])
    # Plot Venn diagram
    plt.figure(figsize=(8, 6))
    #venn3([top_rf_color, top_svm_color, top_rf_all_traits], ('RF Indivisual', 'SVM Individual', 'RF All Traits'))
    from matplotlib_venn import venn3
    venn3([top_rf_color, top_svm_color, top_cg_color], ('RF', 'SVM', 'CG'))
    plt.title(f"Top SNPs Overlap Across RF, SVM, and CG for {re.sub(r'[^A-Za-z0-9]', ' ', trait).title()}")
    output_name = trait+'_venn.pgf'
    output_name = os.path.join(result_dir, output_name)
    plt.savefig(output_name, bbox_inches='tight', dpi=300, format='pgf')
    print("Graph visualization saved as", output_name)


    top_x = 500
    # Select the top SNPs based on importance
    top_rf_color = set(rf_color.max(axis=1).nlargest(top_x).index)
    top_svm_color = set(svm_color.max(axis=1).nlargest(top_x).index)
    #top_rf_all_traits = set(rf_all_traits.max(axis=1).nlargest(top_x).index)
    #top_cg_color = set(cg_color.max(axis=1).nlargest(top_x).index)
    # Plot Venn diagram
    plt.figure(figsize=(8, 6))
    #venn3([top_rf_color, top_svm_color, top_rf_all_traits], ('RF Indivisual', 'SVM Individual', 'RF All Traits'))
    venn3([top_rf_color, top_svm_color, top_cg_color], ('RF', 'SVM', 'CG'))
    plt.title(f"Top SNPs Overlap Across RF, SVM, and CG for {trait}")
    output_name = trait+'_venn_' +str(top_x)+'.pgf'
    output_name = os.path.join(result_dir, output_name)
    plt.savefig(output_name, bbox_inches='tight', dpi=300, format='pgf')
    print("Graph visualization saved as", output_name)

def vis_cg_sunflower(G, target_node, output_name='causal_graph', result_dir="./", ratio=1.2, top_n=20):
    """
    Plots the FULL graph G in a Spiral (Sunflower) layout.
    
    FIXED:
    1. Strictly limits labels to 'top_n' + target.
    2. Draws labels by iterating the filtered list directly (failsafe).
    3. Matches font sizes/styling with the concentric function.
    """
    
    # --- STEP 1: Sort ALL Nodes for Spiral Layout ---
    # We prioritize nodes connected to the target (Neighbors) -> Close to center
    node_scores = []
    
    for node in G.nodes():
        if node == target_node: continue
        
        weight = 0.0
        is_direct = False
        
        # Check for direct connection (In or Out)
        if G.has_edge(node, target_node): # Parent -> Target
            weight = abs(G[node][target_node].get('weight', 0.0))
            is_direct = True
        elif G.has_edge(target_node, node): # Target -> Child
            weight = abs(G[target_node][node].get('weight', 0.0))
            is_direct = True
            
        # Score calculation: Bonus for direct neighbors + weight
        score = (1e6 if is_direct else 0) + weight
        node_scores.append((node, score))

    # Sort descending: Highest score = closest to center
    sorted_nodes = sorted(node_scores, key=lambda x: x[1], reverse=True)
    layout_order = [n for n, s in sorted_nodes]
    
    # Determine which nodes to LABEL (Target + Top N from sorted list)
    # Using a list to preserve order is nice, but set is faster for lookup.
    # Here we just create the list of nodes we WANT to label.
    nodes_to_label_list = [target_node] + layout_order[:top_n]
    
    print(f"Graph has {len(G.nodes())} nodes. Displaying {len(nodes_to_label_list)} labels (Top {top_n} + Target).")

    # --- STEP 2: Spiral Layout Calculation ---
    pos = {}
    pos[target_node] = np.array([0, 0]) # Center
    
    golden_angle = np.pi * (3 - np.sqrt(5))
    scale = 3.5 
    
    for i, node in enumerate(layout_order):
        idx = i + 1
        r = scale * np.sqrt(idx)
        theta = idx * golden_angle
        pos[node] = np.array([r * np.cos(theta), r * np.sin(theta)])

    # --- STEP 3: Styling ---
    # Colors
    try:
        node_colors = [assign_node_color(n) for n in G.nodes()]
    except NameError:
        node_colors = []
        for n in G.nodes():
            if n == target_node: node_colors.append('#FFD700') # Gold
            elif G.has_edge(n, target_node): node_colors.append('#d62728') # Red
            else: node_colors.append('#1f77b4') # Blue

    # LaTeX Config
    plt.rcParams.update({
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",  
        "font.family": "serif",
        "font.size": 10
    })

    # --- STEP 4: Drawing ---
    plt.figure(figsize=(8*ratio, 6*ratio))
    
    # Draw ALL Edges
    edges = G.edges(data=True)
    weights = [data.get('weight', 0) for u, v, data in edges]
    
    if weights and max([abs(w) for w in weights]) > 0:
        max_w = max([abs(w) for w in weights])
        edge_widths = [(abs(w)/max_w * 3.0 + 0.5) for w in weights]
    else:
        edge_widths = [1.0] * len(edges)
    
    nx.draw_networkx_edges(G, pos, 
                           width=edge_widths, 
                           edge_color='gray', 
                           alpha=0.6, 
                           arrowsize=12)

    # Draw ALL Nodes
    sizes = [800 if n == target_node else 400 for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, 
                           node_color=node_colors, 
                           node_size=sizes, 
                           edgecolors='black', 
                           linewidths=0.8)

    # Draw Labels (STRICTLY FILTERED)
    # We iterate ONLY over the list of nodes we decided to label.
    # This guarantees no other labels can appear.
    for node in nodes_to_label_list:
        if node not in pos: continue
        
        x, y = pos[node]
        label = str(node) # Show full name
        
        # Consistent styling with other function
        fontsize = 12 if node == target_node else 9
        fontweight = 'bold' if node == target_node else 'normal'
        
        plt.text(x, y-0.45, label, 
                 ha='center', va='top', 
                 fontsize=fontsize, fontweight=fontweight)

    # --- STEP 5: Legend ---
    unique_colors = set(node_colors)
    legend_elements = []
    if '#FFD700' in unique_colors:
        legend_elements.append(Patch(facecolor='#FFD700', edgecolor='black', label='Target'))
    if '#d62728' in unique_colors:
        legend_elements.append(Patch(facecolor='#d62728', edgecolor='black', label='Direct Parent'))
    if '#1f77b4' in unique_colors:
        legend_elements.append(Patch(facecolor='#1f77b4', edgecolor='black', label='Indirect/Hub'))
        
    if legend_elements:
        plt.legend(handles=legend_elements, loc="upper left", fontsize=10, frameon=True)

    # --- STEP 6: Save ---
    clean_title = re.sub(r'[^A-Za-z0-9]', ' ', str(target_node)).title()
    plt.title(f"Full Causal Network (Top {top_n} Labels): {clean_title}", fontsize=16)
    plt.axis('off')
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    filename_base = f"{output_name}_full"
    full_path_eps = os.path.join(result_dir, filename_base + ".eps")
    full_path_png = os.path.join(result_dir, filename_base + ".pdf")
    
    # Save EPS
    try:
        plt.savefig(full_path_eps, format='eps', bbox_inches='tight')
        print(f"Saved EPS: {full_path_eps}")
    except Exception as e:
        print(f"EPS Error: {e}")
        
    # Save PNG
    plt.savefig(full_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PNG: {full_path_png}")

def vis_cg_sunflower(G, target_node, output_name='causal_graph', result_dir="./", ratio=1.2, top_n=20):
    """
    Plots the FULL graph G in a Spiral (Sunflower) layout.
    Matches styling of vis_cg_sunflower2.
    """
    
    # --- STEP 1: Sort ALL Nodes for Spiral Layout ---
    node_scores = []
    
    for node in G.nodes():
        if node == target_node: continue
        
        weight = 0.0
        is_direct = False
        
        # Check for direct connection
        if G.has_edge(node, target_node): # Parent -> Target
            weight = abs(G[node][target_node].get('weight', 0.0))
            is_direct = True
        elif G.has_edge(target_node, node): # Target -> Child
            weight = abs(G[target_node][node].get('weight', 0.0))
            is_direct = True
            
        score = (1e6 if is_direct else 0) + weight
        node_scores.append((node, score))

    sorted_nodes = sorted(node_scores, key=lambda x: x[1], reverse=True)
    layout_order = [n for n, s in sorted_nodes]
    
    # Filter labels
    nodes_to_label_list = [target_node] + layout_order[:top_n]

    # --- STEP 2: Spiral Layout Calculation ---
    pos = {}
    pos[target_node] = np.array([0, 0]) 
    
    golden_angle = np.pi * (3 - np.sqrt(5))
    scale = 3.5 
    
    for i, node in enumerate(layout_order):
        idx = i + 1
        r = scale * np.sqrt(idx)
        theta = idx * golden_angle
        pos[node] = np.array([r * np.cos(theta), r * np.sin(theta)])

    # --- STEP 3: Styling (Matched to vis_cg_sunflower2) ---
    # Define Colors internally to ensure standalone execution
    node_colors = []
    for n in G.nodes():
        if n == target_node: node_colors.append('#FFD700') # Gold
        elif G.has_edge(n, target_node): node_colors.append('#d62728') # Red
        else: node_colors.append('#1f77b4') # Blue

    labels = {n: str(n) for n in G.nodes()}

    # Matched rcParams
    plt.rcParams.update({
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",  
        "font.family": "serif",
        "font.size": 10
    })

    # --- STEP 4: Drawing ---
    plt.figure(figsize=(6*ratio, 8*ratio)) # Matched size aspect
    
    # Edges
    edges = G.edges(data=True)
    weights = [data.get('weight', 0) for u, v, data in edges]
    
    max_w = max([abs(w) for w in weights]) if weights and max(weights)!=0 else 1
    edge_widths = [(abs(w)/max_w * 2.5 + 0.5) if w!=0 else 0.5 for w in weights]
    
    nx.draw_networkx_edges(G, pos, 
                           width=edge_widths, 
                           edge_color='gray', 
                           alpha=0.5, 
                           arrowsize=10)

    # Nodes (Matched Sizes: 300/200)
    sizes = [300 if n == target_node else 200 for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, 
                           node_color=node_colors, 
                           node_size=sizes, 
                           edgecolors='black', 
                           linewidths=0.5)

    # Labels (Matched Logic: 12/8 fontsize, y-0.35 offset)
    for node in nodes_to_label_list:
        if node not in pos: continue
        x, y = pos[node]
        fontsize = 12 if node == target_node else 8
        
        plt.text(x, y-0.35, labels[node], 
                 ha='center', va='top', 
                 fontsize=fontsize)

    # --- STEP 5: Legend ---
    unique_colors = set(node_colors)
    legend_elements = []
    if '#FFD700' in unique_colors:
        legend_elements.append(Patch(facecolor='#FFD700', edgecolor='black', label='Target'))
    if '#d62728' in unique_colors:
        legend_elements.append(Patch(facecolor='#d62728', edgecolor='black', label='Direct Parent'))
    if '#1f77b4' in unique_colors:
        legend_elements.append(Patch(facecolor='#1f77b4', edgecolor='black', label='Indirect/Hub'))
        
    if legend_elements:
        plt.legend(handles=legend_elements, loc="upper left", fontsize=10, frameon=True)

    # --- STEP 6: Saving (Matched) ---
    clean_title = re.sub(r'[^A-Za-z0-9]', ' ', str(target_node)).title()
    plt.title(f"Full Causal Network (Top {top_n} Labels): {clean_title}")
    plt.axis('off')
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    filename_base = f"{output_name}_full"
    full_path_pgf = os.path.join(result_dir, filename_base + ".pgf")
    full_path_png = os.path.join(result_dir, filename_base + ".png")
    full_path_eps = os.path.join(result_dir, filename_base + ".eps")
    
    try:
        plt.savefig(full_path_pgf)
        print(f"Graph saved as PGF: {full_path_pgf}")
    except Exception as e:
        print(f"PGF Save warning: {e}")

    try:
        plt.savefig(full_path_eps, format='eps')
        print(f"Graph saved as EPS: {full_path_eps}")
    except Exception as e:
        print(f"EPS Save warning: {e}")
        
    plt.savefig(full_path_png, dpi=300)
    print(f"Graph saved as PNG: {full_path_png}")
def vis_cg_concentric_sunflower(G, target_node, output_name='causal_graph', result_dir="./", ratio=1.2, top_k=20):
    """
    Visualizes the Causal Graph in a Concentric 'Sunflower' layout.
    Standardized fonts and styling to match the function above.
    """
    
    # --- 0. Pre-processing ---
    try:
        parents = list(G.predecessors(target_node))
    except:
        print(f"Error: Target node '{target_node}' not found in graph.")
        return

    try:
        ancestors = list(nx.ancestors(G, target_node))
    except:
        ancestors = []
    
    hubs = [n for n in ancestors if n not in parents and n != target_node]

    all_nodes = [target_node] + parents + hubs
    
    if len(all_nodes) > top_k + 1:
        parents.sort(key=lambda n: abs(G[n][target_node]['weight']) if 'weight' in G[n][target_node] else 0, reverse=True)
        hubs.sort(key=lambda n: G.out_degree(n), reverse=True)
        
        keep_parents = parents[:top_k] 
        remaining = top_k - len(keep_parents)
        keep_hubs = hubs[:max(0, remaining)]
        
        keep_nodes = {target_node} | set(keep_parents) | set(keep_hubs)
        G_viz = G.subgraph(list(keep_nodes)).copy()
        
        parents = [n for n in keep_parents if n in G_viz]
        hubs = [n for n in keep_hubs if n in G_viz]
    else:
        G_viz = G.copy()

    # --- A. Layout ---
    pos = {}
    pos[target_node] = np.array([0, 0]) 
    
    r_inner = 3.5
    if parents:
        angle_step = 2 * np.pi / len(parents)
        for i, node in enumerate(parents):
            theta = i * angle_step
            pos[node] = np.array([r_inner * np.cos(theta), r_inner * np.sin(theta)])
            
    r_outer = 7.0
    if hubs:
        angle_step = 2 * np.pi / len(hubs)
        offset = np.pi / len(hubs)
        for i, node in enumerate(hubs):
            theta = i * angle_step + offset
            pos[node] = np.array([r_outer * np.cos(theta), r_outer * np.sin(theta)])

    # --- B. Styling (Standardized) ---
    node_colors = []
    for n in G_viz.nodes():
        if n == target_node: node_colors.append('#FFD700')
        elif n in parents:   node_colors.append('#d62728')
        elif n in hubs:      node_colors.append('#1f77b4')
        else:                node_colors.append('grey')

    labels = {n: str(n) for n in G_viz.nodes()}

    # LaTeX Config (MATCHING BOTH FUNCTIONS)
    plt.rcParams.update({
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",  
        "font.family": "serif",
        "font.size": 10
    })

    # --- C. Drawing ---
    plt.figure(figsize=(8*ratio, 6*ratio))
    
    edges = G_viz.edges(data=True)
    widths = []
    edge_colors = []
    
    for u, v, data in edges:
        if v == target_node: 
            widths.append(2.5)
            edge_colors.append('black')
        else:
            widths.append(1.0)
            edge_colors.append('gray')

    nx.draw_networkx_edges(G_viz, pos, 
                           width=widths, 
                           edge_color=edge_colors, 
                           alpha=0.6, 
                           arrowsize=20,
                           connectionstyle='arc3,rad=0.1')

    sizes = []
    for n in G_viz.nodes():
        if n == target_node: sizes.append(2000)
        elif n in parents:   sizes.append(1000)
        else:                sizes.append(600)
    
    nx.draw_networkx_nodes(G_viz, pos, 
                           node_color=node_colors, 
                           node_size=sizes, 
                           edgecolors='black', 
                           linewidths=1.5)

    # Labels
    for node, (x, y) in pos.items():
        label = labels[node]
        
        # Standardized Font Sizes
        fontsize = 12 if node == target_node else 9
        fontweight = 'bold' if node == target_node else 'normal'

        if node == target_node:
            plt.text(x, y, label, ha='center', va='center', 
                     fontsize=fontsize, fontweight=fontweight)
            continue

        dist = np.sqrt(x*x + y*y)
        if dist > 0:
            label_x = x + (x/dist) * 0.8 
            label_y = y + (y/dist) * 0.8
            
            plt.text(label_x, label_y, label, 
                     ha='center', va='center', 
                     fontsize=fontsize, fontweight=fontweight,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    # --- D. Legend & Save ---
    legend_elements = [
        Patch(facecolor='#FFD700', edgecolor='black', label='Target Phenotype'),
        Patch(facecolor='#d62728', edgecolor='black', label='Direct Parents (High Specificity)'),
        Patch(facecolor='#1f77b4', edgecolor='black', label='Upstream Hubs (High Pleiotropy)')
    ]
    plt.legend(handles=legend_elements, loc="upper right", fontsize=10) # Standardized 10

    clean_title = re.sub(r'[^A-Za-z0-9]', ' ', str(target_node)).title()
    plt.title(f"Causal Architecture: {clean_title}", fontsize=16) # Standardized 16
    plt.axis('off')

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    filepath = os.path.join(result_dir, f"{output_name}_hub_parent.pdf")
    
    # Save with bbox_inches='tight'
    plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(filepath.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved: {filepath}")
    plt.close()

if __name__ == "__main__":
    if True:
        filename = '~/data/plant/alfalfa/cg_stem_color_141_sub.graphml'
        filename = os.path.expanduser(filename)
        from util_cg import load_nxgraph
        nx_graph = load_nxgraph(filename)
        vis_cg_sunflower(nx_graph, 'stem_color', output_name = 'stem_color')
        vis_cg_concentric_sunflower(nx_graph, 'stem_color', output_name='stem_color')
    elif True:
        filename = '~/data/plant/alfalfa/cg_stem_fill_155_sub.graphml'
        filename = os.path.expanduser(filename)
        from util_cg import load_nxgraph
        nx_graph = load_nxgraph(filename)
        vis_cg_sunflower(nx_graph, 'stem_fill', output_name='sunflower_causal_graph_stem_fill_155.png')
    else:
        filename = '~/data/plant/alfalfa/cg_stem_strength_151_sub.graphml'
        filename = os.path.expanduser(filename)
        from util_cg import load_nxgraph
        nx_graph = load_nxgraph(filename)
        vis_cg_sunflower(nx_graph, 'stem_strength', output_name='sunflower_causal_graph_stem_strength_151.png')