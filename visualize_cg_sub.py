import io
import os
import argparse
import pandas as pd

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Edge import Edge

from load_data import load_raw, load_cg
from util_vis import vis_graph_with_nx, venn_compare
from cg_util import create_nx_graph, extract_subgraph, extract_subgraph_direct, save_weight_csv
    
def draw_graph(g, labels, f_name, format=None):
    pyd = GraphUtils.to_pydot(g, labels=labels)
    pyd.set_size('"8.5,11!"')
    #pyd.set_graph_defaults(size="8,11!")
    if format is None:
        tmp_png = pyd.create_png(f="png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        plt.rcParams["figure.figsize"] = [20, 12]
        plt.rcParams["figure.autolayout"] = True
        plt.axis('off')
        plt.imshow(img)
        plt.show()
    elif format == 'pdf':
        pyd.write_pdf(f_name+'.pdf')
    elif format == 'png':
        pyd.write_png(f_name+'.png')
    else:
        raise Exception(f' unsupported format {format}')

def load_graph_as_general_graph(file_path):
    """
    读取 GraphML 文件，并将其完整转换为 causal-learn 的 GeneralGraph 对象。
    这允许您对已保存的图进行后续的因果分析。
    """
    print(f"\n>>> 从文件加载 GeneralGraph: {file_path}")
    import networkx as nx
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
        
    # 1. 先用 NetworkX 读取结构和属性
    try:
        nx_graph = nx.read_graphml(file_path)
    except Exception as e:
        raise ValueError(f"读取 GraphML 失败: {e}")

    # 2. 初始化 GeneralGraph 节点
    nodes = []
    node_map_name_to_obj = {}
    
    # causal-learn 的节点对象
    # 注意：GraphML 中的节点名通常是字符串 ID
    sorted_nodes = sorted(list(nx_graph.nodes())) # 排序以保证确定性
    
    for node_name in sorted_nodes:
        node = GraphNode(node_name)
        nodes.append(node)
        node_map_name_to_obj[node_name] = node
        
    # 创建 GeneralGraph 对象
    cg_graph = GeneralGraph(nodes)
    
    # 3. 转换边 (NetworkX -> GeneralGraph Endpoint)
    # 我们需要解析之前保存的 'relation' 属性来恢复 Endpoint 类型
    
    edge_count = 0
    # NetworkX 的 edges 默认是有向遍历
    for u, v, data in nx_graph.edges(data=True):
        node_u = node_map_name_to_obj[u]
        node_v = node_map_name_to_obj[v]
        
        relation = data.get('relation', '')
        
        # 默认端点 (无边)
        end1 = Endpoint.NULL
        end2 = Endpoint.NULL
        
        # 逻辑逆向映射：根据保存时的规则恢复 Endpoint
        if relation == 'direct':
            # SNP(u) -> Trait(v)
            end1 = Endpoint.TAIL
            end2 = Endpoint.ARROW
        elif relation == 'ld_directed':
            # u -> v
            end1 = Endpoint.TAIL
            end2 = Endpoint.ARROW
        elif relation == 'ld_bidirected':
            # u <-> v
            end1 = Endpoint.ARROW
            end2 = Endpoint.ARROW
        elif relation == 'ld_undirected' or relation == 'ld_uncertain':
            # u --- v (视为无向)
            end1 = Endpoint.TAIL
            end2 = Endpoint.TAIL
        else:
            # 如果没有标记，假设它是 NetworkX 的有向边 u->v
            end1 = Endpoint.TAIL
            end2 = Endpoint.ARROW
            
        # 在 GeneralGraph 中添加边
        # 注意：add_edge(node1, node2, endpoint1, endpoint2)
        e = Edge(node_u, node_v, end1, end2)
        cg_graph.add_edge(e)
        edge_count += 1

    print(f"✅ 转换完成: {len(nodes)} 节点, {edge_count} 边")
    print(f"   对象类型: {type(cg_graph)}")
    
    return cg_graph

def load_causal_graph(file_path):
    """
    加载保存的因果图文件 (GraphML 或 GML)。
    
    参数:
        file_path (str): 图文件的路径 (.graphml 或 .gml)
        
    返回:
        nx.DiGraph: NetworkX 有向图对象
    """
    import networkx as nx
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 错误: 找不到文件 {file_path}")
    
    print(f"📂 正在加载图文件: {file_path}")
    
    try:
        # 1. 尝试加载 GraphML
        if file_path.endswith('.graphml'):
            G = nx.read_graphml(file_path)
            # GraphML 读取时可能会把所有属性读成字符串，这里尝试转换数值属性
            # (如果之前保存时已经是正确格式，这一步是保险)
            for u, v, data in G.edges(data=True):
                if 'weight' in data and isinstance(data['weight'], str):
                    try:
                        data['weight'] = float(data['weight'])
                    except: pass
                if 'p_value' in data and isinstance(data['p_value'], str):
                    try:
                        data['p_value'] = float(data['p_value'])
                    except: pass
                    
        # 2. 尝试加载 GML
        elif file_path.endswith('.gml'):
            G = nx.read_gml(file_path)
            
        # 3. 其他格式不支持
        else:
            raise ValueError("不支持的文件格式。请使用 .graphml 或 .gml")
            
        # 确保是有向图
        if not isinstance(G, nx.DiGraph):
            print("⚠️ 警告: 加载的图不是有向图 (DiGraph)，正在尝试转换...")
            G = nx.DiGraph(G)
            
        print(f"✅ 加载成功! 节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
        return G

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        # 打印更详细的错误信息以便调试
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Parse two input arguments: -n and -t.")

    parser.add_argument("--data_dir", type=str, default="~/data/plant/alfalfa", help="dir to data")
    parser.add_argument("--cg", type=str, default=None, help="cg file")
    parser.add_argument("--trait", type=str, default = 'stem_color', help="Target phenotype column name")
    # Parse arguments
    args = parser.parse_args()

    DATA_DIR = os.path.expanduser(args.data_dir)
    print(f"Data directory: {DATA_DIR}")
    
    PEDIGREES = {"UMN3097": [0, None, 'black'],
                "UMN3355": [1, 'H', 'blue'],
                "UMN3358": [1, 'L', 'yellow'],
                "UMN4016": [2, 'H', 'green'],
                "UMN4351": [2, 'L', 'red']}

    trait = args.trait
    f_name = 'cg_'+str(trait)    
    matching_files = []
    for filename in os.listdir(DATA_DIR):
        full_path = os.path.join(DATA_DIR, filename)
        if os.path.isfile(full_path) and \
           f_name in filename and \
           filename.endswith(f".graphml"):
            matching_files.append(full_path)
    if len(matching_files) == 0:
        print('ERROR: no matching pkl')
    for f in matching_files:
        print(f'find pkl: {f}')
    f_name = matching_files[0]
    if True:
        if f_name.endswith('.pkl'):
            cg = load_cg(f_name)
            G = cg.G
        else:
            G = load_graph_as_general_graph(f_name)#load_causal_graph(f_name)

        layout_names = [
            "spring", "kamada_kawai", "circular", "shell",
            "spectral", "random", "spiral"
        ]

        try:
            pheno_node_idx = G.get_node_names().index(trait)
            print(f"The first occurrence of '{trait}' is at index: {pheno_node_idx}")
        except ValueError:
            print(f"'node {trait}' is not found in the CG.")

        sg_nodes,sg_edges = extract_subgraph(G, pheno_node_idx)
        subgraph = create_nx_graph(G, sg_nodes, sg_edges)
        traits_string = str(trait)
        vis_graph_with_nx(subgraph, output_name= traits_string, ratio=1, layout='spiral', result_dir=DATA_DIR)
 
        
        #vis_graph_with_nx(subgraph_node, output_name= traits_string, ratio=0.4, layout='circular', result_dir=result_dir)

        
    if False:
        f_cg_importance = traits_string+"_importance_"+str(top_x)+".csv"
        if not os.path.isfile(os.path.join(data_dir, f_cg_importance)): # save weights to csv
            print(f"File {f_cg_importance} not found, generating weights")
            if not('cg' in locals()) or cg is None:
                cg = load_cg(f_name, data_dir=data_dir)
            save_weight_csv(cg, weight_thresh=0.05, file_name=f_cg_importance, data_dir=data_dir)

        print(f"Loading {f_cg_importance} from disk")
        cg_color = pd.read_csv(os.path.join(data_dir, f_cg_importance), index_col=0)

        rf_color = pd.read_csv(os.path.join(data_dir, traits_string+"_importance.csv"), index_col=0)
        # Load individual SVM-based importance scores
        svm_color = pd.read_csv(os.path.join(data_dir, f"processed_{traits_string}_svm_importance.csv"), index_col=0)
        # Load CG importance scores
            
        print(f"Data shapes for {traits_string}: \nRF: {rf_color.shape}, \nSVM: {svm_color.shape}, \nCG: {cg_color.shape}")
        venn_compare(rf_color, svm_color, cg_color, threshold=0.0001, top_x=top_x, trait=traits_string)

