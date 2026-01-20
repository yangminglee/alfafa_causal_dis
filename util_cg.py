import networkx as nx
import os

from sklearn.linear_model import LinearRegression

def cg_trim_ancester(G, target_trait):
    relevant_nodes = nx.ancestors(G, target_trait)
    # nx.ancestors 返回的是集合，不包括 target 本身，所以要加上
    relevant_nodes.add(target_trait)
    G_sub = G.subgraph(relevant_nodes).copy()
    print(f"原始节点数: {len(G.nodes())}, 剪枝后关键节点数: {len(G_sub.nodes())}")    
    return G_sub

def cg_quantify_linear(graph, df, target_trait):
    model = LinearRegression()
    
    for node in graph.nodes():
        parents = list(graph.predecessors(node))
        if not parents:
            continue

        #parent_labels = [graph.nodes[p]['label'] for p in parents]
        # 准备数据 (OLS)
        X_parents = df[parents]
        y_target = df[node]
        model.fit(X_parents, y_target)
        # 赋值权重
        for parent, coef in zip(parents, model.coef_):
            graph[parent][node]['weight'] = coef
    return graph
    
def cg2nxgraph(cg):
    nx_graph = nx.DiGraph()
    nodes = range(len(cg.G.graph))
    for i in nodes:
        nx_graph.add_node(cg.G.nodes[i].get_name())

    undirected = cg.find_undirected()
    directed = cg.find_fully_directed()
    bidirected = cg.find_bi_directed()
    for (i, j) in undirected:
        nx_graph.add_edge(cg.G.nodes[i].get_name(),
                          cg.G.nodes[j].get_name(), label='ud')  # Green edge: undirected edge
    for (i, j) in directed:
        nx_graph.add_edge(cg.G.nodes[i].get_name(),
                          cg.G.nodes[j].get_name(), lable='d')  # Blue edge: directed edge
    for (i, j) in bidirected:
        nx_graph.add_edge(cg.G.nodes[i].get_name(),
                          cg.G.nodes[j].get_name(), label='bd')  # Red edge: bidirected edge

    return nx_graph

def save_nxgraph(graph, filename):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
            
    # 2. 检查图是否为空
    if len(graph.nodes()) == 0:
        print("警告：你试图保存一个没有任何节点的空图！")
        
    # 3. 写入文件 (不使用 try-except，以便暴露错误)
    print(f"正在保存图到 {filename}...")
    nx.write_graphml(graph, filename)
    
    # 4. 验证写入结果
    if os.path.getsize(filename) > 0:
        print(f"[Success] 保存成功，文件大小: {os.path.getsize(filename)} bytes")
    else:
        raise ValueError("[Error] 文件已创建但大小为 0，写入失败！")

def load_nxgraph(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件未找到: {filename}")
        
    try:
        graph = nx.read_graphml(filename)
        
        # 类型安全检查：确保 weight 是 float 类型
        # 虽然 write_graphml 通常会保留类型，但在某些版本或跨软件交互中，
        # 属性可能会变成字符串。这里做一个强制类型转换的“清洗”步骤是工程上的保险措施。
        for u, v, data in graph.edges(data=True):
            if 'weight' in data:
                try:
                    data['weight'] = float(data['weight'])
                except ValueError:
                    print(f"警告: 边 {u}->{v} 的 weight 无法转换为 float: {data['weight']}")
        
        # 确保这是一个有向图
        if not isinstance(graph, nx.DiGraph):
            graph = nx.DiGraph(graph)
            
        print(f"[IO Success] 已加载因果图: {len(graph.nodes())} 节点, {len(graph.edges())} 边")
        return graph

    except Exception as e:
        print(f"[IO Error] 读取失败: {e}")
        return None

if __name__ == "__main__":
    filename = '~/data/plant/alfalfa/cg_stem_fill_155.graphml'
    filename = os.path.expanduser(filename)
    nx_graph = load_nxgraph(filename)
    nx_graph = cg_trim_ancester(nx_graph, 'stem_color')
    nx_graph = cg_quantify_linear(nx_graph, df, 'stem_color')
    save_nxgraph(nx_graph, 'cg_stem_color_141_sub.graphml')
