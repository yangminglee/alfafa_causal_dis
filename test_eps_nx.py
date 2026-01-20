import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams.update({
    "text.usetex": False,         # Use mathtext (not actual LaTeX)
    "pgf.texsystem": "pdflatex",  # Required if you switch back to usetex=True
    "font.family": "serif",
    "font.size": 10
})

G = nx.Graph()
G.add_edges_from([(1, 2)])
pos = nx.spring_layout(G, seed=42)

labels = {1: r"$x_1$", 2: r"$x_2$"}

plt.figure(figsize=(4, 3))
nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.5, alpha=0.8)
nx.draw_networkx_nodes(G, pos, node_color=["skyblue", "lightgreen"], edgecolors='black', node_size=700)

for node, (x, y) in pos.items():
    plt.text(x, y, labels[node], ha='center', va='center', fontsize=12)

plt.title(r"Network: $x_1 \leftrightarrow x_2$", fontsize=12)
plt.axis('off')
plt.tight_layout()

# Save as PGF (requires LaTeX installed)
plt.savefig("network_graph_final.pgf")
