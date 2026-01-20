import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

combined_data = pd.read_csv("combined_data_causal.csv", index_col=0)
print(f"Combined Data Shape: {combined_data.shape}")
# ----------------------------
# STEP 2: Prepare Causal Discovery Inputs
# ----------------------------
# Combine genotype and phenotype data into one dataset

# Convert combined dataset to a numpy array for the GPUCSL causal discovery algorithms
data_matrix = combined_data.values  # SNPs + Phenotypes (N x Features)

# Define variable names (SNPs + Traits)
variable_names = combined_data.columns.tolist()

# ----------------------------
# STEP 3: Run GPU-Accelerated Causal Discovery (PC Algorithm)
# ----------------------------
# Set a threshold for conditional independence tests (alpha determines significance level)
alpha = 0.05  # 95% confidence threshold

max_level = 1


(
    (
        directed_graph,
        separation_sets,
        pmax,
        discover_skeleton_runtime,
        edge_orientation_runtime,
        discover_skeleton_kernel_runtime,
        ),
    pc_runtime,
) = (
    DiscretePC(data_matrix, max_level, alpha).set_distribution_specific_options().execute()
)

# ----------------------------
# STEP 4: Extract and Visualize the Causal Graph
# ----------------------------
# Visualize the causal graph using the GPUCSL interface
cg.draw_graph()
plt.title("Causal Graph Between SNPs and Traits")
plt.show()

# ----------------------------
# STEP 5: Identify SNP-Trait Associations
# ----------------------------
# Extract edges representing direct causal relationships between SNPs and phenotypes
edges = cg.edges  # List of (source, target, metadata) tuples

snps_to_traits = []
for edge in edges:
    source, target, data = edge
    if source in geno_data.columns and target in pheno_data.columns:
        snps_to_traits.append((source, target, data.get('weight', 'Not Computed')))
        
        # Print causal associations
        print("\nCausal Associations Between SNPs and Traits:")
        for snp, trait, weight in snps_to_traits:
            print(f"SNP {snp} -> Trait {trait} | Causal Weight: {weight}")
            
