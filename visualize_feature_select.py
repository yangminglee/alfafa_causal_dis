import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from scipy import stats
import os
import argparse

# ==========================================
# Configuration: LaTeX-style Plotting
# ==========================================
def configure_plots(format='pdf'):
    """
    Configures Matplotlib for publication-quality output based on format.
    """
    # Specific backend settings for PGF (Direct LaTeX code)
    if format == 'pgf':
        try:
            matplotlib.use('pgf')
            plt.rcParams.update({
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
            })
        except Exception as e:
            print(f"Warning: Could not switch to PGF backend. Is LaTeX installed? Error: {e}")
    else:
        pass 

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (6, 4),
        "savefig.bbox": "tight",
    })

# ==========================================
# Color Map Definition
# ==========================================
# User defined map: "Color Name" -> "Label"
# We invert this for plotting: "Chr ID" -> "Color Name"
USER_COLOR_MAP = {
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

def get_chr_color(chrom_num):
    """
    Maps chromosome number (float/int) to the specific color.
    """
    try:
        c = int(chrom_num)
        mapping = {
            1: "blue",
            2: "green", 
            3: "orange",
            4: "purple",
            5: "pink",
            6: "yellow",
            7: "cyan",
            8: "magenta"
        }
        return mapping.get(c, "gray")
    except:
        return "gray"

# ==========================================
# Data Processing Module
# ==========================================
def load_and_parse_gwas_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
        
        def get_chr_pos(snp_str):
            try:
                clean_str = str(snp_str).lower().replace('chr', '')
                parts = clean_str.replace(':', '_').split('_')
                chrom = float(parts[0])
                pos = int(parts[1])
                return chrom, pos
            except (ValueError, IndexError):
                return -1, -1

        if 'SNP' in df.columns:
            df[['CHR', 'POS']] = df['SNP'].apply(lambda x: pd.Series(get_chr_pos(x)))
            df = df[df['CHR'] != -1]
            df = df.sort_values(['CHR', 'POS'])
        
        if 'P_value' in df.columns:
            df['logp'] = -np.log10(df['P_value'].replace(0, 1e-300))
            
        return df
    except Exception as e:
        print(f"Error parsing data: {e}")
        return pd.DataFrame()

# ==========================================
# Visualization Class
# ==========================================
class GWASVisualizer:
    def __init__(self, df, output_dir="./plots", fmt="pdf"):
        self.df = df
        self.output_dir = output_dir
        self.fmt = fmt
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _save(self, filename_base):
        output_path = os.path.join(self.output_dir, f"{filename_base}.{self.fmt}")
        plt.savefig(output_path, format=self.fmt)
        plt.close()
        print(f"Saved: {output_path}")

    def plot_manhattan(self, title="Manhattan Plot", filename="Figure1_Manhattan", highlight_threshold=0.05):
        if self.df.empty: return
        
        df_grouped = self.df.groupby('CHR')
        
        fig, ax = plt.subplots(figsize=(5, 5))
        
        x_labels = []
        x_labels_pos = []
        last_pos = 0
        
        # Legend handles for significant chromosomes
        legend_patches = []
        seen_chroms = set()
        
        for num, (name, group) in enumerate(df_grouped):
            group = group.copy()
            group['ind'] = range(len(group))
            group['ind'] += last_pos
            
            sig_mask = group['P_value'] < highlight_threshold
            non_sig = group[~sig_mask]
            sig = group[sig_mask]
            
            chrom_color = get_chr_color(name)
            
            # 1. Non-significant: Gray/Dim
            # Using alternating grays to distinguish chromosomes slightly in background
            bg_color = '#E0E0E0' if num % 2 == 0 else '#F0F0F0'
            alpha_val = 0.5 if self.fmt != 'eps' else 1.0
            
            ax.scatter(non_sig['ind'], non_sig['logp'], c='gray', 
                       s=5, alpha=0.15, linewidths=0)
            
            # 2. Significant: Specific Color
            if not sig.empty:
                ax.scatter(sig['ind'], sig['logp'], c=chrom_color, 
                           s=20, alpha=1.0, linewidths=0, zorder=10)
                
                if name not in seen_chroms:
                    legend_patches.append(mpatches.Patch(color=chrom_color, label=f'Chr {int(name)}'))
                    seen_chroms.add(name)
            
            x_labels.append(f"{int(name)}")
            x_labels_pos.append(group['ind'].mean())
            last_pos += len(group)
        
        threshold_logp = -np.log10(highlight_threshold)
        ax.axhline(threshold_logp, color='black', linestyle='--', linewidth=1.0)
        
        ax.set_xticks(x_labels_pos)
        ax.set_xticklabels(x_labels, rotation=0)
        ax.set_xlabel('Chromosome')
        ax.set_ylabel(r'$-\log_{10}(P)$')
        ax.set_title(title)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend if we have significant hits
        if legend_patches:
            ax.legend(handles=legend_patches, loc='upper right', frameon=True, ncol=2)
        
        plt.tight_layout()
        self._save(filename)

    def plot_qq(self, title="Q-Q Plot", filename="Figure2_QQ"):
        if self.df.empty: return
        n = len(self.df)
        observed = -np.log10(np.sort(self.df['P_value']))
        expected = -np.log10(np.arange(1, n + 1) / (n + 1))
        
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Confidence interval
        try:
            c95 = stats.beta.ppf(0.95, np.arange(1, n+1), np.arange(n, 0, -1))
            c05 = stats.beta.ppf(0.05, np.arange(1, n+1), np.arange(n, 0, -1))
            ax.fill_between(expected, -np.log10(c05), -np.log10(c95), color='gray', alpha=0.2)
        except Exception:
            pass
        
        # QQ plot typically implies general fit, so we keep standard coloring (blue dots)
        # highlighting specific chromosomes here might be cluttered.
        ax.scatter(expected, observed, c='#4c72b0', s=15, alpha=0.8, linewidths=0)
        ax.plot([0, max(expected)], [0, max(expected)], color='#c44e52', linestyle='--')
        
        chi2_obs = stats.norm.ppf(1 - self.df['P_value'] / 2) ** 2
        lambda_gc = np.median(chi2_obs) / stats.chi2.ppf(0.5, 1)
        
        lambda_text = f"$\lambda_{{GC}} = {lambda_gc:.3f}$" if self.fmt in ['pgf', 'pdf'] else f"Lambda GC = {lambda_gc:.3f}"
        ax.text(0.05, 0.95, lambda_text, transform=ax.transAxes, verticalalignment='top', fontweight='bold')
        
        ax.set_xlabel(r'Expected $-\log_{10}(P)$')
        ax.set_ylabel(r'Observed $-\log_{10}(P)$')
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        self._save(filename)

    def plot_volcano(self, title="Volcano Plot", filename="Figure3_Volcano", highlight_threshold=0.05):
        if self.df.empty or 'Coef' not in self.df.columns: return
        
        fig, ax = plt.subplots(figsize=(5, 5))
        threshold_logp = -np.log10(highlight_threshold)
        
        # Logic: 
        # 1. Non-significant -> Gray
        # 2. Significant -> Color by Chromosome (to match the map)
        
        colors = []
        sizes = []
        legend_seen = set()
        legend_patches = []
        
        for idx, row in self.df.iterrows():
            if row['logp'] < threshold_logp:
                colors.append('lightgrey')
                sizes.append(5)
            else:
                c = get_chr_color(row['CHR'])
                colors.append(c)
                sizes.append(20)
                
                # Track for legend
                chr_num = int(row['CHR'])
                if chr_num not in legend_seen:
                    legend_patches.append(mpatches.Patch(color=c, label=f'Chr {chr_num}'))
                    legend_seen.add(chr_num)

        ax.scatter(self.df['Coef'], self.df['logp'], c=colors, s=sizes, alpha=0.6, linewidths=0)
        ax.axhline(threshold_logp, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Effect Size (Coefficient)')
        ax.set_ylabel(r'$-\log_{10}(P)$')
        ax.set_title(title)
        
        # Label top 3
        sig_hits = self.df[self.df['P_value'] < highlight_threshold].sort_values('P_value').head(3)
        for _, row in sig_hits.iterrows():
            try:
                snp_label = str(row['SNP']).split('_')[1] if '_' in str(row['SNP']) else str(row['SNP'])
                ax.annotate(snp_label, (row['Coef'], row['logp']), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
            except: pass
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if legend_patches:
             # Sort legend by chr number
            legend_patches.sort(key=lambda x: int(x.get_label().split()[-1]))
            ax.legend(handles=legend_patches, loc='upper right', frameon=True, fontsize=8, ncol=2)
            
        plt.tight_layout()
        self._save(filename)

    def plot_effect_sizes(self, title="Selected Candidate SNPs", filename="Figure4_EffectSizes", highlight_threshold=0.05):
        if self.df.empty or 'Coef' not in self.df.columns: return
        
        selected = self.df[self.df['P_value'] < highlight_threshold].copy()
        if selected.empty:
            print(f"No SNPs found with P < {highlight_threshold}")
            return

        selected = selected.sort_values('P_value', ascending=False) # Best P at top
        
        if len(selected) > 20:
            print(f"Note: {len(selected)} SNPs significant. Plotting top 20.")
            selected = selected.tail(20)
            
        error = 1.96 * selected['SE']
        
        fig, ax = plt.subplots(figsize=(5, max(4, len(selected)*0.3)))
        
        y_pos = np.arange(len(selected))
        
        # Color by Chromosome
        colors = [get_chr_color(c) for c in selected['CHR']]
        
        ax.errorbar(selected['Coef'], y_pos, xerr=error, fmt='o', color='black', 
                    ecolor='grey', capsize=3, elinewidth=1, markersize=0)
        ax.scatter(selected['Coef'], y_pos, c=colors, s=60, zorder=10)
        
        ax.set_yticks(y_pos)
        clean_labels = [str(l).replace('chr', '') for l in selected['SNP']]
        ax.set_yticklabels(clean_labels)
        
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
        
        ax.set_xlabel('Causal Effect Size (95% CI)')
        ax.set_title(title)
        
        # Legend for Chromosomes
        unique_chrs = sorted(selected['CHR'].unique())
        legend_patches = [mpatches.Patch(color=get_chr_color(c), label=f'Chr {int(c)}') for c in unique_chrs]
        ax.legend(handles=legend_patches, loc='lower right', frameon=True)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        self._save(filename)

# ==========================================
# Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Visualize GWAS Results")
    parser.add_argument("--file", type=str, default="test.csv", help="Input CSV")
    parser.add_argument("--output", type=str, default="./plots", help="Output Directory")
    parser.add_argument("--threshold", type=float, default=0.05, help="P-value threshold")
    parser.add_argument("--format", type=str, default="pdf", choices=['pdf', 'eps', 'pgf', 'png'], help="Output format")
    
    args = parser.parse_args()

    print(f"Configuring for {args.format} format...")
    configure_plots(format=args.format)
    
    print(f"Loading data from {args.file}...")
    df = load_and_parse_gwas_data(args.file)
    
    if df.empty:
        print("Data load failed.")
        return

    viz = GWASVisualizer(df, output_dir=args.output, fmt=args.format)
    
    print(f"Generating figures...")
    viz.plot_manhattan(highlight_threshold=args.threshold)
    viz.plot_qq()
    viz.plot_volcano(highlight_threshold=args.threshold)
    viz.plot_effect_sizes(highlight_threshold=args.threshold)
    
    print("✅ Complete.")

if __name__ == "__main__":
    main()