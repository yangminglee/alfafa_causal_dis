import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_and_plot(file_path):
    # 1. 读取数据
    df = pd.read_csv(file_path)
    
    # 2. 解析 SNP 名称以获取 染色体 和 位置
    # 假设格式为 "chr4.1_6473437"
    # 我们需要把 'chr4.1' 转为数字以便排序 (例如 4.1 -> 4)
    def parse_snp(snp_str):
        try:
            parts = snp_str.split('_')
            chrom_part = parts[0].replace('chr', '')
            pos = int(parts[1])
            # 简单处理：取小数点前的数字作为染色体编号
            chrom = float(chrom_part) 
            return chrom, pos
        except:
            return 0, 0

    df[['CHR', 'POS']] = df['SNP'].apply(lambda x: pd.Series(parse_snp(x)))
    df = df.sort_values(['CHR', 'POS'])
    
    # 计算 -log10(P)
    df['logp'] = -np.log10(df['P_value'])
    
    # 3. 设定阈值
    n_snps = len(df)
    bonferroni_thresh = -np.log10(0.05 / n_snps)
    suggestive_thresh = -np.log10(1e-4)  # 宽松阈值

    # ==========================
    # 图表 1: 曼哈顿图 (Manhattan Plot)
    # ==========================
    plt.figure(figsize=(12, 6))
    
    # 为每个染色体设置颜色和X轴位置
    df['ind'] = range(len(df))
    df_grouped = df.groupby(('CHR'))
    
    colors = ['#4c72b0', '#55a868'] # 交替颜色
    x_labels = []
    x_labels_pos = []
    
    for num, (name, group) in enumerate(df_grouped):
        group = group.reset_index()
        plt.scatter(group['ind'], group['logp'], color=colors[num % len(colors)], s=15, alpha=0.8)
        x_labels.append(name)
        x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0]) / 2))
    
    # 画阈值线
    plt.axhline(y=bonferroni_thresh, color='r', linestyle='--', linewidth=1, label=f'Bonferroni ({bonferroni_thresh:.2f})')
    plt.axhline(y=suggestive_thresh, color='b', linestyle='--', linewidth=1, label='Suggestive (1e-4)')
    
    plt.title('DoubleML GWAS Manhattan Plot (Stem Color)', fontsize=14)
    plt.ylabel('-log10(P-value)')
    plt.xlabel('Chromosome')
    plt.xticks(x_labels_pos, x_labels, rotation=45, fontsize=8)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # ==========================
    # 图表 2: QQ 图 (QQ Plot)
    # ==========================
    plt.figure(figsize=(6, 6))
    
    expected = -np.log10(np.arange(1, n_snps + 1) / (n_snps + 1))
    observed = -np.log10(np.sort(df['P_value']))
    
    plt.scatter(expected, observed, s=15, color='#c44e52')
    plt.plot([0, max(expected)], [0, max(expected)], color='grey', linestyle='--')
    
    plt.xlabel('Expected -log10(P)')
    plt.ylabel('Observed -log10(P)')
    plt.title('QQ Plot')
    
    # 计算膨胀因子 lambda
    chi2_obs = stats.norm.ppf(1 - df['P_value'] / 2) ** 2
    lambda_gc = np.median(chi2_obs) / stats.chi2.ppf(0.5, 1)
    plt.text(0.1, max(observed)*0.9, f'λ GC = {lambda_gc:.3f}', fontsize=12)
    
    plt.tight_layout()
    plt.show()

    # ==========================
    # 图表 3: 火山图 (Volcano Plot)
    # ==========================
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Coef'], df['logp'], s=15, alpha=0.6, c=df['logp'], cmap='viridis')
    plt.axhline(y=bonferroni_thresh, color='r', linestyle='--', linewidth=1)
    plt.xlabel('Causal Effect Size (Coefficient)')
    plt.ylabel('-log10(P-value)')
    plt.title('Volcano Plot: Effect Size vs Significance')
    plt.colorbar(label='-log10(P)')
    plt.show()

    # ==========================
    # 4. 输出 Top Hits 表格
    # ==========================
    print("\n🏆 Top 10 最显著的 SNP:")
    cols = ['SNP', 'Coef', 'SE', 'P_value']
    print(df[cols].head(10).to_markdown(index=False))

# 运行分析
analyze_and_plot('final_gwas_results.csv')
