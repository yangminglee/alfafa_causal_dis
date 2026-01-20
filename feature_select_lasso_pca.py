import pandas as pd
import numpy as np
import argparse
import os
import time
from doubleml import DoubleMLPLR, DoubleMLData
from sklearn.linear_model import LassoCV
import warnings
warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'",
    category=FutureWarning
)
N_PCS = 10                  
N_REPS = 100
DATA_DIR = os.path.expanduser('~/data/plant/alfalfa')
GENO_FILE = os.path.join(DATA_DIR, 'geno.csv')
PHENO_FILE = os.path.join(DATA_DIR, 'pheno.csv')
PCA_FILE = os.path.join(DATA_DIR, 'pca_scores_'+str(N_PCS)+'.csv')

def parse_args():
    parser = argparse.ArgumentParser(description="DoubleML GWAS Full Scan")
    parser.add_argument("--trait", type=str, required=True, help="Target phenotype column name")
    parser.add_argument("--output", type=str, required=True, help="Output filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility and splitting")
    return parser.parse_args()

def load_and_align_data(trait):
    if not os.path.exists(PCA_FILE):
        raise FileNotFoundError(f"找不到 {PCA_FILE}，请确保已生成 PCA 数据。")
    
    geno = pd.read_csv(GENO_FILE)
    pheno = pd.read_csv(PHENO_FILE)
    pca = pd.read_csv(PCA_FILE)
    
    for df in [geno, pheno, pca]:
        if 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'Sample_ID'}, inplace=True)
        if 'Sample_ID' in df.columns:
            df.set_index('Sample_ID', inplace=True)
            
    common = geno.index.intersection(pheno.index).intersection(pca.index)
    if len(common) == 0:
        raise ValueError("没有找到重叠样本，请检查 Sample_ID。")
    
    geno = geno.loc[common]
    pheno = pheno.loc[common]
    pca = pca.loc[common]
    
    if trait not in pheno.columns:
        raise ValueError(f"表型文件中找不到性状: {trait}")
        
    mask = ~pheno[trait].isna()
    
    return geno.loc[mask], pheno.loc[mask], pca.loc[mask]

def main():
    args = parse_args()
    target_trait = args.trait
    output_filename = args.output
    seed = args.seed
    
    np.random.seed(seed)
    
    try:
        df_geno, df_pheno, df_pca = load_and_align_data(target_trait)
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        return

    snp_list = df_geno.columns.tolist()
    pc_cols = df_pca.columns.tolist()
    
    print(f"\n>>> 开始全基因组扫描 (Seed: {seed})")
    print(f"   - 目标性状: {target_trait}")
    print(f"   - 样本量: {len(df_geno)}")
    print(f"   - SNP 数量: {len(snp_list)}")
    
    results = []
    start_time = time.time()
    
    # 预先配置学习器 (LassoCV)
    # n_jobs=1 避免在 Slurm 任务内部再进行多进程争抢
    ml_l = LassoCV(cv=5, n_jobs=-1, random_state=seed)
    ml_m = LassoCV(cv=5, n_jobs=-1, random_state=seed)
    
    for i, snp in enumerate(snp_list):
        # 构造数据块：Y + 当前SNP + PCs
        try:
            local_df = pd.concat([
                df_pheno[[target_trait]], 
                df_geno[[snp]], 
                df_pca
            ], axis=1)
            
            dml_data = DoubleMLData(
                local_df,
                y_col=target_trait,
                d_cols=snp,
                x_cols=pc_cols
            )
            
            # 运行模型 (Standard PLR)
            # 去掉 apply_cross_fitting 参数 (新版默认)
            dml_plr = DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=5, n_rep=N_REPS)
            dml_plr.fit()
            
            # 提取统计量
            summary = dml_plr.summary
            res_dict = {
                'SNP': snp,
                'Coef': summary.iloc[0]['coef'],
                'SE': summary.iloc[0]['std err'],
                'P_value': summary.iloc[0]['P>|t|'],
                'Seed': seed
            }
            results.append(res_dict)
            
        except Exception as e:
            # 遇到个别 SNP 报错跳过，不中断整个任务
            continue

    # 2. 结果保存
    final_df = pd.DataFrame(results)
    if 'P_value' in final_df.columns:
        final_df = final_df.sort_values('P_value')
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(output_filename, index=False)
    
    print(f"\n>>> ✅ 扫描完成！耗时: {(time.time()-start_time)/60:.2f} 分钟 for n_rep {N_REPS}")

if __name__ == "__main__":
    main()
