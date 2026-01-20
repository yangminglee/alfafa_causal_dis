import pandas as pd
import numpy as np
import os
import time
from doubleml import DoubleMLPLR, DoubleMLData
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

N_PCS = 10                  
TEST_SNP_INDEX = 50          # 测试第几个 SNP (跳过开头避免空值)

DATA_DIR = os.path.expanduser('~/data/plant/alfalfa')
GENO_FILE = os.path.join(DATA_DIR, 'geno.csv')
PHENO_FILE = os.path.join(DATA_DIR, 'pheno.csv')
PCA_FILE = os.path.join(DATA_DIR, 'pca_scores_'+str(N_PCS)+'.csv')
TARGET_TRAIT = 'stem_color'

def load_clean_dataframe(filepath):
    """读取并标准化索引的辅助函数"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到文件: {filepath}")
    
    df = pd.read_csv(filepath)
    # 处理常见的索引列名问题
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'Sample_ID'})
    if 'Sample_ID' in df.columns:
        df = df.set_index('Sample_ID')
    return df

def get_pca_data(geno_df):
    """
    获取 PCA 数据：
    1. 如果存在 PCA_FILE，直接读取。
    2. 如果不存在，计算并保存。
    """
    if os.path.exists(PCA_FILE):
        print(f"✅ 发现已保存的 PCA 文件: {PCA_FILE}，正在加载...")
        pca_df = load_clean_dataframe(PCA_FILE)
        # 简单验证一下索引是否匹配
        common = pca_df.index.intersection(geno_df.index)
        if len(common) < len(geno_df) * 0.5:
            print("⚠️ 警告: 保存的 PCA 文件样本与当前基因型文件重叠度较低，建议删除 pca_scores.csv 重新计算。")
        return pca_df
    else:
        print(f"⚡ 未找到 PCA 文件，正在开始计算 (N_PCS={N_PCS})...")
        start_time = time.time()
        
        # 1. 缺失值插补 (Mean Imputation)
        print("   - 正在进行缺失值插补...")
        imputer = SimpleImputer(strategy='mean')
        geno_imp = imputer.fit_transform(geno_df)
        
        # 2. 标准化 (Standardization)
        print("   - 正在进行标准化...")
        scaler = StandardScaler()
        geno_sc = scaler.fit_transform(geno_imp)
        
        # 3. PCA 计算
        print("   - 正在执行 PCA...")
        pca = PCA(n_components=N_PCS)
        pcs = pca.fit_transform(geno_sc)
        
        # 4. 构建 DataFrame 并保存
        pc_cols = [f"PC{i+1}" for i in range(N_PCS)]
        pca_df = pd.DataFrame(pcs, columns=pc_cols, index=geno_df.index)
        
        pca_df.to_csv(PCA_FILE)
        print(f"✅ PCA 计算完成，耗时 {time.time() - start_time:.2f}秒。结果已保存至 {PCA_FILE}")
        print(f"   - 解释方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")
        
        return pca_df

def main():
    print(">>> 1. 加载原始数据...")
    try:
        geno_df = load_clean_dataframe(GENO_FILE)
        pheno_df = load_clean_dataframe(PHENO_FILE)
        print(f"   - Geno shape: {geno_df.shape}")
        print(f"   - Pheno shape: {pheno_df.shape}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        return

    # ===========================
    # 2. 获取/计算 PCA (混杂因子)
    # ===========================
    # 注意：我们在筛选样本前计算 PCA，利用尽可能多的基因型信息来捕捉群体结构
    pca_df = get_pca_data(geno_df)
    pc_cols = pca_df.columns.tolist()

    # ===========================
    # 3. 数据对齐与合并
    # ===========================
    print(f"\n>>> 2. 数据对齐 (Trait: {TARGET_TRAIT})...")
    
    # 找到三者共有的样本
    common_samples = geno_df.index.intersection(pheno_df.index).intersection(pca_df.index)
    print(f"   - 共有样本数: {len(common_samples)}")
    
    if len(common_samples) == 0:
        print("❌ 错误: 没有找到重叠样本，请检查 sample_id 格式。")
        return

    # 筛选数据
    df_geno_s = geno_df.loc[common_samples]
    df_pheno_s = pheno_df.loc[common_samples]
    df_pca_s = pca_df.loc[common_samples]
    
    # 去除表型缺失值
    mask = ~df_pheno_s[TARGET_TRAIT].isna()
    df_geno_final = df_geno_s.loc[mask]
    df_pheno_final = df_pheno_s.loc[mask]
    df_pca_final = df_pca_s.loc[mask]
    
    print(f"   - 去除缺失值后的最终分析样本量 (N): {len(df_geno_final)}")

    # ===========================
    # 4. 运行单点 DoubleML
    # ===========================
    snp_cols = df_geno_final.columns.tolist()
    if TEST_SNP_INDEX >= len(snp_cols):
        print("❌ 错误: 测试索引超出了 SNP 总数。")
        return
        
    test_snp = snp_cols[TEST_SNP_INDEX]
    
    print(f"\n>>> 3. 运行 DoubleML 验证 (SNP: {test_snp})")
    print(f"   - Confounders (X): {pc_cols}")
    
    # 合并为一个大表供 DoubleML 使用
    # 数据结构: [Y, T, PC1, PC2...]
    data_merged = pd.concat([
        df_pheno_final[[TARGET_TRAIT]], 
        df_geno_final[[test_snp]], 
        df_pca_final
    ], axis=1)
    
    # 定义 DoubleML 数据对象
    dml_data = DoubleMLData(
        data_merged,
        y_col=TARGET_TRAIT,
        d_cols=test_snp,
        x_cols=pc_cols  # 这里放入读取好的 PCs
    )
    
    # 模型设置
    ml_l = LassoCV(cv=5, n_jobs=1, max_iter=10000)
    ml_m = LassoCV(cv=5, n_jobs=1, max_iter=10000)
    
    print("   - 正在拟合模型 (fit)...")
    start_time = time.time()
    
    # 初始化模型 (注意：已移除 apply_cross_fitting 参数)
    dml_plr = DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=5, n_rep=1)
    dml_plr.fit()
    
    duration = time.time() - start_time
    print(f"   - 拟合完成，耗时: {duration:.4f} 秒")

    # ===========================
    # 5. 结果与诊断
    # ===========================
    print("\n" + "="*50)
    print("               📊 分析结果报告")
    print("="*50)
    
    # A. 核心统计量
    res = dml_plr.summary
    coef = res.iloc[0]['coef']
    se = res.iloc[0]['std err']
    pval = res.iloc[0]['P>|t|']
    
    print(f"SNP ID:        {test_snp}")
    print(f"因果系数(Coef): {coef:.6f}")
    print(f"标准误(SE):     {se:.6f}")
    print(f"P值 (P-value):  {pval:.6e}")
    
    # B. 模型诊断 (R2)
    print("-" * 50)
    print("🔍 模型诊断 (Lasso 预测能力检查)")
    
    preds = dml_plr.predictions
    y_true = dml_data.y
    d_true = dml_data.d
    
    # 计算 R2
    r2_y = r2_score(y_true, preds['ml_l'].squeeze()) # PCs 预测 性状
    r2_d = r2_score(d_true, preds['ml_m'].squeeze()) # PCs 预测 SNP
    
    print(f"1. 混杂因子预测性状 (PCs -> Y): R2 = {r2_y:.4f}")
    print(f"2. 混杂因子预测 SNP (PCs -> D): R2 = {r2_d:.4f}")
    
    print("-" * 50)
    print("💡 结果解读:")
    
    if r2_y > 0.05:
        print("   ✅ 有效控制: PCs 对性状有解释力，说明群体结构是真正的混杂因子，DoubleML 正在发挥去偏作用。")
    elif r2_y < 0.01:
        print("   ℹ️  弱混杂: PCs 对性状几乎没有解释力。这意味着群体结构对该性状影响不大，结果接近普通回归。")
        
    if r2_d > 0.8:
        print("   ⚠️ 警告: SNP 被 PCs 高度预测。可能存在共线性问题，结果可能不稳定。")
    else:
        print("   ✅ SNP 独立性良好: SNP 变异未被群体结构完全覆盖。")

    print("="*50)

if __name__ == "__main__":
    main()
