"""
简化版动态相互作用分析脚本
快速验证VAR模型和基本分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller

# 设置显示选项
pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = [12, 8]

def quick_analysis():
    """快速分析函数"""
    print("加载数据...")
    
    # 读取数据
    umcsent = pd.read_csv("UMCSENT.csv")
    dspic96 = pd.read_csv("DSPIC96.csv") 
    pce = pd.read_csv("PCE.csv")
    totalsl = pd.read_csv("TOTALSL.csv")
    
    # 处理日期
    for df in [umcsent, dspic96, pce, totalsl]:
        df['observation_date'] = pd.to_datetime(df['observation_date'])
    
    # 合并数据
    data = umcsent.merge(dspic96, on='observation_date', how='inner') \
                  .merge(pce, on='observation_date', how='inner') \
                  .merge(totalsl, on='observation_date', how='inner')
    
    data.set_index('observation_date', inplace=True)
    data.columns = ['消费者信心', '实际收入', '消费支出', '信贷总额']
    
    print(f"数据维度: {data.shape}")
    print(f"时间范围: {data.index[0]} 到 {data.index[-1]}")
    
    # 描述统计
    print("\n描述性统计:")
    print(data.describe())
    
    # 相关性分析
    print("\n相关性矩阵:")
    corr_matrix = data.corr()
    print(corr_matrix)
    
    # 可视化相关性
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f')
    plt.title('变量间相关性热力图')
    plt.tight_layout()
    plt.show()
    
    # 平稳性检验
    print("\n平稳性检验 (ADF):")
    for col in data.columns:
        adf_stat, p_value = adfuller(data[col].dropna())[:2]
        print(f"{col}: ADF统计量={adf_stat:.4f}, p值={p_value:.4f}, "
              f"{'平稳' if p_value < 0.05 else '非平稳'}")
    
    # 一阶差分
    data_diff = data.diff().dropna()
    print("\n一阶差分后的平稳性检验:")
    for col in data_diff.columns:
        adf_stat, p_value = adfuller(data_diff[col].dropna())[:2]
        print(f"{col}: ADF统计量={adf_stat:.4f}, p值={p_value:.4f}, "
              f"{'平稳' if p_value < 0.05 else '非平稳'}")
    
    # VAR模型
    print("\nVAR模型分析:")
    model = VAR(data_diff)
    
    # 选择滞后期
    lag_selection = model.select_order(maxlags=6)
    optimal_lag = lag_selection.aic
    print(f"最优滞后期 (AIC): {optimal_lag}")
    
    # 估计模型
    var_model = model.fit(optimal_lag)
    print(f"\nVAR({optimal_lag})模型估计完成")
    
    # 脉冲响应分析
    print("\n脉冲响应分析...")
    irf = var_model.irf(periods=10)
    
    # 绘制脉冲响应图
    fig = irf.plot(figsize=(15, 12))
    plt.suptitle('脉冲响应函数', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 方差分解
    print("\n方差分解分析...")
    fevd = var_model.fevd(periods=10)
    
    # 绘制方差分解图  
    fig = fevd.plot(figsize=(15, 10))
    plt.suptitle('方差分解', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("\n分析完成！")
    return data, var_model, irf, fevd

if __name__ == "__main__":
    data, model, irf, fevd = quick_analysis()
