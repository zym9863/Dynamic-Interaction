"""
消费者信心、收入、信贷状况与实际消费支出之间的动态相互作用关系分析
使用多元时间序列模型 (VAR - Vector Autoregression)

数据说明:
- UMCSENT: 消费者信心指数
- DSPIC96: 实际可支配个人收入 (2012年美元)
- PCE: 个人消费支出 (十亿美元)
- TOTALSL: 消费者信贷总额 (百万美元)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DynamicInteractionAnalysis:
    def __init__(self, data_path="./"):
        """初始化分析类"""
        self.data_path = data_path
        self.data = None
        self.var_model = None
        self.results = {}
        
    def load_and_merge_data(self):
        """加载并合并所有数据集"""
        print("正在加载数据...")
        
        # 读取各个数据文件
        umcsent = pd.read_csv(f"{self.data_path}UMCSENT.csv")
        dspic96 = pd.read_csv(f"{self.data_path}DSPIC96.csv")
        pce = pd.read_csv(f"{self.data_path}PCE.csv")
        totalsl = pd.read_csv(f"{self.data_path}TOTALSL.csv")
        
        # 转换日期格式
        for df in [umcsent, dspic96, pce, totalsl]:
            df['observation_date'] = pd.to_datetime(df['observation_date'])
        
        # 合并数据
        self.data = umcsent.merge(dspic96, on='observation_date', how='inner') \
                          .merge(pce, on='observation_date', how='inner') \
                          .merge(totalsl, on='observation_date', how='inner')
        
        # 设置日期为索引
        self.data.set_index('observation_date', inplace=True)
        
        # 重命名列以便理解
        self.data.columns = ['消费者信心', '实际收入', '消费支出', '信贷总额']
        
        print(f"数据加载完成，共{len(self.data)}个观测值")
        print(f"时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
        print("\n数据预览:")
        print(self.data.head())
        
        return self.data
    
    def descriptive_statistics(self):
        """描述性统计分析"""
        print("\n" + "="*50)
        print("描述性统计分析")
        print("="*50)
        
        # 基本统计信息
        print("\n基本统计信息:")
        print(self.data.describe())
        
        # 相关性分析
        print("\n相关性矩阵:")
        correlation_matrix = self.data.corr()
        print(correlation_matrix)
        
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 时间序列图
        for i, col in enumerate(self.data.columns):
            row, col_idx = i // 2, i % 2
            axes[row, col_idx].plot(self.data.index, self.data[col])
            axes[row, col_idx].set_title(f'{col}时间序列图')
            axes[row, col_idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.data_path}时间序列图.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 相关性热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.3f')
        plt.title('变量间相关性热力图')
        plt.tight_layout()
        plt.savefig(f'{self.data_path}相关性热力图.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.results['correlation_matrix'] = correlation_matrix
    
    def stationarity_test(self):
        """平稳性检验"""
        print("\n" + "="*50)
        print("平稳性检验 (ADF检验)")
        print("="*50)
        
        adf_results = {}
        
        for col in self.data.columns:
            # 原始序列ADF检验
            adf_stat, p_value, _, _, critical_values, _ = adfuller(self.data[col].dropna())
            
            print(f"\n{col}:")
            print(f"  ADF统计量: {adf_stat:.4f}")
            print(f"  p值: {p_value:.4f}")
            print(f"  临界值: {critical_values}")
            
            if p_value <= 0.05:
                print(f"  结论: {col}是平稳的 (p < 0.05)")
                is_stationary = True
            else:
                print(f"  结论: {col}不是平稳的 (p >= 0.05)")
                is_stationary = False
            
            adf_results[col] = {
                'adf_statistic': adf_stat,
                'p_value': p_value,
                'critical_values': critical_values,
                'is_stationary': is_stationary
            }
        
        self.results['stationarity_test'] = adf_results
        
        # 如果存在非平稳序列，进行一阶差分
        non_stationary = [col for col, result in adf_results.items() 
                         if not result['is_stationary']]
        
        if non_stationary:
            print(f"\n检测到非平稳序列: {non_stationary}")
            print("进行一阶差分处理...")
            
            self.data_diff = self.data.diff().dropna()
            
            print("\n一阶差分后的ADF检验:")
            for col in non_stationary:
                adf_stat, p_value, _, _, critical_values, _ = adfuller(self.data_diff[col].dropna())
                print(f"\n{col} (一阶差分):")
                print(f"  ADF统计量: {adf_stat:.4f}")
                print(f"  p值: {p_value:.4f}")
                
                if p_value <= 0.05:
                    print(f"  结论: 一阶差分后{col}是平稳的")
                else:
                    print(f"  结论: 一阶差分后{col}仍不平稳，可能需要更高阶差分")
        
        return adf_results
    
    def cointegration_test(self):
        """协整检验 (Johansen检验)"""
        print("\n" + "="*50)
        print("协整检验 (Johansen检验)")
        print("="*50)
        
        # 进行Johansen协整检验
        result = coint_johansen(self.data.values, det_order=0, k_ar_diff=1)
        
        print("Johansen协整检验结果:")
        print("迹统计量:")
        for i in range(len(result.lr1)):
            print(f"  r <= {i}: {result.lr1[i]:.4f} (临界值90%: {result.cvt[i, 0]:.4f}, "
                  f"95%: {result.cvt[i, 1]:.4f}, 99%: {result.cvt[i, 2]:.4f})")
        
        print("\n最大特征值统计量:")
        for i in range(len(result.lr2)):
            print(f"  r = {i}: {result.lr2[i]:.4f} (临界值90%: {result.cvm[i, 0]:.4f}, "
                  f"95%: {result.cvm[i, 1]:.4f}, 99%: {result.cvm[i, 2]:.4f})")
        
        # 判断协整关系个数
        cointegration_rank = 0
        for i in range(len(result.lr1)):
            if result.lr1[i] > result.cvt[i, 1]:  # 95%显著性水平
                cointegration_rank = len(result.lr1) - i
                break
        
        print(f"\n在95%显著性水平下，协整关系个数: {cointegration_rank}")
        
        self.results['cointegration'] = {
            'johansen_result': result,
            'cointegration_rank': cointegration_rank
        }
        
        return result
    
    def granger_causality_test(self):
        """格兰杰因果关系检验"""
        print("\n" + "="*50)
        print("格兰杰因果关系检验")
        print("="*50)
        
        variables = self.data.columns.tolist()
        causality_results = {}
        
        # 使用一阶差分数据（如果原数据非平稳）
        if hasattr(self, 'data_diff'):
            test_data = self.data_diff
            print("使用一阶差分数据进行检验")
        else:
            test_data = self.data
            print("使用原始数据进行检验")
        
        print(f"\n检验滞后期: 1-4期")
        
        for i, cause_var in enumerate(variables):
            for j, effect_var in enumerate(variables):
                if i != j:
                    print(f"\n检验 '{cause_var}' 是否格兰杰引起 '{effect_var}':")
                    
                    # 准备数据
                    test_data_pair = test_data[[effect_var, cause_var]].dropna()
                    
                    try:
                        # 进行格兰杰因果关系检验（滞后期1-4）
                        gc_result = grangercausalitytests(test_data_pair, maxlag=4, verbose=False)
                        
                        causality_results[f"{cause_var} -> {effect_var}"] = gc_result
                        
                        # 输出结果
                        for lag in range(1, 5):
                            f_stat = gc_result[lag][0]['ssr_ftest'][0]
                            p_value = gc_result[lag][0]['ssr_ftest'][1]
                            
                            significance = ""
                            if p_value < 0.01:
                                significance = "***"
                            elif p_value < 0.05:
                                significance = "**"
                            elif p_value < 0.1:
                                significance = "*"
                            
                            print(f"  滞后{lag}期: F统计量={f_stat:.4f}, p值={p_value:.4f}{significance}")
                    
                    except Exception as e:
                        print(f"  检验失败: {str(e)}")
        
        self.results['granger_causality'] = causality_results
        
        return causality_results
    
    def var_model_estimation(self):
        """VAR模型估计"""
        print("\n" + "="*50)
        print("VAR模型估计")
        print("="*50)
        
        # 确定使用的数据
        if hasattr(self, 'data_diff'):
            model_data = self.data_diff
            print("使用一阶差分数据建立VAR模型")
        else:
            model_data = self.data
            print("使用原始数据建立VAR模型")
        
        # 确定最优滞后期
        model = VAR(model_data)
        lag_selection = model.select_order(maxlags=8)
        
        print("\n滞后期选择结果:")
        print(lag_selection.summary())
        
        # 选择AIC最小的滞后期
        optimal_lag = lag_selection.aic
        print(f"\n根据AIC准则选择最优滞后期: {optimal_lag}")
        
        # 估计VAR模型
        self.var_model = model.fit(optimal_lag)
        
        print(f"\nVAR({optimal_lag})模型估计结果:")
        print(self.var_model.summary())
        
        # 模型诊断
        print("\n模型诊断:")
        
        # 残差序列相关性检验
        ljung_box_results = {}
        residuals = self.var_model.resid
        for i, col in enumerate(model_data.columns):
            lb_result = acorr_ljungbox(residuals.iloc[:, i], lags=10, return_df=False)
            
            # 处理不同版本的statsmodels返回值
            if isinstance(lb_result, tuple) and len(lb_result) == 2:
                lb_stat, lb_pvalue = lb_result
                if hasattr(lb_stat, '__len__'):
                    lb_stat_val = float(lb_stat[-1])
                else:
                    lb_stat_val = float(lb_stat)
                if hasattr(lb_pvalue, '__len__'):
                    lb_pvalue_val = float(lb_pvalue[-1])
                else:
                    lb_pvalue_val = float(lb_pvalue)
            else:
                # 如果返回的是DataFrame或其他格式
                if hasattr(lb_result, 'iloc'):
                    lb_stat_val = float(lb_result.iloc[-1, 0])
                    lb_pvalue_val = float(lb_result.iloc[-1, 1])
                else:
                    # 尝试直接转换
                    lb_stat_val = float(lb_result)
                    lb_pvalue_val = 0.05  # 默认值
            
            ljung_box_results[col] = {'statistic': lb_stat_val, 'p_value': lb_pvalue_val}
            
            if lb_pvalue_val > 0.05:
                print(f"  {col}残差: 无序列相关性 (Ljung-Box p值: {lb_pvalue_val:.4f})")
            else:
                print(f"  {col}残差: 存在序列相关性 (Ljung-Box p值: {lb_pvalue_val:.4f})")
        
        self.results['var_model'] = {
            'model': self.var_model,
            'optimal_lag': optimal_lag,
            'ljung_box': ljung_box_results
        }
        
        return self.var_model
    
    def impulse_response_analysis(self):
        """脉冲响应分析"""
        print("\n" + "="*50)
        print("脉冲响应分析")
        print("="*50)
        
        if self.var_model is None:
            print("请先估计VAR模型")
            return
        
        # 计算脉冲响应函数
        irf = self.var_model.irf(periods=10)
        
        # 绘制脉冲响应图
        fig = irf.plot(orth=False, figsize=(15, 12))
        plt.suptitle('脉冲响应函数图', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.data_path}脉冲响应图.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 正交化脉冲响应
        fig = irf.plot(orth=True, figsize=(15, 12))
        plt.suptitle('正交化脉冲响应函数图', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.data_path}正交化脉冲响应图.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.results['impulse_response'] = irf
        
        return irf
    
    def variance_decomposition(self):
        """方差分解分析"""
        print("\n" + "="*50)
        print("方差分解分析")
        print("="*50)
        
        if self.var_model is None:
            print("请先估计VAR模型")
            return
        
        # 计算方差分解
        fevd = self.var_model.fevd(periods=10)
        
        # 输出方差分解结果
        print("\n方差分解结果 (10期预测):")
        print(fevd.summary())
        
        # 绘制方差分解图
        fig = fevd.plot(figsize=(15, 10))
        plt.suptitle('方差分解图', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.data_path}方差分解图.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.results['variance_decomposition'] = fevd
        
        return fevd
    
    def generate_report(self):
        """生成分析报告"""
        print("\n" + "="*50)
        print("分析报告总结")
        print("="*50)
        
        report = []
        report.append("消费者信心、收入、信贷状况与实际消费支出动态相互作用分析报告")
        report.append("="*60)
        
        # 数据基本信息
        report.append(f"\n1. 数据基本信息")
        report.append(f"   - 观测期间: {self.data.index[0].strftime('%Y-%m')} 到 {self.data.index[-1].strftime('%Y-%m')}")
        report.append(f"   - 观测数量: {len(self.data)}个月度数据")
        report.append(f"   - 变量数量: {len(self.data.columns)}个")
        
        # 相关性分析结果
        if 'correlation_matrix' in self.results:
            corr = self.results['correlation_matrix']
            report.append(f"\n2. 变量间相关性分析")
            
            # 找出最强的正相关和负相关
            corr_flat = corr.values[np.triu_indices_from(corr.values, k=1)]
            max_corr = np.max(corr_flat)
            min_corr = np.min(corr_flat)
            
            report.append(f"   - 最强正相关: {max_corr:.3f}")
            report.append(f"   - 最强负相关: {min_corr:.3f}")
        
        # 平稳性检验结果
        if 'stationarity_test' in self.results:
            stationary_vars = [var for var, result in self.results['stationarity_test'].items() 
                             if result['is_stationary']]
            non_stationary_vars = [var for var, result in self.results['stationarity_test'].items() 
                                 if not result['is_stationary']]
            
            report.append(f"\n3. 平稳性检验结果")
            if stationary_vars:
                report.append(f"   - 平稳变量: {', '.join(stationary_vars)}")
            if non_stationary_vars:
                report.append(f"   - 非平稳变量: {', '.join(non_stationary_vars)}")
                report.append(f"   - 处理方法: 一阶差分")
        
        # 协整关系
        if 'cointegration' in self.results:
            cointegration_rank = self.results['cointegration']['cointegration_rank']
            report.append(f"\n4. 协整关系")
            report.append(f"   - 协整关系个数: {cointegration_rank}")
            if cointegration_rank > 0:
                report.append(f"   - 结论: 变量间存在长期均衡关系")
            else:
                report.append(f"   - 结论: 变量间不存在长期均衡关系")
        
        # VAR模型结果
        if 'var_model' in self.results:
            optimal_lag = self.results['var_model']['optimal_lag']
            report.append(f"\n5. VAR模型")
            report.append(f"   - 最优滞后期: {optimal_lag}")
            report.append(f"   - 模型类型: VAR({optimal_lag})")
        
        # 格兰杰因果关系
        if 'granger_causality' in self.results:
            report.append(f"\n6. 格兰杰因果关系 (5%显著性水平)")
            causality_found = False
            
            for relationship, results in self.results['granger_causality'].items():
                for lag in range(1, 5):
                    if lag in results:
                        p_value = results[lag][0]['ssr_ftest'][1]
                        if p_value < 0.05:
                            report.append(f"   - {relationship} (滞后{lag}期): 显著 (p={p_value:.3f})")
                            causality_found = True
            
            if not causality_found:
                report.append(f"   - 在5%显著性水平下未发现显著的格兰杰因果关系")
        
        # 政策建议
        report.append(f"\n7. 政策建议")
        report.append(f"   - 基于模型结果，建议关注各变量间的动态相互作用")
        report.append(f"   - 消费者信心可能对实际消费支出有重要影响")
        report.append(f"   - 收入和信贷政策的调整需要考虑其对消费的滞后效应")
        report.append(f"   - 建议结合脉冲响应分析制定具体的政策措施")
        
        # 保存报告
        report_text = '\n'.join(report)
        with open(f'{self.data_path}分析报告.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        
        return report_text
    
    def run_full_analysis(self):
        """运行完整分析流程"""
        print("开始进行消费者信心、收入、信贷状况与实际消费支出动态相互作用分析")
        print("="*70)
        
        # 1. 加载数据
        self.load_and_merge_data()
        
        # 2. 描述性统计
        self.descriptive_statistics()
        
        # 3. 平稳性检验
        self.stationarity_test()
        
        # 4. 协整检验
        self.cointegration_test()
        
        # 5. 格兰杰因果关系检验
        self.granger_causality_test()
        
        # 6. VAR模型估计
        self.var_model_estimation()
        
        # 7. 脉冲响应分析
        self.impulse_response_analysis()
        
        # 8. 方差分解
        self.variance_decomposition()
        
        # 9. 生成报告
        self.generate_report()
        
        print("\n分析完成！")
        print("生成的文件:")
        print("- 时间序列图.png")
        print("- 相关性热力图.png") 
        print("- 脉冲响应图.png")
        print("- 正交化脉冲响应图.png")
        print("- 方差分解图.png")
        print("- 分析报告.txt")


if __name__ == "__main__":
    # 创建分析实例
    analyzer = DynamicInteractionAnalysis()
    
    # 运行完整分析
    analyzer.run_full_analysis()
