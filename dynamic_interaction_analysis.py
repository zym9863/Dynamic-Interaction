"""
消费者信心、收入、信贷状况与实际消费支出之间的动态相互作用关系分析
使用多元时间序列模型 (VAR - Vector Autoregression)

数据说明:
- UMCSENT: 消费者信心指数
- DSPIC96: 实际可支配个人收入 (2012年美元)
- PCE: 个人消费支出 (十亿美元)
- TOTALSL: 消费者信贷总额 (百万美元)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import jarque_bera
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DynamicInteractionAnalysis:
    def __init__(self, data_path="./"):
        """初始化分析类

        参数:
            data_path (str): 数据与输出文件所在目录路径
        """
        self.data_path = data_path
        self.data = None
        self.var_model = None
        self.results = {}
        self.data_diff = None
        
    def load_and_merge_data(self):
        """加载并合并所有数据集（含异常处理与文件检查）

        返回:
            pd.DataFrame: 合并后的数据，索引为日期
        异常:
            FileNotFoundError: 当必需数据文件缺失时
            ValueError: 当数据列缺失或合并失败时
            Exception: 其他不可预期错误
        """
        print("正在加载数据...")

        required_files = ["UMCSENT.csv", "DSPIC96.csv", "PCE.csv", "TOTALSL.csv"]

        # 确保路径存在
        for fname in required_files:
            fpath = os.path.join(self.data_path, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"数据文件缺失: {fpath}")

        try:
            # 读取各个数据文件
            umcsent = pd.read_csv(os.path.join(self.data_path, "UMCSENT.csv"))
            dspic96 = pd.read_csv(os.path.join(self.data_path, "DSPIC96.csv"))
            pce = pd.read_csv(os.path.join(self.data_path, "PCE.csv"))
            totalsl = pd.read_csv(os.path.join(self.data_path, "TOTALSL.csv"))

            # 基本列校验
            for name, df in {
                "UMCSENT": umcsent,
                "DSPIC96": dspic96,
                "PCE": pce,
                "TOTALSL": totalsl,
            }.items():
                if "observation_date" not in df.columns:
                    raise ValueError(f"{name} 缺少 observation_date 列")
                if df.shape[1] < 2:
                    raise ValueError(f"{name} 数据列不足，期望2列以上")

            # 转换日期格式
            for df in [umcsent, dspic96, pce, totalsl]:
                df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
            
            # 丢弃无效日期
            for name, df in {"UMCSENT": umcsent, "DSPIC96": dspic96, "PCE": pce, "TOTALSL": totalsl}.items():
                before = len(df)
                df.dropna(subset=["observation_date"], inplace=True)
                after = len(df)
                if after < before:
                    print(f"警告: {name} 含有无效日期记录，已丢弃 {before-after} 行")

            # 合并数据
            data = umcsent.merge(dspic96, on="observation_date", how="inner") \
                           .merge(pce, on="observation_date", how="inner") \
                           .merge(totalsl, on="observation_date", how="inner")

            if data.empty:
                raise ValueError("合并后数据为空，请检查各文件的日期范围是否有交集")

            # 设置日期为索引并重命名列
            data.set_index("observation_date", inplace=True)

            # 自动识别指标列（每个文件假定除日期列外仅一列数值列）
            cols = []
            for df, cname in [
                (umcsent, "消费者信心"),
                (dspic96, "实际收入"),
                (pce, "消费支出"),
                (totalsl, "信贷总额"),
            ]:
                value_cols = [c for c in df.columns if c != "observation_date"]
                if not value_cols:
                    raise ValueError("无法识别数值列")
                cols.append(value_cols[0])

            # 重命名到统一中文名
            data.columns = ["消费者信心", "实际收入", "消费支出", "信贷总额"] if len(data.columns) == 4 else data.columns

            self.data = data

            print(f"数据加载完成，共{len(self.data)}个观测值")
            print(f"时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
            print("\n数据预览:")
            print(self.data.head())
            return self.data

        except pd.errors.EmptyDataError as e:
            raise ValueError(f"读取到空数据文件: {e}") from e
        except Exception as e:
            # 将异常抛出，以便上层感知
            raise
    
    def descriptive_statistics(self):
        """描述性统计分析

        生成基础统计、时间序列图与相关性热力图，并将相关性矩阵保存到 results。
        """
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

    def validate_data(self):
        """数据质量验证（缺失/类型/异常值/日期连续性）

        返回:
            dict: 数据质量报告，包括缺失值计数、异常值计数等
        """
        print("\n" + "="*50)
        print("数据质量检测")
        print("="*50)

        report = {
            "missing": self.data.isnull().sum().to_dict(),
            "outliers": {},
            "date_freq": None,
            "non_numeric": [],
        }

        # 非数值列（除索引外）检查
        non_numeric_cols = self.data.select_dtypes(include=[object]).columns.tolist()
        for col in non_numeric_cols:
            report["non_numeric"].append(col)
            print(f"警告: 列 {col} 含非数值类型")

        # 缺失值报告
        if any(v > 0 for v in report["missing"].values()):
            print(f"发现缺失值: {report['missing']}")
        else:
            print("未发现缺失值")

        # 异常值（IQR）
        for col in self.data.select_dtypes(include=[np.number]).columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.data[col] < (Q1 - 1.5 * IQR)) | (self.data[col] > (Q3 + 1.5 * IQR))).sum()
            report["outliers"][col] = int(outliers)
            if outliers > 0:
                print(f"{col} 发现 {outliers} 个潜在异常值")

        # 日期频率与连续性检查（按月）
        try:
            inferred = pd.infer_freq(self.data.index)
            report["date_freq"] = inferred
            if inferred is None:
                print("提示: 无法推断固定频率，可能存在日期缺口")
            else:
                # 简要缺口检查
                full_range = pd.date_range(self.data.index.min(), self.data.index.max(), freq=inferred)
                missing_dates = full_range.difference(self.data.index)
                if len(missing_dates) > 0:
                    print(f"日期缺口数量: {len(missing_dates)}")
        except Exception:
            print("日期频率推断失败")

        self.results["data_quality"] = report
        return report
    
    def stationarity_test(self):
        """平稳性检验（ADF）

        对每个变量进行ADF检验，并在必要时自动构造一阶差分数据。
        返回:
            dict: 每个变量的ADF统计与结论
        """
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
        
        # 仅保存必要的摘要，避免存储大型对象
        self.results['cointegration'] = {
            'trace_stats': result.lr1.tolist(),
            'trace_crit_95': result.cvt[:, 1].tolist(),
            'maxeig_stats': result.lr2.tolist(),
            'maxeig_crit_95': result.cvm[:, 1].tolist(),
            'cointegration_rank': int(cointegration_rank)
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
        """VAR模型估计

        自动选择滞后阶，拟合模型并执行基础残差诊断。仅在results中保存轻量摘要。
        返回:
            VARResults: 拟合后的VAR模型
        """
        print("\n" + "="*50)
        print("VAR模型估计")
        print("="*50)

        # 确定使用的数据
        if hasattr(self, 'data_diff') and self.data_diff is not None:
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
        optimal_lag = int(lag_selection.aic)
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

        # 只保存必要摘要，避免大型对象
        self.results['var_model'] = {
            'optimal_lag': int(optimal_lag),
            'nobs': int(self.var_model.nobs),
            'aic': float(self.var_model.aic) if hasattr(self.var_model, 'aic') else None,
            'bic': float(self.var_model.bic) if hasattr(self.var_model, 'bic') else None,
            'ljung_box': ljung_box_results,
        }

        return self.var_model
    
    def impulse_response_analysis(self):
        """脉冲响应分析

        计算标准与正交化脉冲响应，绘图并在results保存轻量数值数组。
        返回:
            Irf: statsmodels 的 IRF 对象（不存入results以节省内存）
        """
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

        # 仅保存轻量数组
        try:
            self.results['impulse_response'] = {
                'periods': 10,
                'irfs': irf.irfs.tolist(),           # shape: (periods+1, k, k)
                'orth_irfs': irf.orth_irfs.tolist()  # shape: (periods+1, k, k)
            }
        except Exception:
            self.results['impulse_response'] = {'periods': 10}

        return irf
    
    def variance_decomposition(self):
        """方差分解分析

        进行FEVD并绘图，在results中仅保存分解矩阵。
        返回:
            FEVD: statsmodels 的 FEVD 对象（不存入results以节省内存）
        """
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

        # 仅保存轻量数组
        try:
            self.results['variance_decomposition'] = {
                'periods': 10,
                'decomp': fevd.decomp.tolist()
            }
        except Exception:
            self.results['variance_decomposition'] = {'periods': 10}

        return fevd
    
    def generate_report(self):
        """生成分析报告

        汇总各步骤分析结果，输出到文本文件。
        返回:
            str: 报告文本
        """
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
        
        # 模型增强诊断（如已计算）
        if 'diagnostics' in self.results:
            diag = self.results['diagnostics']
            report.append(f"\n7. 模型诊断增强")
            report.append(f"   - 残差正态性(JB)显著比例: {diag.get('jb_significant_ratio', 'N/A')}")
            report.append(f"   - 异方差(ARCH)显著比例: {diag.get('arch_significant_ratio', 'N/A')}")
            report.append(f"   - 模型稳定性: {'稳定' if diag.get('is_stable', False) else '不稳定'}")

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
        """运行完整分析流程

        步骤: 加载与合并 -> 数据质量 -> 描述统计 -> 平稳性 -> 协整 -> 因果 -> VAR估计 -> IRF -> FEVD -> 报告
        """
        print("开始进行消费者信心、收入、信贷状况与实际消费支出动态相互作用分析")
        print("="*70)

        # 1. 加载数据
        self.load_and_merge_data()

        # 1.1 数据质量
        self.validate_data()

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

        # 6.1 模型增强诊断
        self.enhanced_diagnostics()

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

    def enhanced_diagnostics(self):
        """增强模型诊断（正态性、异方差、稳定性、预测评估）

        基于VAR残差进行Jarque-Bera正态性检验、ARCH异方差检验；
        使用根模数判断稳定性；并给出简单的滚动预测精度评估（RMSE）。
        """
        if self.var_model is None:
            print("请先估计VAR模型")
            return

        residuals = self.var_model.resid
        k = residuals.shape[1]

        jb_sig = 0
        arch_sig = 0
        jb_results = {}
        arch_results = {}

        for col in residuals.columns:
            # Jarque-Bera 正态性
            try:
                jb_stat, jb_pvalue, _, _ = jarque_bera(residuals[col])
                jb_results[col] = {"stat": float(jb_stat), "p_value": float(jb_pvalue)}
                if jb_pvalue < 0.05:
                    jb_sig += 1
            except Exception:
                jb_results[col] = {"stat": None, "p_value": None}

            # ARCH 异方差（滞后10）
            try:
                arch_stat, arch_pvalue, _, _ = het_arch(residuals[col], nlags=10)
                arch_results[col] = {"stat": float(arch_stat), "p_value": float(arch_pvalue)}
                if arch_pvalue < 0.05:
                    arch_sig += 1
            except Exception:
                arch_results[col] = {"stat": None, "p_value": None}

        # 稳定性（所有根模数 < 1）
        try:
            is_stable = bool(self.var_model.is_stable(verbose=False)) if hasattr(self.var_model, 'is_stable') else np.all(np.abs(self.var_model.roots) < 1)
        except Exception:
            is_stable = False

        diag = {
            "jb": jb_results,
            "arch": arch_results,
            "jb_significant_ratio": round(jb_sig / k, 3) if k else None,
            "arch_significant_ratio": round(arch_sig / k, 3) if k else None,
            "is_stable": is_stable,
        }

        # 简单预测评估（留出末尾12期，如不足则取 len//5）
        try:
            total_n = len(self.data)
            holdout = max(6, min(12, total_n // 5))
            if holdout >= 4 and total_n - holdout > 20:
                model_data = self.data_diff if self.data_diff is not None else self.data
                train = model_data.iloc[:-holdout]
                test = model_data.iloc[-holdout:]
                model = VAR(train)
                lag = int(self.results.get('var_model', {}).get('optimal_lag', 1))
                fitted = model.fit(lag)
                fc = fitted.forecast(train.values[-lag:], steps=holdout)
                fc = pd.DataFrame(fc, index=test.index, columns=test.columns)
                rmse = ((fc - test) ** 2).mean().pow(0.5).to_dict()
                diag["forecast_rmse"] = {k: float(v) for k, v in rmse.items()}
        except Exception:
            pass

        self.results['diagnostics'] = diag


if __name__ == "__main__":
    # 创建分析实例
    analyzer = DynamicInteractionAnalysis()
    
    # 运行完整分析
    analyzer.run_full_analysis()
