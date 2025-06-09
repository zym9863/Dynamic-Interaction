[English](README_EN.md) | [中文](README.md)

# 消费者信心、收入、信贷状况与实际消费支出动态相互作用分析

## 项目概述

本项目使用多元时间序列模型（VAR - Vector Autoregression）分析消费者信心、收入、信贷状况与实际消费支出之间的动态相互作用关系。

## 数据说明

- **UMCSENT**: 消费者信心指数 (University of Michigan Consumer Sentiment)
- **DSPIC96**: 实际可支配个人收入 (2012年美元)
- **PCE**: 个人消费支出 (Personal Consumption Expenditures, 十亿美元)
- **TOTALSL**: 消费者信贷总额 (百万美元)

## 环境配置

### 1. 创建conda虚拟环境

```bash
# 创建新的conda环境
conda create -n dynamic_interaction python=3.9

# 激活环境
conda activate dynamic_interaction
```

### 2. 安装依赖包

```bash
# 安装所需的Python包
pip install -r requirements.txt

# 或者使用conda安装
conda install pandas numpy matplotlib seaborn scipy
conda install -c conda-forge statsmodels
```

### 3. 验证安装

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
print("所有包安装成功！")
```

## 文件结构

```
Dynamic Interaction/
├── UMCSENT.csv          # 消费者信心数据
├── DSPIC96.csv          # 实际收入数据  
├── PCE.csv              # 消费支出数据
├── TOTALSL.csv          # 信贷总额数据
├── requirements.txt     # 依赖包列表
├── dynamic_interaction_analysis.py  # 完整分析脚本
├── quick_analysis.py    # 快速分析脚本
└── README.md           # 说明文档
```

## 运行方式

### 方式1: 完整分析 (推荐)

```bash
python dynamic_interaction_analysis.py
```

这将运行完整的分析流程，包括：
- 数据加载和预处理
- 描述性统计分析
- 平稳性检验 (ADF检验)
- 协整检验 (Johansen检验)
- 格兰杰因果关系检验
- VAR模型估计
- 脉冲响应分析
- 方差分解分析
- 生成分析报告

### 方式2: 快速分析

```bash
python quick_analysis.py
```

这将运行简化版分析，适合快速验证和初步探索。

## 分析方法

### 1. 平稳性检验
使用ADF检验判断时间序列的平稳性，对非平稳序列进行差分处理。

### 2. 协整检验
使用Johansen检验判断变量间是否存在长期均衡关系。

### 3. 格兰杰因果关系检验
检验变量间的因果关系方向和强度。

### 4. VAR模型
建立向量自回归模型，分析变量间的动态相互作用。

### 5. 脉冲响应分析
分析一个变量的冲击对其他变量的动态影响。

### 6. 方差分解
分析各变量对预测误差方差的贡献度。

## 输出结果

运行完整分析后将生成以下文件：

1. **时间序列图.png** - 各变量的时间趋势图
2. **相关性热力图.png** - 变量间相关性矩阵热力图
3. **脉冲响应图.png** - 脉冲响应函数图
4. **正交化脉冲响应图.png** - 正交化脉冲响应函数图
5. **方差分解图.png** - 方差分解图
6. **分析报告.txt** - 详细的分析报告

## 注意事项

1. 确保所有CSV文件都在同一目录下
2. 数据文件的日期格式应为 'YYYY-MM-DD'
3. 如果遇到中文字体显示问题，请安装相应的中文字体
4. 建议在Jupyter Notebook中运行以获得更好的交互体验

## 技术要求

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- statsmodels >= 0.13.0
- scipy >= 1.7.0

## 联系信息

如有问题或建议，请通过GitHub Issues联系。
