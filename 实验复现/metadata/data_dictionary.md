# 数据字典（简版）

## 1. intermediate/merged_dataset.csv
- `LN_IC50`：GDSC 药敏目标变量（自然对数尺度）
- `DRUG_*`：药物 one-hot 指示变量
- `EXP_*`：基因表达特征（脚本中可转换为 `log2(RPKM+1)`）
- `ERBB2_CNV`：ERBB2 拷贝数相关特征
- `normalized_name`：标准化细胞系名称，用于跨库匹配

## 2. results/table1_model_performance.csv
- `模型`：模型名称（随机森林、线性回归、SVR）
- `目标变量`：默认 `LN_IC50`
- `RMSE`：测试集均方根误差
- `R2`：测试集决定系数
- `训练集占比` / `测试集占比`：数据划分比例
- `交叉验证折数`：交叉验证折数
- `随机种子`：随机种子
- `特征规模`：建模特征数量
- `样本量`：建模样本数

## 3. results/table2_top10_features.csv
- `rank`：重要性排名
- `feature_name`：特征名称
- `data_type`：特征类型（表达/拷贝数/其他）
- `biological_annotation`：生物学注释
- `importance`：特征重要性权重

## 4. results/figure3_predictions.csv
- `observed_log2_ic50`：观测值
- `predicted_log2_ic50`：预测值
- `residual`：残差（观测-预测）
