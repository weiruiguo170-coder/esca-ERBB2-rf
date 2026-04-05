# 数据准备说明

本仓库默认不下载、不附带大体量原始数据。请自行从官方来源下载并整理。

## 推荐流程

1. 从 GDSC、CCLE 下载原始数据。
2. 根据论文筛选上消化道/食管相关细胞系。
3. 提取两种代表性药物（Trastuzumab、Lapatinib）对应 IC50。
4. 合并特征与药敏到单个输入 CSV（格式见 `data/README.md`）。
5. 将整理后的 CSV 放入 `data/input/`，并在 `configs/config.example.yaml` 设置路径。

## 提醒

- Demo 不依赖真实大文件。
- 正式复现时请在文稿中记录数据版本、下载日期和筛选规则。

