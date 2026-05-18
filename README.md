# Knowledge-Popularity-for-LLM-Knowledge-Boundary-Perception

本仓库配套论文与代码：用于研究大规模语言模型（LLM）知识边界感知与实体/知识流行度（popularity）相关的方法、数据生成与分析。

**主要内容**
- 论文：paper.md
- 代码：`code/`（数据预处理、分析与绘图脚本）
- QA 与生成：`qa_generation/`（基于 LLM 的 QA / 生成脚本）
- LLM 输出：`llm_pop_generation/`（不同模型与参数下的生成结果）
- 流行度生成工具：`pop_generation/`（解析与处理 LLM 输出的工具和 API）
- 数据：`data/`（示例数据集，如篮球、电影、歌曲等）
- 结果：`res/`（实验结果与汇总）

## 目录结构（摘要）
- code/
  - analysis_correlation/: 相关性分析与绘图脚本
  - my_utils/: 项目共用工具
- qa_generation/: 生成 QA 与相关实验脚本
- llm_pop_generation/: 不同数据集与模型的生成输出（jsonl）
- pop_generation/: 解析 LLM 输出、运行 API 的脚本和工具
- data/: 小型示例数据（jsonl）
- res/: 实验结果与可视化输出
- paper.md: 项目论文草稿

## 功能概览
- 生成与收集：通过不同 LLM（ChatGPT、Llama、Qwen 等）采集生成结果并存储于 `llm_pop_generation/`。
- 解析与聚合：`pop_generation/parse_llm_outputs.py` 提供解析 LLM 输出并将其转为可分析格式的工具。
- QA 生成：`qa_generation/` 包含用于生成问题、运行评测与预处理的脚本。
- 分析与绘图：`code/analysis_correlation/` 提供相关性分析与绘图脚本，用于生成论文图表。

## 依赖与运行环境
- 推荐 Python 3.8+ / 3.10
- 常见依赖（示例）：

```bash
python -m pip install -r requirements.txt
# 若无 requirements.txt，可安装常见包：
python -m pip install numpy pandas matplotlib seaborn tqdm transformers
```

（如需环境文件，我可以为仓库生成 `requirements.txt` 或 `environment.yml`。）

## 快速开始示例
1. 克隆仓库并进入目录：

```bash
git clone <your-repo-url>
cd Knowledge-Popularity-for-LLM-Knowledge-Boundary-Perception
```

2. 运行示例脚本（示例：解析 LLM 输出并生成统计）：

```bash
python pop_generation/parse_llm_outputs.py --input llm_pop_generation/basketball/ --output res/basketball_parsed.jsonl
```

3. 运行分析脚本生成图表：

```bash
python code/analysis_correlation/acl_plot.py --input res/basketball_parsed.jsonl --out res/plots/
```

4. 若要启动本地 API（仓库中已有 run_api.sh）：

```bash
bash pop_generation/run_api.sh
# 或在需要时使用 run_api.py 启动服务
python pop_generation/run_api.py
```

## 数据说明
- `data/` 包含小型示例数据（jsonl 格式），用于快速测试脚本。
- 真实或更大规模的数据应根据论文与实验需要另行下载并放置到合适路径，脚本中通常使用相对路径指向 `data/` 或 `llm_pop_generation/`。

## 贡献与沟通
- 欢迎 Issue 与 PR。若要复现实验或运行脚本，建议先创建虚拟环境并安装依赖。
- 如需我帮你添加 `requirements.txt`、CI 配置或示例运行 notebook，我可以继续实现。

## 许可证
默认不包含 LICENSE，请根据需要补充（例如 MIT、Apache-2.0 等）。

---
（自动生成 README，如需调整语言风格、增加示例或补充依赖，请告诉我具体需求。）
