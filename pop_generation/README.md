# LLM-generated Popularity Signals

此仓库收集并说明项目中用于让大模型（LLM）生成“popularity”指标的关键代码、运行方法与改进建议。

目录结构（说明，不复制所有代码）：

- `run_api.py` - 主入口脚本：解析参数、构建 `QADataset`、批量调用 LLM 并将结果保存为 JSONL。
- `utils/prompt.py` - Prompt 模板：构造要求模型返回 1-10 整数的提示（支持 few-shot）。
- `utils/data_api.py` - 数据集处理：读取清洗数据、对指定字段分层采样以产生 few-shot 示例。
- `utils/llm_api.py` - 与 LLM 的交互封装：并发调用、重试与顺序保证，返回原始文本 `Res`。
- `data/read.py` - 清洗/合并脚本示例：将已有实体与外部 popularity 数据合并成带 `question_pop`/`gene_pop`/`coo_pop` 的文件。
- `run_api.sh` - 批量运行示例脚本，展示参数组合与输出路径（`./llm_pop_generation/...`）。

快速开始

1. 将本仓库根目录指向本项目工作目录（已放置于 `work/pop_generation_repo`，但实际代码仍在父目录）。

2. 运行示例（参考 `run_api.sh` 或直接运行 `run_api.py`）：

```bash
# 使用 run_api.sh 批量生成（示例）
bash run_api.sh

# 或单次运行示例：
python -u run_api.py \
  --source ./data/clean_data_for_pop_generation/movies_chatgpt_temperature1.jsonl \
  --type qa_pop_rank_diverse \
  --outfile ./llm_pop_generation/movies/movies_chatgpt_question_pop_0.jsonl \
  --model gpt-3.5-turbo-1106 \
  --batch_size 3 \
  --gene_type question \
  --dataset_name movies \
  --temperature 0.0 \
  --n_shot 0
```

输出

- 每行 JSON 包含 `qa_prompt` 与 `Res` 字段：`Res` 为模型返回的文本（期望为 1-10 的整数）。输出文件位于 `./llm_pop_generation/{dataset}/`。

建议与改进

- 自动解析并校验 `Res` 为整数（1-10）：在 `utils/llm_api.py` 或写入阶段增加解析逻辑，并对异常或范围外值进行回退或标记。
- 增加 `--parse` 或 `--strict` 参数以控制是否把 `Res` 转为 `int` 并在输出中加入 `pop_value` 字段。
- 提供简单脚本将生成的 LLM 输出与原始统计 `question_pop`/`gene_pop` 做对比与归一化，便于后续分析与分级。

如需，我可以：
- 在 `utils/llm_api.py` 中实现 `Res` 解析与校验并把结果写回（自动修正/标注异常）。
- 添加一个小脚本 `parse_llm_outputs.py` 来批量解析 `llm_pop_generation/` 下的输出并汇总统计。

---

作者注：此仓库为项目导出说明与运行指引，实际代码文件仍留在父目录 `work/` 下。将来可按需将相关模块复制到本仓库并初始化为独立 Git 仓库。