# 去重配置示例

本目录包含了各种去重方式的配置示例脚本。

## 配置文件列表

### 1. 严格哈希去重 (`config_strict_hash.sh`)
- **特点**: 完全匹配去重，速度最快，零误删
- **适用场景**: 需要精确去重，只删除完全相同的数据
- **性能**: 最快，内存占用最小

### 2. N-gram LSH去重 (`config_ngram_lsh.sh`)
- **特点**: 文本相似度去重，可识别相似文本
- **适用场景**: 需要删除高度相似的文本内容
- **性能**: 中等速度，适中内存占用

### 3. Embedding去重 - 精确模式 (`config_embedding.sh`)
- **特点**: 语义相似度去重，最高精度
- **适用场景**: 需要识别语义相似的内容，数据量中等（百万级）
- **性能**: 较慢，但精度最高
- **要求**: 需要预先生成embedding向量

### 4. Embedding LSH去重 - 优化模式 (`config_embedding_lsh.sh`)
- **特点**: 语义相似度去重 + LSH加速
- **适用场景**: 千万级、亿级数据的语义去重
- **性能**: 比精确模式快10-100倍，略微降低召回率（<1%）
- **要求**: 需要预先生成embedding向量

## 使用方法

1. 选择适合的配置文件
2. 修改配置文件中的路径和参数
3. 运行脚本：
```bash
bash examples/config_strict_hash.sh
```

或者直接修改参数运行：
```bash
chmod +x examples/*.sh
./examples/config_ngram_lsh.sh
```

## 参数说明

### 基础参数
- `--method`: 去重方法（strict_hash/ngram_lsh/embedding）
- `--input_file`: 输入JSONL文件路径
- `--output_file`: 输出JSONL文件路径
- `--content_keys`: 用于去重的字段（可多个，如：prompt instruction）
- `--num_workers`: 并行worker数量（默认16）
- `--batch_size`: 批处理大小（默认10000）

### N-gram LSH专用参数
- `--ngram_size`: N-gram大小（默认10，推荐8-12）
- `--jaccard_threshold`: Jaccard相似度阈值（默认0.85，推荐0.8-0.9）
- `--num_permutations`: MinHash排列数（默认128，推荐64-256）

### Embedding专用参数
- `--embeddings_file`: Embedding文件路径
- `--embeddings_files`: 多个Embedding文件（仅JSONL格式）
- `--embeddings_format`: 文件格式（npy/jsonl）
- `--prompt_id_key`: 数据文件中的ID字段名
- `--embedding_id_key`: Embedding文件中的ID字段名
- `--embedding_vector_key`: Embedding文件中的向量字段名
- `--threshold`: 相似度阈值（默认0.95，推荐0.9-0.98）
- `--top_k`: Top-K检索数量（默认10，推荐5-20）
- `--use_gpu`: 启用GPU加速
- `--gpu_device`: GPU设备ID

### LSH优化参数（适用于大规模数据）
- `--use_lsh`: 启用LSH优化
- `--lsh_num_tables`: LSH哈希表数量（默认10，推荐5-20）
- `--lsh_hash_size`: LSH哈希位数（默认10，推荐8-12）

## 性能优化建议

### 内存优化
- 减小 `--batch_size`
- 减小 `--num_workers`

### 速度优化
- 增加 `--num_workers`（不超过CPU核心数的2倍）
- 增加 `--batch_size`（如果内存充足）
- 使用 `--use_gpu`（如果有GPU且安装了faiss-gpu）
- 对于超大数据集，使用 `--use_lsh`

### LSH参数调优
- **高精度要求**: `--lsh_num_tables 15-20`
- **一般场景**: `--lsh_num_tables 10`
- **追求速度**: `--lsh_num_tables 5-8`
- **超大数据集(>1000万)**: `--lsh_hash_size 10-12`
- **中等数据集**: `--lsh_hash_size 8-10`

## 示例数据格式

### 输入数据格式 (JSONL)
```json
{"prompt_id": "001", "prompt": "什么是机器学习？", "instruction": "回答问题"}
{"prompt_id": "002", "prompt": "机器学习是什么？", "instruction": "回答问题"}
```

### Embedding文件格式 (JSONL)
```json
{"prompt_id": "001", "embedding": [0.1, 0.2, 0.3, ...]}
{"prompt_id": "002", "embedding": [0.15, 0.25, 0.35, ...]}
```

### Embedding文件格式 (NPY)
- 二维numpy数组，形状为 (样本数, 向量维度)
- 行顺序必须与输入数据文件完全一致

## 常见问题

### 1. 如何选择去重方法？
- **完全匹配去重**: 使用 `strict_hash`
- **文本相似去重**: 使用 `ngram_lsh`
- **语义相似去重**: 使用 `embedding`（精确）或 `embedding + use_lsh`（大规模）

### 2. 如何选择阈值？
- **Jaccard阈值**: 0.8-0.9，值越高越严格
- **Embedding阈值**: 0.9-0.98，值越高越严格

### 3. 何时使用LSH优化？
- 数据量 > 1000万条
- 需要加速embedding去重
- 可以接受略微降低召回率（<1%）

### 4. 如何生成embedding？
参考主项目README中的embedding生成示例。

