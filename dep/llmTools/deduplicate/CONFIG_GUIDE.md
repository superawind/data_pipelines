# 去重配置参数完整指南

## 配置参数分类

配置参数按功能分为以下几类：
- **基础配置**: 必需的基本参数
- **性能配置**: 控制性能和资源使用
- **方法专属配置**: 特定去重方法的参数
- **LSH优化配置**: 大规模数据加速参数

---

## 1. 基础配置

### `method` (必选)
- **类型**: `DedupMethod` 枚举
- **可选值**:
  - `DedupMethod.STRICT_HASH` 或 `"strict_hash"`: 严格哈希去重（完全匹配）
  - `DedupMethod.NGRAM_LSH` 或 `"ngram_lsh"`: N-gram LSH去重（文本相似）
  - `DedupMethod.EMBEDDING` 或 `"embedding"`: Embedding去重（语义相似）
- **说明**: 选择去重算法，字符串会自动转换为枚举类型

**示例**:
```python
# 方式1: 使用枚举
from deduplicate.dep_code.config import DedupConfig, DedupMethod
config = DedupConfig(method=DedupMethod.STRICT_HASH, ...)

# 方式2: 使用字符串（自动转换）
config = DedupConfig(method="strict_hash", ...)
```

---

### `input_file` (必选)
- **类型**: `str`
- **说明**: 输入JSONL文件路径，必须存在
- **验证**: 文件必须存在，否则抛出 `FileNotFoundError`

---

### `output_file` (必选)
- **类型**: `str`
- **说明**: 输出JSONL文件路径

---

### `content_keys` (可选)
- **类型**: `List[str]`
- **默认值**: `["prompt"]`
- **说明**: 用于去重的内容字段列表
- **示例**: `["prompt"]`, `["prompt", "instruction"]`
- **验证**: 不能为空，必须是列表类型

---

### `field_order` (可选)
- **类型**: `List[str]`
- **默认值**: `["prompt", "instruction", "input", "output"]`
- **说明**: 控制JSON输出字段的顺序

---

## 2. 通用性能配置

### `num_workers` (可选)
- **类型**: `Optional[int]`
- **默认值**: `None` (自动检测)
- **范围**: 1-128
- **推荐**: CPU核心数的1-2倍
- **验证**: 
  - 小于1: 抛出错误
  - 大于128: 显示警告

---

### `batch_size` (可选)
- **类型**: `int`
- **默认值**: `10000`
- **范围**: 1000-100000
- **推荐**: 10000-50000
- **验证**:
  - 小于1: 抛出错误
  - 小于1000: 显示警告
  - 大于100000: 显示警告

---

### `log_dir` (可选)
- **类型**: `str`
- **默认值**: `None` (自动创建 `global_dedup_logs/` 目录)
- **说明**: 指定日志文件保存目录

---

## 3. N-gram LSH 专属配置

仅在 `method="ngram_lsh"` 时使用。

### `ngram_size`
- **类型**: `int`
- **默认值**: `10`
- **范围**: 3-20
- **推荐**: 8-12
- **说明**: N-gram的大小，控制文本分割粒度
- **验证**:
  - 小于1: 抛出错误
  - 大于20: 显示警告

---

### `jaccard_threshold`
- **类型**: `float`
- **默认值**: `0.85`
- **范围**: 0.0-1.0 (严格限制)
- **推荐**: 0.8-0.9
- **说明**: Jaccard相似度阈值，值越高去重越严格
- **验证**: 超出范围抛出错误

---

### `num_permutations`
- **类型**: `int`
- **默认值**: `128`
- **范围**: 32-512
- **推荐**: 64-256
- **说明**: MinHash排列数，值越大精度越高但速度越慢
- **验证**:
  - 小于1: 抛出错误
  - 大于512: 显示警告

---

## 4. Embedding 专属配置

仅在 `method="embedding"` 时使用。

### `embeddings_file` (embedding方法必需)
- **类型**: `Optional[str]`
- **默认值**: `None`
- **说明**: Embedding文件路径（.npy或.jsonl）
- **验证**: 文件必须存在

---

### `embeddings_files` (可选)
- **类型**: `Optional[List[str]]`
- **默认值**: `None`
- **说明**: 多个Embedding文件路径列表（仅JSONL格式）
- **使用场景**: 大文件分割成多个小文件时使用

---

### `embeddings_format`
- **类型**: `EmbeddingFileFormat` 枚举
- **默认值**: `EmbeddingFileFormat.JSONL`
- **可选值**:
  - `EmbeddingFileFormat.NPY` 或 `"npy"`: numpy数组格式
  - `EmbeddingFileFormat.JSONL` 或 `"jsonl"`: JSONL格式，支持ID匹配
- **说明**: 会根据文件扩展名自动检测

**示例**:
```python
# 方式1: 使用枚举
config = DedupConfig(
    embeddings_format=EmbeddingFileFormat.JSONL, ...
)

# 方式2: 使用字符串（自动转换）
config = DedupConfig(
    embeddings_format="jsonl", ...
)
```

---

### `prompt_id_key`
- **类型**: `str`
- **默认值**: `"prompt_id"`
- **说明**: 数据文件中的ID字段名
- **使用场景**: 仅在 `embeddings_format=JSONL` 时使用

---

### `embedding_id_key`
- **类型**: `str`
- **默认值**: `"prompt_id"`
- **说明**: Embedding文件中的ID字段名
- **使用场景**: 仅在 `embeddings_format=JSONL` 时使用

---

### `embedding_vector_key`
- **类型**: `str`
- **默认值**: `"embedding"`
- **说明**: Embedding文件中的向量字段名
- **使用场景**: 仅在 `embeddings_format=JSONL` 时使用

---

### `threshold`
- **类型**: `float`
- **默认值**: `0.95`
- **范围**: 0.0-1.0 (严格限制)
- **推荐**: 0.9-0.98
- **说明**: 向量相似度阈值，值越高去重越严格
- **验证**: 超出范围抛出错误

---

### `top_k`
- **类型**: `int`
- **默认值**: `10`
- **范围**: 1-100
- **推荐**: 5-20
- **说明**: Top-K检索数量，值越大召回越高但速度越慢
- **验证**:
  - 小于1: 抛出错误
  - 大于100: 显示警告

---

### `use_gpu`
- **类型**: `bool`
- **默认值**: `False`
- **可选值**: `True`, `False`
- **说明**: 是否使用GPU加速
- **依赖**: 
  - `True`: 需要 `faiss-gpu`
  - `False`: 需要 `faiss-cpu`

---

### `gpu_device`
- **类型**: `int`
- **默认值**: `0`
- **范围**: 0-7 (严格限制)
- **说明**: GPU设备ID
- **使用场景**: 仅在 `use_gpu=True` 时生效

---

## 5. LSH优化配置

适用于 `ngram_lsh` 和 `embedding` 两种方法的大规模数据加速。

### `use_lsh`
- **类型**: `bool`
- **默认值**: `False`
- **可选值**: `True`, `False`
- **说明**: 是否启用LSH优化
- **适用场景**: 千万级、亿级数据

---

### `lsh_num_tables`
- **类型**: `int`
- **默认值**: `10`
- **范围**: 3-50
- **推荐**: 5-20
- **说明**: LSH哈希表数量，值越大召回率越高但速度越慢
- **验证**:
  - 小于1: 抛出错误
  - 大于50: 显示警告

---

### `lsh_hash_size`
- **类型**: `int`
- **默认值**: `10`
- **范围**: 6-16
- **推荐**: 8-12
- **说明**: LSH哈希位数，值越大bucket越多，比较次数越少
- **验证**:
  - 小于1: 抛出错误
  - 大于16: 显示警告

---

## 配置验证规则

配置类在初始化时会自动进行以下验证：

### 1. 类型验证
- `method`: 必须是 `DedupMethod` 枚举或有效字符串
- `embeddings_format`: 必须是 `EmbeddingFileFormat` 枚举或有效字符串
- `content_keys`: 必须是列表类型
- `use_gpu`, `use_lsh`: 必须是布尔值

### 2. 文件验证
- `input_file`: 必须存在
- `embeddings_file`: 必须存在（如果指定）
- `embeddings_files`: 所有文件都必须存在（如果指定）

### 3. 范围验证
严格验证（超出范围抛出错误）:
- `jaccard_threshold`: 0.0-1.0
- `threshold`: 0.0-1.0
- `gpu_device`: 0-7

宽松验证（超出推荐范围显示警告）:
- `ngram_size`: 推荐 8-12
- `num_permutations`: 推荐 64-256
- `top_k`: 推荐 5-20
- `batch_size`: 推荐 10000-50000
- `lsh_num_tables`: 推荐 5-20
- `lsh_hash_size`: 推荐 8-12

### 4. 逻辑验证
- embedding方法必须指定 `embeddings_file` 或 `embeddings_files`
- `content_keys` 不能为空

---

## 配置示例

### 示例1: 严格哈希去重（最简配置）
```python
config = DedupConfig(
    method="strict_hash",
    input_file="data.jsonl",
    output_file="output.jsonl"
)
```

### 示例2: N-gram LSH去重（完整配置）
```python
config = DedupConfig(
    method="ngram_lsh",
    input_file="data.jsonl",
    output_file="output.jsonl",
    content_keys=["prompt", "instruction"],
    ngram_size=10,
    jaccard_threshold=0.85,
    num_permutations=128,
    batch_size=20000,
    num_workers=16
)
```

### 示例3: Embedding去重（JSONL格式）
```python
config = DedupConfig(
    method="embedding",
    input_file="data.jsonl",
    output_file="output.jsonl",
    embeddings_file="embeddings.jsonl",
    embeddings_format="jsonl",  # 可省略，自动检测
    prompt_id_key="prompt_id",
    embedding_id_key="prompt_id",
    embedding_vector_key="embedding",
    threshold=0.95,
    top_k=10,
    use_gpu=True,
    gpu_device=0
)
```

### 示例4: Embedding LSH去重（大规模数据）
```python
config = DedupConfig(
    method="embedding",
    input_file="large_data.jsonl",
    output_file="output.jsonl",
    embeddings_file="embeddings.jsonl",
    threshold=0.95,
    use_lsh=True,          # 启用LSH加速
    lsh_num_tables=10,
    lsh_hash_size=10,
    batch_size=10000
)
```

---

## 错误处理

### 常见错误及解决方案

#### 1. 无效的method
```
ValueError: 无效的去重方法: invalid_method
可选值: strict_hash, ngram_lsh, embedding
```
**解决**: 使用正确的method值

#### 2. 文件不存在
```
FileNotFoundError: 输入文件不存在: data.jsonl
```
**解决**: 检查文件路径是否正确

#### 3. 阈值超出范围
```
ValueError: threshold必须在0.0-1.0范围内，当前值: 1.5
推荐值: 0.9-0.98
```
**解决**: 使用有效范围内的值

#### 4. embedding方法缺少必需参数
```
ValueError: embedding方法必须指定embeddings_file或embeddings_files
示例：--embeddings_file embeddings.jsonl
```
**解决**: 添加 `embeddings_file` 参数

---

## 性能调优建议

### 内存优化
- 减小 `batch_size`: 10000 → 5000
- 减小 `num_workers`: 16 → 8

### 速度优化
- 增加 `num_workers`: 8 → 16
- 增加 `batch_size`: 10000 → 20000
- 使用 `use_gpu=True`（需要GPU）
- 对于超大数据集，使用 `use_lsh=True`

### LSH参数调优
- **高精度要求**: `lsh_num_tables=15-20`
- **一般场景**: `lsh_num_tables=10`
- **追求速度**: `lsh_num_tables=5-8`
- **超大数据集(>1000万)**: `lsh_hash_size=10-12`
- **中等数据集**: `lsh_hash_size=8-10`

---

## 参考文档

- 配置示例: `deduplicate/examples/`
- 完整文档: `deduplicate/README.md`
- 重构总结: `deduplicate/REFACTORING_SUMMARY_V2.md`

