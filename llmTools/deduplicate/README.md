# 数据去重工具

高性能的数据去重工具，支持多种去重算法和处理模式，专为大规模数据处理而设计。

## 🚀 快速安装

### 基础安装
```bash
# 克隆项目
git clone <repository-url>
cd llmTools/deduplicate

# 安装依赖
pip install -r requirements.txt
```

### GPU加速（可选）
如果需要GPU加速向量去重，请安装GPU版本的FAISS：
```bash
# 先卸载CPU版本
pip uninstall faiss-cpu

# 安装GPU版本
pip install faiss-gpu
```

### 性能优化库（推荐）
为了获得最佳性能，建议安装以下加速库：
```bash
# 高速哈希库
pip install xxhash

# 高速JSON解析库  
pip install orjson
```

## ⚡ 快速开始

```bash
# 严格哈希去重（完全匹配）
python -m deduplicate --method strict_hash --input_file data.jsonl --output_file output.jsonl

# N-gram LSH去重（相似度匹配）  
python -m deduplicate --method ngram_lsh --input_file data.jsonl --output_file output.jsonl \
    --jaccard_threshold 0.85 --ngram_size 10

# Embedding去重（语义匹配）
python -m deduplicate --method embedding --input_file data.jsonl --output_file output.jsonl \
    --embeddings_file embeddings.jsonl --threshold 0.95
```

## 🔧 去重方法

### 1. 严格哈希去重 (`strict_hash`)
**原理**：基于内容的哈希值进行完全匹配去重，只有完全相同的内容才会被识别为重复。

**特点**：
- 速度最快，内存占用最小
- 零误删，保证数据完整性
- 适用于需要精确去重的场景

**支持模式**：`streaming` | `global`

### 2. N-gram LSH去重 (`ngram_lsh`)  
**原理**：将文本分割为N-gram片段，使用MinHash和LSH算法计算Jaccard相似度，检测近似重复内容。

**特点**：
- 可检测相似但不完全相同的文本
- 通过Jaccard阈值控制相似度敏感性
- 平衡速度和精度

**支持模式**：`streaming` | `global`

### 3. 向量语义去重 (`embedding`)
**原理**：基于预训练模型的向量表示，使用余弦相似度检测语义相似的内容。

**特点**：
- 最高精度的语义级去重
- 可检测表达相同意思但用词不同的文本
- 支持多种嵌入模型和文件格式

**支持模式**：`streaming` | `global`

## 🎛️ 处理模式

### 流式模式 (`streaming`)
- **特点**：低内存占用，零延迟启动，边读边处理
- **适用**：超大文件（>10GB），内存受限环境
- **日志目录**：`stream_dedup_logs/`

### 全局模式 (`global`) 
- **特点**：真正的全局去重，最高精度，完整相似度图构建
- **适用**：中等规模文件（<10GB），追求最高去重精度
- **日志目录**：`global_dedup_logs/`

## ⚙️ 配置参数

### 基础必需配置

#### `--method`
- **类型**：枚举 (`strict_hash` | `ngram_lsh` | `embedding`)
- **作用**：指定使用的去重算法
- **必需**：是

#### `--input_file` 
- **类型**：字符串
- **作用**：输入JSONL文件路径，支持任意大小文件的流式读取
- **必需**：是

#### `--output_file`
- **类型**：字符串  
- **作用**：输出JSONL文件路径，保存去重后的唯一数据
- **必需**：是

#### `--content_keys`
- **类型**：字符串列表
- **默认**：`["prompt"]`
- **作用**：指定用于去重比较的JSON字段名，支持多字段组合
- **原理**：多个字段的内容会被连接后进行去重判断

#### `--field_order`
- **类型**：字符串列表
- **默认**：`["prompt", "instruction", "input", "output"]`
- **作用**：控制输出JSON文件中字段的排列顺序
- **原理**：指定字段按此顺序输出，其余字段追加在后

### 通用性能配置

#### `--num_workers`
- **类型**：整数
- **默认**：16
- **作用**：并行处理的worker数量
- **原理**：控制CPU并行度，建议设为CPU核心数的1-2倍

#### `--chunk_size_mb`
- **类型**：整数
- **默认**：16
- **作用**：分块处理的大小（MB）
- **原理**：较大值减少I/O开销但增加内存使用，较小值降低内存压力但增加I/O开销

#### `--queue_maxsize`
- **类型**：整数
- **默认**：8
- **作用**：队列最大缓存块数
- **原理**：控制内存中同时缓存的数据块数量，总内存使用 = chunk_size_mb × queue_maxsize

#### `--buffer_size_mb`
- **类型**：整数
- **默认**：2
- **作用**：I/O缓冲区大小（MB）
- **原理**：较大值减少系统调用次数，提升文件读写性能

#### `--log_dir`
- **类型**：字符串
- **默认**：根据模式自动选择
- **作用**：日志文件存储目录
- **原理**：存储重复数据记录、性能统计和处理日志

### N-gram LSH专用配置

#### `--ngram_size`
- **类型**：整数
- **默认**：10
- **作用**：N-gram分割的大小
- **原理**：控制文本分割粒度，值越大精度越高但计算越慢

#### `--jaccard_threshold`
- **类型**：浮点数 (0.0-1.0)
- **默认**：0.85
- **作用**：Jaccard相似度阈值
- **原理**：判定文本相似的最低Jaccard系数，值越高去重越严格

#### `--num_permutations`
- **类型**：整数
- **默认**：128
- **作用**：MinHash排列数量
- **原理**：控制LSH算法的精度和速度平衡，值越大精度越高但内存消耗增加

### Embedding专用配置

#### `--embeddings_file`
- **类型**：字符串
- **作用**：单个嵌入向量文件路径
- **支持格式**：`.npy`（NumPy数组）| `.jsonl`（JSON行）
- **原理**：NPY格式要求向量顺序与输入文件一致，JSONL格式支持基于ID匹配

#### `--embeddings_files`
- **类型**：字符串列表
- **作用**：多个嵌入向量文件路径列表
- **支持格式**：仅支持JSONL格式
- **原理**：当大的embedding文件被分割成多个小文件时使用，基于ID进行匹配

#### `--embeddings_format`
- **类型**：枚举 (`npy` | `jsonl`)
- **默认**：根据文件扩展名自动检测
- **作用**：指定嵌入向量文件格式
- **原理**：NPY格式按行索引匹配，JSONL格式按ID字段匹配

#### `--prompt_id_key`
- **类型**：字符串
- **默认**：`"prompt_id"`
- **作用**：输入数据中的ID字段名
- **原理**：用于与embedding文件中的ID进行匹配（仅JSONL格式）

#### `--embedding_id_key`
- **类型**：字符串
- **默认**：`"prompt_id"`
- **作用**：embedding文件中的ID字段名
- **原理**：embedding JSONL文件中存储唯一标识符的字段名

#### `--embedding_vector_key`
- **类型**：字符串
- **默认**：`"embedding"`
- **作用**：embedding文件中的向量字段名
- **原理**：embedding JSONL文件中存储向量数据的字段名

#### `--threshold`
- **类型**：浮点数 (0.0-1.0)
- **默认**：0.95
- **作用**：向量相似度阈值
- **原理**：判定向量相似的最低余弦相似度，值越高去重越严格

#### `--top_k`
- **类型**：整数
- **默认**：10
- **作用**：向量检索的Top-K数量
- **原理**：每次查询检索最相似的K个候选向量，值越大召回越高但计算时间增加

#### `--use_gpu`
- **类型**：布尔值
- **默认**：False
- **作用**：是否使用GPU加速
- **原理**：启用GPU加速的FAISS向量检索，需要安装faiss-gpu

#### `--gpu_device`
- **类型**：整数
- **默认**：0
- **作用**：GPU设备ID
- **原理**：指定使用的GPU设备编号（仅在use_gpu=True时有效）

### 全局模式专用配置

#### `--batch_size`
- **类型**：整数
- **默认**：10000
- **作用**：批处理大小
- **原理**：控制全局去重时的批量处理大小，影响内存使用和处理效率

## 💡 算法原理

### 严格哈希去重
使用快速哈希算法（xxHash/Blake2b）对内容生成唯一指纹，通过哈希值比较实现O(1)时间复杂度的重复检测。

### N-gram LSH去重  
1. 将文本分割为长度为N的字符片段
2. 为每个片段生成MinHash签名
3. 使用LSH算法快速找到相似的MinHash
4. 计算精确的Jaccard相似度确认重复

### 向量语义去重
1. 加载预训练模型的向量表示
2. 对向量进行L2归一化
3. 使用FAISS构建高效的相似度搜索索引
4. 通过余弦相似度检测语义相似内容
5. 全局模式额外构建相似度图发现传递相似关系

## 🚀 使用方式

```bash
python -m deduplicate [配置参数]
```

所有配置参数都可以通过命令行参数传递，参数名为配置项名前加`--`。

## ⚡ 大规模数据性能说明

### 千万级数据处理
当处理1000万以上数据时，全局模式下的相似度图构建是计算密集型操作：

**时间复杂度**：O(N × top_k × log(N))，其中N为数据量
- 1000万数据 × top_k=10 ≈ 1亿次向量检索
- 预计处理时间：CPU 2-8小时，GPU 0.5-2小时

**优化建议**：
- `batch_size`: 增加到20000-50000减少批次数
- `top_k`: 降低到5-8减少检索开销  
- `use_gpu`: 强烈建议启用GPU加速
- 考虑使用`streaming`模式避免构建完整相似度图

**进度监控**：
- 首个批次可能需要数分钟，属于正常现象
- 进度条会显示详细的时间预估和边数统计
- 建议监控日志文件获取详细处理信息

## 📋 使用注意事项

### 依赖库说明
- **numpy**: 核心数值计算，支持内存映射大文件处理
- **faiss-cpu/faiss-gpu**: 高效向量相似度搜索
- **datasketch**: MinHash和LSH算法实现
- **xxhash** (可选): 高性能哈希算法，比标准库快3-5倍
- **orjson** (可选): 高性能JSON解析，比标准库快2-3倍