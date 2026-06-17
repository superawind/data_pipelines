# JSONL 批量验证工作流程

本文档介绍如何使用AI验证系统的JSONL批量处理功能。

## 📋 工作流程概述

1. **准备数据** - 创建包含`prompt`和`response`字段的JSONL文件
2. **实现LLM接口** - 自定义LLM调用函数
3. **生成验证契约** - 使用`generate_contracts.py`生成`verify`字段
4. **执行验证** - 使用`run_verification.py`生成验证结果

## 🔧 步骤1：实现LLM接口

在开始之前，您需要实现`src/ai_verifier/llm_interface.py`中的两个函数：

```python
def call_llm_api(
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    json_mode: bool = True
) -> str:
    """
    调用LLM API的统一接口
    
    示例实现（OpenAI）：
    """
    import openai
    
    client = openai.OpenAI(api_key="your-api-key")
    
    messages = [{"role": "user", "content": prompt}]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=temperature or 0.1,
        max_tokens=max_tokens or 2000,
        response_format={"type": "json_object"} if json_mode else None
    )
    
    return response.choices[0].message.content


def is_llm_available() -> bool:
    """
    检查LLM服务是否可用
    
    示例实现：
    """
    import os
    return bool(os.getenv("OPENAI_API_KEY"))
```

## 📄 步骤2：准备JSONL数据文件

创建包含以下字段的JSONL文件：

```jsonl
{"prompt": "用Python写一个函数 add_numbers(a, b) 计算两个数的和", "response": "def add_numbers(a, b):\n    return a + b"}
{"prompt": "写一篇100字以内的关于人工智能的文章", "response": "人工智能是一门前沿的技术学科..."}
```

### 必需字段

- `prompt` (string): 用户提示
- `response` (string): AI生成的回应

### 可选字段

- 其他自定义字段将在处理过程中保留

## 🎯 步骤3：生成验证契约

使用`generate_contracts.py`脚本为每条数据生成验证契约：

```bash
# 基本用法（创建新文件）
python scripts/generate_contracts.py data/sample_data.jsonl

# 就地更新原文件（推荐）
python scripts/generate_contracts.py data/sample_data.jsonl --in-place

# 指定输出文件
python scripts/generate_contracts.py data/sample_data.jsonl -o data/data_with_contracts.jsonl

# 覆盖已存在的verify字段
python scripts/generate_contracts.py data/sample_data.jsonl --in-place --overwrite-verify

# 调试模式
python scripts/generate_contracts.py data/sample_data.jsonl --in-place --log-level DEBUG
```

### 输出结果

**就地更新模式**（使用`--in-place`，推荐）：
脚本会直接在原文件中添加`verify`字段，每条数据逐步完善：

```jsonl
{"prompt": "...", "response": "...", "verify": {"code_verification": {...}, "llm_verification": {...}}}
```

**新建文件模式**（不使用`--in-place`）：
脚本会创建新文件，原文件保持不变：
- 输入：`data.jsonl` 
- 输出：`data_with_contracts.jsonl`

### 命令行参数

- `input_file`: 输入的JSONL文件路径
- `-o, --output`: 输出文件路径（默认：输入文件名_with_contracts.jsonl）
- `--in-place`: 就地更新原文件，而不是创建新文件（推荐）
- `--overwrite-verify`: 覆盖已存在的verify字段
- `--log-level`: 日志级别（DEBUG, INFO, WARNING, ERROR）

## ✅ 步骤4：执行验证

使用`run_verification.py`脚本执行验证：

```bash
# 基本用法（创建新文件）
python scripts/run_verification.py data/sample_data.jsonl

# 就地更新原文件（推荐）
python scripts/run_verification.py data/sample_data.jsonl --in-place

# 就地更新并生成汇总报告
python scripts/run_verification.py data/sample_data.jsonl \
    --in-place \
    --summary data/summary_report.json

# 覆盖已存在的验证结果
python scripts/run_verification.py data/sample_data.jsonl --in-place --overwrite-result

# 调试模式
python scripts/run_verification.py data/sample_data.jsonl --in-place --log-level DEBUG
```

### 输出结果

**就地更新模式**（使用`--in-place`，推荐）：
脚本会直接在原文件中添加验证结果字段，形成完整记录：

```jsonl
{
  "prompt": "...",
  "response": "...", 
  "verify": {...},
  "verification_result": {...},
  "scores": {
    "code_verification": 1,
    "function_name_accuracy": 1,
    "summary": {"total_points": 2, "max_points": 2, "pass_rate": 100.0, "verdict": "PASSED"}
  }
}
```

**新建文件模式**（不使用`--in-place`）：
脚本会创建新文件，原文件保持不变：
- 输入：`data.jsonl` 
- 输出：`data_verified.jsonl`

**添加的字段**：
- `verification_result`: 完整的验证结果
- `scores`: 简化的得分信息
- `verification_error`: 验证错误信息（如有）

### 命令行参数

- `input_file`: 输入的JSONL文件路径（需包含verify字段）
- `-o, --output`: 输出文件路径（默认：输入文件名_verified.jsonl）
- `--in-place`: 就地更新原文件，而不是创建新文件（推荐）
- `--summary`: 汇总报告输出路径（JSON格式）
- `--overwrite-result`: 覆盖已存在的verification_result字段
- `--log-level`: 日志级别

## 📊 汇总报告

使用`--summary`参数可以生成详细的汇总报告：

```json
{
  "total_items": 100,
  "successful_verifications": 95,
  "failed_verifications": 3,
  "error_verifications": 2,
  "overall_pass_rate": 85.5,
  "average_scores": {
    "code_verification": 0.892,
    "content_quality": 0.945,
    "topic_relevance": 0.987
  },
  "statistics": {
    "verification_success_rate": 95.0,
    "passed_items": 85
  },
  "processing_time_seconds": 128.5
}
```

## 🔄 完整工作流程示例

```bash
# 1. 实现LLM接口（编辑 src/ai_verifier/llm_interface.py）

# 2. 生成验证契约（就地更新原文件）
python scripts/generate_contracts.py data/sample_data.jsonl \
    --in-place \
    --log-level INFO

# 3. 执行验证（就地更新原文件）
python scripts/run_verification.py data/sample_data.jsonl \
    --in-place \
    --summary data/report.json \
    --log-level INFO

# 4. 查看结果
cat data/report.json
head -n 1 data/sample_data.jsonl | jq .  # 查看一条完整记录
```

## ⚠️ 注意事项

### LLM接口实现

- 确保`call_llm_api`函数正确处理JSON模式
- 实现适当的错误处理和重试机制
- 考虑API调用频率限制

### 数据格式

- JSONL文件每行必须是有效的JSON对象
- `prompt`和`response`字段不能为空
- 特殊字符需要正确转义

### 性能优化

- 大文件建议分批处理
- 合理设置日志级别
- 监控API调用成本

### 错误处理

- 脚本会跳过格式错误的数据行
- 验证失败的项目会保留错误信息
- 使用详细日志排查问题

## 🛠️ 故障排除

### 常见问题

1. **LLM接口未实现**
   ```
   ERROR: 用户LLM接口不可用: This function is not implemented
   ```
   解决：实现`src/ai_verifier/llm_interface.py`中的函数

2. **JSON解析错误**
   ```
   ERROR: 第X行JSON解析失败
   ```
   解决：检查JSONL文件格式，确保每行是有效JSON

3. **缺少必需字段**
   ```
   WARNING: 第X条数据缺少字段: ['prompt']
   ```
   解决：确保数据包含必需的`prompt`和`response`字段

4. **验证超时**
   ```
   ERROR: 代码执行超时
   ```
   解决：检查AI生成的代码是否包含死循环

### 调试技巧

- 使用`--log-level DEBUG`获取详细信息
- 先用小样本测试
- 检查LLM API的响应格式
- 验证JSONL文件格式

## 📈 最佳实践

1. **数据准备**
   - 保持prompt描述清晰准确
   - 确保response质量一致
   - 备份原始数据

2. **就地更新优势**（推荐使用`--in-place`）
   - 所有数据集中在一个文件中，便于管理
   - 支持增量处理，可随时中断和恢复
   - 避免文件名混乱，减少磁盘空间占用
   - 每条记录逐步完善：原始数据 → +verify → +验证结果

3. **批处理**
   - 大数据集可以分批处理，使用`--in-place`累积结果
   - 设置合适的超时时间
   - 监控处理进度
   - 使用`--overwrite-verify`和`--overwrite-result`重新处理特定数据

4. **结果分析**
   - 定期查看汇总报告
   - 分析失败模式
   - 持续优化验证策略

5. **成本控制**
   - 合理使用LLM API
   - 避免重复处理（脚本会自动跳过已处理数据）
   - 使用缓存机制 