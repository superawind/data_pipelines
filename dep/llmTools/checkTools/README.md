# AI回应验证系统 - 工业级混合验证模式

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

这是一个完整的、工业级的AI回应验证系统，采用"两阶段、统一契约"模型，支持代码验证和LLM判断的双重验证机制。每个验证点采用0/1得分制，确保评估结果的准确性和可量化性。

## 🎯 核心特性

- **🔄 智能验证模式**：根据任务类型智能选择验证方式，简单函数使用代码验证，其他任务使用LLM验证
- **📊 得分制评估**：每个验证点给出0分（失败）或1分（通过）的明确评分
- **🛡️ 安全执行**：代码在隔离的临时环境中安全执行
- **⚙️ 高度可配置**：支持多种配置方式和参数调整
- **📈 批量处理**：支持大规模数据的批量验证
- **🚀 工业级设计**：模块化架构，易于扩展和维护
- **📂 就地更新**：智能的文件管理，支持增量处理和恢复

## 📁 项目结构

```
checkTools/
├── src/ai_verifier/                # 核心代码
│   ├── __init__.py                 # 包初始化
│   ├── config.py                   # 配置管理
│   ├── llm_client.py              # LLM客户端
│   ├── llm_interface.py           # LLM接口抽象层
│   ├── contract_generator.py       # 验证契约生成器
│   ├── code_executor.py           # 代码执行器
│   ├── judge_evaluator.py         # 得分评判器
│   ├── score_calculator.py        # 得分计算器
│   └── verifier.py                # 主验证器
├── prompts/                        # Prompt模板
│   ├── contract_generator.txt      # 契约生成器prompt
│   └── score_evaluator.txt        # 评分器prompt
├── config/                         # 配置文件
│   └── default.json               # 默认配置
├── data/                          # 数据文件
│   └── sample_data.jsonl          # 示例数据
├── scripts/                       # 脚本工具
│   ├── demo.py                    # 演示脚本
│   ├── demo_in_place.py           # 就地更新演示
│   ├── generate_contracts.py     # 生成验证契约
│   └── run_verification.py       # 执行批量验证
├── tests/                         # 测试文件
├── docs/                          # 文档
│   ├── jsonl_workflow.md          # JSONL工作流程文档
├── setup.py                       # 安装配置
├── requirements.txt               # 依赖文件
├── Makefile                       # 构建脚本
├── .gitignore                     # Git忽略文件
└── README.md                      # 本文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd checkTools

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
make install
# 或
pip install -r requirements.txt
```

### 2. 运行演示

```bash
# 基础演示（模拟模式）
make demo

# 就地更新功能演示
python scripts/demo_in_place.py

# 启用真实LLM验证（需要API密钥）
export OPENAI_API_KEY="your-api-key-here"
make demo
```

### 3. 编程接口使用

```python
from ai_verifier import AIVerifier, Config

# 初始化验证器
verifier = AIVerifier()

# 单条验证
prompt = "用Python写一个函数 add(a, b) 返回两数之和"
response = "def add(a, b):\n    return a + b"

result = verifier.verify_response(prompt, response)

# 获取简化得分
scores = verifier.create_simple_score_output(result)
print(scores)
# 输出: {"code_verification": 1, "eval_point_1": 1, "summary": {...}}

# 批量验证
data = [
    {"prompt": "提示1", "response": "回应1"},
    {"prompt": "提示2", "response": "回应2"}
]
batch_result = verifier.verify_batch(data)
```

## 🔄 JSONL批量处理详细指南

系统提供了强大的JSONL批量处理功能，适用于大规模数据验证。以下是完整的使用步骤：

### 步骤1：实现LLM接口

首先在`src/ai_verifier/llm_interface.py`中实现您的LLM调用逻辑：

```python
def call_llm_api(
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    json_mode: bool = True
) -> str:
    """
    调用LLM API的统一接口
    
    Args:
        prompt: 输入提示
        temperature: 温度参数（0-2），控制输出随机性
        max_tokens: 最大输出令牌数
        json_mode: 是否启用JSON模式输出
        
    Returns:
        LLM的响应文本
        
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
    
    Returns:
        是否可用
        
    示例实现：
    """
    import os
    return bool(os.getenv("OPENAI_API_KEY"))
```

### 步骤2：准备JSONL数据文件

创建包含必需字段的JSONL文件：

```jsonl
{"prompt": "用Python写一个函数 add_numbers(a, b) 计算两个数的和", "response": "def add_numbers(a, b):\n    return a + b"}
{"prompt": "写一篇100字以内的关于人工智能的文章", "response": "人工智能是一门前沿的技术学科..."}
{"prompt": "用Flask创建一个Web应用", "response": "from flask import Flask\napp = Flask(__name__)..."}
```

**必需字段**：
- `prompt` (string): 用户提示
- `response` (string): AI生成的回应

**可选字段**：
- 其他自定义字段将在处理过程中保留

### 步骤3：生成验证契约

使用`generate_contracts.py`脚本为每条数据生成验证契约：

#### 基本用法

```bash
# 就地更新原文件（推荐）
python scripts/generate_contracts.py data/sample_data.jsonl --in-place

# 创建新文件
python scripts/generate_contracts.py data/sample_data.jsonl

# 指定输出文件
python scripts/generate_contracts.py data/sample_data.jsonl -o data/processed.jsonl
```

#### 高级选项

```bash
# 覆盖已存在的verify字段，重新生成所有契约
python scripts/generate_contracts.py data/sample_data.jsonl --in-place --overwrite-verify

# 启用详细调试日志
python scripts/generate_contracts.py data/sample_data.jsonl --in-place --log-level DEBUG

# 仅显示警告和错误
python scripts/generate_contracts.py data/sample_data.jsonl --in-place --log-level WARNING
```

#### 命令行参数详解

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `input_file` | string | ✅ | - | 输入的JSONL文件路径 |
| `-o, --output` | string | ❌ | `{input}_with_contracts.jsonl` | 输出文件路径（不使用--in-place时） |
| `--in-place` | flag | ❌ | false | 就地更新原文件，不创建新文件 |
| `--overwrite-verify` | flag | ❌ | false | 覆盖已存在的verify字段 |
| `--log-level` | choice | ❌ | INFO | 日志级别：DEBUG, INFO, WARNING, ERROR |

#### 输出结果

**就地更新模式**（使用`--in-place`，推荐）：
- 直接在原文件中添加`verify`字段
- 保持数据顺序和其他字段不变
- 自动跳过已有`verify`字段的数据

**新建文件模式**：
- 创建新文件，原文件保持不变
- 默认命名：`{input_filename}_with_contracts.jsonl`

### 步骤4：执行验证

使用`run_verification.py`脚本执行验证：

#### 基本用法

```bash
# 就地更新原文件（推荐）
python scripts/run_verification.py data/sample_data.jsonl --in-place

# 创建新文件
python scripts/run_verification.py data/sample_data.jsonl

# 生成汇总报告
python scripts/run_verification.py data/sample_data.jsonl --in-place --summary report.json
```

#### 高级选项

```bash
# 覆盖已存在的验证结果，重新验证所有数据
python scripts/run_verification.py data/sample_data.jsonl --in-place --overwrite-result

# 启用详细调试日志
python scripts/run_verification.py data/sample_data.jsonl --in-place --log-level DEBUG

# 指定自定义输出文件
python scripts/run_verification.py data/sample_data.jsonl -o results.jsonl --summary detailed_report.json
```

#### 命令行参数详解

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `input_file` | string | ✅ | - | 输入的JSONL文件路径（需包含verify字段） |
| `-o, --output` | string | ❌ | `{input}_verified.jsonl` | 输出文件路径（不使用--in-place时） |
| `--in-place` | flag | ❌ | false | 就地更新原文件，不创建新文件 |
| `--summary` | string | ❌ | - | 汇总报告输出路径（JSON格式） |
| `--overwrite-result` | flag | ❌ | false | 覆盖已存在的verification_result字段 |
| `--log-level` | choice | ❌ | INFO | 日志级别：DEBUG, INFO, WARNING, ERROR |

#### 输出结果

**就地更新模式**（使用`--in-place`，推荐）：
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

**添加的字段**：
- `verification_result`: 完整的验证结果详情
- `scores`: 简化的得分信息
- `verification_error`: 验证错误信息（如出现错误）

### 步骤5：查看结果

```bash
# 查看汇总报告
cat report.json

# 查看完整数据文件
head -n 3 data/sample_data.jsonl

# 使用jq查看格式化的单条记录（需要安装jq）
head -n 1 data/sample_data.jsonl | jq .

# 统计验证通过率
grep '"verdict":"PASSED"' data/sample_data.jsonl | wc -l
```

## ⚙️ 完整配置选项

### 环境变量配置

系统支持通过环境变量进行配置：

| 环境变量 | 对应配置 | 类型 | 默认值 | 说明 |
|----------|----------|------|--------|------|
| `OPENAI_API_KEY` | openai_api_key | string | - | OpenAI API密钥 |
| `OPENAI_MODEL` | openai_model | string | gpt-4-turbo-preview | OpenAI模型名称 |
| `OPENAI_BASE_URL` | openai_base_url | string | - | OpenAI API基础URL |
| `AI_VERIFIER_LOG_LEVEL` | log_level | string | INFO | 日志级别 |
| `AI_VERIFIER_LOG_FILE` | log_file | string | - | 日志文件路径 |
| `AI_VERIFIER_STRICT_MODE` | strict_mode | boolean | true | 严格模式 |

**示例配置**：
```bash
# 基础配置
export OPENAI_API_KEY="sk-your-api-key-here"
export OPENAI_MODEL="gpt-4"
export AI_VERIFIER_LOG_LEVEL="DEBUG"

# 高级配置
export OPENAI_BASE_URL="https://api.openai.com/v1"
export AI_VERIFIER_STRICT_MODE="true"
export AI_VERIFIER_LOG_FILE="/var/log/ai_verifier.log"
```

### 配置文件选项

在`config/default.json`中的完整配置选项：

#### LLM相关配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `openai_model` | string | "gpt-4-turbo-preview" | OpenAI模型名称 |
| `contract_generation_temperature` | float | 0.2 | 生成验证契约时的温度参数 |
| `evaluation_temperature` | float | 0.1 | 执行评估时的温度参数 |
| `max_tokens_contract` | integer | 4000 | 生成契约的最大令牌数 |
| `max_tokens_evaluation` | integer | 2000 | 执行评估的最大令牌数 |

#### 代码执行配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `code_execution_timeout` | integer | 60 | 代码执行超时时间（秒） |
| `pytest_timeout` | integer | 30 | pytest测试超时时间（秒） |
| `enable_code_sandboxing` | boolean | true | 是否启用代码沙箱 |

#### 验证配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `pass_threshold` | float | 0.6 | 验证通过阈值（0-1） |
| `strict_mode` | boolean | true | 严格模式 |

#### 性能配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `max_concurrent_requests` | integer | 5 | 最大并发请求数 |
| `request_retry_count` | integer | 3 | 请求重试次数 |
| `request_timeout` | integer | 30 | 请求超时时间（秒） |

#### 日志配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `log_level` | string | "INFO" | 日志级别 |

### 自定义配置文件

```python
from ai_verifier import Config

# 从自定义配置文件加载
config = Config.load_from_file("my_config.json")

# 程序化配置
config = Config(
    openai_model="gpt-3.5-turbo",
    pass_threshold=0.8,
    strict_mode=False,
    code_execution_timeout=120
)

# 保存配置到文件
config.save_to_file("saved_config.json")
```

## 📊 汇总报告详解

使用`--summary`参数可以生成详细的处理报告：

### 报告结构

```json
{
  "total_items": 100,                    // 总数据条数
  "successful_verifications": 95,        // 验证成功条数
  "failed_verifications": 3,             // 验证失败条数
  "error_verifications": 2,              // 验证出错条数
  "overall_pass_rate": 85.5,            // 总体通过率
  "average_scores": {                    // 平均得分
    "code_verification": 0.892,
    "content_quality": 0.945,
    "topic_relevance": 0.987
  },
  "statistics": {                        // 统计信息
    "verification_success_rate": 95.0,   // 验证成功率
    "passed_items": 85                   // 通过验证的条数
  },
  "processing_time_seconds": 128.5       // 处理时间（秒）
}
```

### 报告分析

通过汇总报告可以：
- **监控处理进度**：跟踪验证成功率和错误率
- **质量评估**：分析各项得分的平均水平
- **性能优化**：了解处理时间和效率
- **问题排查**：识别验证失败的模式

## 📋 验证契约结构

系统根据任务类型生成不同的验证契约：

### 简单函数验证契约（代码验证 + LLM验证）

```json
{
  "code_verification": {
    "validation_type": "response_code_test",
    "input_source_description": "从AI回应中提取代码，导入函数进行测试",
    "validation_code": "完整的pytest测试套件",
    "expected_success_criteria": "所有测试用例通过，退出代码为0"
  },
  "llm_verification": {
    "evaluation_points": [
      {
        "id": "function_name_accuracy",
        "description": "检查函数名是否符合要求",
        "weight": 1.0
      },
      {
        "id": "code_quality",
        "description": "评估代码质量和可读性",
        "weight": 1.0
      }
    ]
  }
}
```

### 其他任务验证契约（仅LLM验证）

```json
{
  "llm_verification": {
    "evaluation_points": [
      {
        "id": "content_length",
        "description": "检查内容长度是否符合要求",
        "weight": 1.0
      },
      {
        "id": "topic_relevance",
        "description": "评估内容与主题的相关性",
        "weight": 1.0
      },
      {
        "id": "content_quality",
        "description": "评估内容质量和表达清晰度",
        "weight": 1.0
      }
    ]
  }
}
```

## 🎯 验证策略详解

系统采用**智能验证策略**，根据任务特性选择最适合的验证方式：

### ✅ 代码验证适用条件（需同时满足所有条件）

1. **明确的函数编写要求**：用户明确要求编写特定函数（有具体函数名）
2. **简单明确的功能**：函数功能简单，易于理解和测试
3. **可预测的输入输出**：函数的输入输出关系明确，可编写准确测试用例
4. **标准算法或数学计算**：数学运算、字符串处理、数据结构操作等
5. **无外部依赖**：函数不依赖外部API、文件系统、数据库等

**典型适用场景**：
- `add(a, b)` - 数学运算函数
- `reverse_string(s)` - 字符串处理函数
- `factorial(n)` - 数学计算函数
- `sort_list(lst)` - 数据结构操作

### 📋 LLM验证适用场景

- **输出格式限制**：字数限制、长度要求、特定格式要求
- **内容创作任务**：文章、故事、诗歌、翻译、摘要等
- **观点分析任务**：分析、解释、评价、建议、讨论等
- **数据转换任务**：JSON/XML转换、格式化、数据清洗等
- **复杂代码任务**：多文件项目、类设计、复杂逻辑、架构设计
- **系统集成代码**：Web应用、API开发、数据库操作、框架使用
- **界面相关代码**：HTML、CSS、前端组件、UI设计等

## 📖 实际使用案例

### 案例1：简单函数验证（代码验证 + LLM验证）

**输入**：
```json
{
  "prompt": "写一个计算两数之和的函数 add_numbers(a, b)",
  "response": "def add_numbers(a, b):\n    return a + b"
}
```

**生成的验证契约**：
- 代码验证：自动生成pytest测试用例
- LLM验证：函数名准确性、代码质量评估

**验证结果**：
```json
{
  "scores": {
    "code_verification": 1,
    "function_name_accuracy": 1,
    "code_quality": 1,
    "summary": {"total_points": 3, "max_points": 3, "pass_rate": 100.0, "verdict": "PASSED"}
  }
}
```

### 案例2：内容创作验证（仅LLM验证）

**输入**：
```json
{
  "prompt": "写一篇100字以内的关于AI的文章",
  "response": "人工智能是一门前沿的技术学科，通过模拟人类智能来解决复杂问题..."
}
```

**生成的验证契约**：
- LLM验证：字数限制、内容质量、主题相关性

**验证结果**：
```json
{
  "scores": {
    "word_count_check": 1,
    "content_quality": 1,
    "topic_relevance": 1,
    "summary": {"total_points": 3, "max_points": 3, "pass_rate": 100.0, "verdict": "PASSED"}
  }
}
```

### 案例3：复杂代码验证（仅LLM验证）

**输入**：
```json
{
  "prompt": "用Flask写一个Web应用，包含用户登录功能",
  "response": "from flask import Flask, request, session\n@app.route('/login', methods=['POST'])\ndef login():..."
}
```

**生成的验证契约**：
- LLM验证：框架使用正确性、功能实现完整性、代码结构合理性

**验证结果**：
```json
{
  "scores": {
    "framework_usage": 1,
    "login_functionality": 1,
    "code_structure": 1,
    "summary": {"total_points": 3, "max_points": 3, "pass_rate": 100.0, "verdict": "PASSED"}
  }
}
```

## 🔄 完整工作流程示例

### 方式1：命令行批处理

```bash
# 1. 设置环境变量
export OPENAI_API_KEY="your-api-key"
export AI_VERIFIER_LOG_LEVEL="INFO"

# 2. 生成验证契约（就地更新）
python scripts/generate_contracts.py data/my_data.jsonl \
    --in-place \
    --log-level INFO

# 3. 执行验证（就地更新，生成报告）
python scripts/run_verification.py data/my_data.jsonl \
    --in-place \
    --summary data/verification_report.json \
    --log-level INFO

# 4. 查看结果
cat data/verification_report.json
head -n 1 data/my_data.jsonl | jq .
```

### 方式2：Python脚本自动化

```python
#!/usr/bin/env python3
import subprocess
import json
import sys
from pathlib import Path

def run_batch_verification(data_file: str, report_file: str = None):
    """运行批量验证流程"""
    
    data_path = Path(data_file)
    if not data_path.exists():
        print(f"数据文件不存在: {data_file}")
        return False
    
    try:
        # 步骤1: 生成验证契约
        print("🎯 生成验证契约...")
        result = subprocess.run([
            "python", "scripts/generate_contracts.py",
            str(data_path),
            "--in-place",
            "--log-level", "INFO"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"生成验证契约失败: {result.stderr}")
            return False
        
        # 步骤2: 执行验证
        print("✅ 执行验证...")
        cmd = [
            "python", "scripts/run_verification.py",
            str(data_path),
            "--in-place",
            "--log-level", "INFO"
        ]
        
        if report_file:
            cmd.extend(["--summary", report_file])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"执行验证失败: {result.stderr}")
            return False
        
        # 步骤3: 显示结果
        if report_file and Path(report_file).exists():
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            print(f"📊 验证完成!")
            print(f"   总数据: {report['total_items']}")
            print(f"   成功验证: {report['successful_verifications']}")
            print(f"   通过率: {report['overall_pass_rate']:.1f}%")
            print(f"   处理时间: {report['processing_time_seconds']:.1f}秒")
        
        return True
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python batch_verify.py <data_file> [report_file]")
        sys.exit(1)
    
    data_file = sys.argv[1]
    report_file = sys.argv[2] if len(sys.argv) > 2 else "verification_report.json"
    
    success = run_batch_verification(data_file, report_file)
    sys.exit(0 if success else 1)
```

### 方式3：增量处理大数据集

```bash
# 处理大型数据集，支持中断和恢复

# 第一批：处理前1000条
head -n 1000 large_dataset.jsonl > batch_1.jsonl
python scripts/generate_contracts.py batch_1.jsonl --in-place
python scripts/run_verification.py batch_1.jsonl --in-place --summary report_1.json

# 第二批：处理接下来的1000条
sed -n '1001,2000p' large_dataset.jsonl > batch_2.jsonl
python scripts/generate_contracts.py batch_2.jsonl --in-place
python scripts/run_verification.py batch_2.jsonl --in-place --summary report_2.json

# 合并结果
cat batch_*.jsonl > final_results.jsonl

# 或者使用就地更新处理完整文件（推荐）
python scripts/generate_contracts.py large_dataset.jsonl --in-place
python scripts/run_verification.py large_dataset.jsonl --in-place --summary final_report.json
```

## 🧪 测试

```bash
# 运行所有测试
make test

# 代码格式化
make format

# 类型检查
make type-check

# 代码检查
make lint

# 全面检查
make check

# 运行演示测试
python scripts/demo.py
python scripts/demo_in_place.py
```

## 📈 最佳实践

### 1. 数据准备

- **prompt清晰明确**：使用具体、明确的任务描述
- **response质量一致**：确保AI生成的回应格式规范
- **数据备份**：处理前备份原始数据
- **分批处理**：大数据集建议分批处理，避免内存问题

### 2. 配置优化

```bash
# 生产环境配置
export OPENAI_MODEL="gpt-4"
export AI_VERIFIER_LOG_LEVEL="INFO"
export AI_VERIFIER_STRICT_MODE="true"

# 开发环境配置
export OPENAI_MODEL="gpt-3.5-turbo"
export AI_VERIFIER_LOG_LEVEL="DEBUG"
export AI_VERIFIER_STRICT_MODE="false"
```

### 3. 性能优化

- **并发处理**：调整`max_concurrent_requests`参数
- **超时设置**：根据任务复杂度调整超时时间
- **内存管理**：大文件使用流式处理
- **缓存利用**：避免重复处理相同数据

### 4. 错误处理

- **日志监控**：使用适当的日志级别
- **错误重试**：配置合理的重试次数
- **数据验证**：处理前验证JSONL格式
- **进度跟踪**：定期检查处理进度

### 5. 成本控制

- **API用量监控**：跟踪LLM API调用次数
- **模型选择**：根据任务复杂度选择合适模型
- **批处理优化**：合理设置批处理大小
- **缓存策略**：避免重复验证相同内容

## 🛡️ 安全考虑

- **代码沙箱**：所有用户代码在临时隔离环境执行
- **关键词过滤**：检测并阻止危险系统调用
- **超时控制**：防止长时间运行或死循环
- **权限限制**：最小化执行权限
- **输入验证**：严格验证所有输入数据
- **错误隔离**：单个验证失败不影响整体处理

## ⚠️ 故障排除

### 常见问题

1. **LLM接口未实现**
   ```
   ERROR: 用户LLM接口不可用: This function is not implemented
   ```
   **解决**：在`src/ai_verifier/llm_interface.py`中实现`call_llm_api`和`is_llm_available`函数

2. **API密钥问题**
   ```
   ERROR: OpenAI API调用失败
   ```
   **解决**：检查`OPENAI_API_KEY`环境变量是否正确设置

3. **JSON解析错误**
   ```
   ERROR: 第X行JSON解析失败
   ```
   **解决**：验证JSONL文件格式，确保每行是有效JSON

4. **缺少必需字段**
   ```
   WARNING: 第X条数据缺少字段: ['prompt']
   ```
   **解决**：确保数据包含必需的`prompt`和`response`字段

5. **验证超时**
   ```
   ERROR: 代码执行超时
   ```
   **解决**：检查AI生成的代码是否包含死循环，调整超时设置

### 调试技巧

```bash
# 启用详细调试
python scripts/generate_contracts.py data.jsonl --in-place --log-level DEBUG

# 测试单条数据
head -n 1 data.jsonl > test_single.jsonl
python scripts/generate_contracts.py test_single.jsonl --in-place --log-level DEBUG

# 检查LLM接口
python -c "from ai_verifier.llm_interface import is_llm_available; print(is_llm_available())"

# 验证JSONL格式
python -c "
import json
with open('data.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
            print(f'第{i}行：OK')
        except Exception as e:
            print(f'第{i}行错误：{e}')
"
```

## 🔧 扩展开发

### 添加新的验证类型

1. 修改`prompts/contract_generator.txt`定义新验证类型
2. 在`code_executor.py`中实现执行逻辑
3. 更新`prompts/score_evaluator.txt`评估标准

### 集成其他LLM

修改`llm_interface.py`支持新的API提供商：

```python
def call_llm_api(prompt: str, temperature=None, max_tokens=None, json_mode=True) -> str:
    # 根据配置选择不同的LLM提供商
    provider = os.getenv("LLM_PROVIDER", "openai")
    
    if provider == "anthropic":
        return call_anthropic_api(prompt, temperature, max_tokens, json_mode)
    elif provider == "azure":
        return call_azure_api(prompt, temperature, max_tokens, json_mode)
    else:
        return call_openai_api(prompt, temperature, max_tokens, json_mode)
```

### 自定义验证策略

通过修改契约生成器的prompt来调整验证策略：

```python
from ai_verifier import ContractGenerator, Config

# 自定义契约生成器
class CustomContractGenerator(ContractGenerator):
    def generate_verification_contract(self, prompt: str, response: str) -> dict:
        # 实现自定义逻辑
        pass

# 使用自定义生成器
config = Config()
generator = CustomContractGenerator(config)
```

## 🤝 贡献指南

1. Fork本项目
2. 创建功能分支：`git checkout -b feature/new-feature`
3. 提交更改：`git commit -am 'Add new feature'`
4. 推送分支：`git push origin feature/new-feature`
5. 创建Pull Request

## 📄 许可证

本项目使用MIT许可证 - 详见[LICENSE](LICENSE)文件

## 🆘 支持与反馈

- **问题报告**：[Issues](https://github.com/ai-verifier/issues)
- **功能请求**：[Discussions](https://github.com/ai-verifier/discussions)
- **文档**：[Wiki](https://github.com/ai-verifier/wiki)

## 🔖 版本历史

- **v2.0.0** - 添加就地更新功能，智能验证策略优化
- **v1.0.0** - 初始版本，支持混合验证模式
- 更多版本信息请查看[CHANGELOG.md](CHANGELOG.md)

---

**⭐ 如果这个项目对您有帮助，请给我们一个Star！** 