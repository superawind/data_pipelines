#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
去重框架命令行入口 - 零延迟启动，实时进度显示

核心特性：
- 🚀 零延迟启动 - 立即开始处理，无需等待数据加载
- 📊 实时进度显示 - 进度条实时更新，避免假死现象
- 🌊 流式架构 - 边读边处理边写入，支持任意大小文件
- 💓 心跳机制 - 定期更新状态，确保用户感知处理进度
- 📋 详细日志 - 记录重复数据和阈值边界示例，便于效果评估

使用示例：
    # 严格哈希去重（完全匹配）
    python -m deduplicate --method strict_hash --input_file data.jsonl --output_file output.jsonl
    
    # N-gram LSH去重（相似度匹配）  
    python -m deduplicate --method ngram_lsh --input_file data.jsonl --output_file output.jsonl \
        --jaccard-threshold 0.85 --ngram-size 10
    
    # Embedding去重（语义匹配）
    python -m deduplicate --method embedding --input_file data.jsonl --output_file output.jsonl \
        --embeddings-file embeddings.npy --threshold 0.95
        
    # Embedding去重（使用JSONL格式的embedding文件）
    python -m deduplicate --method embedding --input_file data.jsonl --output_file output.jsonl \
        --embeddings-file embeddings.jsonl --threshold 0.95 \
        --prompt-id-key prompt_id --embedding-id-key prompt_id --embedding-vector-key embedding
"""

import sys
import os
import argparse
import time

# 添加dep_code到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
dep_code_path = os.path.join(current_dir, 'dep_code')
sys.path.insert(0, dep_code_path)

# 导入必要的模块
from config import DedupMethod, DedupConfig

# 导入全局去重版本
try:
    sys.path.append(current_dir)  # 将主目录添加到路径
    from global_strict_hash_dedup import GlobalStrictHashDeduplicator
    from global_ngram_lsh_dedup import GlobalNgramLshDeduplicator
    from global_embedding_dedup import GlobalEmbeddingDeduplicator
    GLOBAL_MODES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 全局模式不可用: {e}")
    GLOBAL_MODES_AVAILABLE = False


def main():
    """去重命令行主入口"""
    parser = argparse.ArgumentParser(
        description="去重框架 - 零延迟启动，实时进度显示，支持大文件处理",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # ===== 基础必需参数 =====
    parser.add_argument(
        "--method",
        choices=[m.value for m in DedupMethod],
        required=True,
        help="去重方法选择:\n"
             "  strict_hash - 严格哈希匹配（完全相同）\n"
             "  ngram_lsh   - N-gram LSH相似度匹配（文本相似）\n"
             "  embedding   - 向量语义匹配（语义相似）"
    )
    
    parser.add_argument(
        "--input_file",
        required=True,
        help="输入JSONL文件路径（支持任意大小，采用流式读取）"
    )
    
    parser.add_argument(
        "--output_file", 
        required=True,
        help="输出JSONL文件路径（去重后的唯一数据）"
    )
    
    parser.add_argument(
        "--content_keys",
        nargs="+",
        default=["prompt"],
        help="用于去重的内容字段列表（默认: prompt）\n"
             "示例: --content_keys prompt instruction"
    )
    
    parser.add_argument(
        "--field_order",
        nargs="+",
        default=["prompt", "instruction", "input", "output"],
        help="输出JSON字段顺序（默认: prompt instruction input output）"
    )
    
    # ===== 通用性能参数 =====
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="并行worker数量（默认: 16）"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="批处理大小（默认: 10000）"
    )
    
    parser.add_argument(
        "--log_dir",
        help="日志目录路径（可选，默认自动选择）"
    )
    
    # ===== N-gram LSH 专用参数 =====
    ngram_group = parser.add_argument_group('N-gram LSH专用参数')
    
    ngram_group.add_argument(
        "--ngram_size",
        type=int,
        default=10,
        help="N-gram大小（默认: 10，推荐: 8-12）"
    )
    
    ngram_group.add_argument(
        "--jaccard_threshold",
        type=float,
        default=0.85,
        help="Jaccard相似度阈值（默认: 0.85，推荐: 0.8-0.9）"
    )
    
    ngram_group.add_argument(
        "--num_permutations",
        type=int,
        default=128,
        help="MinHash排列数（默认: 128，推荐: 64-256）"
    )
    
    # ===== Embedding 专用参数 =====
    embedding_group = parser.add_argument_group('Embedding专用参数')
    
    embedding_group.add_argument(
        "--embeddings_file",
        help="Embedding文件路径（.npy或.jsonl）"
    )
    
    embedding_group.add_argument(
        "--embeddings_files",
        nargs="+",
        help="多个Embedding文件路径（仅JSONL格式）"
    )
    
    embedding_group.add_argument(
        "--embeddings_format",
        choices=["npy", "jsonl"],
        default="jsonl",
        help="Embedding文件格式（默认: jsonl）"
    )
    
    embedding_group.add_argument(
        "--prompt_id_key",
        default="prompt_id",
        help="数据文件中的ID字段名（默认: prompt_id）"
    )
    
    embedding_group.add_argument(
        "--embedding_id_key", 
        default="prompt_id",
        help="Embedding文件中的ID字段名（默认: prompt_id）"
    )
    
    embedding_group.add_argument(
        "--embedding_vector_key",
        default="embedding", 
        help="Embedding文件中的向量字段名（默认: embedding）"
    )
    
    embedding_group.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="相似度阈值（默认: 0.95，推荐: 0.9-0.98）"
    )
    
    embedding_group.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-K检索数量（默认: 10，推荐: 5-20）"
    )
    
    embedding_group.add_argument(
        "--use_gpu",
        action="store_true",
        help="启用GPU加速"
    )
    
    embedding_group.add_argument(
        "--gpu_device",
        type=int,
        default=0,
        help="GPU设备ID（默认: 0）"
    )
    
    # ===== LSH优化参数（适用于大规模数据） =====
    lsh_group = parser.add_argument_group('LSH优化参数（适用于大规模数据）')
    
    lsh_group.add_argument(
        "--use_lsh",
        action="store_true",
        help="启用LSH优化（适用于千万级、亿级数据）"
    )
    
    lsh_group.add_argument(
        "--lsh_num_tables",
        type=int,
        default=10,
        help="LSH哈希表数量（默认: 10，推荐: 5-20）"
    )
    
    lsh_group.add_argument(
        "--lsh_hash_size",
        type=int,
        default=10,
        help="LSH哈希位数（默认: 10，推荐: 8-12）"
    )
    
    args = parser.parse_args()
    
    try:
        # 验证embedding参数
        if args.method == "embedding":
            if not args.embeddings_file and not args.embeddings_files:
                print("❌ 错误: embedding方法必须指定 --embeddings_file 或 --embeddings_files 之一")
                print("💡 支持两种向量文件格式:")
                print("   .npy格式：numpy数组文件，与输入文件行数一致")
                print("   .jsonl格式：每行包含prompt_id和embedding字段的JSON文件")
                print("💡 多文件支持：--embeddings_files file1.jsonl file2.jsonl ...")
                sys.exit(1)
            
            if args.embeddings_file and args.embeddings_files:
                print("❌ 错误: 不能同时指定 --embeddings_file 和 --embeddings_files")
                sys.exit(1)
            
            if args.embeddings_files and len(args.embeddings_files) < 2:
                print("❌ 错误: --embeddings_files 必须包含至少2个文件")
                sys.exit(1)
        
        # 处理embedding格式参数
        embeddings_format = None
        if args.method == "embedding":
            if hasattr(args, 'embeddings_format') and args.embeddings_format:
                # 用户明确指定了格式
                from dep_code.config import EmbeddingFileFormat
                if args.embeddings_format == "jsonl":
                    embeddings_format = EmbeddingFileFormat.JSONL
                else:
                    embeddings_format = EmbeddingFileFormat.NPY
            else:
                # 根据文件扩展名自动检测格式
                from dep_code.config import EmbeddingFileFormat
                if args.embeddings_file and args.embeddings_file.endswith('.npy'):
                    embeddings_format = EmbeddingFileFormat.NPY
                elif args.embeddings_file and args.embeddings_file.endswith('.jsonl'):
                    embeddings_format = EmbeddingFileFormat.JSONL
                elif args.embeddings_files:
                    # 多文件默认为JSONL格式
                    embeddings_format = EmbeddingFileFormat.JSONL
                else:
                    # 默认为JSONL格式
                    embeddings_format = EmbeddingFileFormat.JSONL
        
        # 创建去重配置
        config_kwargs = {
            "method": args.method,
            "input_file": args.input_file,
            "output_file": args.output_file,
            "content_keys": args.content_keys,
            "field_order": args.field_order,
            "num_workers": args.num_workers,
            "batch_size": args.batch_size,
            "ngram_size": args.ngram_size,
            "jaccard_threshold": args.jaccard_threshold,
            "num_permutations": args.num_permutations,
            "embeddings_file": args.embeddings_file,
            "embeddings_files": args.embeddings_files,
            "embeddings_format": embeddings_format,
            "prompt_id_key": args.prompt_id_key,
            "embedding_id_key": args.embedding_id_key,
            "embedding_vector_key": args.embedding_vector_key,
            "threshold": args.threshold,
            "top_k": args.top_k,
            "use_gpu": args.use_gpu,
            "gpu_device": args.gpu_device,
            "use_lsh": args.use_lsh,
            "lsh_num_tables": args.lsh_num_tables,
            "lsh_hash_size": args.lsh_hash_size
        }
        
        # 添加可选参数（只有非None时才添加）
        if args.log_dir is not None:
            config_kwargs["log_dir"] = args.log_dir
        
        config = DedupConfig(**config_kwargs)
        
        # 显示启动信息
        print(f"🚀 去重框架启动")
        print("=" * 60)
        print(f"📁 输入文件: {args.input_file}")
        print(f"💾 输出文件: {args.output_file}")
        print(f"🔧 去重方法: {args.method}")
        print(f"📊 内容字段: {args.content_keys}")
        print("=" * 60)
        print()
        
        # 根据方法选择对应的去重器
        deduplicator = None
        
        if args.method == "strict_hash":
            print("🚀 使用严格哈希去重")
            print("💡 特点: 完全匹配去重，零误删")
            
            from global_strict_hash_dedup import GlobalStrictHashDeduplicator
            deduplicator = GlobalStrictHashDeduplicator(config)
        
        elif args.method == "ngram_lsh":
            print("🚀 使用N-gram LSH去重")
            print("💡 特点: 文本相似度去重")
            print(f"💡 配置: {args.ngram_size}-gram, 阈值{args.jaccard_threshold}, {args.num_permutations}排列")
            
            from global_ngram_lsh_dedup import GlobalNgramLshDeduplicator
            deduplicator = GlobalNgramLshDeduplicator(config)
            
        elif args.method == "embedding":
            # 显示embedding配置信息
            if args.embeddings_file:
                print(f"🎯 Embedding文件: {args.embeddings_file}")
            elif args.embeddings_files:
                print(f"🎯 Embedding文件: {len(args.embeddings_files)} 个文件")
            
            # 显示格式
            from dep_code.config import EmbeddingFileFormat
            format_name = "npy" if embeddings_format == EmbeddingFileFormat.NPY else "jsonl"
            print(f"📊 文件格式: {format_name}")
            
            print("🚀 使用Embedding去重")
            print(f"💡 配置: 阈值{args.threshold}, Top-{args.top_k}检索")
            if args.use_lsh:
                print(f"💡 LSH优化: 启用（表={args.lsh_num_tables}, 位={args.lsh_hash_size}）")
            if args.use_gpu:
                print("💡 GPU加速: 启用")
            
            # 根据use_lsh选择去重器
            if args.use_lsh:
                from global_embedding_lsh_dedup import GlobalEmbeddingLSHDeduplicator
                deduplicator = GlobalEmbeddingLSHDeduplicator(config)
            else:
                from global_embedding_dedup import GlobalEmbeddingDeduplicator
                deduplicator = GlobalEmbeddingDeduplicator(config)
        
        else:
            print(f"❌ 不支持的去重方法: {args.method}")
            sys.exit(1)
        
        # 执行去重
        print()
        start_time = time.time()

        # 极速模式直接执行，内部处理统计
        deduplicator.deduplicate()
        # 为了统一接口，创建基本的stats
        stats = {'total_time': time.time() - start_time}
        
        total_time = time.time() - start_time
        
        
        print("=" * 60)
        
        # 日志文件信息
        print("📁 详细日志文件位置:")
        print(f"   📄 重复数据记录: global_dedup_logs/duplicates_{args.method}_*.jsonl")
        print(f"   📄 阈值边界示例: global_dedup_logs/threshold_examples_{args.method}_*.jsonl")
        print(f"   📄 处理日志: global_dedup_logs/global_dedup_{args.method}_*.log")
        print(f"   📄 性能统计: global_dedup_logs/performance_{args.method}_*.json")
        
        # 性能评估和建议
        file_size_gb = os.path.getsize(args.input_file) / (1024**3)
        processing_time = stats.get('total_time', total_time)
        
        if file_size_gb > 50:  # 大文件建议
            print("💡 大文件处理建议:")
            print("   - 确保充足的磁盘空间（至少2倍文件大小）")
            print("   - 使用SSD硬盘以提高I/O性能")
            print("   - 适当增加--num-workers参数值")
        
        print("🎯 任务完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ 文件不存在: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ 参数错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 去重失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()