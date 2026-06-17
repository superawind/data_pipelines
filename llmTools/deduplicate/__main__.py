#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
流式去重框架命令行入口 - 零延迟启动，实时进度显示

支持三种流式去重算法：
1. 流式严格哈希去重 (StreamingStrictHashDeduplicator) - 完全匹配，速度最快
2. 流式N-gram LSH去重 (StreamingNgramLshDeduplicator) - 近似匹配，平衡速度和精度
3. 流式Embedding去重 (StreamingEmbeddingDeduplicator) - 语义匹配，精度最高

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
from streaming_strict_hash import StreamingStrictHashDeduplicator
from streaming_ngram_lsh import StreamingNgramLshDeduplicator
from streaming_embedding import StreamingEmbeddingDeduplicator

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
    """流式去重命令行主入口"""
    parser = argparse.ArgumentParser(
        description="流式去重框架 - 零延迟启动，实时进度显示，支持大文件处理",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # ===== 基础必需参数 =====
    parser.add_argument(
        "--method",
        choices=[m.value for m in DedupMethod],
        required=True,
        help="去重方法选择:\n"
             "  strict_hash - 严格哈希匹配（最快，完全相同才去重）\n"
             "  ngram_lsh   - N-gram LSH相似度匹配（平衡速度和精度）\n"
             "  embedding   - 向量语义匹配（最准确，需要向量文件）"
    )
    
    parser.add_argument(
        "--mode",
        choices=["streaming", "global"],
        default="streaming",
        help="处理模式选择:\n"
             "  streaming - 流式模式（低内存占用，零延迟启动，支持超大文件）\n"
             "  global    - 全局模式（高精度，真正的全局去重，适合中等文件）\n"
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
        help="并行worker数量（默认: 16\n"
             "建议值: CPU核心数的1-2倍，大文件可设置16+"
    )
    
    parser.add_argument(
        "--chunk_size_mb",
        type=int,
        default=16,
        help="分块大小（MB）（默认: 16）\n"
             "控制每个处理块的大小，影响内存使用和性能\n"
             "推荐值: 16MB（标准），64MB（高性能），128MB（极致性能）"
    )
    
    parser.add_argument(
        "--queue_maxsize",
        type=int,
        default=8,
        help="队列最大缓存块数（默认: 8）\n"
             "控制内存中同时缓存的数据块数量\n"
             "总内存使用 = chunk_size_mb × queue_maxsize"
    )
    
    parser.add_argument(
        "--buffer_size_mb",
        type=int,
        default=2,
        help="I/O缓冲区大小（MB）（默认: 2）\n"
             "控制文件读写的缓冲区大小\n"
             "推荐值: 2MB（标准），8MB（高性能），16MB（极致性能）"
    )
    
    parser.add_argument(
        "--log_dir",
        help="日志目录路径（可选）\n"
             "指定日志文件保存目录，如果不指定将使用默认目录\n"
             "默认: 根据模式自动选择（stream_dedup_logs/ 或 global_dedup_logs/）"
    )
    
    # ===== N-gram LSH 专用参数 =====
    ngram_group = parser.add_argument_group('N-gram LSH专用参数', '仅在--method ngram_lsh时使用')
    
    ngram_group.add_argument(
        "--ngram_size",
        type=int,
        default=10,
        help="N-gram大小，控制文本分割粒度（默认: 10）\n"
             "值越大越精确但越慢，推荐范围: 8-12"
    )
    
    ngram_group.add_argument(
        "--jaccard_threshold",
        type=float,
        default=0.85,
        help="Jaccard相似度阈值（默认: 0.85）\n"
             "范围: 0.0-1.0，值越高越严格，推荐: 0.8-0.9"
    )
    
    ngram_group.add_argument(
        "--num_permutations",
        type=int,
        default=128,
        help="MinHash排列数，控制精度和速度平衡（默认: 128）\n"
             "值越大越精确但内存消耗越大，推荐: 64-256"
    )
    
    # ===== Embedding 专用参数 =====
    embedding_group = parser.add_argument_group('Embedding专用参数', '仅在--method embedding时使用')
    
    embedding_group.add_argument(
        "--embeddings_file",
        help="嵌入向量文件路径（embedding方法必需）\n"
             "支持两种格式：\n"
             "  .npy格式：numpy数组文件，向量顺序必须与输入文件行顺序一致\n"
             "  .jsonl格式：每行包含prompt_id和embedding字段的JSON文件"
    )
    
    embedding_group.add_argument(
        "--embeddings_files",
        nargs="+",
        help="多个嵌入向量文件路径列表（用于大文件分割场景）\n"
             "当大的embedding文件被分割成多个小文件时使用\n"
             "支持JSONL格式的多文件加载\n"
             "示例: --embeddings_files part1.jsonl part2.jsonl part3.jsonl"
    )
    
    embedding_group.add_argument(
        "--embeddings_format",
        choices=["npy", "jsonl"],
        default="jsonl",
        help="Embedding文件格式（默认: jsonl）\n"
             "  npy  - 传统numpy数组格式，要求与输入文件行顺序一致\n"
             "  jsonl - 新格式，每行包含prompt_id和embedding，支持基于ID匹配"
    )
    
    embedding_group.add_argument(
        "--prompt_id_key",
        default="prompt_id",
        help="输入数据中prompt_id字段名（默认: prompt_id）\n"
             "用于与embedding文件中的prompt_id进行匹配"
    )
    
    embedding_group.add_argument(
        "--embedding_id_key", 
        default="prompt_id",
        help="embedding文件中ID字段名（默认: prompt_id）\n"
             "embedding JSONL文件中存储ID的字段名\n"
             "可自定义为其他字段名，如'id', 'sample_id'等"
    )
    
    embedding_group.add_argument(
        "--embedding_vector_key",
        default="embedding", 
        help="embedding文件中向量字段名（JSONL格式）"
    )
    
    # 全局模式专用参数组
    global_group = parser.add_argument_group('全局模式专用参数', '仅在--mode global时使用')
    global_group.add_argument(
        "--batch_size", type=int, default=10000,
        help="批处理大小，控制内存使用和处理效率"
    )
    
    embedding_group.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="向量相似度阈值（默认: 0.95）\n"
             "范围: 0.0-1.0，值越高越严格，推荐: 0.9-0.98"
    )
    
    embedding_group.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-K相似向量检索数量（默认: 10）\n"
             "值越大召回越高但越慢，推荐: 5-20"
    )
    
    embedding_group.add_argument(
        "--use_gpu",
        action="store_true",
        help="启用GPU加速FAISS向量检索（需要faiss-gpu）"
    )
    
    embedding_group.add_argument(
        "--gpu_device",
        type=int,
        default=0,
        help="GPU设备ID（默认: 0，仅在--use_gpu时有效）"
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
            "chunk_size_mb": args.chunk_size_mb,
            "queue_maxsize": args.queue_maxsize,
            "buffer_size_mb": args.buffer_size_mb,
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
            "batch_size": args.batch_size
        }
        
        # 添加可选参数（只有非None时才添加）
        if args.log_dir is not None:
            config_kwargs["log_dir"] = args.log_dir
        
        config = DedupConfig(**config_kwargs)
        
        # 检查模式兼容性
        
        if args.mode == "global" and not GLOBAL_MODES_AVAILABLE:
            print("❌ 错误: 全局模式不可用，请使用streaming模式")
            sys.exit(1)
        
        # 显示启动信息
        mode_name = "流式模式" if args.mode == "streaming" else "全局模式"
        print(f"🚀 去重框架启动 - {mode_name}")
        print("=" * 60)
        print(f"📁 输入文件: {args.input_file}")
        print(f"💾 输出文件: {args.output_file}")
        print(f"🔧 去重方法: {args.method}")
        print(f"⚡ 处理模式: {mode_name}")
        print(f"📊 内容字段: {args.content_keys}")
        print("=" * 60)
        print()
        
        # 根据方法和模式选择对应的去重器
        deduplicator = None
        
        if args.method == "strict_hash":
            if args.mode == "streaming":
                print("🌊 使用流式严格哈希去重算法")
                print("💡 特点: 完全匹配去重，零延迟启动，内存占用低")
                print("💡 适用: 需要精确去重，文件很大或内存有限的场景")
                print("💡 建议安装加速库: pip install xxhash orjson")
                deduplicator = StreamingStrictHashDeduplicator(config)
            else:  # global mode
                print("🚀 使用全局严格哈希去重算法")
                print("💡 特点: 基于完全匹配的真正全局去重，零误删")
                print("💡 适用: 中等规模文件，追求最高精度的完全匹配去重")
                print("💡 建议安装加速库: pip install xxhash orjson")
                
                from global_strict_hash_dedup import GlobalStrictHashDeduplicator
                deduplicator = GlobalStrictHashDeduplicator(config)
        
        elif args.method == "ngram_lsh":
            if args.mode == "streaming":
                print("🌊 使用流式N-gram LSH去重算法")
                print("💡 特点: 基于文本相似度的智能去重，零延迟启动")
                print("💡 适用: 需要检测相似文本，内存有限的场景")
                print(f"💡 配置: {args.ngram_size}-gram, 阈值{args.jaccard_threshold}, {args.num_permutations}排列")
                print("💡 建议安装加速库: pip install xxhash orjson datasketch")
                deduplicator = StreamingNgramLshDeduplicator(config)
            else:  # global mode
                print("🚀 使用全局N-gram LSH去重算法")
                print("💡 特点: 基于文本相似度的真正全局去重，智能模糊匹配")
                print("💡 适用: 中等规模文件，追求高精度的文本相似度去重")
                print(f"💡 配置: {args.ngram_size}-gram, 阈值{args.jaccard_threshold}, {args.num_permutations}排列")
                print("💡 建议安装加速库: pip install xxhash orjson datasketch")
                
                from global_ngram_lsh_dedup import GlobalNgramLshDeduplicator
                deduplicator = GlobalNgramLshDeduplicator(config)
            
        elif args.method == "embedding":
            # 显示embedding配置信息
            if args.embeddings_file:
                print(f"🎯 Embedding文件: {args.embeddings_file}")
            elif args.embeddings_files:
                print(f"🎯 Embedding文件: {len(args.embeddings_files)} 个文件")
                for i, f in enumerate(args.embeddings_files[:3], 1):
                    print(f"    {i}. {f}")
                if len(args.embeddings_files) > 3:
                    print(f"    ... 还有 {len(args.embeddings_files) - 3} 个文件")
            
            # 显示检测到的格式
            from dep_code.config import EmbeddingFileFormat
            format_name = "npy" if embeddings_format == EmbeddingFileFormat.NPY else "jsonl"
            print(f"📊 文件格式: {format_name}")
            
            if embeddings_format == EmbeddingFileFormat.JSONL:
                print(f"🔗 ID字段映射: {args.prompt_id_key} -> {args.embedding_id_key}")
                print(f"📐 向量字段: {args.embedding_vector_key}")
            else:
                print(f"�� NPY格式: 按行索引严格对应")
            
            if args.mode == "streaming":
                print("🌊 使用流式Embedding去重算法")
                print("💡 特点: 基于向量语义的高精度去重，零延迟启动")
                print("💡 适用: 需要语义级别去重，内存有限的场景")
                print(f"💡 配置: 阈值{args.threshold}, Top-{args.top_k}检索")
                if args.use_gpu:
                    print("🔥 GPU加速模式已启用")
                    print("💡 建议安装: pip install xxhash orjson faiss-gpu")
                else:
                    print("💡 建议安装加速库: pip install xxhash orjson faiss-cpu")
                deduplicator = StreamingEmbeddingDeduplicator(config)
            else:  # global mode
                print("🚀 使用全局Embedding去重算法")
                print("💡 特点: 基于向量语义的真正全局去重，最高精度")
                print("💡 适用: 中等规模文件，追求最高去重精度的场景")
                print(f"💡 配置: 阈值{args.threshold}, Top-{args.top_k}检索")
                if args.use_gpu:
                    print("🔥 GPU加速模式已启用")
                    print("💡 建议安装: pip install xxhash orjson faiss-gpu")
                else:
                    print("💡 建议安装加速库: pip install xxhash orjson faiss-cpu")
                
                from global_embedding_dedup import GlobalEmbeddingDeduplicator
                deduplicator = GlobalEmbeddingDeduplicator(config)
        
        else:
            print(f"❌ 不支持的去重方法: {args.method}")
            sys.exit(1)
        
        # 执行去重
        print()
        start_time = time.time()
        
        if args.mode == "streaming":
            # 流式模式返回 (keep_indices, remove_indices, stats)
            _, _, stats = deduplicator.deduplicate()
        else:
            # 极速模式直接执行，内部处理统计
            deduplicator.deduplicate()
            # 为了统一接口，创建基本的stats
            stats = {'total_time': time.time() - start_time}
        
        total_time = time.time() - start_time
        
        # 极速模式内部已经输出了完整统计，流式模式需要额外输出
        if args.mode == "streaming":
            # 输出详细的结果统计
            print()
            mode_title = "流式去重完成!" 
            print(f"🎉 {mode_title}")
            print("=" * 60)
            print(f"📊 总行数: {stats.get('total_lines', 0):,}")
            print(f"✅ 保留行数: {stats.get('unique_lines', 0):,}")  
            print(f"🗑️  删除行数: {stats.get('duplicate_lines', 0):,}")
            
            if 'dedup_ratio' in stats:
                print(f"📉 去重比例: {stats['dedup_ratio']:.2%}")
            
            # 性能统计
            if 'processing_speed_mb_s' in stats:
                print(f"🚀 处理速度: {stats['processing_speed_mb_s']:.1f} MB/秒")
                gb_per_hour = stats['processing_speed_mb_s'] * 3.6
                print(f"⚡ 处理能力: {gb_per_hour:.1f} GB/小时")
                
            if 'lines_per_second' in stats:
                print(f"📈 行处理速度: {stats['lines_per_second']:,.0f} 行/秒")
                
            # 时间统计
            if 'total_time' in stats:
                if stats['total_time'] > 3600:
                    print(f"⏱️  总耗时: {stats['total_time']/3600:.1f} 小时")
                elif stats['total_time'] > 60:
                    print(f"⏱️  总耗时: {stats['total_time']/60:.1f} 分钟")
                else:
                    print(f"⏱️  总耗时: {stats['total_time']:.1f} 秒")
            else:
                print(f"⏱️  总耗时: {total_time:.1f} 秒")
            
            # 算法特定统计信息
            if args.method == "ngram_lsh":
                if 'lsh_comparisons' in stats:
                    print(f"🔍 LSH比较次数: {stats['lsh_comparisons']:,}")
                if 'similarity_checks' in stats:
                    print(f"🔍 相似度计算次数: {stats['similarity_checks']:,}")
                    
            elif args.method == "embedding":
                if 'vector_searches' in stats:
                    print(f"🔍 向量搜索次数: {stats['vector_searches']:,}")
                if 'similarity_comparisons' in stats:
                    print(f"🔍 相似度比较次数: {stats['similarity_comparisons']:,}")
        else:
            # 极速模式已经内部输出了统计信息
            print(f"\n⏱️ 执行完成，总耗时: {total_time:.1f} 秒")
        
        print("=" * 60)
        
        # 日志文件信息
        print("📁 详细日志文件位置:")
        print(f"   📄 重复数据记录: {args.mode}_dedup_logs/duplicates_{args.method}_*.jsonl")
        print(f"   📄 阈值边界示例: {args.mode}_dedup_logs/threshold_examples_{args.method}_*.jsonl")
        print(f"   📄 处理日志: {args.mode}_dedup_logs/{args.mode}_dedup_{args.method}_*.log")
        print(f"   📄 性能统计: {args.mode}_dedup_logs/performance_{args.method}_*.json")
        
        # 性能评估和建议
        file_size_gb = os.path.getsize(args.input_file) / (1024**3)
        processing_time = stats.get('total_time', total_time)
        
        print()
        if processing_time <= 1800:  # 30分钟内
            print("🏆 性能优秀！零延迟启动和实时进度显示效果完美！")
        elif processing_time <= 3600:  # 1小时内
            print("✅ 性能良好！流式处理架构运行稳定！")
        elif processing_time <= 7200:  # 2小时内
            print("✅ 性能可接受！大文件处理成功完成！")
        else:
            print("⚠️  处理时间较长，建议优化参数或增加硬件配置")
        
        if file_size_gb > 50:  # 大文件建议
            print("💡 大文件处理建议:")
            print("   - 确保充足的磁盘空间（至少2倍文件大小）")
            print("   - 使用SSD硬盘以提高I/O性能")
            print("   - 适当增加--num-workers参数值")
        
        print("🎯 任务完成！享受零延迟启动和实时进度显示的流式去重体验！")
        
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