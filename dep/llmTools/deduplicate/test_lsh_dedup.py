#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试LSH去重器

创建模拟数据和embedding，测试LSH去重功能
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# 添加当前目录到path
sys.path.insert(0, str(Path(__file__).parent))


def create_test_data(num_samples=1000, num_duplicates=100, embedding_dim=128):
    """
    创建测试数据和embeddings
    
    Args:
        num_samples: 总样本数
        num_duplicates: 重复样本数
        embedding_dim: embedding维度
    """
    print(f"🔧 创建测试数据...")
    print(f"  📊 样本数: {num_samples}")
    print(f"  🔄 重复数: {num_duplicates}")
    print(f"  📐 维度: {embedding_dim}")
    
    # 创建测试目录
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    
    # 生成唯一的embedding向量
    unique_embeddings = []
    for i in range(num_samples - num_duplicates):
        # 生成随机向量并归一化
        vec = np.random.randn(embedding_dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        unique_embeddings.append(vec)
    
    # 添加重复向量（随机选择一些向量进行复制）
    all_embeddings = unique_embeddings.copy()
    duplicate_indices = np.random.choice(len(unique_embeddings), num_duplicates, replace=True)
    
    for idx in duplicate_indices:
        # 添加轻微噪声使其不完全相同但高度相似
        duplicate_vec = unique_embeddings[idx] + np.random.randn(embedding_dim).astype(np.float32) * 0.01
        duplicate_vec = duplicate_vec / np.linalg.norm(duplicate_vec)
        all_embeddings.append(duplicate_vec)
    
    # 转换为numpy数组
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    
    # 创建JSONL数据文件
    data_file = test_dir / "test_input.jsonl"
    print(f"\n📝 写入数据文件: {data_file}")
    
    with open(data_file, 'w', encoding='utf-8') as f:
        for i in range(len(embeddings_array)):
            data = {
                'prompt_id': f'sample_{i:06d}',
                'prompt': f'This is test sample number {i}. Content for testing deduplication.',
                'instruction': f'Instruction {i}',
                'output': f'Output {i}'
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"✅ 数据文件创建完成: {len(embeddings_array)} 条")
    
    # 保存embeddings - NPY格式
    npy_file = test_dir / "test_embeddings.npy"
    np.save(npy_file, embeddings_array)
    print(f"✅ NPY格式embedding保存: {npy_file}")
    
    # 保存embeddings - JSONL格式
    jsonl_file = test_dir / "test_embeddings.jsonl"
    print(f"📝 写入JSONL格式embedding: {jsonl_file}")
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for i in range(len(embeddings_array)):
            emb_data = {
                'prompt_id': f'sample_{i:06d}',
                'embedding': embeddings_array[i].tolist()
            }
            f.write(json.dumps(emb_data, ensure_ascii=False) + '\n')
    
    print(f"✅ JSONL格式embedding保存: {jsonl_file}")
    
    # 计算理论去重率
    expected_dedup_ratio = num_duplicates / num_samples
    print(f"\n📈 预期去重率: {expected_dedup_ratio:.2%}")
    print(f"  📥 总样本: {num_samples}")
    print(f"  ✅ 预期保留: {num_samples - num_duplicates}")
    print(f"  🗑️  预期删除: ~{num_duplicates}")
    
    return data_file, npy_file, jsonl_file


def test_lsh_dedup(data_file, embeddings_file, use_jsonl_format=False):
    """
    测试LSH去重
    
    Args:
        data_file: 数据文件路径
        embeddings_file: embedding文件路径
        use_jsonl_format: 是否使用JSONL格式
    """
    from dep_code.config import DedupConfig, DedupMethod, EmbeddingFileFormat
    from global_embedding_lsh_dedup import GlobalEmbeddingLSHDeduplicator
    
    print(f"\n" + "="*70)
    print(f"🧪 测试LSH去重")
    print(f"="*70)
    
    # 输出文件
    test_dir = Path(data_file).parent
    output_file = test_dir / "test_output_lsh.jsonl"
    
    # 配置
    config = DedupConfig(
        method=DedupMethod.EMBEDDING,
        input_file=str(data_file),
        output_file=str(output_file),
        embeddings_file=str(embeddings_file),
        embeddings_format=EmbeddingFileFormat.JSONL if use_jsonl_format else EmbeddingFileFormat.NPY,
        threshold=0.95,
        use_lsh=True,
        lsh_num_tables=10,
        lsh_hash_size=10,
        content_keys=['prompt'],
        prompt_id_key='prompt_id',
        embedding_id_key='prompt_id',
        embedding_vector_key='embedding'
    )
    
    # 执行去重
    deduplicator = GlobalEmbeddingLSHDeduplicator(config)
    stats = deduplicator.deduplicate()
    
    return stats


def test_exact_dedup(data_file, embeddings_file, use_jsonl_format=False):
    """
    测试精确去重（作为对照）
    
    Args:
        data_file: 数据文件路径
        embeddings_file: embedding文件路径
        use_jsonl_format: 是否使用JSONL格式
    """
    from dep_code.config import DedupConfig, DedupMethod, EmbeddingFileFormat
    from global_embedding_dedup import GlobalEmbeddingDeduplicator
    
    print(f"\n" + "="*70)
    print(f"🧪 测试精确去重（对照组）")
    print(f"="*70)
    
    # 输出文件
    test_dir = Path(data_file).parent
    output_file = test_dir / "test_output_exact.jsonl"
    
    # 配置
    config = DedupConfig(
        method=DedupMethod.EMBEDDING,
        input_file=str(data_file),
        output_file=str(output_file),
        embeddings_file=str(embeddings_file),
        embeddings_format=EmbeddingFileFormat.JSONL if use_jsonl_format else EmbeddingFileFormat.NPY,
        threshold=0.95,
        use_lsh=False,  # 不使用LSH
        content_keys=['prompt'],
        prompt_id_key='prompt_id',
        embedding_id_key='prompt_id',
        embedding_vector_key='embedding',
        batch_size=5000  # 小数据集用较小的batch
    )
    
    # 执行去重
    deduplicator = GlobalEmbeddingDeduplicator(config)
    stats = deduplicator.deduplicate()
    
    return stats


def compare_results(lsh_stats, exact_stats):
    """比较LSH和精确方法的结果"""
    print(f"\n" + "="*70)
    print(f"📊 结果对比")
    print(f"="*70)
    
    print(f"\n去重效果对比:")
    print(f"{'指标':<20} {'LSH模式':>15} {'精确模式':>15} {'差异':>15}")
    print(f"-" * 70)
    
    # 去重数量
    lsh_removed = lsh_stats['duplicate_items']
    exact_removed = exact_stats['duplicate_items']
    diff_removed = lsh_removed - exact_removed
    print(f"{'删除数据':<20} {lsh_removed:>15,} {exact_removed:>15,} {diff_removed:>+15,}")
    
    # 保留数量
    lsh_kept = lsh_stats['unique_items']
    exact_kept = exact_stats['unique_items']
    diff_kept = lsh_kept - exact_kept
    print(f"{'保留数据':<20} {lsh_kept:>15,} {exact_kept:>15,} {diff_kept:>+15,}")
    
    # 去重率
    lsh_ratio = lsh_stats['dedup_ratio']
    exact_ratio = exact_stats['dedup_ratio']
    diff_ratio = lsh_ratio - exact_ratio
    print(f"{'去重率':<20} {lsh_ratio:>14.2%} {exact_ratio:>14.2%} {diff_ratio:>+14.2%}")
    
    print(f"\n性能对比:")
    print(f"{'指标':<20} {'LSH模式':>15} {'精确模式':>15} {'加速比':>15}")
    print(f"-" * 70)
    
    # 总时间
    lsh_time = lsh_stats['total_time']
    exact_time = exact_stats['total_time']
    time_speedup = exact_time / lsh_time if lsh_time > 0 else 0
    print(f"{'总时间(秒)':<20} {lsh_time:>15.2f} {exact_time:>15.2f} {time_speedup:>14.2f}x")
    
    # 去重时间
    lsh_dedup = lsh_stats.get('dedup_time', 0)
    exact_dedup = exact_stats.get('dedup_time', 0)
    dedup_speedup = exact_dedup / lsh_dedup if lsh_dedup > 0 else 0
    print(f"{'去重时间(秒)':<20} {lsh_dedup:>15.2f} {exact_dedup:>15.2f} {dedup_speedup:>14.2f}x")
    
    # 处理速度
    lsh_speed = lsh_stats['processing_speed']
    exact_speed = exact_stats['processing_speed']
    print(f"{'处理速度(条/秒)':<20} {lsh_speed:>15,.0f} {exact_speed:>15,.0f} {'-':>15}")
    
    # LSH特有指标
    if 'speedup' in lsh_stats:
        print(f"\nLSH优化效果:")
        print(f"  🔄 比较次数减少: {lsh_stats['speedup']:.1f}x")
        print(f"  📊 哈希表数量: {lsh_stats.get('lsh_num_tables', 'N/A')}")
        print(f"  📏 哈希位数: {lsh_stats.get('lsh_hash_size', 'N/A')}")
    
    # 准确性评估
    accuracy = (lsh_removed / exact_removed * 100) if exact_removed > 0 else 0
    print(f"\n✅ 准确性评估:")
    print(f"  LSH召回率: {accuracy:.2f}%")
    if accuracy > 95:
        print(f"  ✨ 优秀！LSH在保持高准确度的同时大幅提升了速度")
    elif accuracy > 90:
        print(f"  ✅ 良好！LSH提供了较好的速度-准确度平衡")
    else:
        print(f"  ⚠️  建议增加lsh_num_tables或调整lsh_hash_size以提高准确度")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试LSH去重器")
    parser.add_argument("--num_samples", type=int, default=1000, help="样本数量")
    parser.add_argument("--num_duplicates", type=int, default=100, help="重复样本数量")
    parser.add_argument("--embedding_dim", type=int, default=128, help="embedding维度")
    parser.add_argument("--format", choices=["npy", "jsonl"], default="jsonl", help="embedding格式")
    parser.add_argument("--skip_exact", action="store_true", help="跳过精确模式测试")
    
    args = parser.parse_args()
    
    # 创建测试数据
    data_file, npy_file, jsonl_file = create_test_data(
        num_samples=args.num_samples,
        num_duplicates=args.num_duplicates,
        embedding_dim=args.embedding_dim
    )
    
    # 选择embedding文件格式
    embeddings_file = jsonl_file if args.format == "jsonl" else npy_file
    use_jsonl = args.format == "jsonl"
    
    # 测试LSH去重
    lsh_stats = test_lsh_dedup(data_file, embeddings_file, use_jsonl)
    
    # 测试精确去重（对照）
    if not args.skip_exact:
        exact_stats = test_exact_dedup(data_file, embeddings_file, use_jsonl)
        
        # 比较结果
        compare_results(lsh_stats, exact_stats)
    else:
        print(f"\n✅ LSH测试完成，跳过精确模式对照")
    
    print(f"\n" + "="*70)
    print(f"🎉 测试完成！")
    print(f"="*70)
    print(f"📁 测试文件位于: {Path(data_file).parent}")

