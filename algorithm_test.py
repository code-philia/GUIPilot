import cv2
import json
import time
import numpy as np
from datetime import datetime
from guipilot.entities import Bbox, Widget, WidgetType, Screen
from guipilot.checker import GVT as Checker
from guipilot.matcher import GUIPilotV2 as Matcher

# 导入优化后的算法
from guipilot.matcher.advanced import AdvancedMatcher, SimilarityConfig
from guipilot.matcher.hierarchical import HierarchicalMatcher, LayerConfig
from guipilot.matcher.optimized import OptimizedMatcher, CacheConfig
from guipilot.privacy.masking import PrivacyManager, MaskingConfig, SensitivityLevel


def run_comparison_analysis(screenA_path, screenB_path):
    """运行完整的算法对比分析"""
    
    print("=" * 80)
    print("GUIPilot 算法优化对比分析")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 加载图像
    print("\n1. 加载图像...")
    screenA_image = cv2.imread(screenA_path)
    screenB_image = cv2.imread(screenB_path)
    
    screenA = Screen(screenA_image)
    screenB = Screen(screenB_image)
    
    # 检测部件
    print("\n2. 执行部件检测...")
    screenA.detect()
    screenB.detect()
    
    # OCR识别
    print("\n3. 执行OCR识别...")
    screenA.ocr()
    screenB.ocr()
    
    print(f"屏幕A检测到部件数: {len(screenA.widgets)}")
    print(f"屏幕B检测到部件数: {len(screenB.widgets)}")
    
    # 隐私保护分析
    print("\n4. 隐私保护分析...")
    privacy_manager = PrivacyManager()
    
    # 分析屏幕A的隐私风险
    risk_analysis_A = privacy_manager.analyze_privacy_risk(screenA)
    risk_analysis_B = privacy_manager.analyze_privacy_risk(screenB)
    
    print(f"屏幕A隐私风险分数: {risk_analysis_A['risk_score']:.2f}")
    print(f"屏幕B隐私风险分数: {risk_analysis_B['risk_score']:.2f}")
    
    if risk_analysis_A['detected_info']:
        print(f"屏幕A检测到的敏感信息: {risk_analysis_A['detected_info']}")
    if risk_analysis_B['detected_info']:
        print(f"屏幕B检测到的敏感信息: {risk_analysis_B['detected_info']}")
    
    # 初始化一致性检查器
    checker = Checker()
    
    # 测试不同的匹配算法
    results = {}
    
    # 1. 原始算法
    print("\n5. 测试原始匹配算法...")
    start_time = time.time()
    matcher = Matcher()
    pairs_original, scores_original, match_time_original = matcher.match(screenA, screenB)
    original_time = (time.time() - start_time) * 1000
    
    # 对原始算法进行一致性检查
    inconsistencies_original, check_time_original = checker.check(screenA, screenB, pairs_original)
    
    results['original'] = {
        'pairs': pairs_original,
        'scores': scores_original,
        'time': original_time,
        'match_count': len(pairs_original),
        'inconsistencies': len(inconsistencies_original),
        'check_time': check_time_original
    }
    
    # 2. 高级匹配算法 - 第一次运行（冷启动）
    print("\n6. 测试高级匹配算法 - 第一次运行（冷启动）...")
    start_time = time.time()
    advanced_matcher = AdvancedMatcher()
    pairs_advanced_cold, scores_advanced_cold, match_time_advanced_cold = advanced_matcher.match(screenA, screenB)
    advanced_time_cold = (time.time() - start_time) * 1000
    
    # 对高级算法进行一致性检查
    inconsistencies_advanced_cold, check_time_advanced_cold = checker.check(screenA, screenB, pairs_advanced_cold)
    
    # 获取缓存统计
    cache_stats_advanced_cold = {
        'similarity_cache_size': len(advanced_matcher._similarity_cache),
        'matching_cache_size': len(advanced_matcher._matching_cache)
    }
    
    results['advanced_cold'] = {
        'pairs': pairs_advanced_cold,
        'scores': scores_advanced_cold,
        'time': advanced_time_cold,
        'match_count': len(pairs_advanced_cold),
        'inconsistencies': len(inconsistencies_advanced_cold),
        'check_time': check_time_advanced_cold,
        'cache_stats': cache_stats_advanced_cold
    }
    
    # 3. 高级匹配算法 - 第二次运行（热启动，验证缓存）
    print("\n7. 测试高级匹配算法 - 第二次运行（热启动）...")
    start_time = time.time()
    pairs_advanced_hot, scores_advanced_hot, match_time_advanced_hot = advanced_matcher.match(screenA, screenB)
    advanced_time_hot = (time.time() - start_time) * 1000
    
    # 对高级算法进行一致性检查
    inconsistencies_advanced_hot, check_time_advanced_hot = checker.check(screenA, screenB, pairs_advanced_hot)
    
    # 获取缓存统计
    cache_stats_advanced_hot = {
        'similarity_cache_size': len(advanced_matcher._similarity_cache),
        'matching_cache_size': len(advanced_matcher._matching_cache)
    }
    
    results['advanced_hot'] = {
        'pairs': pairs_advanced_hot,
        'scores': scores_advanced_hot,
        'time': advanced_time_hot,
        'match_count': len(pairs_advanced_hot),
        'inconsistencies': len(inconsistencies_advanced_hot),
        'check_time': check_time_advanced_hot,
        'cache_stats': cache_stats_advanced_hot
    }
    
    # 4. 分层匹配算法 - 第一次运行（冷启动）
    print("\n8. 测试分层匹配算法 - 第一次运行（冷启动）...")
    start_time = time.time()
    hierarchical_matcher = HierarchicalMatcher()
    pairs_hierarchical_cold = hierarchical_matcher.hierarchical_match(screenA, screenB)
    hierarchical_time_cold = (time.time() - start_time) * 1000
    
    # 对分层算法进行一致性检查
    inconsistencies_hierarchical_cold, check_time_hierarchical_cold = checker.check(screenA, screenB, pairs_hierarchical_cold)
    
    # 获取缓存统计
    cache_stats_hierarchical_cold = hierarchical_matcher.get_cache_stats()
    
    results['hierarchical_cold'] = {
        'pairs': pairs_hierarchical_cold,
        'scores': [1.0] * len(pairs_hierarchical_cold),  # 简化处理
        'time': hierarchical_time_cold,
            'match_count': len(pairs_hierarchical_cold),
            'inconsistencies': len(inconsistencies_hierarchical_cold),
            'check_time': check_time_hierarchical_cold,
            'cache_stats': cache_stats_hierarchical_cold
        }
    
    # 5. 分层匹配算法 - 第二次运行（热启动，验证缓存）
    print("\n9. 测试分层匹配算法 - 第二次运行（热启动）...")
    start_time = time.time()
    pairs_hierarchical_hot = hierarchical_matcher.hierarchical_match(screenA, screenB)
    hierarchical_time_hot = (time.time() - start_time) * 1000
    
    # 对分层算法进行一致性检查
    inconsistencies_hierarchical_hot, check_time_hierarchical_hot = checker.check(screenA, screenB, pairs_hierarchical_hot)
    
    # 获取缓存统计
    cache_stats_hierarchical_hot = hierarchical_matcher.get_cache_stats()
    
    results['hierarchical_hot'] = {
        'pairs': pairs_hierarchical_hot,
        'scores': [1.0] * len(pairs_hierarchical_hot),  # 简化处理
        'time': hierarchical_time_hot,
        'match_count': len(pairs_hierarchical_hot),
        'inconsistencies': len(inconsistencies_hierarchical_hot),
        'check_time': check_time_hierarchical_hot,
        'cache_stats': cache_stats_hierarchical_hot
    }
    
    # 6. 优化匹配算法 - 第一次运行（冷启动）
    print("\n10. 测试优化匹配算法 - 第一次运行（冷启动）...")
    start_time = time.time()
    optimized_matcher = OptimizedMatcher()
    pairs_optimized_cold = optimized_matcher.optimized_match(screenA, screenB)
    optimized_time_cold = (time.time() - start_time) * 1000
    
    # 获取性能统计
    performance_stats_cold = optimized_matcher.get_performance_stats()
    
    # 对优化算法进行一致性检查
    inconsistencies_optimized_cold, check_time_optimized_cold = checker.check(screenA, screenB, pairs_optimized_cold)
    
    results['optimized_cold'] = {
        'pairs': pairs_optimized_cold,
        'scores': [1.0] * len(pairs_optimized_cold),  # 简化处理
        'time': optimized_time_cold,
        'match_count': len(pairs_optimized_cold),
        'inconsistencies': len(inconsistencies_optimized_cold),
        'check_time': check_time_optimized_cold,
        'performance_stats': performance_stats_cold
    }
    
    # 7. 优化匹配算法 - 第二次运行（热启动，验证缓存）
    print("\n11. 测试优化匹配算法 - 第二次运行（热启动）...")
    start_time = time.time()
    pairs_optimized_hot = optimized_matcher.optimized_match(screenA, screenB)
    optimized_time_hot = (time.time() - start_time) * 1000
    
    # 获取性能统计
    performance_stats_hot = optimized_matcher.get_performance_stats()
    
    # 对优化算法进行一致性检查
    inconsistencies_optimized_hot, check_time_optimized_hot = checker.check(screenA, screenB, pairs_optimized_hot)
    
    results['optimized_hot'] = {
        'pairs': pairs_optimized_hot,
        'scores': [1.0] * len(pairs_optimized_hot),  # 简化处理
        'time': optimized_time_hot,
        'match_count': len(pairs_optimized_hot),
        'inconsistencies': len(inconsistencies_optimized_hot),
        'check_time': check_time_optimized_hot,
        'performance_stats': performance_stats_hot
    }
    
    # 输出对比结果
    print("\n" + "=" * 80)
    print("算法性能对比结果")
    print("=" * 80)
    
    print(f"\n{'算法名称':<20} {'匹配数量':<10} {'执行时间(ms)':<15} {'性能提升':<12} {'不一致数量':<12} {'检查时间(ms)':<15}")
    print("-" * 90)
    
    baseline_time = results['original']['time']
    
    for algo_name, result in results.items():
        match_count = result['match_count']
        exec_time = result['time']
        inconsistencies = result['inconsistencies']
        check_time = result['check_time']
        
        # 计算性能提升（相对于原始算法）
        if algo_name == 'original':
            improvement = "基准"
        else:
            improvement_percent = ((baseline_time - exec_time) / baseline_time * 100)
            improvement = f"{improvement_percent:.1f}%"
        
        # 显示算法名称（冷/热启动）
        display_name = algo_name
        if algo_name == 'optimized_cold':
            display_name = 'optimized (冷启动)'
        elif algo_name == 'optimized_hot':
            display_name = 'optimized (热启动)'
        
        print(f"{display_name:<20} {match_count:<10} {exec_time:<15.2f} {improvement:<12} {inconsistencies:<12} {check_time:<15.2f}")
    
    # 输出所有算法的缓存统计
    print("\n缓存使用情况统计:")
    
    # 高级匹配算法缓存统计
    if 'advanced_cold' in results and 'cache_stats' in results['advanced_cold']:
        cache_advanced_cold = results['advanced_cold']['cache_stats']
        cache_advanced_hot = results['advanced_hot']['cache_stats']
        
        print(f"高级匹配算法:")
        print(f"  冷启动 - 相似度缓存: {cache_advanced_cold['similarity_cache_size']}, 匹配缓存: {cache_advanced_cold['matching_cache_size']}")
        print(f"  热启动 - 相似度缓存: {cache_advanced_hot['similarity_cache_size']}, 匹配缓存: {cache_advanced_hot['matching_cache_size']}")
        
        # 计算缓存效果
        advanced_cache_improvement = ((results['advanced_cold']['time'] - results['advanced_hot']['time']) / results['advanced_cold']['time'] * 100)
        print(f"  缓存性能提升: {advanced_cache_improvement:.1f}%")
    
    # 分层匹配算法缓存统计
    if 'hierarchical_cold' in results and 'cache_stats' in results['hierarchical_cold']:
        cache_hierarchical_cold = results['hierarchical_cold']['cache_stats']
        cache_hierarchical_hot = results['hierarchical_hot']['cache_stats']
        
        print(f"分层匹配算法:")
        print(f"  冷启动 - 特征缓存: {cache_hierarchical_cold['feature_cache_size']}, 匹配缓存: {cache_hierarchical_cold['matching_cache_size']}")
        print(f"  热启动 - 特征缓存: {cache_hierarchical_hot['feature_cache_size']}, 匹配缓存: {cache_hierarchical_hot['matching_cache_size']}")
        
        # 计算缓存效果
        hierarchical_cache_improvement = ((results['hierarchical_cold']['time'] - results['hierarchical_hot']['time']) / results['hierarchical_cold']['time'] * 100)
        print(f"  缓存性能提升: {hierarchical_cache_improvement:.1f}%")
    
    # 优化算法性能统计
    print("\n优化算法性能统计:")
    stats_cold = results['optimized_cold']['performance_stats']
    stats_hot = results['optimized_hot']['performance_stats']
    
    cache_hit_rate_cold = stats_cold['cache_hits'] / (stats_cold['cache_hits'] + stats_cold['cache_misses']) * 100 if (stats_cold['cache_hits'] + stats_cold['cache_misses']) > 0 else 0
    cache_hit_rate_hot = stats_hot['cache_hits'] / (stats_hot['cache_hits'] + stats_hot['cache_misses']) * 100 if (stats_hot['cache_hits'] + stats_hot['cache_misses']) > 0 else 0
    
    print(f"  冷启动:")
    print(f"    - 缓存命中率: {cache_hit_rate_cold:.1f}%")
    print(f"    - 向量化操作比例: {stats_cold['vectorized_operations'] / stats_cold['total_operations'] * 100:.1f}%")
    
    print(f"  热启动:")
    print(f"    - 缓存命中率: {cache_hit_rate_hot:.1f}%")
    print(f"    - 向量化操作比例: {stats_hot['vectorized_operations'] / stats_hot['total_operations'] * 100:.1f}%")
    
    # 计算缓存效果
    cache_improvement = ((results['optimized_cold']['time'] - results['optimized_hot']['time']) / results['optimized_cold']['time'] * 100)
    
    print(f"\n缓存效果分析:")
    print(f"  冷启动执行时间: {results['optimized_cold']['time']:.2f}ms")
    print(f"  热启动执行时间: {results['optimized_hot']['time']:.2f}ms")
    print(f"  缓存带来的性能提升: {cache_improvement:.1f}%")
    
    # 性能优化评估
    print("\n性能优化评估:")
    
    
    # 准确性评估
    min_inconsistencies = min([result['inconsistencies'] for result in results.values()])
    best_algorithms = [algo for algo, result in results.items() if result['inconsistencies'] == min_inconsistencies]
    
    print(f"\n准确性评估:")
    print(f"  最低不一致数量: {min_inconsistencies}")
    print(f"  最佳算法: {', '.join(best_algorithms)}")
    
    for algo_name, result in results.items():
        if result['inconsistencies'] == min_inconsistencies:
            print(f"    ✅ {algo_name}: {result['inconsistencies']} 个不一致")
        else:
            print(f"    ⚠️  {algo_name}: {result['inconsistencies']} 个不一致")
    
    # 保存详细结果到JSON文件
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'screenA_path': screenA_path,
        'screenB_path': screenB_path,
        'screenA_widgets': len(screenA.widgets),
        'screenB_widgets': len(screenB.widgets),
        'privacy_analysis': {
            'screenA': risk_analysis_A,
            'screenB': risk_analysis_B
        },
        'algorithm_results': {},
        'performance_summary': {
            'fastest_algorithm': min(results.items(), key=lambda x: x[1]['time'])[0],
            'most_accurate_algorithm': min(results.items(), key=lambda x: x[1]['inconsistencies'])[0],
            'cache_improvement_percent': cache_improvement
        }
    }
    
    for algo_name, result in results.items():
        output_data['algorithm_results'][algo_name] = {
            'match_count': result['match_count'],
            'execution_time': result['time'],
            'inconsistencies': result['inconsistencies'],
            'check_time': result['check_time'],
            'average_score': np.mean(result['scores']) if result['scores'] else 0
        }
        
        # 包含性能统计（如果存在）
        if 'performance_stats' in result:
            output_data['algorithm_results'][algo_name]['performance_stats'] = result['performance_stats']
    
    # 保存到文件
    output_file = f"comparison_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {output_file}")
    
    return results


def demo_privacy_masking():
    """演示隐私保护功能"""
    print("\n" + "=" * 80)
    print("隐私保护功能演示")
    print("=" * 80)
    
    # 创建测试数据
    test_widgets = [
        {"texts": ["用户邮箱: example@email.com"], "type": "textview"},
        {"texts": ["联系电话: 138-1234-5678"], "type": "textview"},
        {"texts": ["密码: mysecretpassword"], "type": "inputbox"},
    ]
    
    privacy_manager = PrivacyManager()
    
    print("\n原始文本:")
    for widget in test_widgets:
        for text in widget['texts']:
            print(f"  - {text}")
    
    print("\n检测敏感信息:")
    for widget in test_widgets:
        for text in widget['texts']:
            detected = privacy_manager.masker.detector.detect_sensitive_info(text)
            if detected:
                for info_type, matched_text, confidence in detected:
                    print(f"  - 检测到 {info_type.value}: {matched_text} (置信度: {confidence:.2f})")
            else:
                print(f"  - 未检测到敏感信息: {text}")
    
    print("\n应用隐私保护后:")
    for widget in test_widgets:
        for text in widget['texts']:
            masked_text = privacy_manager.masker._mask_text(text)
            print(f"  - {masked_text}")


if __name__ == "__main__":
    # 设置测试文件路径
    screenA_path = "datasets/new/Adobe/1.jpg"
    screenB_path = "datasets/new/Bitget1/1.jpg"
    
    try:
        # 运行完整的对比分析
        results = run_comparison_analysis(screenA_path, screenB_path)
        
        # 演示隐私保护功能
        demo_privacy_masking()
        
        print("\n" + "=" * 80)
        print("算法优化完成!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        print("请检查文件路径是否正确，或尝试使用其他测试图像。")