from __future__ import annotations
import typing
import numpy as np
from dataclasses import dataclass
from functools import lru_cache
import time
from collections import defaultdict

from .advanced import AdvancedMatcher, SimilarityConfig
from .hierarchical import HierarchicalMatcher, LayerConfig

if typing.TYPE_CHECKING:
    from guipilot.entities import Screen, Widget


@dataclass
class CacheConfig:
    """缓存配置"""
    max_size: int = 1000
    ttl_seconds: int = 300  # 缓存存活时间（秒）
    enabled: bool = True


class LRUCache:
    """LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        
    def get(self, key):
        if key not in self.cache:
            return None
        
        # 检查是否过期
        if time.time() - self.access_times[key] > self.ttl:
            del self.cache[key]
            del self.access_times[key]
            return None
        
        # 更新访问时间
        self.access_times[key] = time.time()
        return self.cache[key]
    
    def put(self, key, value):
        # 如果达到最大容量，移除最久未使用的
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()


class OptimizedMatcher:
    """优化后的匹配器，包含向量化计算和缓存机制"""
    
    def __init__(self, 
                 similarity_config: SimilarityConfig = None,
                 layer_config: LayerConfig = None,
                 cache_config: CacheConfig = None):
        
        self.similarity_config = similarity_config or SimilarityConfig()
        self.layer_config = layer_config or LayerConfig()
        self.cache_config = cache_config or CacheConfig()
        
        # 初始化子匹配器
        self.advanced_matcher = AdvancedMatcher(self.similarity_config)
        self.hierarchical_matcher = HierarchicalMatcher(self.layer_config, self.similarity_config)
        
        # 初始化缓存
        self._similarity_cache = LRUCache(self.cache_config.max_size, self.cache_config.ttl_seconds)
        self._feature_cache = LRUCache(self.cache_config.max_size, self.cache_config.ttl_seconds)
        
        # 性能统计
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_operations': 0,
            'vectorized_operations': 0
        }
    
    def vectorized_similarity_matrix(self, screen1: Screen, screen2: Screen) -> np.ndarray:
        """向量化计算相似度矩阵（优化缓存）"""
        self._stats['total_operations'] += 1
        self._stats['vectorized_operations'] += 1
        
        # 优化的缓存检查
        cache_key = self._generate_cache_key(screen1, screen2)
        if self.cache_config.enabled and cache_key:
            cached_result = self._similarity_cache.get(cache_key)
            if cached_result is not None:
                self._stats['cache_hits'] += 1
                return cached_result
            else:
                self._stats['cache_misses'] += 1
        else:
            self._stats['cache_misses'] += 1
        
        widgets1 = list(screen1.widgets.values())
        widgets2 = list(screen2.widgets.values())
        
        if not widgets1 or not widgets2:
            return np.zeros((0, 0))
        
        # 向量化计算位置特征
        positions1 = self._vectorized_positions(widgets1, screen1)
        positions2 = self._vectorized_positions(widgets2, screen2)
        
        # 向量化计算形状特征
        shapes1 = self._vectorized_shapes(widgets1)
        shapes2 = self._vectorized_shapes(widgets2)
        
        # 计算相似度矩阵
        similarity_matrix = self._compute_similarity_matrix(positions1, positions2, shapes1, shapes2, widgets1, widgets2)
        
        if self.cache_config.enabled:
            self._similarity_cache.put(cache_key, similarity_matrix)
        
        return similarity_matrix
    
    def _vectorized_positions(self, widgets: list[Widget], screen: Screen) -> np.ndarray:
        """向量化计算位置特征"""
        positions = np.zeros((len(widgets), 2))
        
        screen_h, screen_w, _ = screen.image.shape
        
        for i, widget in enumerate(widgets):
            # 归一化位置
            x = widget.bbox.xmin / screen_w
            y = widget.bbox.ymin / screen_h
            positions[i] = [x, y]
        
        return positions
    
    def _vectorized_shapes(self, widgets: list[Widget]) -> np.ndarray:
        """向量化计算形状特征"""
        shapes = np.zeros((len(widgets), 2))
        
        for i, widget in enumerate(widgets):
            # 宽高比和面积（归一化）
            aspect_ratio = widget.width / widget.height
            normalized_area = widget.area / (widget.width * widget.height)  # 相对面积
            shapes[i] = [aspect_ratio, normalized_area]
        
        return shapes
    
    def _compute_similarity_matrix(self, positions1: np.ndarray, positions2: np.ndarray,
                                  shapes1: np.ndarray, shapes2: np.ndarray,
                                  widgets1: list[Widget], widgets2: list[Widget]) -> np.ndarray:
        """计算相似度矩阵"""
        n1, n2 = len(widgets1), len(widgets2)
        similarity_matrix = np.zeros((n1, n2))
        
        # 向量化计算位置相似度
        if n1 > 0 and n2 > 0:
            # 扩展维度以进行广播计算
            pos1_expanded = positions1[:, np.newaxis, :]  # (n1, 1, 2)
            pos2_expanded = positions2[np.newaxis, :, :]  # (1, n2, 2)
            
            # 计算欧几里得距离
            pos_distances = np.sqrt(np.sum((pos1_expanded - pos2_expanded) ** 2, axis=2))
            pos_similarities = 1.0 / (1.0 + pos_distances)
            
            # 向量化计算形状相似度
            shape1_expanded = shapes1[:, np.newaxis, :]  # (n1, 1, 2)
            shape2_expanded = shapes2[np.newaxis, :, :]  # (1, n2, 2)
            
            # 计算形状距离
            shape_distances = np.sqrt(np.sum((shape1_expanded - shape2_expanded) ** 2, axis=2))
            shape_similarities = 1.0 / (1.0 + shape_distances)
            
            # 组合相似度
            similarity_matrix = (pos_similarities + shape_similarities) / 2
            
            # 添加类型相似度
            for i in range(n1):
                for j in range(n2):
                    type_similarity = self._type_similarity(widgets1[i], widgets2[j])
                    similarity_matrix[i, j] = (similarity_matrix[i, j] + type_similarity) / 2
        
        return similarity_matrix
    
    def _type_similarity(self, widget1: Widget, widget2: Widget) -> float:
        """计算类型相似度"""
        if widget1.type == widget2.type:
            return 1.0
        
        # 类型兼容性矩阵（严格版本，与advanced.py保持一致）
        type_compatibility = {
            ('textbutton', 'combinedbutton'): 0.9,  # 提高相似度阈值
            ('iconbutton', 'combinedbutton'): 0.9,
            ('textview', 'inputbox'): 0.8,  # 提高相似度阈值
            # 移除textbutton和textview的兼容性
        }
        
        key = (widget1.type.value, widget2.type.value)
        reverse_key = (widget2.type.value, widget1.type.value)
        
        if key in type_compatibility:
            return type_compatibility[key]
        elif reverse_key in type_compatibility:
            return type_compatibility[reverse_key]
        else:
            return 0.1
    
    def optimized_match(self, screen1: Screen, screen2: Screen) -> list[tuple[int, int]]:
        """优化后的匹配函数"""
        # 使用向量化计算相似度矩阵
        similarity_matrix = self.vectorized_similarity_matrix(screen1, screen2)
        
        if similarity_matrix.size == 0:
            return []
        
        # 使用匈牙利算法进行最优匹配
        from scipy.optimize import linear_sum_assignment
        
        # 由于匈牙利算法最小化成本，我们需要将相似度转换为成本
        cost_matrix = 1.0 - similarity_matrix
        
        # 应用匈牙利算法
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # 构建匹配对
        pairs = []
        widget_keys1 = list(screen1.widgets.keys())
        widget_keys2 = list(screen2.widgets.keys())
        
        for i, j in zip(row_indices, col_indices):
            # 提高阈值并添加类型检查（提高准确性）
            if (similarity_matrix[i, j] > 0.8 and 
                self._is_valid_match(widget_keys1[i], widget_keys2[j], screen1, screen2)):
                pairs.append((widget_keys1[i], widget_keys2[j]))
        
        return pairs
    
    def _is_type_compatible(self, widget1: Widget, widget2: Widget) -> bool:
        """检查类型是否兼容"""
        if widget1.type == widget2.type:
            return True
        
        # 定义兼容类型对（严格版本，与advanced.py保持一致）
        compatible_types = {
            ('textbutton', 'combinedbutton'),
            ('iconbutton', 'combinedbutton'),
            ('textview', 'inputbox'),
            # 移除textbutton和textview的兼容性
        }
        
        key = (widget1.type.value, widget2.type.value)
        reverse_key = (widget2.type.value, widget1.type.value)
        
        return key in compatible_types or reverse_key in compatible_types

    def _is_valid_match(self, idx1: int, idx2: int, screen1: Screen, screen2: Screen) -> bool:
        """验证匹配是否有效"""
        widget1 = screen1.widgets[idx1]
        widget2 = screen2.widgets[idx2]
        
        # 类型必须兼容
        if not self._is_type_compatible(widget1, widget2):
            return False
        
        return True

    def get_performance_stats(self) -> dict:
        """获取性能统计"""
        return self._stats.copy()
    
    def _generate_cache_key(self, screen1: Screen, screen2: Screen) -> str:
        """生成优化的缓存键，提高缓存命中率"""
        try:
            # 使用更稳定的特征作为缓存键
            key_parts = []
            
            # 屏幕基本特征
            if hasattr(screen1, 'image') and screen1.image is not None:
                h1, w1, _ = screen1.image.shape
                key_parts.append(f"s1_{w1}_{h1}")
            
            if hasattr(screen2, 'image') and screen2.image is not None:
                h2, w2, _ = screen2.image.shape
                key_parts.append(f"s2_{w2}_{h2}")
            
            # 部件数量特征（使用哈希值提高稳定性）
            if hasattr(screen1, 'widgets') and screen1.widgets is not None:
                widget_count1 = len(screen1.widgets)
                key_parts.append(f"wc1_{widget_count1}")
            
            if hasattr(screen2, 'widgets') and screen2.widgets is not None:
                widget_count2 = len(screen2.widgets)
                key_parts.append(f"wc2_{widget_count2}")
            
            # 部件类型分布特征（简化的哈希）
            if hasattr(screen1, 'widgets') and screen1.widgets:
                type_signature1 = self._get_type_signature(screen1.widgets)
                key_parts.append(f"ts1_{type_signature1}")
            
            if hasattr(screen2, 'widgets') and screen2.widgets:
                type_signature2 = self._get_type_signature(screen2.widgets)
                key_parts.append(f"ts2_{type_signature2}")
            
            return ':'.join(key_parts) if key_parts else None
            
        except Exception:
            # 如果生成缓存键失败，返回None（不使用缓存）
            return None
    
    def _get_type_signature(self, widgets) -> str:
        """生成类型分布的签名"""
        type_counts = {}
        for widget in widgets.values():
            widget_type = str(widget.type) if hasattr(widget, 'type') else 'unknown'
            type_counts[widget_type] = type_counts.get(widget_type, 0) + 1
        
        # 按类型字母顺序排序，确保稳定性
        sorted_types = sorted(type_counts.items(), key=lambda x: x[0])
        
        # 生成简化的签名（避免过长的缓存键）
        signature = '_'.join([f"{k[:3]}_{v}" for k, v in sorted_types[:5]])  # 最多5种类型
        
        return signature
    
    def clear_cache(self):
        """清空缓存"""
        self._similarity_cache.clear()
        self._feature_cache.clear()
        self.advanced_matcher.clear_cache()
        self.hierarchical_matcher.clear_cache()
        
        # 重置统计
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_operations': 0,
            'vectorized_operations': 0
        }


class MultiLevelCache:
    """多级缓存系统"""
    
    def __init__(self, levels: int = 3):
        self.levels = levels
        self.caches = [LRUCache(100 * (i + 1), 300) for i in range(levels)]
        
    def get(self, key):
        # 从最高级开始查找
        for i in range(self.levels - 1, -1, -1):
            value = self.caches[i].get(key)
            if value is not None:
                # 提升到更高级缓存
                if i < self.levels - 1:
                    self.caches[i + 1].put(key, value)
                return value
        return None
    
    def put(self, key, value):
        # 放入所有级别的缓存
        for cache in self.caches:
            cache.put(key, value)
    
    def clear(self):
        for cache in self.caches:
            cache.clear()


class BatchProcessor:
    """批处理处理器"""
    
    def __init__(self, matcher: OptimizedMatcher):
        self.matcher = matcher
        
    def batch_match(self, screen_pairs: list[tuple[Screen, Screen]]) -> list[list[tuple[int, int]]]:
        """批量匹配"""
        results = []
        
        for screen1, screen2 in screen_pairs:
            # 使用优化后的匹配器
            pairs = self.matcher.optimized_match(screen1, screen2)
            results.append(pairs)
        
        return results
    
    def _generate_cache_key(self, screen1: Screen, screen2: Screen) -> str:
        try:
            # 使用更稳定的特征作为缓存键
            key_parts = []
            
            # 屏幕基本特征
            if hasattr(screen1, 'image') and screen1.image is not None:
                h1, w1, _ = screen1.image.shape
                key_parts.append(f"s1_{w1}_{h1}")
            
            if hasattr(screen2, 'image') and screen2.image is not None:
                h2, w2, _ = screen2.image.shape
                key_parts.append(f"s2_{w2}_{h2}")
            
            # 部件数量特征（使用哈希值提高稳定性）
            if hasattr(screen1, 'widgets') and screen1.widgets is not None:
                widget_count1 = len(screen1.widgets)
                key_parts.append(f"wc1_{widget_count1}")
            
            if hasattr(screen2, 'widgets') and screen2.widgets is not None:
                widget_count2 = len(screen2.widgets)
                key_parts.append(f"wc2_{widget_count2}")
            
            # 部件类型分布特征（简化的哈希）
            if hasattr(screen1, 'widgets') and screen1.widgets:
                type_signature1 = self._get_type_signature(screen1.widgets)
                key_parts.append(f"ts1_{type_signature1}")
            
            if hasattr(screen2, 'widgets') and screen2.widgets:
                type_signature2 = self._get_type_signature(screen2.widgets)
                key_parts.append(f"ts2_{type_signature2}")
            
            return ':'.join(key_parts) if key_parts else None
            
        except Exception:
            # 如果生成缓存键失败，返回None（不使用缓存）
            return None
    
    def _get_type_signature(self, widgets) -> str:
        """生成类型分布的签名"""
        type_counts = {}
        for widget in widgets.values():
            widget_type = str(widget.type) if hasattr(widget, 'type') else 'unknown'
            type_counts[widget_type] = type_counts.get(widget_type, 0) + 1
        
        # 按类型字母顺序排序，确保稳定性
        sorted_types = sorted(type_counts.items(), key=lambda x: x[0])
        
        # 生成简化的签名（避免过长的缓存键）
        signature = '_'.join([f"{k[:3]}_{v}" for k, v in sorted_types[:5]])  # 最多5种类型
        
        return signature
    
    def parallel_match(self, screen_pairs: list[tuple[Screen, Screen]], 
                      num_threads: int = 4) -> list[list[tuple[int, int]]]:
        """并行匹配（伪代码，实际实现需要根据具体环境调整）"""
        import concurrent.futures
        
        def match_pair(pair):
            screen1, screen2 = pair
            return self.matcher.optimized_match(screen1, screen2)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(match_pair, screen_pairs))
        
        return results
    
    def _generate_cache_key(self, screen1: Screen, screen2: Screen) -> str:
        """生成优化的缓存键，提高缓存命中率"""
        try:
            # 使用更稳定的特征作为缓存键
            key_parts = []
            
            # 屏幕基本特征
            if hasattr(screen1, 'image') and screen1.image is not None:
                h1, w1, _ = screen1.image.shape
                key_parts.append(f"s1_{w1}_{h1}")
            
            if hasattr(screen2, 'image') and screen2.image is not None:
                h2, w2, _ = screen2.image.shape
                key_parts.append(f"s2_{w2}_{h2}")
            
            # 部件数量特征（使用哈希值提高稳定性）
            if hasattr(screen1, 'widgets') and screen1.widgets is not None:
                widget_count1 = len(screen1.widgets)
                key_parts.append(f"wc1_{widget_count1}")
            
            if hasattr(screen2, 'widgets') and screen2.widgets is not None:
                widget_count2 = len(screen2.widgets)
                key_parts.append(f"wc2_{widget_count2}")
            
            # 部件类型分布特征（简化的哈希）
            if hasattr(screen1, 'widgets') and screen1.widgets:
                type_signature1 = self._get_type_signature(screen1.widgets)
                key_parts.append(f"ts1_{type_signature1}")
            
            if hasattr(screen2, 'widgets') and screen2.widgets:
                type_signature2 = self._get_type_signature(screen2.widgets)
                key_parts.append(f"ts2_{type_signature2}")
            
            return ':'.join(key_parts) if key_parts else None
            
        except Exception:
            # 如果生成缓存键失败，返回None（不使用缓存）
            return None
    
    def _get_type_signature(self, widgets) -> str:
        """生成类型分布的签名"""
        type_counts = {}
        for widget in widgets.values():
            widget_type = str(widget.type) if hasattr(widget, 'type') else 'unknown'
            type_counts[widget_type] = type_counts.get(widget_type, 0) + 1
        
        # 按类型字母顺序排序，确保稳定性
        sorted_types = sorted(type_counts.items(), key=lambda x: x[0])
        
        # 生成简化的签名（避免过长的缓存键）
        signature = '_'.join([f"{k[:3]}_{v}" for k, v in sorted_types[:5]])  # 最多5种类型
        
        return signature