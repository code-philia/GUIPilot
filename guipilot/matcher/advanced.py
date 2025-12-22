from __future__ import annotations
import typing
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import re
from difflib import SequenceMatcher

if typing.TYPE_CHECKING:
    from guipilot.entities import Screen, Widget


class SimilarityType(Enum):
    POSITION = "position"
    SHAPE = "shape"
    TYPE = "type"
    SEMANTIC = "semantic"
    VISUAL = "visual"


@dataclass
class SimilarityConfig:
    """相似度计算配置"""
    position_weight: float = 0.3
    shape_weight: float = 0.25
    type_weight: float = 0.2
    semantic_weight: float = 0.15
    visual_weight: float = 0.1
    
    # 容错阈值
    fuzzy_threshold: float = 0.75
    strict_threshold: float = 0.9
    
    # 文本相似度配置
    text_similarity_threshold: float = 0.8
    
    def normalize_weights(self):
        """归一化权重"""
        total = self.position_weight + self.shape_weight + self.type_weight + self.semantic_weight + self.visual_weight
        if total != 1.0:
            self.position_weight /= total
            self.shape_weight /= total
            self.type_weight /= total
            self.semantic_weight /= total
            self.visual_weight /= total


class AdvancedMatcher:
    """高级匹配器，支持多维度相似度计算和容错机制"""
    
    def __init__(self, config: SimilarityConfig = None):
        self.config = config or SimilarityConfig()
        self.config.normalize_weights()
        
        # 缓存机制
        self._similarity_cache = {}
        self._matching_cache = {}  # 新增匹配缓存
        
    def calculate_similarity(self, widget1: Widget, widget2: Widget, screen1: Screen, screen2: Screen) -> float:
        """计算多维度相似度（性能优化版本）"""
        # 改进缓存键：基于部件特征而不是内存地址
        cache_key = self._generate_cache_key(widget1, widget2)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        # 快速过滤：如果类型完全不兼容，直接返回低相似度
        
        if not self._is_type_compatible(widget1, widget2):
            self._similarity_cache[cache_key] = 0.05
            return 0.05  # 更低的不兼容分数
    
        # 位置相似度（快速计算）
        position_sim = self._position_similarity(widget1, widget2, screen1, screen2)
        
        # 更严格的提前终止条件
        if position_sim < 0.3:  # 提高阈值
            return position_sim * 0.5
        
        # 形状相似度
        shape_sim = self._shape_similarity(widget1, widget2)
        
        # 类型相似度
        type_sim = self._type_similarity(widget1, widget2)
        
        # 只在高度相似时才计算语义相似度
        semantic_sim = 0.0
        if position_sim > 0.6 and shape_sim > 0.6 and type_sim > 0.7:
            semantic_sim = self._semantic_similarity(widget1, widget2)
        
        # 更严格的权重策略（提高准确性）
        if position_sim > 0.8:  # 提高位置相似度阈值
            total_similarity = (
                0.6 * position_sim +    # 进一步提高位置权重
                0.25 * shape_sim +
                0.1 * type_sim +
                0.05 * semantic_sim     # 进一步降低语义权重
            )
        else:
            # 基础权重保持不变，但增加类型权重的严格性
            total_similarity = (
                0.35 * position_sim +
                0.3 * shape_sim +
                0.3 * type_sim +        # 提高类型权重
                0.05 * semantic_sim
            )
        
        self._similarity_cache[cache_key] = total_similarity
        return total_similarity
    
    def _position_similarity(self, widget1: Widget, widget2: Widget, screen1: Screen, screen2: Screen) -> float:
        s1_h, s1_w, _ = screen1.image.shape
        s2_h, s2_w, _ = screen2.image.shape
        
        x1, y1 = widget1.bbox.xmin / s1_w, widget1.bbox.ymin / s1_h
        x2, y2 = widget2.bbox.xmin / s2_w, widget2.bbox.ymin / s2_h
    

        distance = abs(x1 - x2) + abs(y1 - y2)

        similarity = 1.0 / (1.0 + distance)
        
        if similarity > 0.5:

            region1 = "左上" if x1 < 0.5 and y1 < 0.5 else "右上" if x1 >= 0.5 and y1 < 0.5 else "左下" if x1 < 0.5 and y1 >= 0.5 else "右下"
            region2 = "左上" if x2 < 0.5 and y2 < 0.5 else "右上" if x2 >= 0.5 and y2 < 0.5 else "左下" if x2 < 0.5 and y2 >= 0.5 else "右下"
            
            if region1 == region2:
                similarity = min(similarity * 1.2, 1.0)
        
        return similarity
    
    def _shape_similarity(self, widget1: Widget, widget2: Widget) -> float:
        """形状相似度计算"""
        # 面积比
        area_ratio = min(widget1.area, widget2.area) / max(widget1.area, widget2.area)
        
        # 宽高比相似度
        aspect1 = widget1.width / widget1.height
        aspect2 = widget2.width / widget2.height
        aspect_ratio = min(aspect1, aspect2) / max(aspect1, aspect2)
        
        # 综合形状相似度
        shape_similarity = (area_ratio + aspect_ratio) / 2
        return shape_similarity
    
    def _type_similarity(self, widget1: Widget, widget2: Widget) -> float:
        """类型相似度计算（改进版本，提高准确性）"""
        if widget1.type == widget2.type:
            return 1.0
        
        # 类型兼容性矩阵（严格版本，减少误匹配）
        type_compatibility = {
            ('textbutton', 'combinedbutton'): 0.9,  # 提高相似度阈值
            ('iconbutton', 'combinedbutton'): 0.9,
            ('textview', 'inputbox'): 0.8,  # 提高相似度阈值
            # 移除textbutton和textview的兼容性，强制精确匹配
        }
        
        key = (widget1.type.value, widget2.type.value)
        reverse_key = (widget2.type.value, widget1.type.value)
        
        if key in type_compatibility:
            return type_compatibility[key]
        elif reverse_key in type_compatibility:
            return type_compatibility[reverse_key]
        else:
            return 0.05  # 降低最低相似度，提高严格性
    
    def _semantic_similarity(self, widget1: Widget, widget2: Widget) -> float:
        """语义相似度计算"""
        if not widget1.texts or not widget2.texts:
            return 0.5  # 中性相似度
        
        # 文本相似度
        text_similarities = []
        for text1 in widget1.texts:
            for text2 in widget2.texts:
                # 清理文本
                clean_text1 = re.sub(r'[^a-zA-Z0-9]', '', text1).lower()
                clean_text2 = re.sub(r'[^a-zA-Z0-9]', '', text2).lower()
                
                if not clean_text1 or not clean_text2:
                    continue
                
                similarity = SequenceMatcher(None, clean_text1, clean_text2).ratio()
                text_similarities.append(similarity)
        
        if not text_similarities:
            return 0.3
        
        return max(text_similarities)
    
    def fuzzy_match(self, score: float, threshold_type: str = "fuzzy") -> bool:
        """模糊匹配判定"""
        if threshold_type == "strict":
            return score >= self.config.strict_threshold
        else:
            return score >= self.config.fuzzy_threshold
    
    def _is_type_compatible(self, widget1: Widget, widget2: Widget) -> bool:
        """检查类型是否兼容（改进版本，提高严格性）"""
        if widget1.type == widget2.type:
            return True
        
        # 定义兼容类型对（严格版本，减少误匹配）
        compatible_types = {
            ('textbutton', 'combinedbutton'),
            ('iconbutton', 'combinedbutton'),
            ('textview', 'inputbox'),
            # 移除textbutton和textview的兼容性
        }
        
        key = (widget1.type.value, widget2.type.value)
        reverse_key = (widget2.type.value, widget1.type.value)
        
        return key in compatible_types or reverse_key in compatible_types
    
    def _generate_cache_key(self, widget1: Widget, widget2: Widget) -> str:
        """生成基于部件特征的缓存键"""
        try:
            key_parts = []
            
            # 类型特征
            key_parts.append(f"type_{widget1.type.value}_{widget2.type.value}")
            
            # 尺寸特征
            key_parts.append(f"size_{int(widget1.width)}_{int(widget1.height)}_{int(widget2.width)}_{int(widget2.height)}")
            
            # 位置特征（相对位置）
            key_parts.append(f"pos_{int(widget1.bbox.xmin)}_{int(widget1.bbox.ymin)}_{int(widget2.bbox.xmin)}_{int(widget2.bbox.ymin)}")
            
            # 文本特征（简化）
            if hasattr(widget1, 'texts') and widget1.texts:
                text1_key = ''.join(widget1.texts)[:20].replace(' ', '_')
                key_parts.append(f"text1_{text1_key}")
            
            if hasattr(widget2, 'texts') and widget2.texts:
                text2_key = ''.join(widget2.texts)[:20].replace(' ', '_')
                key_parts.append(f"text2_{text2_key}")
            
            return '_'.join(key_parts)
            
        except Exception:
            # 如果生成失败，回退到原始方法
            return f"{id(widget1)}_{id(widget2)}"
    
    def match(self, screen1, screen2) -> tuple[list, list, float]:
        """完整的匹配方法，计算相似度矩阵并使用匈牙利算法进行匹配"""
        import time
        from scipy.optimize import linear_sum_assignment
        
        # 生成缓存键
        cache_key = self._generate_screen_cache_key(screen1, screen2)
        
        # 检查匹配缓存
        if cache_key in self._matching_cache:
            return self._matching_cache[cache_key]
        
        start_time = time.time()
        
        # 获取部件列表
        widgets1 = list(screen1.widgets.values())
        widgets2 = list(screen2.widgets.values())
        
        # 计算相似度矩阵
        similarity_matrix = np.zeros((len(widgets1), len(widgets2)))
        for i, widget1 in enumerate(widgets1):
            for j, widget2 in enumerate(widgets2):
                similarity = self.calculate_similarity(widget1, widget2, screen1, screen2)
                similarity_matrix[i, j] = similarity
        
        # 使用匈牙利算法进行匹配
        cost_matrix = 1.0 - similarity_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # 构建匹配结果
        pairs = []
        scores = []
        for i, j in zip(row_indices, col_indices):
            if similarity_matrix[i, j] > self.config.fuzzy_threshold:  # 使用配置的阈值
                widget_keys1 = list(screen1.widgets.keys())
                widget_keys2 = list(screen2.widgets.keys())
                pairs.append((widget_keys1[i], widget_keys2[j]))
                scores.append(similarity_matrix[i, j])
        
        match_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 缓存结果
        self._matching_cache[cache_key] = (pairs, scores, match_time)
        
        return pairs, scores, match_time
    
    def _generate_screen_cache_key(self, screen1, screen2) -> str:
        """生成屏幕级别的缓存键"""
        try:
            # 使用屏幕基本特征和部件统计作为缓存键
            key_parts = []
            
            # 屏幕尺寸特征
            if hasattr(screen1, 'image') and screen1.image is not None:
                h1, w1, _ = screen1.image.shape
                key_parts.append(f"s1_{w1}_{h1}")
            
            if hasattr(screen2, 'image') and screen2.image is not None:
                h2, w2, _ = screen2.image.shape
                key_parts.append(f"s2_{w2}_{h2}")
            
            # 部件数量特征
            if hasattr(screen1, 'widgets') and screen1.widgets is not None:
                widget_count1 = len(screen1.widgets)
                key_parts.append(f"wc1_{widget_count1}")
            
            if hasattr(screen2, 'widgets') and screen2.widgets is not None:
                widget_count2 = len(screen2.widgets)
                key_parts.append(f"wc2_{widget_count2}")
            
            # 部件类型分布特征
            if hasattr(screen1, 'widgets') and screen1.widgets:
                type_signature1 = self._get_type_signature(screen1.widgets)
                key_parts.append(f"ts1_{type_signature1}")
            
            if hasattr(screen2, 'widgets') and screen2.widgets:
                type_signature2 = self._get_type_signature(screen2.widgets)
                key_parts.append(f"ts2_{type_signature2}")
            
            return ':'.join(key_parts) if key_parts else None
            
        except Exception:
            return None
    
    def _get_type_signature(self, widgets) -> str:
        """生成类型分布的签名"""
        type_counts = {}
        for widget in widgets.values():
            widget_type = str(widget.type) if hasattr(widget, 'type') else 'unknown'
            type_counts[widget_type] = type_counts.get(widget_type, 0) + 1
        
        # 按类型字母顺序排序，确保稳定性
        sorted_types = sorted(type_counts.items(), key=lambda x: x[0])
        
        # 生成简化的签名
        signature = '_'.join([f"{k[:3]}_{v}" for k, v in sorted_types[:5]])  # 最多5种类型
        
        return signature
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        return {
            'similarity_cache_size': len(self._similarity_cache),
            'matching_cache_size': len(self._matching_cache)
        }
    
    def _generate_screen_cache_key(self, screen1, screen2) -> str:
        """生成屏幕级别的缓存键"""
        try:
            # 使用屏幕基本特征和部件统计作为缓存键
            key_parts = []
            
            # 屏幕尺寸特征
            if hasattr(screen1, 'image') and screen1.image is not None:
                h1, w1, _ = screen1.image.shape
                key_parts.append(f"s1_{w1}_{h1}")
            
            if hasattr(screen2, 'image') and screen2.image is not None:
                h2, w2, _ = screen2.image.shape
                key_parts.append(f"s2_{w2}_{h2}")
            
            # 部件数量特征
            if hasattr(screen1, 'widgets') and screen1.widgets is not None:
                widget_count1 = len(screen1.widgets)
                key_parts.append(f"wc1_{widget_count1}")
            
            if hasattr(screen2, 'widgets') and screen2.widgets is not None:
                widget_count2 = len(screen2.widgets)
                key_parts.append(f"wc2_{widget_count2}")
            
            # 部件类型分布特征
            if hasattr(screen1, 'widgets') and screen1.widgets:
                type_signature1 = self._get_type_signature(screen1.widgets)
                key_parts.append(f"ts1_{type_signature1}")
            
            if hasattr(screen2, 'widgets') and screen2.widgets:
                type_signature2 = self._get_type_signature(screen2.widgets)
                key_parts.append(f"ts2_{type_signature2}")
            
            return ':'.join(key_parts) if key_parts else None
            
        except Exception:
            return None
    
    def _get_type_signature(self, widgets) -> str:
        """生成类型分布的签名"""
        type_counts = {}
        for widget in widgets.values():
            widget_type = str(widget.type) if hasattr(widget, 'type') else 'unknown'
            type_counts[widget_type] = type_counts.get(widget_type, 0) + 1
        
        # 按类型字母顺序排序，确保稳定性
        sorted_types = sorted(type_counts.items(), key=lambda x: x[0])
        
        # 生成简化的签名
        signature = '_'.join([f"{k[:3]}_{v}" for k, v in sorted_types[:5]])  # 最多5种类型
        
        return signature
    
    def clear_cache(self):
        """清空缓存"""
        self._similarity_cache.clear()
        self._matching_cache.clear()
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        return {
            'similarity_cache_size': len(self._similarity_cache),
            'matching_cache_size': len(self._matching_cache)
        }