from __future__ import annotations
import typing
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops

from .advanced import AdvancedMatcher, SimilarityConfig

if typing.TYPE_CHECKING:
    from guipilot.entities import Screen, Widget


class MatchingLayer(Enum):
    """匹配层级"""
    VISUAL = "visual"  # 视觉特征层
    STRUCTURAL = "structural"  # 结构特征层
    SEMANTIC = "semantic"  # 语义特征层


@dataclass
class LayerConfig:
    """分层匹配配置"""
    visual_weight: float = 0.4
    structural_weight: float = 0.35
    semantic_weight: float = 0.25
    
    # 视觉特征配置
    lbp_radius: int = 1
    lbp_points: int = 8
    
    # 结构特征配置
    glcm_distances: list = field(default_factory=lambda: [1])
    glcm_angles: list = field(default_factory=lambda: [0, np.pi/4, np.pi/2, 3*np.pi/4])
    
    def normalize_weights(self):
        """归一化权重"""
        total = self.visual_weight + self.structural_weight + self.semantic_weight
        if total != 1.0:
            self.visual_weight /= total
            self.structural_weight /= total
            self.semantic_weight /= total


class HierarchicalMatcher:
    """分层匹配器，支持三层匹配架构"""
    
    def __init__(self, layer_config: LayerConfig = None, similarity_config: SimilarityConfig = None):
        self.layer_config = layer_config or LayerConfig()
        self.layer_config.normalize_weights()
        
        self.advanced_matcher = AdvancedMatcher(similarity_config)
        
        # 缓存机制
        self._feature_cache = {}
        self._matching_cache = {}  # 屏幕级别的匹配结果缓存
        self._matching_cache = {}  # 屏幕级别的匹配结果缓存
    
    def hierarchical_match(self, screen1: Screen, screen2: Screen) -> list[tuple[int, int]]:
        """分层匹配主函数"""
        # 检查缓存
        cache_key = self._generate_screen_cache_key(screen1, screen2)
        if cache_key and cache_key in self._matching_cache:
            return self._matching_cache[cache_key]
        
        # 第一层：视觉特征匹配
        visual_pairs = self._visual_layer_match(screen1, screen2)
        
        # 第二层：结构特征匹配（排除已匹配的部件）
        structural_pairs = self._structural_layer_match(screen1, screen2, visual_pairs)
        
        # 第三层：语义特征匹配
        semantic_pairs = self._semantic_layer_match(screen1, screen2, visual_pairs + structural_pairs)
        
        # 合并结果
        all_pairs = visual_pairs + structural_pairs + semantic_pairs
        final_pairs = self._resolve_conflicts(all_pairs)
        
        # 缓存结果
        if cache_key:
            self._matching_cache[cache_key] = final_pairs
        
        return final_pairs
    
    def _visual_layer_match(self, screen1: Screen, screen2: Screen) -> list[tuple[int, int]]:
        """优化视觉层匹配"""
        pairs = []
        
        # 预先计算所有部件的视觉特征 - 使用带缓存的方法
        visual_features1 = {}
        for i, widget1 in screen1.widgets.items():
            visual_features1[i] = self._extract_cached_visual_features(widget1, screen1)
        
        visual_features2 = {}
        for j, widget2 in screen2.widgets.items():
            visual_features2[j] = self._extract_cached_visual_features(widget2, screen2)
        
        # 使用更高效的匹配策略
        candidate_pairs = []
        for i, features1 in visual_features1.items():
            for j, features2 in visual_features2.items():
                similarity = self._fast_visual_similarity(features1, features2)
                if similarity > 0.85:  # 进一步提高视觉层阈值，提高准确性
                    candidate_pairs.append((i, j, similarity))
        
        # 按相似度排序并选择最佳匹配
        candidate_pairs.sort(key=lambda x: x[2], reverse=True)
        
        used_widgets1 = set()
        used_widgets2 = set()
        
        for i, j, similarity in candidate_pairs:
            if i not in used_widgets1 and j not in used_widgets2:
                pairs.append((i, j))
                used_widgets1.add(i)
                used_widgets2.add(j)
        
        return pairs

    def _fast_visual_similarity(self, features1: dict, features2: dict) -> float:
        """快速视觉相似度计算"""
        # 尺寸相似度 (40%)
        size_sim = 1.0 - abs(features1['area'] - features2['area']) / max(features1['area'], features2['area'])
        
        # 宽高比相似度 (30%)
        aspect_sim = 1.0 - abs(features1['aspect_ratio'] - features2['aspect_ratio']) / max(features1['aspect_ratio'], features2['aspect_ratio'])
        
        # 位置相似度 (30%)
        if 'normalized_position' in features1 and 'normalized_position' in features2:
            pos_distance = np.linalg.norm(np.array(features1['normalized_position']) - np.array(features2['normalized_position']))
            pos_sim = 1.0 / (1.0 + pos_distance)
        else:
            # 回退到中心点距离
            center1 = features1.get('center', (0.5, 0.5))
            center2 = features2.get('center', (0.5, 0.5))
            pos_distance = np.linalg.norm(np.array(center1) - np.array(center2))
            pos_sim = 1.0 / (1.0 + pos_distance)
        
        return 0.4 * max(0, size_sim) + 0.3 * max(0, aspect_sim) + 0.3 * pos_sim
    
    def _structural_layer_match(self, screen1: Screen, screen2: Screen, 
                              exclude_pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """结构特征层匹配"""
        pairs = []
        
        # 获取未匹配的部件
        matched_widgets2 = {pair[1] for pair in exclude_pairs}
        unmatched_widgets2 = set(j for j in screen2.widgets.keys() if j not in matched_widgets2)
        
        # 预先计算所有部件的结构特征（使用缓存）
        structural_features1 = {}
        for i, widget1 in screen1.widgets.items():
            if i not in [pair[0] for pair in exclude_pairs]:
                structural_features1[i] = self._extract_structural_features(widget1, screen1)
        
        structural_features2 = {}
        for j in unmatched_widgets2:
            widget2 = screen2.widgets[j]
            structural_features2[j] = self._extract_structural_features(widget2, screen2)
        
        for i, features1 in structural_features1.items():
            best_match = None
            best_score = 0
            
            # 只考虑当前仍然可用的未匹配部件
            available_widgets = unmatched_widgets2.copy()
            
            for j in available_widgets:
                features2 = structural_features2[j]
                # 计算结构相似度
                structural_score = self._structural_similarity(features1, features2)
                
                if structural_score > best_score and structural_score > 0.85:  # 提高结构层阈值，提高准确性
                    best_score = structural_score
                    best_match = j
            
            if best_match is not None and best_match in unmatched_widgets2:
                pairs.append((i, best_match))
                # 从unmatched_widgets2中移除已匹配的部件
                unmatched_widgets2.remove(best_match)
        
        return pairs
    
    def _semantic_layer_match(self, screen1: Screen, screen2: Screen, 
                             exclude_pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """语义特征层匹配"""
        pairs = []
        
        # 获取未匹配的部件
        matched_widgets2 = {pair[1] for pair in exclude_pairs}
        unmatched_widgets2 = [j for j in screen2.widgets.keys() if j not in matched_widgets2]
        
        for i, widget1 in screen1.widgets.items():
            if i in [pair[0] for pair in exclude_pairs]:
                continue
            
            best_match = None
            best_score = 0
            
            for j in unmatched_widgets2:
                widget2 = screen2.widgets[j]
                
                # 使用高级匹配器计算语义相似度
                semantic_score = self.advanced_matcher.calculate_similarity(
                    widget1, widget2, screen1, screen2
                )
                
                if semantic_score > best_score and semantic_score > 0.85:  # 提高语义层阈值，提高准确性
                    best_score = semantic_score
                    best_match = j
            
            if best_match is not None:
                pairs.append((i, best_match))
        
        return pairs
    
    def _extract_visual_features(self, widget: Widget, screen: Screen) -> dict:
        """提取简化的视觉特征（性能优化版本）"""
        cache_key = f"visual_{widget.bbox.xmin}_{widget.bbox.ymin}_{widget.width}_{widget.height}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        # 直接使用简化的几何特征，避免昂贵的图像处理
        features = {
            'size': (widget.width, widget.height),
            'aspect_ratio': widget.width / max(widget.height, 1),
            'area': widget.width * widget.height,
            'position': (widget.bbox.xmin, widget.bbox.ymin),
            'normalized_position': (
                widget.bbox.xmin / screen.image.shape[1],
                widget.bbox.ymin / screen.image.shape[0]
            )
        }
        
        self._feature_cache[cache_key] = features
        return features
    
    def _extract_cached_visual_features(self, widget: Widget, screen: Screen) -> dict:
        """提取带缓存的视觉特征"""
        cache_key = f"visual_{widget.bbox.xmin}_{widget.bbox.ymin}_{widget.width}_{widget.height}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        # 使用简单的几何特征，避免图像处理
        features = {
            'width': widget.width,
            'height': widget.height,
            'aspect_ratio': widget.width / max(widget.height, 1),
            'area': widget.width * widget.height,
            'position': (widget.bbox.xmin, widget.bbox.ymin),  # 左上角坐标
            'center': ((widget.bbox.xmin + widget.bbox.xmax) / 2, (widget.bbox.ymin + widget.bbox.ymax) / 2),  # 中心点
            'normalized_position': (
                widget.bbox.xmin / screen.image.shape[1],
                widget.bbox.ymin / screen.image.shape[0]
            )
        }
        
        # 添加类型特征
        features['type'] = str(widget.type) if hasattr(widget, 'type') else 'unknown'
        
        # 缓存特征
        self._feature_cache[cache_key] = features
        return features
    
    def _extract_simple_visual_features(self, widget: Widget) -> dict:
        """提取简化的视觉特征（用于性能优化）"""
        # 使用简单的几何特征，避免图像处理
        features = {
            'width': widget.width,
            'height': widget.height,
            'aspect_ratio': widget.width / max(widget.height, 1),
            'area': widget.width * widget.height,
            'position': (widget.bbox.xmin, widget.bbox.ymin),  # 左上角坐标
            'center': ((widget.bbox.xmin + widget.bbox.xmax) / 2, (widget.bbox.ymin + widget.bbox.ymax) / 2)  # 中心点
        }
        
        # 添加类型特征
        features['type'] = str(widget.type) if hasattr(widget, 'type') else 'unknown'
        
        # 添加归一化位置（如果可能）
        if hasattr(widget, 'normalized_position'):
            features['normalized_position'] = widget.normalized_position
    
        return features
    
    def _simple_visual_similarity(self, features1: dict, features2: dict) -> float:
        """计算简化视觉相似度"""
        similarity = 0.0
        
        # 尺寸相似度 (权重: 0.4)
        size_sim = 1.0 - abs(features1['area'] - features2['area']) / max(features1['area'], features2['area'])
        similarity += 0.4 * max(0, size_sim)
        
        # 长宽比相似度 (权重: 0.3)
        aspect_sim = 1.0 - abs(features1['aspect_ratio'] - features2['aspect_ratio']) / max(features1['aspect_ratio'], features2['aspect_ratio'])
        similarity += 0.3 * max(0, aspect_sim)
        
        # 类型相似度 (权重: 0.3)
        type_sim = 1.0 if features1['type'] == features2['type'] else 0.0
        similarity += 0.3 * type_sim
        
        return similarity
    
    def _extract_structural_features(self, widget: Widget, screen: Screen) -> dict:
        cache_key = f"structural_{widget.bbox.xmin}_{widget.bbox.ymin}_{widget.width}_{widget.height}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        features = {
            'position': (widget.bbox.xmin, widget.bbox.ymin),
            'relative_position': (
                widget.bbox.xmin / screen.image.shape[1],
                widget.bbox.ymin / screen.image.shape[0]
            ),
            'size_ratio': (widget.width / screen.image.shape[1], widget.height / screen.image.shape[0]),
            'aspect_ratio': widget.width / max(widget.height, 1),
            'area_ratio': (widget.width * widget.height) / (screen.image.shape[1] * screen.image.shape[0])
        }
        
        self._feature_cache[cache_key] = features
        return features
    
    def _visual_similarity(self, features1: dict, features2: dict) -> float:
        """计算视觉相似度"""
        # LBP特征相似度（余弦相似度）
        lbp_sim = np.dot(features1['lbp'], features2['lbp']) / (
            np.linalg.norm(features1['lbp']) * np.linalg.norm(features2['lbp']) + 1e-8
        )
        
        # 尺寸相似度
        size_sim = min(features1['size'][0], features2['size'][0]) / max(features1['size'][0], features2['size'][0]) *\
                  min(features1['size'][1], features2['size'][1]) / max(features1['size'][1], features2['size'][1])
        
        # 宽高比相似度
        aspect_sim = min(features1['aspect_ratio'], features2['aspect_ratio']) / \
                    max(features1['aspect_ratio'], features2['aspect_ratio'])
        
        return (lbp_sim + size_sim + aspect_sim) / 3
    
    def _structural_similarity(self, features1: dict, features2: dict) -> float:
        """计算简化的结构相似度（性能优化版本）"""
        # 位置相似度（主要特征）
        pos_distance = np.linalg.norm(
            np.array(features1['relative_position']) - np.array(features2['relative_position'])
        )
        pos_sim = 1.0 / (1.0 + pos_distance)
        
        # 尺寸相似度
        size_distance = np.linalg.norm(
            np.array(features1['size_ratio']) - np.array(features2['size_ratio'])
        )
        size_sim = 1.0 / (1.0 + size_distance)
        
        # 宽高比相似度
        aspect_sim = 1.0 - abs(features1['aspect_ratio'] - features2['aspect_ratio']) / max(features1['aspect_ratio'], features2['aspect_ratio'])
        
        # 综合相似度（加权平均）
        return (0.5 * pos_sim + 0.3 * size_sim + 0.2 * aspect_sim)
    
    def _resolve_conflicts(self, pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """解决匹配冲突"""
        # 按部件ID分组
        widget1_to_widget2 = {}
        widget2_to_widget1 = {}
        
        for pair in pairs:
            if pair[0] in widget1_to_widget2:
                # 冲突：一个部件匹配到多个目标部件
                # 这里可以添加更复杂的冲突解决策略
                continue
            if pair[1] in widget2_to_widget1:
                # 冲突：一个目标部件匹配到多个源部件
                continue
            
            widget1_to_widget2[pair[0]] = pair[1]
            widget2_to_widget1[pair[1]] = pair[0]
        
        return [(k, v) for k, v in widget1_to_widget2.items()]
    
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
            
            # 添加部件位置特征，增强缓存键的精确性
            if hasattr(screen1, 'widgets') and screen1.widgets:
                pos_signature1 = self._get_position_signature(screen1.widgets)
                key_parts.append(f"ps1_{pos_signature1}")
            
            if hasattr(screen2, 'widgets') and screen2.widgets:
                pos_signature2 = self._get_position_signature(screen2.widgets)
                key_parts.append(f"ps2_{pos_signature2}")
            
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
    
    def _get_position_signature(self, widgets) -> str:
        """生成部件位置分布的签名"""
        if not widgets:
            return "empty"
        
        # 计算部件的中心点位置并分桶
        positions = []
        for widget in widgets.values():
            if hasattr(widget, 'bbox') and hasattr(widget.bbox, 'xmin'):
                center_x = (widget.bbox.xmin + widget.bbox.xmax) / 2
                center_y = (widget.bbox.ymin + widget.bbox.ymax) / 2
                # 将位置分成5x5的网格
                grid_x = min(int(center_x / 200), 4)  # 假设屏幕宽度约1000px
                grid_y = min(int(center_y / 200), 4)
                positions.append(f"{grid_x}_{grid_y}")
        
        # 统计每个网格中的部件数量
        grid_counts = {}
        for pos in positions:
            grid_counts[pos] = grid_counts.get(pos, 0) + 1
        
        # 生成签名
        sorted_grids = sorted(grid_counts.items(), key=lambda x: x[0])
        signature = '_'.join([f"{k}_{v}" for k, v in sorted_grids[:10]])  # 最多10个网格
        
        return signature
    
    def clear_cache(self):
        """清空缓存"""
        self._feature_cache.clear()
        self._matching_cache.clear()
        self.advanced_matcher.clear_cache()
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        return {
            'feature_cache_size': len(self._feature_cache),
            'matching_cache_size': len(self._matching_cache),
            'advanced_matcher_cache_stats': self.advanced_matcher.get_cache_stats() if hasattr(self.advanced_matcher, 'get_cache_stats') else {}
        }