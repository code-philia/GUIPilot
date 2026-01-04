from __future__ import annotations
import typing
import re
import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional

if typing.TYPE_CHECKING:
    from guipilot.entities import Widget, Screen


class SensitivityLevel(Enum):
    """敏感度级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SensitiveInfoType(Enum):
    """敏感信息类型"""
    EMAIL = "email"
    PHONE = "phone"
    PASSWORD = "password"
    IP_ADDRESS = "ip_address"
    CREDIT_CARD = "credit_card"
    PERSONAL_ID = "personal_id"


@dataclass
class MaskingConfig:
    """masking配置"""
    sensitivity_level: SensitivityLevel = SensitivityLevel.MEDIUM
    
    # 检测阈值
    confidence_threshold: float = 0.8
    
    # masking样式
    mask_char: str = "*"
    mask_ratio: float = 0.7  # 遮挡比例
    
    # 不同敏感度级别的检测规则
    detection_rules: dict = None
    
    def __post_init__(self):
        if self.detection_rules is None:
            self.detection_rules = self._get_default_rules()
    
    def _get_default_rules(self) -> dict:
        """获取默认检测规则"""
        return {
            SensitivityLevel.LOW: {
                SensitiveInfoType.EMAIL: {
                    'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    'confidence': 0.9
                },
                SensitiveInfoType.PASSWORD: {
                    'pattern': r'(?i)(password|pwd|pass|secret|key)\s*[:=]\s*[^\s]+',
                    'confidence': 0.8
                }
            },
            SensitivityLevel.MEDIUM: {
                SensitiveInfoType.EMAIL: {
                    'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    'confidence': 0.8
                },
                SensitiveInfoType.PHONE: {
                    'pattern': r'(\+?\d{1,3}[-.\s]?)?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
                    'confidence': 0.7
                },
                SensitiveInfoType.PASSWORD: {
                    'pattern': r'(?i)(password|pwd|pass|secret|key)\s*[:=]\s*([^\s]+)',
                    'confidence': 0.7
                }
            },
            SensitivityLevel.HIGH: {
                SensitiveInfoType.EMAIL: {
                    'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    'confidence': 0.7
                },
                SensitiveInfoType.PHONE: {
                    'pattern': r'(\+?\d{1,3}[-.\s]?)?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
                    'confidence': 0.6
                },
                SensitiveInfoType.PASSWORD: {
                    'pattern': r'(?i)(password|pwd|pass|secret|key)\s*[:=]\s*[^\s]+',
                    'confidence': 0.6
                },

                SensitiveInfoType.PERSONAL_ID: {
                    'pattern': r'\b\d{17}[\dXx]\b|\b\d{15}\b',  # 身份证号
                    'confidence': 0.9
                }
            }
        }


class SensitiveInfoDetector:
    """敏感信息检测器"""
    
    def __init__(self, config: MaskingConfig = None):
        self.config = config or MaskingConfig()
    
    def detect_sensitive_info(self, text: str) -> List[Tuple[SensitiveInfoType, str, float]]:
        """检测文本中的敏感信息"""
        detected_info = []
        
        # 获取当前敏感度级别的检测规则
        rules = self.config.detection_rules.get(self.config.sensitivity_level, {})
        
        # 增强检测：先使用关键词检测，再精确匹配
        for info_type, rule in rules.items():
            pattern = rule['pattern']
            confidence_threshold = rule.get('confidence', self.config.confidence_threshold)
            
            # 第一步：检测关键词
            keywords = self._get_keywords_for_type(info_type)
            has_keyword = any(keyword.lower() in text.lower() for keyword in keywords)
            
            if has_keyword:
                # 第二步：从文本中提取信息值
                info_value = self._extract_info_value(text, info_type)
                if info_value:
                    confidence = self._calculate_confidence(info_value, info_type)
                    if confidence >= confidence_threshold:
                        detected_info.append((info_type, info_value, confidence))
                        continue
            
            # 第三步：直接模式匹配
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                matched_text = match.group()
                confidence = self._calculate_confidence(matched_text, info_type)
                if confidence >= confidence_threshold:
                    detected_info.append((info_type, matched_text, confidence))
        
        return detected_info
    
    def _get_keywords_for_type(self, info_type: SensitiveInfoType) -> list:
        """获取每种信息类型的关键词"""
        keywords_map = {
            SensitiveInfoType.EMAIL: ['邮箱', 'email', 'mail', '电子邮箱'],
            SensitiveInfoType.PHONE: ['电话', '手机', 'phone', '联系电话', '手机号'],
            SensitiveInfoType.PASSWORD: ['密码', 'password', 'pwd', 'pass', 'secret'],

            SensitiveInfoType.PERSONAL_ID: ['身份证', '证件', 'id', '身份证号']
        }
        return keywords_map.get(info_type, [])
    
    def _extract_info_value(self, text: str, info_type: SensitiveInfoType) -> str:
        """从包含标签的文本中提取信息值"""
        keywords = self._get_keywords_for_type(info_type)
        
        for keyword in keywords:
            if keyword.lower() in text.lower():
                # 查找关键字后的内容
                keyword_index = text.lower().find(keyword.lower())
                if keyword_index != -1:
                    # 提取冒号或等号后面的内容
                    value_start = text.find(':', keyword_index)
                    if value_start == -1:
                        value_start = text.find('：', keyword_index)  # 中文冒号
                    if value_start == -1:
                        value_start = text.find('=', keyword_index)
                    
                    if value_start != -1:
                        value_text = text[value_start + 1:].strip()
                        # 提取第一个有效值（改进正则表达式）
                        if info_type == SensitiveInfoType.IP_ADDRESS:
                            # 使用更严格的IP地址正则表达式
                            value_match = re.search(r'((?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?))', value_text)
                        elif info_type == SensitiveInfoType.CREDIT_CARD:
                            # 使用更严格的信用卡正则表达式
                            value_match = re.search(r'(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9]{2})[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})', value_text)
                        else:
                            value_match = re.search(r'([^\s]+)', value_text)
                        
                        if value_match:
                            return value_match.group(1)
        
        return ""
    
    def _calculate_confidence(self, text: str, info_type: SensitiveInfoType) -> float:
        """计算检测置信度"""
        base_confidence = 0.8  # 提高基础置信度
        
        # 根据信息类型调整置信度
        type_weights = {
            SensitiveInfoType.EMAIL: 0.95,
            SensitiveInfoType.PHONE: 0.85,
            SensitiveInfoType.PASSWORD: 0.98,
            SensitiveInfoType.IP_ADDRESS: 0.8,
            SensitiveInfoType.CREDIT_CARD: 0.9,
            SensitiveInfoType.PERSONAL_ID: 0.95
        }
        
        base_confidence *= type_weights.get(info_type, 0.7)
        
        # 根据匹配质量调整置信度
        if info_type == SensitiveInfoType.EMAIL and '@' in text and '.' in text:
            base_confidence *= 1.1
        elif info_type == SensitiveInfoType.PHONE and len(text.replace('-', '').replace(' ', '')) >= 10:
            base_confidence *= 1.1
        elif info_type == SensitiveInfoType.CREDIT_CARD and len(text.replace('-', '').replace(' ', '')) == 16:
            base_confidence *= 1.2
        
        return min(base_confidence, 1.0)


class PrivacyMasker:
    """隐私保护masking器"""
    
    def __init__(self, config: MaskingConfig = None):
        self.config = config or MaskingConfig()
        self.detector = SensitiveInfoDetector(config)
    
    def mask_sensitive_info(self, widget: Widget, screen: Screen) -> Widget:
        """对部件中的敏感信息进行masking"""
        # 创建部件的副本以避免修改原始数据
        masked_widget = self._copy_widget(widget)
        
        # 检测并masking文本
        masked_texts = []
        for text in widget.texts:
            masked_text = self._mask_text(text)
            masked_texts.append(masked_text)
        
        masked_widget.texts = masked_texts
        
        # 对部件图像进行masking（如果需要）
        if self.config.sensitivity_level == SensitivityLevel.HIGH:
            masked_widget = self._mask_widget_image(masked_widget, screen)
        
        return masked_widget
    
    def _mask_text(self, text: str) -> str:
        """对文本进行masking"""
        detected_info = self.detector.detect_sensitive_info(text)
        
        if not detected_info:
            return text
        
        masked_text = text
        
        for info_type, matched_text, confidence in detected_info:
            # 计算masking长度
            mask_length = int(len(matched_text) * self.config.mask_ratio)
            mask_length = max(1, mask_length)  # 至少masking一个字符
            
            # 生成masking字符串
            mask_str = self.config.mask_char * mask_length
            
            # 保留部分原文本
            keep_length = len(matched_text) - mask_length
            if keep_length > 0:
                # 保留开头部分
                masked_part = matched_text[:keep_length] + mask_str
            else:
                masked_part = mask_str
            
            # 替换敏感信息（使用正则替换确保精确匹配）
            import re
            pattern = re.escape(matched_text)
            masked_text = re.sub(pattern, masked_part, masked_text)
        
        return masked_text
    
    def _mask_widget_image(self, widget: Widget, screen: Screen) -> Widget:
        """对部件图像进行masking"""
        # 在实际实现中，这里可以对部件图像进行像素级的masking
        # 目前返回原始部件，需要时再实现
        return widget
    
    def _copy_widget(self, widget: Widget) -> Widget:
        """创建部件的深拷贝"""
        # 简化实现，实际项目中应该使用copy.deepcopy
        import copy
        return copy.deepcopy(widget)
    
    def mask_screen(self, screen: Screen) -> Screen:
        """对整个屏幕进行masking"""
        # 创建屏幕的副本
        import copy
        masked_screen = copy.deepcopy(screen)
        
        # 对每个部件进行masking
        for widget_id, widget in masked_screen.widgets.items():
            masked_screen.widgets[widget_id] = self.mask_sensitive_info(widget, masked_screen)
        
        return masked_screen


class PrivacyManager:
    """隐私管理器"""
    
    def __init__(self, config: MaskingConfig = None):
        self.config = config or MaskingConfig()
        self.masker = PrivacyMasker(config)
    
    def set_sensitivity_level(self, level: SensitivityLevel):
        """设置敏感度级别"""
        self.config.sensitivity_level = level
        self.masker.config.sensitivity_level = level
    
    def get_sensitivity_level(self) -> SensitivityLevel:
        """获取当前敏感度级别"""
        return self.config.sensitivity_level
    
    def analyze_privacy_risk(self, screen: Screen) -> dict:
        """分析隐私风险"""
        risk_analysis = {
            'total_widgets': len(screen.widgets),
            'sensitive_widgets': 0,
            'detected_info': {},
            'risk_score': 0.0
        }
        
        total_confidence = 0.0
        detected_count = 0
        
        for widget_id, widget in screen.widgets.items():
            for text in widget.texts:
                detected_info = self.masker.detector.detect_sensitive_info(text)
                
                if detected_info:
                    risk_analysis['sensitive_widgets'] += 1
                    
                    for info_type, matched_text, confidence in detected_info:
                        if info_type.value not in risk_analysis['detected_info']:
                            risk_analysis['detected_info'][info_type.value] = 0
                        risk_analysis['detected_info'][info_type.value] += 1
                        
                        total_confidence += confidence
                        detected_count += 1
        
        # 计算风险分数
        if detected_count > 0:
            avg_confidence = total_confidence / detected_count
            widget_ratio = risk_analysis['sensitive_widgets'] / risk_analysis['total_widgets']
            risk_analysis['risk_score'] = avg_confidence * widget_ratio * 100
        
        return risk_analysis
    
    def apply_privacy_protection(self, screen: Screen) -> Screen:
        """应用隐私保护"""
        return self.masker.mask_screen(screen)