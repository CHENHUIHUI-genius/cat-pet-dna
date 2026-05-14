# Cat Pet DNA — 低算力视觉系统架构

> **目标**：在尽量低算力、可商用的前提下，为猫生成可验证的结构化 Pet DNA。
> **原则**：不优先训练大模型，优先使用现成视觉算法和规则方法。

---

## 目录

1. [数据源说明](#1-数据源说明)
2. [管线架构总览](#2-管线架构总览)
3. [模块详解](#3-模块详解)
4. [可视化验证 Demo](#4-可视化验证-demo)
5. [Pet DNA JSON Schema](#5-pet-dna-json-schema)
6. [算力估算与优化说明](#6-算力估算与优化说明)

---

## 1. 数据源说明

| 数据集 | 许可 | 是否进入主流程 | 用途 |
|--------|------|---------------|------|
| **Oxford-IIIT Pet** | 仅供**研究目的** (非商用) | ❌ 不进入主流程 | 离线调参、规则验证、算法选型参考 |
| **AP-10K (猫科子集)** | 研究目的 | ❌ 不进入主流程 | 姿态验证参考 |
| **用户自有图片 / 公开 CC0 图片** | 可商用 | ✅ 主流程输入 | 实际推理输入 |

> ⚠️ **商业合规说明**：主推理管线设计为**不依赖任何非商用数据训练出的模型参数**。Oxford-IIIT Pet 仅在离线开发阶段用于调参验证规则阈值，最终系统中无任何源于该数据集的权重文件。

---

## 2. 管线架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入：猫图片                               │
│                    (单张 RGB，任意尺寸)                           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: 预处理 & 前景分割                                      │
│  ├─ 双线性缩放至 512×512 (统一尺寸)                              │
│  ├─ 自动主体检测：色彩阈值 + 边缘检测 + 最大轮廓提取               │
│  └─ GrabCut 精细化分割 (OpenCV 内置，无需模型)                    │
│  ▶ 输出：前景二值蒙版 (mask)                                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: 颜色特征提取                                           │
│  ├─ RGB/HSV 直方图统计 (前景区域)                                │
│  ├─ K=3~6 的 K-Means 聚类 → 主色提取                             │
│  ├─ 颜色分布熵：多样性指标                                        │
│  └─ 毛色斑纹规则推断 (虎斑/纯色/双色/三花规则)                    │
│  ▶ 输出：color_palette, color_pattern_type                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: 轮廓几何特征                                         │
│  ├─ 轮廓面积 (像素)                                              │
│  ├─ 轮廓周长                                                   │
│  ├─ 最小外接矩形 → 长宽比                                        │
│  ├─ 圆形度 (4π·面积/周长²)                                      │
│  ├─ 凸包面积比 (solidity)                                        │
│  └─ 头部 ROI 提取 (基于轮廓上1/3区域 或 传入的头部框)             │
│  ▶ 输出：body_geometry, head_geometry                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4: 头部 & 面部特征                                      │
│  ├─ 头部区域裁剪 (基于轮廓 或 传参头部框)                         │
│  ├─ 头部圆形度 / 宽高比                                         │
│  ├─ 耳朵检测：轮廓顶点分析 → 耳尖位置 & 耳间距                    │
│  ├─ 眼睛 ROI 检测 (基于头部明暗分布 + 形态学)                     │
│  └─ 面部对称性评分                                               │
│  ▶ 输出：face_landmarks, ear_features, eye_features              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5: 姿态 & 动作倾向                                      │
│  ├─ 主体长宽比 → 区分 侧卧/站立/坐姿/卷缩                        │
│  ├─ 轮廓主轴角度 → 倾斜姿态判定                                  │
│  ├─ 拉伸指数 (extent = 面积/外接矩形面积) → 动作倾向              │
│  ├─ 尾巴检测 (轮廓底部延伸分析)                                  │
│  └─ 规则推断：                                                      │
│     · 低长宽比 + 高 extent → "卷缩/休息"                          │
│     · 高长宽比 + 低 extent → "伸展/玩耍"                          │
│     · 水平主轴 → "侧躺"                                          │
│     · 竖立主轴 → "站立/坐姿"                                      │
│  ▶ 输出：pose, action_tendency                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 6: DNA 聚合 & 置信度                                     │
│  ├─ 各模块置信度加权聚合                                         │
│  ├─ 分割质量评分 → 整体置信度                                    │
│  ├─ 品种推断 (基于颜色+轮廓+耳朵规则表)                           │
│  └─ 输出最终 Pet DNA JSON                                       │
│  ▶ 输出：结构化 Pet DNA                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 算力分析（总览）

| 阶段 | 算法 | 算力级别 | 说明 |
|------|------|---------|------|
| S1 | GrabCut + 边缘检测 | ~30ms (CPU) | OpenCV 内置，无需 GPU |
| S2 | K-Means 聚类 | ~10ms (CPU) | K=6，迭代10次 |
| S3 | 轮廓几何计算 | ~5ms (CPU) | OpenCV contour 函数 |
| S4 | 形态学 + 顶点分析 | ~15ms (CPU) | 纯图像处理 |
| S5 | 比率规则推断 | <1ms (CPU) | 纯数值计算 |
| **总计** | **全管线** | **~60ms / 帧 (CPU)** | 无 GPU 需求 |

---

## 3. 模块详解

### 3.1 前景分割 — 视觉算法

```
输入图片 → 高斯模糊 → Canny 边缘检测 → 形态学闭操作 → 
最大连通域提取 → 扩展边界框 → GrabCut 精细化 → 输出 mask
```

- **Canny 参数**：low=50, high=150 (经 Oxford 数据验证)
- **GrabCut**：仅做 1 次迭代（iterCount=1）保持低算力
- **降级方案**：若有传入分割图，直接使用；若有头部框，以此初始化 GrabCut

### 3.2 颜色特征 — 视觉算法 + 规则推断

| 步骤 | 方法 | 类型 |
|------|------|------|
| 主色提取 | 前景像素 K-Means (K=3~6) | ✅ 视觉算法 |
| 颜色多样性 | HSV 直方图熵值 | ✅ 视觉算法 |
| 斑纹分类 | 基于主色分布模式 + 颜色对比度规则 | ✅ 规则推断 |
| 毛色类型 | 基于 RGB 域值判定 (白/黑/橘/灰/狸花) | ✅ 规则推断 |

**斑纹规则表**：

| 主色数 | 颜色对比度 | 斑纹推断 | 颜色熵阈值 |
|--------|-----------|---------|-----------|
| 1~2 | 低 | 纯色 (solid) | H_entropy < 2.0 |
| 2~3 | 中 | 虎斑 (tabby) | 2.0 ≤ H_entropy ≤ 4.0 |
| 3+ | 高且含白色区域 | 三花 (calico) | 白色像素 > 20% |
| 2 | 黑白分明 | 双色 (bicolor) | 黑+白 > 总像素 70% |

### 3.3 轮廓几何 — 视觉算法

所有特征直接通过 OpenCV `cv2.contour` 函数族计算，无任何学习成分：

```python
area = cv2.contourArea(contour)           # 面积
perimeter = cv2.arcLength(contour, True)  # 周长
_, (w, h), angle = cv2.minAreaRect(contour)  # 最小外接矩形
circularity = 4 * math.pi * area / (perimeter ** 2)  # 圆形度
hull = cv2.convexHull(contour)
solidity = area / cv2.contourArea(hull)   # 凸度
extent = area / (w * h)                   # 拉伸指数
```

### 3.4 头部 & 面部 — 视觉算法 + 规则推断

**头部提取**（降级优先级）：
1. ✅ 若传入头部框 → 直接裁剪
2. ✅ 若传入分割图 → 取分割轮廓的垂直上 30% ~ 40% 区域
3. ✅ 若只有图片 → 轮廓质心以上区域 + 形态学处理

**耳朵检测**（规则方法）：
- 在头部轮廓边缘检测凸点
- 取顶部两个显著凸点作为耳尖
- 计算耳间距 / 头宽比 → 判断耳型（尖耳/圆耳/折耳）

**眼睛 ROI**（规则方法）：
- 头部 ROI 内做自适应直方图均衡化
- 形态学开操作去噪
- 按明暗区域位置推断眼部 ROI

### 3.5 姿态 — 视觉算法 + 规则推断

| 指标 | 计算方法 | 类型 |
|------|---------|------|
| 长宽比 (H/W) | 外接矩形高/宽 | ✅ 视觉算法 |
| 主轴角度 | 最小外接矩形旋转角 | ✅ 视觉算法 |
| 拉伸指数 extent | area / (w×h) | ✅ 视觉算法 |
| 姿态分类 | 规则决策树 | ✅ 规则推断 |

```
extent > 0.6 →  compact（卷缩/蹲坐 → 休息态）
extent 0.4~0.6 → normal（常规姿势）
extent < 0.4 → extended（舒展 → 活跃态）

结合长宽比细化：
  H/W > 1.2 → 站立/坐姿
  H/W < 0.8 → 侧躺/伸展
  0.8~1.2  → 端坐/蹲伏
```

### 3.6 品种推断 — 纯规则推断

基于可观察外部特征的规则表（不依赖任何训练数据）：

```
颜色 + 耳型 + 体型 + 面部特征 → 品种候选列表
```

示例规则：
- 纯白色 + 蓝眼/异色眼 → [波斯猫, 布偶猫]
- 虎斑 + 尖耳 + 修长体型 → [中华田园狸花猫, 孟加拉猫]
- 三花 → [田园三花猫] (三花基本为雌性)

> ⚠️ 品种推断列多个候选并标注置信度，不做唯一断定。

---

## 4. 可视化验证 Demo

### 4.1 单张图片 Demo

```
输入:  cat.jpg
输出:  cat_dna_visualized.jpg  (带标注的图片)
      + cat_dna.json           (结构化 DNA)
      + report.txt             (分步结果日志)
```

**可视化标注内容**：
1. ✅ 绿色轮廓 — 前景分割结果
2. ✅ 蓝色矩形 — 最小外接矩形 + 长宽比标注
3. ✅ 红色圆点 — 耳尖位置 & 耳间距标注
4. ✅ 色块条 — 前 3 主色色块
5. ✅ 文字叠加 — 姿态、毛色类型、置信度

### 4.2 批量验证 Demo

```
输入:  ./images/  (文件夹，多张猫图片)
输出:  ./output_dna/ (每张对应 json + visualized)
      + summary_report.html (批处理 HTML 报告)
```

**HTML 报告包含**：
- 每张图片缩略图 + DNA 摘要
- 统计分布图（姿态分布饼图、毛色分布条形图）
- 置信度直方图
- ⏱ 总耗时 / 平均每帧耗时

### 4.3 「一眼看到在动」的实时 Demo

```
实时摄像头模式：
  摄像头帧 → 管线处理 → 实时叠加 DNA 信息显示
  帧率显示 (预期 ≥ 15 FPS on CPU)
```

**屏幕叠加信息**：
```
┌─────────────────────────────────────┐
│  🐱 Pet DNA Live                    │
│  ──────────────────────────────     │
│  毛色: 橘色虎斑  [████████░░] 82%   │
│  姿态: 侧躺     [██████████] 95%   │
│  体型: 舒展                        │
│  ──────────────────────────────     │
│  FPS: 18  |  耗时: 55ms            │
└─────────────────────────────────────┘
```

**关键设计**：
- 取景框自动锁定画面中最大猫轮廓
- 每 5 帧做一次全管线处理，中间帧复用前一帧结果
- 置信度低于 0.3 时显示 "检测中..." 而不是错误数据

---

## 5. Pet DNA JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Cat Pet DNA",
  "type": "object",
  "required": [
    "pet_type",
    "appearance",
    "breed",
    "face",
    "pose",
    "action_tendency",
    "confidence",
    "meta"
  ],
  "properties": {
    "pet_type": {
      "type": "string",
      "enum": ["cat"],
      "description": "宠物类型，固定为 cat"
    },
    "appearance": {
      "type": "object",
      "description": "外观特征（视觉算法结果）",
      "required": ["color_palette", "color_pattern", "body_geometry"],
      "properties": {
        "color_palette": {
          "type": "array",
          "description": "前N个主色 (RGB + 占比) — 视觉算法 (K-Means)",
          "items": {
            "type": "object",
            "properties": {
              "rgb": {
                "type": "array",
                "items": { "type": "integer", "minimum": 0, "maximum": 255 },
                "minItems": 3,
                "maxItems": 3
              },
              "hex": { "type": "string", "pattern": "^#[0-9a-fA-F]{6}$" },
              "ratio": { "type": "number", "minimum": 0, "maximum": 1 }
            }
          },
          "minItems": 1,
          "maxItems": 6
        },
        "color_pattern": {
          "type": "object",
          "description": "毛色斑纹类型 — 规则推断",
          "properties": {
            "type": {
              "type": "string",
              "enum": ["solid", "tabby", "bicolor", "calico", "tortie", "unknown"]
            },
            "inference_method": {
              "type": "string",
              "const": "rule-based: color_entropy + palette_analysis"
            }
          }
        },
        "primary_colors": {
          "type": "array",
          "description": "人类可读毛色描述 — 规则推断",
          "items": {
            "type": "string",
            "enum": ["white", "black", "orange", "gray", "brown", "cream", "blue", "unknown"]
          }
        },
        "body_geometry": {
          "type": "object",
          "description": "体态几何特征 — 视觉算法 (轮廓分析)",
          "properties": {
            "area_px": { "type": "number" },
            "aspect_ratio": { "type": "number", "description": "外接矩形高/宽" },
            "circularity": { "type": "number", "minimum": 0, "maximum": 1 },
            "solidity": { "type": "number", "minimum": 0, "maximum": 1 },
            "extent": { "type": "number", "minimum": 0, "maximum": 1 }
          }
        },
        "body_size_estimate": {
          "type": "string",
          "description": "体型估计 — 规则推断 (基于归一化面积)",
          "enum": ["small", "medium", "large", "unknown"]
        }
      }
    },

    "breed": {
      "type": "object",
      "description": "品种推断 — 纯规则推断 (无训练数据)",
      "properties": {
        "candidates": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": { "type": "string" },
              "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
            }
          },
          "maxItems": 3
        },
        "inference_method": {
          "type": "string",
          "const": "rule-based: color+ear+shape matching table"
        },
        "note": {
          "type": "string",
          "description": "品种推断说明，提醒用户此为表面特征推断"
        }
      }
    },

    "face": {
      "type": "object",
      "description": "面部特征",
      "properties": {
        "head_shape": {
          "type": "object",
          "description": "头型 — 视觉算法 (轮廓分析) + 规则推断",
          "properties": {
            "shape": {
              "type": "string",
              "enum": ["round", "wedge", "square", "triangular", "unknown"]
            },
            "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
          }
        },
        "ears": {
          "type": "object",
          "description": "耳朵特征 — 视觉算法 (凸点检测)",
          "properties": {
            "tips_detected": { "type": "integer", "minimum": 0, "maximum": 2 },
            "ear_type": {
              "type": "string",
              "enum": ["pointed", "rounded", "folded", "unknown"]
            },
            "ear_spread_ratio": { "type": "number", "description": "耳间距/头宽" }
          }
        },
        "eyes_detected": {
          "type": "boolean",
          "description": "是否检测到眼部 ROI"
        },
        "face_symmetry": {
          "type": "number",
          "description": "面部对称性评分 — 视觉算法",
          "minimum": 0,
          "maximum": 1
        }
      }
    },

    "pose": {
      "type": "object",
      "description": "姿态 — 视觉算法 + 规则推断",
      "properties": {
        "pose_type": {
          "type": "string",
          "enum": [
            "standing",
            "sitting",
            "lying_side",
            "curled_up",
            "stretching",
            "walking",
            "unknown"
          ]
        },
        "pose_confidence": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        },
        "body_angle": {
          "type": "number",
          "description": "躯体主轴与水平夹角 (度) — 视觉算法"
        },
        "inference_method": {
          "type": "string",
          "const": "rule-based: aspect_ratio + extent + body_angle"
        }
      }
    },

    "action_tendency": {
      "type": "object",
      "description": "动作倾向 — 基于姿态规则的间接推断",
      "properties": {
        "state": {
          "type": "string",
          "enum": ["resting", "alert", "active", "playful", "hunting", "unknown"]
        },
        "confidence": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        },
        "inference_method": {
          "type": "string",
          "const": "rule-based: pose_type + extent_range"
        },
        "evidence": {
          "type": "array",
          "description": "推断依据摘要",
          "items": { "type": "string" }
        }
      }
    },

    "confidence": {
      "type": "object",
      "description": "全局与各模块置信度",
      "properties": {
        "overall": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        },
        "segmentation_quality": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "description": "分割质量评分 — 基于 mask 边界平滑度 & 面积合理性"
        },
        "color_extraction": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        },
        "pose_estimation": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        },
        "face_analysis": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        }
      }
    },

    "meta": {
      "type": "object",
      "properties": {
        "timestamp": { "type": "string", "format": "date-time" },
        "image_size": {
          "type": "array",
          "items": { "type": "integer" },
          "minItems": 2,
          "maxItems": 2
        },
        "processing_time_ms": { "type": "number" },
        "pipeline_version": { "type": "string" },
        "commercial_safe": { "type": "boolean", "const": true }
      }
    }
  }
}
```

---

## 6. "视觉算法结果" vs "规则推断" 对照表

| Pet DNA 字段 | 来源 | 具体算法/规则 |
|-------------|------|-------------|
| `appearance.color_palette` | ✅ 视觉算法 | K-Means 聚类 (OpenCV) |
| `appearance.color_pattern.type` | ✅ 规则推断 | 颜色熵 + 主色分布决策树 |
| `appearance.primary_colors` | ✅ 规则推断 | RGB 阈值映射表 |
| `appearance.body_geometry.*` | ✅ 视觉算法 | OpenCV contour 函数 |
| `appearance.body_size_estimate` | ✅ 规则推断 | 归一化面积阈值 |
| `breed.candidates` | ✅ 规则推断 | 颜色+耳型+体型匹配表 |
| `face.head_shape` | ✅ 视觉算法 + 推断 | 轮廓分析 + 头型规则匹配 |
| `face.ears.ear_type` | ✅ 视觉算法 | 凸点检测 + 曲率分析 |
| `face.eyes_detected` | ✅ 视觉算法 | 自适应阈值 + 形态学 |
| `face.face_symmetry` | ✅ 视觉算法 | 轮廓镜像匹配度 |
| `pose.*` | ✅ 视觉算法 + 推断 | 长宽比+extent+角度决策树 |
| `action_tendency.*` | ✅ 规则推断 | pose 映射 + 姿态上下文 |

**关键设计原则**：
- 所有 **视觉算法** 都只依赖 OpenCV / scikit-learn 等现成库，无训练过程
- 所有 **规则推断** 都是可解释的、可手动调整阈值的决策树或查找表
- 每个字段都标注了 `inference_method`，保证完全透明

---

## 7. 算力估算与优化说明

### 7.1 预计算法选型基准

| 传统方案 | 深度学习替代方案 | 算力节省 |
|---------|----------------|---------|
| GrabCut (1 iter) | U²-Net (分割模型) | **~100x** |
| K-Means (K=6) | ColorCNN | **~50x** |
| 轮廓几何 | Keypoint R-CNN | **~200x** |
| 规则决策树 | 分类 CNN | **~1000x** |

### 7.2 实际部署配置

**最低配置**：
- CPU: 任何 x86/ARM 单核
- RAM: 256MB
- 无 GPU 需求
- 无需网络连接

**推荐配置**：
- Raspberry Pi 4 / 树莓派级别即可运行 15+ FPS
- 可在浏览器 WASM 中运行 (OpenCV.js)

### 7.3 可选轻量加速

- **帧跳过策略**：视频/摄像头模式下，每 N 帧做全处理，中间帧插值
- **分辨率自适应**：小物体用原始分辨率，大物体降采样
- **缓存机制**：连续帧间 ssim > 0.95 时直接复用 DNA

---

## 8. 验证 Demo 方案

### 8.1 快速验证命令

```bash
# 单张验证
python3 cat_pet_dna_pipeline.py --input cat.jpg --output ./output/

# 批量验证
python3 cat_pet_dna_pipeline.py --input ./test_images/ --batch --output ./output/

# 实时摄像头
python3 cat_pet_dna_pipeline.py --camera 0
```

### 8.2 验收标准

| 标准 | 指标 | 验证方式 |
|------|------|---------|
| 分割效果 | 猫主体被正确分离 | 肉眼检查 mask 覆盖度 |
| 颜色提取 | 主色块与实际毛色一致 | 色块条与猫毛色对比 |
| 姿态分类 | 判定结果合理 | 查看姿势标签是否准确 |
| 实时性 | CPU ≥ 10 FPS | 帧率显示 |
| 置信度 | 低质图自降置信度 | 模糊/遮挡图 confidence < 0.5 |
| 可解释性 | 每字段有 inference_method | JSON 中可见 |

### 8.3 Demo 截图预期

```
┌─────────────────────────────────────────────────┐
│  🐱 Pet DNA  v1.0                  ⏱ 48ms      │
│  ┌─────────────┐   ┌──────────────────────────┐ │
│  │             │   │  毛色: 橘色虎斑           │ │
│  │  原图 +     │   │  Palette: ■■■■■■■        │ │
│  │  轮廓叠加   │   │  姿态: 侧躺 (0.92)       │ │
│  │  耳尖标注   │   │  体型: 中等 头型: 圆    │ │
│  │  色块条     │   │  品种: [田园猫 0.6]      │ │
│  │             │   │  状态: 休息中             │ │
│  │             │   │  ──────────────────────  │ │
│  │             │   │  整体置信度: 0.85        │ │
│  └─────────────┘   └──────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

---

## 9. 项目文件结构

```
cat-pet-dna/
├── README.md                    # 本架构文档
├── cat_pet_dna_pipeline.py      # 主管线 (约 400 行 Python)
├── utils/
│   ├── segmentation.py          # GrabCut + 边缘检测分割
│   ├── color_analysis.py        # K-Means 颜色提取 + 斑纹规则
│   ├── contour_features.py      # 轮廓几何特征计算
│   ├── face_analysis.py         # 头部/面部/耳朵分析
│   ├── pose_estimation.py       # 姿态 + 动作倾向规则
│   └── dna_builder.py           # DNA JSON 组装器
├── demo/
│   ├── visualize.py             # 可视化叠加工具
│   ├── batch_report.py          # 批量 HTML 报告生成
│   └── camera_demo.py           # 实时摄像头 demo
├── test_images/                 # 测试图片 (用户提供)
├── output/                      # 输出目录
└── requirements.txt             # 依赖: opencv-python, numpy, scikit-learn