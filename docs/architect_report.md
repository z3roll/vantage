# Vantage TE 框架整理报告

> 分支: nova | 日期: 2026-03-27

## 1. 当前结构概览

```
src/vantage/
├── analysis/            # 离线指标计算（正确）
├── common/              # 共享常量与地理工具（正确）
├── config/              # 数据预处理管线（正确）
├── control/             # TE 控制器框架
│   └── policy/          # 5 种控制策略
│       └── common/      # sat_cost 预计算 + 工具函数
├── domain/              # 不可变领域对象（正确）
├── engine/              # 引擎编排（epoch 循环）
├── traffic/             # 流量生成与人口模型
├── world/               # 物理基础设施模型
│   ├── ground/          # 地面段：延迟、知识库、基础设施
│   └── satellite/       # 卫星段：星座、拓扑、路由、可见性
├── forward.py           # 数据平面：PoP 选择 + 延迟计算
├── probe.py             # 主动探测 + 被动采样
├── engine_feedback.py   # 反馈观察者（应属于 engine/）
├── main.py              # 实验入口（地理目的地版）
└── main_service.py      # 实验入口（服务类别版）
```

**总体评价**: 架构分层清晰（domain → world → control → engine → forward），Protocol-based 设计良好。
主要问题是包根目录下散落了本应归属子包的模块，以及存在少量代码重复和过时文档。

---

## 2. 具体问题列表

### 2.1 模块位置问题

| # | 文件 | 问题 | 严重度 |
|---|------|------|--------|
| P1 | `engine_feedback.py` (根) | FeedbackObserver 是 engine 循环的一部分，应在 `engine/` 内 | 中 |
| P2 | `forward.py` (根) | 数据平面逻辑，engine/run.py 的核心依赖。可考虑移入 engine/ | 低（风险较高，暂不执行） |
| P3 | `probe.py` (根) | 测量子系统，与 engine 循环耦合。可考虑独立为 `measurement/` | 低（风险较高，暂不执行） |

### 2.2 代码重复

| # | 文件A | 文件B | 重复内容 |
|---|-------|-------|----------|
| D1 | `forward.py:37-62` (`_resolve_local_time`) | `control/policy/service_aware.py:36-42` (`_local_time`) | 时区转换逻辑完全相同 |

### 2.3 未使用的 import / 变量

| # | 文件:行号 | 问题 |
|---|-----------|------|
| U1 | `forward.py:20` | `haversine_km` 导入但未使用 (F401) |
| U2 | `world/satellite/constellation.py:19` | `EARTH_RADIUS_KM` 导入但未使用 (F401) |
| U3 | `main.py:89` | `n_pops` 变量赋值后未使用 (F841) |

### 2.4 Import 顺序问题 (I001)

共 8 处 import 块排序不规范，分布在：
- `control/policy/common/utils.py:10`
- `engine/run.py:20-25`
- `main.py:6-18`
- `main_service.py:10-31`
- 及其他 4 处

### 2.5 过时/错误的文档

| # | 文件:行号 | 问题 |
|---|-----------|------|
| S1 | `control/policy/__init__.py:4` | 引用 `vantage.policy` — 该路径不存在 |
| S2 | `control/policy/common/utils.py:7` | 引用 `candidate.py` 和 `scoring.py` — 这些文件不存在 |

### 2.6 Re-export 缺失

| # | 文件 | 问题 |
|---|------|------|
| R1 | `control/__init__.py` | 空文件，未 re-export `TEController`, `create_controller` |
| R2 | `common/__init__.py` | 未 re-export `EARTH_RADIUS_M`（在 constants.py 中定义） |
| R3 | `world/__init__.py` | 空文件，未 re-export `WorldModel` |

### 2.7 TC001/TC002 类型检查 import（67 处）

大量运行时 import 可以移入 `TYPE_CHECKING` 块。这些是 ruff 规范建议，
但对于模拟器（非库）项目影响较小，**本次不处理**。

---

## 3. 整理方案

### 3.1 本次执行（保守改动）

1. **移动 `engine_feedback.py` → `engine/feedback.py`**
   - 更新所有 import 引用（`engine/run.py`）
   - 更新 `engine/__init__.py` re-export

2. **提取共享时区工具**
   - 将 `_resolve_local_time` 提取到 `common/time.py`
   - `forward.py` 和 `service_aware.py` 改为 import

3. **清理未使用 import / 变量**
   - 删除 `forward.py` 中的 `haversine_km`
   - 删除 `constellation.py` 中的 `EARTH_RADIUS_KM`
   - 删除 `main.py` 中的 `n_pops`

4. **修复过时文档**
   - `control/policy/__init__.py` docstring
   - `control/policy/common/utils.py` docstring

5. **运行 `ruff --fix`** 自动修复 import 排序

6. **补充 re-export**
   - `control/__init__.py`: TEController, create_controller, SupportsGroundFeedback
   - `common/__init__.py`: EARTH_RADIUS_M
   - `world/__init__.py`: WorldModel

### 3.2 标记但不执行（风险较高）

- **移动 `forward.py` / `probe.py`**: 这两个模块被 `main.py`、`main_service.py`、`engine/run.py` 多处引用，移动后需更新大量路径，且与外部脚本可能有耦合
- **合并 `main.py` + `main_service.py`**: 两个实验入口有相似的 setup 代码，但它们是独立的实验脚本，合并反而增加复杂度
- **TC001/TC002 类型检查 import**: 67 处，影响文件过多，建议作为独立 PR 处理

---

## 4. 执行步骤（按优先级）

1. ✅ 移动 engine_feedback.py → engine/feedback.py，更新引用
2. ✅ 提取 _resolve_local_time 到 common/time.py
3. ✅ 清理未使用 import / 变量
4. ✅ 修复过时 docstring
5. ✅ ruff --fix 自动修复 import 排序
6. ✅ 补充 __init__.py re-export
7. ✅ 验证 ruff check + pytest
