# 流量生成 & 地面段延迟模型

## 流量生成器

| 模型 | 空间分布 | 时间模式 | 流大小 / 速率 | 用途 |
|---|---|---|---|---|
| `UniformGenerator` | all-to-all 均匀 | 每 epoch 固定 demand | 固定 | 调试 |
| `GravityGenerator` | demand ∝ 1/distance | 每 epoch 静态 | 总量按权重摊 | 简单 baseline |
| `RealisticGenerator` | Zipf 目标分布 + 渐进发现 | 每 epoch 新增少量目的地 | 固定 | 中度真实 |
| `FlowLevelGenerator`（生产用） | 按城市 gravity（基于到 service 节点最近距离） | 按城市本地时区 diurnal 曲线（24 小时）+ AR(1) burstiness | 用户类型混合（heavy / medium / light），Bounded Pareto 流大小 + 双模态 LogNormal 速率，Poisson 到达（Palm-Khintchine 算 concurrent flows，Little's law） | `run_1hour.py` / `main.py` 默认 |

## 地面段延迟（PoP → service）

`run_1hour.py` 用 `GeographicGroundDelay`：

1. 每个 `(PoP, service)` pair 提前算：
   - `min_dist = haversine(PoP, 该 service 最近节点)`
   - `distance_ms = min_dist × 1.4 / C_FIBER × 1000`
     （detour factor 1.4 模拟光纤绕路；C_FIBER ≈ 200,000 km/s 折射率 1.5）
   - `median = 5ms (base) + distance_ms`（base 模拟路由 / 处理 overhead）
2. 每次 `estimate()` 从 **LogNormal(μ = log median, σ = 0.3)** 采样得到带 jitter 的 one-way RTT。
3. Round trip = `estimate × 2`。
4. 找不到 service 落到 default = 20 ms one-way。

另两种实现备用：

- **`MeasuredGroundDelay`**：直接读 `data/probe_trace/traceroute/{service}_summary.json` 的真实 traceroute 平均（3 service × 29 PoP 覆盖）。严格，未测对会 raise `KeyError`。
- **`TracerouteReplayDelay`**：按 epoch 时钟回放真实 traceroute 时间序列（30 天 × 90s 间隔，按 hour-of-day 索引）。

## 服务节点位置来源

`src/vantage/config/service_prefixes.json`，结构：

```json
{
  "google": {
    "asns": [...],                  // BGP AS 号
    "traffic_weight": 1.0,
    "prefixes": [...],              // BGP 前缀（从 RIPE RIS 拉的）
    "locations": [
      {"city": "Buenos Aires", "country": "AR", "lat": -34.59, "lon": -58.47},
      ...                           // 该服务的物理 PoP / IXP / DC 位置
    ]
  },
  ...  // 共 13 个服务
}
```

**13 个服务**：google, netflix, amazon, meta, akamai, cloudflare, microsoft, apple, tencent, alibaba, fastly, twitter_x, discord。

每服务的 `locations` 字段是该服务在全球的物理节点坐标，**来自 PeeringDB facility 表 + 该服务 ASN 的 PeeringDB 关联**。例如 google 有 93 个节点（CDN、edge、IXP 入网点）。

- `GeographicGroundDelay` 计算 PoP → service 距离时，对每个 `(PoP, service)` 都取**该服务全部节点中离 PoP 最近的那一个**作为有效目标点（min haversine distance）。
- `FlowLevelGenerator` 算 city → service 的 gravity 权重时，也用同样的 "city 到 service 最近节点距离"。
