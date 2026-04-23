# `FlowLevelGenerator` 流量模型

`run_1hour.py` / `main.py` 默认使用的流量生成器。**不追踪个体流，每 epoch 解析地聚合**到 per-(city, service) 的 Gbps 需求 —— 复杂度 `O(cities × profiles × services) ≈ 64K ops/epoch`。

## 1. 输入

| 输入 | 来源 | 说明 |
|---|---|---|
| 城市集合 + 用户数 `N_c` | `EndpointPopulation.from_starlink_users(starlink_users.json, world_cities.json)` | 101 国 / 1773 城市 / ~640 万用户（×USER_SCALE 缩放） |
| 服务集合 `D` 及节点位置 | `service_prefixes.json` | 13 个服务，每个含全球节点坐标 |
| 用户档案 | `user_profiles.json` | heavy / medium / light 三档 |
| Diurnal 曲线 | `diurnal_curve.json` | 24 小时活跃度（0.03–0.30） |

## 2. 用户档案（`user_profiles.json`）

每档定义"用户的行为画像"：

| 档位 | 占比 | 流到达率 r (flows/sec/user) | 流大小（Bounded Pareto） | 流速率（双模 LogNormal） |
|---|---|---|---|---|
| **heavy** | 20% | 0.10 | 1 – 5 MB，α=1.2 | 80% fast (~33 Mbps) + 20% slow (~4.5 Mbps) |
| **medium** | 50% | 0.05 | 100 KB – 5 MB，α=1.2 | 40% fast (~20 Mbps) + 60% slow (~2.7 Mbps) |
| **light** | 30% | 0.01 | 1 – 100 KB，α=1.2 | 10% fast (~12 Mbps) + 90% slow (~1.6 Mbps) |

预计算两个常数（不变）：
- `E[size]_bits` = Bounded Pareto 解析均值 × 8
- `E[rate]_gbps` = 双模 LogNormal 加权均值，并被 `bottleneck_gbps = 50 Mbps` 上限 cap

## 3. 时间维度

### 3.1 Diurnal（一天 24 小时活跃曲线）

`diurnal_curve.json` 给一天 24 小时的活跃度系数（典型范围 0.03 – 0.30）：

```
凌晨 0–4:  0.03–0.04   (低谷)
早上 8–10: 0.17–0.18
下午 16:   0.16
晚高峰 19–20: 0.30      (峰值)
深夜 23:   0.10
```

**每个城市按本地时区独立查表**：UTC 偏移 = `round(lon / 15°)`，`local_hour = (utc_hour + offset) mod 24`，再小时内线性插值。所以美国晚高峰时欧洲已在睡觉，模型自动错峰。

### 3.2 Burstiness（短时抖动）

每个城市独立维护 AR(1) 噪声状态：

```
ε ~ Normal(0, σ=0.3)
state[c] ← φ × state[c] + ε        (φ = 0.95)
burst[c] = exp(state[c])
```

强相关（φ=0.95）→ 流量短时间内波动有"惯性"，不是白噪声。倍率范围实测大致在 [0.5, 2.0]。

## 4. 空间分布（Gravity）

每个城市 `c` 对每个服务 `d` 算一组归一化权重 `w_{c,d}`：

```
dist_{c,d} = min over loc in d.locations: haversine(c, loc)
raw_{c,d}  = traffic_weight[d] / max(dist_{c,d}, 100 km)        (gravity ∝ 1/distance)
w_{c,d}    = raw_{c,d} / sum_{d'} raw_{c,d'}                     (per-city normalize)
```

直觉：城市离一个服务的最近节点越近，流量比例越大。100 km 下界避免 PoP 就在城市头顶时除零。

## 5. 单 epoch 的需求计算

对每个 (城市 c, 用户档位 p)：

```
arrival_rate = N_c × frac_p × active(local_hour, c) × r_p × burst[c]      [flows/sec]
E[duration]  = E[size]_p / E[rate]_p                                       [sec]
concurrent   = arrival_rate × E[duration]                                  (Little's Law)
n_concurrent ~ Poisson(concurrent)                                         (per-epoch noise)
demand_gbps  = n_concurrent × E[rate]_p
```

然后按 gravity 权重分配到各服务：

```
for each service d:
    flow[(c, d)] += demand_gbps × w_{c,d}
```

**Poisson 采样的意义**：当 `concurrent < 1`（小城市深夜）时，自然产出 0 流量，避免假性的"半个 flow"。当 concurrent 大时近似高斯。

## 6. 完整公式

每个 (city, service) pair 的瞬时需求：

```
demand(c, d) = Σ_p [
    Poisson(N_c × f_p × A(t,c) × r_p × B_t(c) × E[size]_p / E[rate]_p)
    × E[rate]_p
] × gravity(c, d)
```

符号汇总：

| 符号 | 含义 |
|---|---|
| `N_c` | 城市 c 的 Starlink 用户数 |
| `f_p` | 用户档位 p 的人口占比（heavy 0.2 / medium 0.5 / light 0.3） |
| `A(t, c)` | t 时刻在 c 城市本地时区的 diurnal 活跃度（[0.03, 0.30]） |
| `r_p` | 档位 p 的人均流到达率（flows/sec/user） |
| `B_t(c)` | c 城市当前 AR(1) burstiness 倍率 |
| `E[size]_p` | 档位 p 的平均流大小（bits） |
| `E[rate]_p` | 档位 p 的平均流速率（Gbps，被 50 Mbps bottleneck cap） |
| `gravity(c, d)` | c 到 d 的 1/distance 归一化权重 |

## 7. 关键设计选择

- **Analytical（解析）而非 individual-flow**：不追踪每条流的 start/end/state，所以不会随仿真时间积累状态，每 epoch 独立计算 ≈ 23 ms。代价：丢失流级 RTT-throughput 反馈，不能精确建模 TCP 慢启动等行为。
- **Palm-Khintchine 简化**：假设流到达 = Poisson 过程；多用户聚合后近似仍 Poisson。配合 Little's Law 算并发数。
- **Two-level temporal modulation**：长尺度（24 小时 diurnal） + 短尺度（AR(1) burst），两者乘性叠加。
- **Per-city RNG state**：1773 个 AR(1) 状态独立演化，所以邻近城市的 burst 不强相关 —— 跨地区错峰自然涌现。

## 8. 配置文件位置

- `src/vantage/config/user_profiles.json`
- `src/vantage/config/diurnal_curve.json`
- `src/vantage/config/starlink_users.json`（每国用户数）
- `src/vantage/config/world_cities.json`（城市坐标 + 人口权重）
- `src/vantage/config/service_prefixes.json`（13 服务的节点 / BGP 数据）
