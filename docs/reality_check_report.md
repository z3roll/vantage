# Vantage TE 模拟器现实性评估报告

**评估者**: Realist Agent
**日期**: 2026-03-27
**代码版本**: nova 分支 (commit f16f955)

---

## 执行摘要

Vantage TE 是一个模拟 Starlink 端到端流量工程的系统，涵盖卫星星座、地面段、流量模型和控制器策略。整体架构设计合理，模块解耦清晰。然而，从 Starlink 实际运营角度审视，该模拟器在多个维度存在简化假设，其中部分简化对研究结论的影响需要谨慎评估。

**严重性分级**:
- **Critical**: 可能导致研究结论无效
- **Major**: 显著影响结果精度，但不改变定性结论
- **Minor**: 对结果有轻微影响，可接受的工程简化
- **Info**: 设计选择说明，不影响研究价值

---

## A. 星座模型

### A1. 轨道参数 [Minor]

**模拟器配置** (`Starlink.xml`):
- 6 个 shell，总计 ~8000 颗卫星（72×32 + 72×22 + 48×30 + 48×27 + 36×20 + 6×58）
- 高度范围: 476–572 km
- 倾角: 53°, 43°, 70°, 97.5°

**Starlink 现实**:
- 截至 2025 年初，Starlink 在轨 ~6600+ 颗卫星（其中 ~6000 运营）
- FCC 授权的 Gen1 配置: 550km/53°(1584), 570km/70°(720), 560km/97.6°(348)
- Gen2 (v2 mini) 正在部署: 530km/43°, 525.5km/53°, 等
- XML 中的参数来自 2026-02-10 的轨道分析，合理反映了实际部署趋势

**评估**: XML 中的 shell 参数（高度、倾角、轨道面数）与 FCC 申请和实际 TLE 数据基本一致。但模拟器仅使用 `shell_id=1`（476km/53°/72×32=2304 颗），忽略了其他 5 个 shell。这意味着：
- 高纬度覆盖（70° 和 97.5° shell）被忽略
- 中纬度覆盖密度被低估（缺少 43° shell）
- 实际网络的卫星密度约为模拟的 3-4 倍

**影响**: 单 shell 模拟低估了可用路径多样性和覆盖冗余。对于 TE 策略比较而言，如果所有策略面临相同的星座，定性结论可能仍然成立，但绝对延迟值会偏高。

### A2. +Grid ISL 拓扑 [Major]

**模拟器实现** (`topology.py`):
- 每颗卫星 4 条 ISL: 2 条同轨（intra-orbit）+ 2 条跨轨（inter-orbit）
- 极区（80°-100° 倾角）禁止最后一个轨道面到第一个轨道面的 inter-orbit ISL wrap-around
- ISL 延迟 = haversine 距离 / 真空光速

**Starlink 现实**:
- **ISL 部署状态**: Starlink v1.5 及之后的卫星搭载激光 ISL（自 2021 年 9 月起）。截至 2025 年初，大部分在轨卫星已具备 ISL 能力，但 **并非所有卫星都启用了 ISL**
- **ISL 数量**: 根据 SpaceX 的 FCC 文件和公开技术描述，每颗卫星配备 **4 个激光终端**，这与 +Grid 假设一致
- **拓扑结构**: +Grid 是 Walker-delta 星座的自然选择，学术界广泛采用（Handley 2018, Bhattacherjee & Singla 2019）。SpaceX 未公开实际拓扑策略，但 +Grid 是合理的第一近似
- **ISL 容量限制**: 模拟器设置 ISL `capacity_gbps=20.0` 但从未实际使用——路由仅基于延迟，不考虑容量约束

**关键偏差**:
1. **极区 ISL 断开**: 代码中对 `is_polar`（80°-100° 倾角）禁止 wrap-around ISL。实际上，53° 倾角的 shell（模拟器使用的 shell 1）**不受此限制**。但即使在非极轨的 53° shell 上，当卫星经过极区附近（纬度 > ~50°）时，相邻轨道面间距会收缩，ISL 指向角变化剧烈，实际中可能需要断开或重建 ISL。模拟器没有建模这种**动态 ISL 连通性变化**
2. **ISL 传播介质**: 代码使用真空光速（300,000 km/s）。自由空间光学 ISL 的实际传播速度确实接近真空光速（~99.97%），这个假设是合理的
3. **ISL 建立/拆除延迟**: 激光 ISL 的指向、捕获、跟踪（PAT）过程需要时间（典型值数秒到数十秒）。模拟器假设 ISL 瞬时建立，不考虑 PAT 开销

**影响**: 对于 53° shell，极区 ISL 处理不是问题。但缺少动态 ISL 连通性变化（特别是高纬度区域 ISL 可能临时断开）会高估网络连通性。

### A3. 时间步长 [Minor]

**模拟器配置**: 星座位置计算步长 `dt_s=15.0`（15 秒），epoch 间隔 300s 或 3600s。

**评估**:
- LEO 卫星轨道周期约 ~94 分钟（5638s），15s 步长意味着每个轨道周期 ~376 个采样点
- 15s 内，550km LEO 卫星移动约 ~115 km（轨道速度 ~7.6 km/s）
- 对于 ISL 延迟变化（平滑渐变），15s 足够精细
- 对于用户终端卫星切换（handover），实际发生在 ~15-30s 级别，15s 步长可能刚好处于临界值
- epoch 间隔 300s（main.py）/ 3600s（main_service.py）是 TE 决策粒度，而非物理仿真粒度。TE 决策每 5-60 分钟更新一次在工程上是合理的

**影响**: 星座步长 15s 对延迟计算足够。TE epoch 间隔的选择取决于研究目标——5 分钟适合展示动态性，1 小时适合展示昼夜变化。

### A4. SGP4 传播器使用 [Minor]

**模拟器实现**: 使用 SGP4/Skyfield 从轨道参数生成位置，而非使用实际 TLE 数据。

**评估**: 这是标准做法。使用 Walker-delta 理想构型而非实际 TLE 数据的好处是可重复性和 shell 完整性（实际中很多卫星仍在升轨）。缺点是忽略了轨道维持误差和不完整的 shell 填充。对于 TE 策略评估，理想构型是合理的基准。

---

## B. 地面段

### B1. GS↔PoP 连接模型 [Major]

**模拟器实现** (`preprocess.py:preprocess_gs_pop_edges`):
- 每个 GS 仅连接**最近的 1 个 PoP**
- 孤立 PoP 反向连接到最近的 GS
- 回程延迟 = haversine × 1.1 detour / fiber_speed

**Starlink 现实**:
- Starlink 的地面段架构中，GS 通过光纤回程连接到**多个 PoP**
- 根据 Starlink AS14593 的 BGP 公告和 PeeringDB 数据，单个地理区域的 GS 可能连接到该区域内的 2-5 个 PoP
- GS 和 PoP 之间通常有**冗余光纤路径**
- SpaceX 使用主流 ISP（如 Zayo、Lumen/CenturyLink）的暗光纤和 Metro Ethernet 服务

**影响**:
- 1:1 映射严重限制了 TE 的灵活性。现实中，一个 GS 可以选择不同的 PoP 出口，这本身就是 TE 优化的一部分
- 对于 nearest_pop baseline，影响较小（总是选最近的）
- 对于 greedy 策略，低估了优化空间——实际中可以通过选择不同的 GS→PoP 路径来优化地面段延迟

### B2. 光纤回程延迟模型 [Major]

**模拟器提供两个模型**:

1. **HaversineDelay** (`delay.py`): `distance × 1.5 detour / 200,000 km/s`
   - detour_factor=1.5 是地面光纤路由的常用经验值
   - 但 GS→PoP 的回程距离通常较短（<500km），在这个距离上 detour_factor 变异大
   - 城市内/近郊: detour ~1.1-1.2; 跨区域: detour ~1.3-1.8

2. **FiberGraphDelay** (`delay.py`): ITU + 海缆拓扑上的 Dijkstra
   - 使用 ITU 光纤数据库和海缆数据构建全球光纤图
   - snap-to-grid 精度 0.01°（~1km），对于长距离路由足够
   - 但该模型似乎主要用于 PoP→目的地的地面延迟，而非 GS→PoP 回程

3. **ProfiledGroundDelay** (`profiled_delay.py`): 使用 traceroute 实测校准
   - 基于 google/facebook/wikipedia 的 traceroute 地面段 RTT
   - 按 PoP 聚合，通过 anchor mapping 映射到 10 个服务类别
   - **这是最接近现实的模型**

**评估**:
- GS→PoP 回程使用 `haversine × 1.1 / fiber_speed`（preprocess.py 中 detour=1.1），合理
- PoP→目的地使用 `haversine × 1.5 / fiber_speed`（delay.py 中 detour=1.5），偏保守但可接受
- ProfiledGroundDelay 使用实测值，是最佳选择。但仅有 3 个 anchor（google, facebook, wikipedia），覆盖面有限
- **关键问题**: 模型假设地面延迟是静态的（不随时间变化），但实际中光纤网络的延迟有时间依赖性（拥塞、路由变化、跨洋路径选择）

### B3. PoP 容量 [Critical]

**模拟器状态**: PoP 没有容量模型。`PoP` dataclass 仅包含 (site_id, code, name, lat_deg, lon_deg)，无容量字段。

**Starlink 现实**:
- 每个 PoP 的对外互联容量有限，取决于 peering 协议和 transit 带宽
- 根据 PeeringDB，Starlink PoP 的 peering 容量从几十 Gbps 到数百 Gbps 不等
- **热门 PoP**（如 Ashburn、Frankfurt、Tokyo）可能达到 Tbps 级别
- **边缘 PoP**（如新兴市场节点）可能仅有 10-50 Gbps
- 当 PoP 过载时，会导致丢包和排队延迟——这是 TE 需要解决的核心问题之一

**影响**: 不建模 PoP 容量意味着模拟器允许所有流量集中到延迟最低的 PoP，而不会产生拥塞。这在实际中不可能发生。greedy 策略在无容量约束下可能表现过于乐观。**如果研究目标是 TE 策略比较，缺少 PoP 容量约束会使结论不完整**。

### B4. 地面段拥塞 [Critical]

**模拟器状态**: 完全不建模地面段拥塞。所有延迟组件都是传播延迟，没有排队延迟。

**Starlink 现实**:
- GS→PoP 回程链路和 PoP 的互联端口都有带宽限制
- Starlink 用户在高峰期（当地时间 18:00-22:00）经常报告性能下降，这部分归因于地面段拥塞
- Ookla Speedtest 数据显示 Starlink 在北美高峰期下载速度可降至 25-50 Mbps（非峰值 100-200+ Mbps）
- 拥塞不仅影响吞吐量，还显著增加延迟（排队延迟可达 10-50ms）

**影响**: 与 B3 类似，缺少拥塞模型使得 TE 策略的负载均衡价值无法体现。nearest_pop 策略在拥塞场景下表现会更差，而 greedy 策略的优势会更明显。**当前模拟可能低估了 TE 优化的收益**。

---

## C. 流量模型

### C1. 服务类别划分 [Minor]

**模拟器定义** (`service.py`):
```
video_streaming, social_media, messaging, music_audio, news,
generative_ai, gaming, financial_services, ecommerce, general_web
```

**Cloudflare Radar 类别**: Cloudflare Radar 确实按类似分类报告流量（Technology, Entertainment, Shopping, etc.）。模拟器的 10 个类别是对 Radar 类别的合理映射。

**评估**: 分类粒度适中。缺少的重要类别包括:
- **VPN/Enterprise**: Starlink Business 用户显著增长
- **IoT/M2M**: Starlink 的海事、航空、车载场景
- **P2P**: BitTorrent 等仍占可观流量

对于 TE 策略研究，这些缺失类别的影响较小，因为 TE 决策主要基于延迟，而非应用类别。

### C2. Cloudflare Radar 数据代表性 [Major]

**模拟器使用**: Cloudflare Radar 提供的每小时流量权重和服务类别混合比例。

**Starlink 用户流量特征**:
- Starlink 用户 ≠ 一般互联网用户。Starlink 用户以**农村/偏远地区居民**为主，使用模式可能不同:
  - 视频流（Netflix/YouTube）占比可能更高（有限的替代娱乐选择）
  - Enterprise/VPN 在 Starlink Business 用户中占比更高
  - 游戏流量可能因延迟敏感性而偏低
- Cloudflare 看到的是**通过 Cloudflare CDN 的流量**，而非全网流量。Starlink 用户访问 Netflix/YouTube 的流量大多不经过 Cloudflare
- **Radar 的全球 profile 被直接应用到 Starlink 用户**，没有考虑 Starlink 用户的人口统计偏差

**影响**: 如果研究重点是 TE 策略的**相对比较**，流量分布的绝对准确性不是关键。但如果要声称"对 video_streaming 的 TE 优化效果是 X%"，需要对流量分布的代表性做 caveat。

### C3. 终端密度与分布 [Major]

**模拟器实现**: 从 `starlink_probes.json`（RIPE Atlas 探针）加载终端位置。

**评估**:
- RIPE Atlas 上的 Starlink 探针数量约**数百个**，而 Starlink 实际用户数 **400 万+**（截至 2025 年初）
- RIPE Atlas 探针**严重偏向欧洲和北美**（技术社区参与者部署），对亚太、非洲、南美代表性不足
- 探针位置代表了"有技术意愿部署 RIPE Atlas 的 Starlink 用户"，而非 Starlink 用户的真实地理分布
- Starlink 实际覆盖 **90+ 个国家**，但密度极不均匀: 北美 >60%，欧洲 ~20%，其余 <20%

**影响**: 终端分布直接影响流量矩阵和 PoP 负载分配。使用 RIPE Atlas 探针可能导致:
- 某些区域过度代表，某些区域完全缺失
- 流量模式（哪些 PoP 承载流量）可能不反映实际

### C4. 流量量级 [Minor]

**模拟器配置**: `base_demand_gbps=0.01`（10 Mbps per flow）。

**Starlink 现实**:
- Starlink 住宅套餐: 50-200 Mbps 下载
- 平均用户并发流量: ~5-20 Mbps（Netflix 4K ~25 Mbps，网页浏览 1-5 Mbps）
- 10 Mbps per flow 作为一个聚合流（代表一个终端到一个服务类别的总流量）是合理的

**评估**: 绝对流量量级在当前模拟中不影响结果，因为路由决策完全基于延迟，不考虑容量约束。如果未来加入容量建模，流量量级的校准将变得重要。

---

## D. 路由与决策

### D1. 终端侧 PoP 选择机制 [Major]

**模拟器实现** (`forward.py`):
```
best_pop = argmin over pop: sat_cost[ingress, pop] + ground_cost[pop, dest]
```

**Starlink 现实**:
- Starlink 终端**不直接选择 PoP**。实际流程:
  1. 终端连接到**可见卫星**（由卫星调度器分配）
  2. 流量通过 ISL 路由到一个**GS**
  3. GS 回程到其连接的 **PoP**
  4. PoP 通过 **BGP** 向互联网公告 Starlink 前缀
  5. 回程流量通过 BGP 最佳路径返回

- **关键区别**: 在现实中，PoP 选择更像是 **GS 选择的副产品**，而非独立决策。终端→卫星→ISL→GS 这条路径决定了流量到达哪个 GS，而 GS 决定了流量到达哪个 PoP
- SpaceX 的 TE 系统（推测）在卫星网络内部控制 GS 出口选择，而非在终端侧做 PoP 选择
- BGP anycast 和 ECMP 在地面段提供额外的路径多样性

**影响**: 模拟器的抽象——终端直接选 PoP——简化了实际的多层决策过程。在概念上等价于"系统选择最优出口"，但忽略了:
- GS 容量和可用性约束
- BGP 路由策略的影响
- 终端不具备全局视图的现实

### D2. 控制器预计算 + 广播模型 [Minor]

**模拟器实现**:
- 控制器预计算 `sat_cost` 和 `ground_cost` 表
- 这些表隐式"广播"给所有终端
- 终端本地执行 `argmin` 选择

**Starlink 现实**:
- SpaceX 使用**集中式 SDN 控制器**（根据 Starlink 工程师的公开演讲和专利）
- 路由决策在**卫星或地面控制器**中做出，而非在终端
- 决策更新频率: 推测秒级到分钟级
- 不需要"广播 cost tables"——控制器直接下发路由指令

**评估**: 模拟器的"precompute + broadcast"是一个有效的抽象，等价于集中式控制器做出相同的 argmin 决策。关键差异在于:
- 实际控制器可能使用更复杂的目标函数（多目标优化，包括容量约束）
- 更新频率可能更高（实时 vs epoch-based）

### D3. 测量机制 [Minor]

**模拟器实现** (`probe.py`):
- **被动采样**: 从实际流量中提取 ground_rtt（30% 采样率）
- **主动探测**: PoP 向目标 ping，每 PoP 每轮 2-10 个目标
- **目标策略**: TrafficDrivenPolicy（按流量频率排序）

**Starlink 现实**:
- SpaceX 确实在 PoP 侧进行主动探测（traceroute/ping）来测量地面延迟
- RIPE Atlas 探针提供了来自用户侧的测量数据
- SpaceX 可能使用自有的遥测数据（终端报告的 RTT、丢包率等）
- 被动测量（从流量中提取 RTT）在 CDN/ISP 中广泛使用

**评估**: 测量机制在概念上是合理的。被动采样 + 主动探测的组合是实际 TE 系统的标准做法。主要简化:
- 模拟器的"ground truth"是 HaversineDelay 估计，而非真实的网络延迟
- 没有测量噪声/误差模型
- 不考虑探测包与数据包的路由差异

### D4. 缓存驱逐策略 [Minor]

**模拟器实现** (`knowledge.py`): 提供 LRU、LFU、TTL、TrafficWeighted 四种驱逐策略，per-PoP 容量限制。

**评估**: 缓存模型设计合理，但在实际 TE 系统中:
- 地面延迟数据通常存储在**时序数据库**（如 InfluxDB/Prometheus）中，不需要 LRU 驱逐
- 容量限制更多体现在**测量覆盖率**而非存储——测量的瓶颈是探测预算和处理能力
- 缓存驱逐策略在模拟器中用于模拟"渐进学习"过程，这是一个有效的实验设计

---

## E. 缺失的关键要素

### E1. Beam 容量限制 [Critical]

**现状**: 完全未建模。

**Starlink 现实**:
- 每颗 Starlink 卫星使用**相控阵天线**，形成多个波束覆盖地面
- 每颗 v1.5 卫星总容量估计 ~17-23 Gbps（2022 年数据），v2 mini 可能更高
- 波束容量在该波束覆盖区域内的所有用户间共享
- **密集城市区域**可能出现单颗卫星容量不足的情况
- SpaceX 通过**频率复用**和**波束成形**来管理容量，但物理限制仍然存在

**影响**: 不建模 beam 容量意味着模拟器假设每颗卫星可以无限制服务所有可见用户。这与 PoP 容量缺失（B3）一起，构成了模拟器的核心局限——**延迟优化在无拥塞假设下过于乐观**。

### E2. 用户调度与多用户竞争 [Critical]

**现状**: 每个终端独立选择最高仰角卫星（`find_ingress_satellite`），不考虑其他用户。

**Starlink 现实**:
- 同一卫星 beam 下的多个用户通过 **TDMA/OFDMA** 调度共享接入带宽
- 卫星侧调度器决定时隙分配，考虑用户优先级、队列深度、信道质量
- **用户密集区域**的接入延迟显著高于稀疏区域（接入竞争 + 排队）
- Starlink 在高峰期对"去优先级"用户降速，体现了用户间的资源竞争

**影响**: 不建模用户调度意味着每个终端独立地获得最佳卫星接入，没有竞争开销。这高估了用户实际可获得的服务质量。

### E3. 天气影响 [Minor]

**现状**: 完全未建模。

**Starlink 现实**:
- Ka/Ku 波段（Starlink 使用的频段）对**降雨衰减（rain fade）**敏感
- 暴雨可导致 10-20 dB 信号衰减，严重时造成链路中断
- 降雪和云层也会影响信号质量
- SpaceX 通过自适应编码调制（ACM）和功率控制来缓解天气影响

**影响**: 天气是一个随机因素，对 TE 策略的平均性能影响有限，但对**可用性和 tail latency** 有显著影响。对于长期平均延迟的研究，忽略天气是可接受的简化。

### E4. Handover（终端切换卫星）[Major]

**现状**: 每个 epoch，终端选择当前最高仰角卫星。epoch 之间没有 handover 建模。

**Starlink 现实**:
- Starlink 终端大约每 **15-30 秒**切换一次服务卫星
- handover 过程中可能出现**短暂中断**（10-50ms 的丢包/延迟抖动）
- handover 频率取决于卫星密度、用户位置、和调度策略
- 这是 Starlink 用户报告"延迟尖刺"的主要原因之一

**影响**: 模拟器的 epoch 粒度（300s/3600s）掩盖了 handover 引起的延迟变化。对于评估**平均延迟**，影响较小。对于评估**延迟稳定性和 tail latency**，handover 是关键缺失因素。

### E5. 排队延迟 [Critical]

**现状**: 所有延迟组件都是**传播延迟**（propagation delay），没有排队延迟（queuing delay）。

**Starlink 现实**:
- 实际 E2E 延迟 = 传播延迟 + 处理延迟 + 排队延迟 + 传输延迟
- Starlink 用户报告的典型 RTT: 20-60ms（理想条件），50-150ms（拥塞/高峰）
- 排队延迟在高负载下可达 **10-50ms**，在拥塞时甚至更多
- 卫星接入段、ISL 中继、GS 回程、PoP 互联——每个环节都可能产生排队延迟

**影响**: 仅建模传播延迟会导致:
- 绝对延迟值被低估
- 不同策略在拥塞下的差异被掩盖
- **TE 的核心价值之一就是避免拥塞点**——如果不建模排队延迟，TE 的优化空间被缩小

### E6. 多 Shell / 多星座 [Minor]

**现状**: 仅使用 shell 1。

**Starlink 现实**: 6 个 shell 协同工作，不同 shell 覆盖不同纬度带。

**影响**: 单 shell 简化是常见做法（降低计算复杂度）。对于 TE 策略评估，如果所有策略使用相同星座，相对比较结论不受影响。

### E7. 接入链路仰角阈值 [Minor]

**模拟器配置**: `DEFAULT_MIN_ELEVATION_DEG = 25.0`

**Starlink 现实**:
- Starlink 终端的最低工作仰角约 **25°**（早期 v1/v1.5 圆盘天线），这与模拟器一致
- 新一代方形天线和 v2 终端可能支持更低的仰角（~20°）
- 最低仰角影响可见卫星数量和覆盖可用性

**评估**: 25° 是合理的保守选择。

---

## F. 综合评估与建议

### F1. 模拟器的定位

当前模拟器更适合定位为**延迟优化策略的概念验证**（proof-of-concept），而非精确的性能预测工具。其核心价值在于:

1. **展示 TE 策略的相对优劣**: nearest_pop vs greedy 的比较在定性上是有效的
2. **验证渐进学习的可行性**: 被动采样 + 主动探测逐步构建 ground knowledge 的设计是合理的
3. **服务类别感知 TE 的框架**: 基于 Radar 数据的时变流量模型是有创意的

### F2. 建议的优先改进

按影响大小排序:

| 优先级 | 改进项 | 复杂度 | 影响 |
|--------|--------|--------|------|
| 1 | 添加 PoP 容量约束 | 中 | Critical — 使 TE 优化有意义 |
| 2 | 添加简单排队延迟模型（M/M/1 或 M/D/1） | 中 | Critical — 反映拥塞现实 |
| 3 | GS↔PoP 多连接 | 低 | Major — 扩大 TE 优化空间 |
| 4 | Beam 容量模型（简化版：每卫星总容量限制） | 中 | Critical — 接入段瓶颈 |
| 5 | Handover 建模（每 15-30s 切换） | 中 | Major — tail latency |
| 6 | 多 shell 支持 | 高 | Minor — 更真实但非必需 |

### F3. 论文写作建议

在使用此模拟器发表论文时，建议:

1. **明确声明假设**: 无拥塞模型、单 shell、传播延迟 only
2. **定位研究贡献**: 重点放在**TE 策略设计**（算法层面），而非**性能预测**
3. **讨论局限性**: 在 Limitations 部分说明缺少 beam 容量、排队延迟、用户调度的影响
4. **使用相对比较**: 避免声称"RTT 改善 X ms"的绝对值，改用"相对改善 Y%"
5. **与实测数据对比**: 使用 RIPE Atlas 和 Ookla 数据验证延迟范围是否在合理区间

---

## 附录: 参考资料

1. **Starlink FCC filings**: FCC IBFS, SAT-MOD-20200417-00037 (Gen1), SAT-LOA-20200526-00055 (Gen2)
2. **Handley, M.** (2018). "Delay is Not an Option: Low Latency Routing in Space." HotNets.
3. **Bhattacherjee & Singla** (2019). "Network Topology Design at 27,000 km/hour." CoNEXT.
4. **Kassem et al.** (2022). "A Browser-side View of Starlink Connectivity." IMC.
5. **Michel et al.** (2022). "A First Look at Starlink Performance." IMC.
6. **Ma et al.** (2024). "Network Characteristics of LEO Satellite Constellations: A Starlink-Based Measurement from End Users." IMC.
7. **Cloudflare Radar**: https://radar.cloudflare.com/
8. **PeeringDB**: AS14593 (SpaceX/Starlink)
9. **Ookla Speedtest Global Index**: Starlink 季度报告
10. **Starlink 专利**: US Patent 11,159,228 (Satellite constellation beamforming), US Patent 11,082,131 (Inter-satellite laser communication)
