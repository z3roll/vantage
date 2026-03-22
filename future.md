# Future Architectural and Design Recommendations for Vantage

针对基于星链（Starlink）场景的流量工程（TE）系统，结合低轨卫星网络（LEO）高度动态、拓扑时变、规模庞大的特性，未来可从以下几个分层设计（Layered Design）和模块化设计（Modular Design）角度进行优化：

## 一、 分层设计建议 (Layered Design)

低轨卫星网络的 TE 不仅仅是传统的图论路由，它强依赖于物理层的约束。

### 1. 物理与轨道层 (Physical & Orbital Layer -> `world/`)
明确区分**“可预测状态”**与**“不可预测状态”**：
*   **可预测部分（Predictable）：** 卫星的星历轨迹、可见性、默认的星间链路（ISL）通断。这些可以通过时间 $t$ 提前计算。
*   **不可预测部分（Stochastic）：** 天气导致的信道衰落（影响容量）、硬件故障导致的链路中断、地面站（Gateway）的拥塞。
*   **建议：** `WorldModel` 提供两个接口：一个是基于时间的确定性拓扑，另一个是叠加了随机扰动的实际物理快照。这有助于测试 Controller 在不确定性下的鲁棒性。

### 2. 拓扑与图抽象层 (Topology & Graph Layer)
*   **痛点：** Starlink 拥有数千颗卫星，每个 Time Step 重新构建全网图模型并计算路由代价极其昂贵。
*   **建议：** 引入**“时变图（Time-Varying Graph, TVG）”**抽象（介于 `world` 和 `model` 之间）。相邻 Epoch 之间的 `NetworkSnapshot` 应通过 Delta（增减的边）来更新，而不是完全重建。这将大幅提高仿真引擎的性能。

### 3. 控制面抽象层 (Control Plane Layer -> `control/`)
TE 通常是分级的，建议支持**“双层控制结构”**：
*   **集中式 TE（Global/Centralized）：** 在地面计算中心运行，按分钟级别下发全局路由策略或流量分片比例。
*   **分布式/本地控制（Local/Distributed）：** 卫星节点本身应对微秒/毫秒级别的链路切换（Handover）或突发链路断开。
*   **建议：** `TEController` 作为集中式控制器下发 Intent，而 `forward.py`（模拟数据面）中应当包含某种本地回退（Fallback）逻辑，以模拟卫星在没有收到最新指令时基于旧表的转发行为。

## 二、 模块化设计建议 (Modular Design)

### 1. 完善遥测与反馈闭环 (Telemetry & Feedback Loop)
*   **建议：** 将遥测正式抽取为一个模块或标准化接口。`realize()` 函数除了输出 `EpochResult`，还可以提取出独立的 `TelemetryReport`（包含链路利用率、队列深度、实际 RTT）传递给下个 epoch 的 Controller。这更贴合现实，也方便基于强化学习（RL）的控制器接入。

### 2. 路由意图 (Routing Intent) 的解耦抽象
*   **建议：** 确保 `RoutingIntent` 是一个高度抽象的声明式接口，而不是具体的流表。在端到端路径易断裂的场景下，路由意图更可能是：
    *   **Segment Routing 标签栈**（指定经过某几个中继点）。
    *   **下一跳权重矩阵**（基于目的地的负载均衡）。
    *   将意图的“实例化”交给 `forward` 模块，解耦控制平面策略与数据平面转发。

### 3. 流量模型的时空特性 (Spatio-Temporal Traffic Module -> `traffic.py`)
*   **特点：** LEO 网络流量需求极度不均匀且随地球自转呈潮汐波动。
*   **建议：** `TrafficGenerator` 应提供基于真实地理坐标和时区的需求矩阵（Gravity Model），与 `world` 模块共享坐标信息，产生具备时空动态特性的流量。

### 4. 从时间步 (Time-Stepped) 到混合事件驱动 (Hybrid Event-Driven)
*   **痛点：** 关键事件（如 ISL 断开、终端切换）可能发生在 Epoch 的中途。
*   **建议：** 在 `epoch` 内部引入微观的事件队列（Event Queue）。集中式 `TEController` 依然固定频率运行，但 `forward.realize()` 可以模拟在一个 Epoch 内拓扑渐变对现有流带来的微观影响，以准确评估控制频率对系统稳定性的影响。