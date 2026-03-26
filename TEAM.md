# Vantage TE Simulator — Team Plan

## Team Roster

| # | Role | Codename | Responsibility | Priority |
|---|------|----------|---------------|----------|
| 1 | Framework Architect | `architect` | 代码整理、模块合并、规范化 | 🔴 P0 (first) |
| 2 | Reality Checker | `realist` | 从 Starlink 实际运营角度审查模型假设 | 🟡 P1 (parallel) |
| 3 | Performance Engineer | `perf` | 性能瓶颈分析与优化 | 🟠 P1 (after architect) |
| 4 | Capacity Developer | `capacity` | 容量约束、用户扩展实验 | 🟠 P2 (after architect) |
| 5 | Code Reviewer | `reviewer` | 每次修改后的质量审核 | 🔵 Gate (after each PR) |

## Execution Order

```
Phase 1 (parallel):
  [architect] → 框架整理、模块合并
  [realist]   → 现实偏差分析报告 (read-only)

Phase 2 (after architect, then review):
  [reviewer]  → 审核 architect 的改动
  [perf]      → 性能优化

Phase 3 (after perf):
  [reviewer]  → 审核 perf 的改动
  [capacity]  → 容量约束实验

Phase 4:
  [reviewer]  → 最终审核
```

## Rules
- All code managed with `uv`
- Code placement must be intentional and clean
- Every change goes through `reviewer` before merge
- Branch: `nova` (current working branch)
