# LifeOS-Agent 文档中心

文档已经按“学习、使用、工程、运维”四类整理。旧版重复教程已合并到统一教材，避免同一公式和过期训练状态散落在多个文件中。

## 学习路线

1. [训练方法全解](TRAINING_METHODS_COMPLETE_GUIDE.md)：主教材，从 JSON 样本、张量维度到 SFT/DPO/PPO/GRPO/Agent RL loss。
2. [Agent RL 全链路教程](AGENT_RL_COMPLETE_GUIDE.md)：独立进阶教材，用一条工具调用完整推导多轮数据流、张量维度、reward、advantage、KL 与 CISPO token loss。
2. [数学推导附录](../MATHEMATICAL_DERIVATIONS.md)：Softmax、交叉熵、DPO、Policy Gradient、GAE、PPO 与 GRPO 的详细推导。
3. [训练流程](TRAINING_FLOW.md)：项目训练流水线和阶段关系。
4. [最终项目报告](../FINAL_PROJECT_REPORT.md)：本项目做了什么、训练结果和问题总结。

## 使用与评测

- [用户指南](USER_GUIDE.md)：运行和对话方式。
- [Agent Demo 说明](../lifeos_agent/README.md)：Tool Calling 外部循环。
- [自测报告](../lifeos_agent/SELFTEST.md)：v0.1 工具链检查。
- [100 条评测集](../eval/README.md)：模型横向比较标准。

## 工程设计

- [系统架构](ARCHITECTURE.md)：模块边界和调用链。
- [实现记录](../IMPLEMENTATION.md)：代码改动和实现细节。
- [路线图](ROADMAP.md)：从 fake tools 到真实 LifeOS。

## 部署运维

- [远程 3090 Ti 环境](../REMOTE_SETUP.md)：服务器路径、环境和运行方式。

## 配图

- [五种训练方法总览](assets/five_stage_training_overview.svg)
- [网络张量维度](assets/network_tensor_dimensions.svg)
- [损失计算对比](assets/loss_computation_comparison.svg)
- [训练流水线](assets/lifeos_training_pipeline.svg)
- [数据 Token 长度](assets/training_data_token_lengths.svg)
- [训练指标](assets/training_metrics.svg)
- [训练目标地图](assets/training_objectives_map.svg)
- [Agent RL 多轮工具数据流](assets/agent_rl_multiturn_dataflow.svg)
- [Agent RL 单样例端到端 Trace](assets/agent_rl_end_to_end_trace.svg)
