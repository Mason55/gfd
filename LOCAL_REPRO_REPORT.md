# GFD 本机复现实验记录

日期：2026-05-25
目录：`/data1/lmy/gfd`

## 1. 目标

在当前机器上完成 GFD 仓库的本地构建与实验复现，确认：

- 仓库是否能在本机编译通过
- 单卡 / 多卡 benchmark 是否能实际跑通
- SG 路径是否能通过端到端正确性验证
- README 中的实验趋势是否能在不同硬件上复现

## 2. 本机环境

### 2.1 硬件

- GPU：`8 x NVIDIA GeForce RTX 3090`
- GPU Compute Capability：`8.6`
- 单卡显存：`24576 MiB`
- CPU：`Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz`
- 在线 CPU：`64`
- NUMA：`2` 节点
- NUMA CPU 拓扑：
  - `node0`: `0-15,32-47`
  - `node1`: `16-31,48-63`
- CPU 指令集：包含 `avx512f`

### 2.2 软件

- NVIDIA Driver：`580.126.09`
- `nvidia-smi` 报告 CUDA Version：`13.0`
- `nvcc`：`12.9.86`
- `cmake`：`4.2.1`
- `gcc/g++`：`11.4.0`
- `libnuma`：已安装

## 3. 与 README 官方结果的差异

本机可以复现“实验流程”和“相对趋势”，但**不能直接复现 README 中的绝对数值**。

主要原因：

- README 使用的 GPU 是 `RTX PRO 5000 72GB (Blackwell, sm_120)`
- README 的 CPU 是 `256 核 / 2 NUMA`
- README 配置中 staging buffer 使用了更强的 hugepage/NUMA 条件
- 本机实际日志显示多处 staging 为 `hugepage=no`
- 本机 GPU 为 `RTX 3090 (sm_86)`，PCIe / 架构 / CE 行为都不同

因此应把本次结果理解为：

- 验证仓库在本机可运行
- 验证 GFD 相对 `cudaMemcpy(N)` 的性能趋势成立
- 验证多卡 / warp-spec / SG 路径可以正常执行

## 4. 为适配本机做的最小修改

为保证仓库能在当前机器与 CUDA 12.9 上跑通，做了最小兼容性修复。

### 4.1 CUDA 12.9 `cudaMemcpyBatchAsync` 签名兼容

文件：

- `examples/04_benchmark.cu`
- `examples/05_multi_gpu_benchmark.cu`

说明：

- CUDA 12.9 的 `cudaMemcpyBatchAsync` 比旧调用方式多了 `failIdx` 参数
- 原 benchmark 示例按旧签名调用，编译失败
- 处理方式：加了一个很小的兼容封装，旧逻辑不变，只补齐新参数

### 4.2 多卡示例 CPU/NUMA 绑核修正

文件：

- `include/gfd/pcie_topology.h`
- `examples/05_multi_gpu_benchmark.cu`
- `examples/06_multi_gpu_direct.cu`
- `examples/08_multi_gpu_warp_spec.cu`

说明：

- 原多卡示例写死了类似 `0-63`、`64-127` 的核心分配假设
- 这台机器在线 CPU 实际只有 `0-63`，且 NUMA cpulist 是交错布局
- 原逻辑会把 NUMA1 的 poller / worker 绑到不存在的 CPU 编号，结果不可信
- 修复后改为：
  - 从 `/sys/devices/system/node/node*/cpulist` 读取真实 cpulist
  - 根据真实 NUMA 节点为每个 GPU 分配一段合法 CPU
  - 给控制线程预留 1 个核心，避免和 poller + gather worker 抢占同一组 CPU

## 5. 构建方式

本机必须显式指定 CUDA 架构为 `sm_86`：

```bash
cd /data1/lmy/gfd
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j$(nproc)
```

结果：

- 全量目标编译通过
- 示例、benchmark、测试目标均成功生成

## 6. 实际执行过的程序

已成功运行：

- `CUDA_VISIBLE_DEVICES=0 ./build/gfd_benchmark`
- `./build/gfd_multi_gpu_benchmark`
- `./build/gfd_multi_gpu_warp_spec`
- `./build/gfd_test_sg_e2e`
- `./build/gfd_test_sg_gpu_submit`

## 7. 单卡 benchmark 结果

命令：

```bash
CUDA_VISIBLE_DEVICES=0 ./build/gfd_benchmark
```

运行环境摘要：

- GPU：`NVIDIA GeForce RTX 3090`
- 最大总传输：`128 MB`
- CPU buffer：`256 MB`
- token 布局：`2x stride` 离散锁页 host 内存
- Warmup：`15`
- Iterations：`50`

### 7.1 Group A: 固定 token_size = 4KB，变化 num_tokens

| Config | Total | Memcpy(N) P50 us | BatchAsync P50 us | GFD Queue P50 us | GFD Direct P50 us |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 x 4KB | 64KB | 42.4 | 39.8 | 118.4 | 15.3 |
| 64 x 4KB | 256KB | 155.0 | 120.0 | 115.4 | 44.6 |
| 256 x 4KB | 1MB | 618.5 | 465.9 | 1101.0 | 153.0 |
| 1024 x 4KB | 4MB | 2465.9 | 1701.2 | 1633.8 | 465.2 |
| 2048 x 4KB | 8MB | 4894.5 | 3082.3 | 2283.0 | 939.4 |
| 4096 x 4KB | 16MB | 10561.7 | 6065.7 | 2980.7 | 1835.1 |
| 8192 x 4KB | 32MB | 19757.6 | 12559.4 | 6363.5 | 3894.4 |

对应带宽：

| Config | Total | Memcpy(N) GB/s | BatchAsync GB/s | GFD Queue GB/s | GFD Direct GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 x 4KB | 64KB | 1.54 | 1.65 | 0.55 | 4.28 |
| 64 x 4KB | 256KB | 1.69 | 2.18 | 2.27 | 5.88 |
| 256 x 4KB | 1MB | 1.70 | 2.25 | 0.95 | 6.85 |
| 1024 x 4KB | 4MB | 1.70 | 2.47 | 2.57 | 9.02 |
| 2048 x 4KB | 8MB | 1.71 | 2.72 | 3.67 | 8.93 |
| 4096 x 4KB | 16MB | 1.59 | 2.77 | 5.63 | 9.14 |
| 8192 x 4KB | 32MB | 1.70 | 2.67 | 5.27 | 8.62 |

### 7.2 Group B: 固定 num_tokens = 2048，变化 token_size

| Config | Total | Memcpy(N) P50 us | BatchAsync P50 us | GFD Queue P50 us | GFD Direct P50 us |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048 x 512B | 1MB | 5015.3 | 1386.0 | 823.8 | 124.8 |
| 2048 x 1KB | 2MB | 5283.1 | 1829.1 | 1346.2 | 228.1 |
| 2048 x 2KB | 4MB | 4484.8 | 2262.7 | 1071.4 | 452.7 |
| 2048 x 4KB | 8MB | 4820.0 | 3167.4 | 1716.8 | 920.3 |
| 2048 x 8KB | 16MB | 5788.2 | 4939.7 | 2950.6 | 1835.9 |
| 2048 x 16KB | 32MB | 7620.5 | 8863.9 | 5397.0 | 3626.8 |
| 2048 x 32KB | 64MB | 10644.5 | 6778.0 | 10660.5 | 7895.8 |
| 2048 x 64KB | 128MB | 16137.8 | 12209.9 | 22011.3 | 16380.4 |

对应带宽：

| Config | Total | Memcpy(N) GB/s | BatchAsync GB/s | GFD Queue GB/s | GFD Direct GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048 x 512B | 1MB | 0.21 | 0.76 | 1.27 | 8.40 |
| 2048 x 1KB | 2MB | 0.40 | 1.15 | 1.56 | 9.19 |
| 2048 x 2KB | 4MB | 0.94 | 1.85 | 3.91 | 9.26 |
| 2048 x 4KB | 8MB | 1.74 | 2.65 | 4.89 | 9.11 |
| 2048 x 8KB | 16MB | 2.90 | 3.40 | 5.69 | 9.14 |
| 2048 x 16KB | 32MB | 4.40 | 3.79 | 6.22 | 9.25 |
| 2048 x 32KB | 64MB | 6.30 | 9.90 | 6.30 | 8.50 |
| 2048 x 64KB | 128MB | 8.32 | 10.99 | 6.10 | 8.19 |

### 7.3 Group C: 固定 token_size = 64KB，变化 num_tokens

| Config | Total | Memcpy(N) P50 us | BatchAsync P50 us | GFD Queue P50 us | GFD Direct P50 us |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 x 64KB | 1MB | 130.9 | 102.0 | 317.0 | 146.9 |
| 64 x 64KB | 4MB | 508.9 | 387.7 | 717.8 | 599.7 |
| 256 x 64KB | 16MB | 2021.7 | 1532.0 | 2559.4 | 2348.7 |
| 1024 x 64KB | 64MB | 8068.8 | 6107.9 | 10467.6 | 7838.4 |
| 2048 x 64KB | 128MB | 16158.0 | 12210.4 | 22314.2 | 16445.4 |

对应带宽：

| Config | Total | Memcpy(N) GB/s | BatchAsync GB/s | GFD Queue GB/s | GFD Direct GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 x 64KB | 1MB | 8.01 | 10.28 | 3.31 | 7.14 |
| 64 x 64KB | 4MB | 8.24 | 10.82 | 5.84 | 6.99 |
| 256 x 64KB | 16MB | 8.30 | 10.95 | 6.56 | 7.14 |
| 1024 x 64KB | 64MB | 8.32 | 10.99 | 6.41 | 8.56 |
| 2048 x 64KB | 128MB | 8.31 | 10.99 | 6.01 | 8.16 |

### 7.4 单卡结论

- 小 token 场景下，`GFD Direct` 明显优于 `Memcpy(N)`
- `BatchAsync` 相比 `Memcpy(N)` 有改善；但在 Group B 的大 token 段，`BatchAsync` 已经高过 `GFD Direct`
- `GFD Queue` 在中等规模场景有收益，但本机上整体弱于 `BatchAsync` 和 README 中的 Blackwell 结果
- 3090 上 `GFD Direct` 单卡大致稳定在 `8-9 GB/s`

## 8. 多卡 benchmark 结果

命令：

```bash
./build/gfd_multi_gpu_benchmark
```

配置：

- 每 GPU 传输：`2048 tokens x 4096 bytes = 8 MB`
- Warmup：`15`
- Iterations：`50`
- 绑定后的拓扑：
  - GPU0: `NUMA0 cores 0-7`
  - GPU1: `NUMA0 cores 8-15`
  - GPU2: `NUMA0 cores 32-39`
  - GPU3: `NUMA0 cores 40-47`
  - GPU4: `NUMA1 cores 16-23`
  - GPU5: `NUMA1 cores 24-31`
  - GPU6: `NUMA1 cores 48-55`
  - GPU7: `NUMA1 cores 56-63`

### 8.1 Test 1: Per-GPU sequential bandwidth

每张卡单独跑一遍，同一 config 下顺序测 `Memcpy(N)`、`BatchAsync`、`GFD Direct`，看单卡基线和 NUMA 差异。

| GPU | NUMA | Memcpy(N) | BatchAsync | GFD Direct |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 1.6 GB/s | 2.5 GB/s | 8.2 GB/s |
| 1 | 0 | 1.6 GB/s | 2.6 GB/s | 8.2 GB/s |
| 2 | 0 | 1.6 GB/s | 2.6 GB/s | 8.1 GB/s |
| 3 | 0 | 1.6 GB/s | 2.5 GB/s | 8.2 GB/s |
| 4 | 1 | 1.6 GB/s | 2.6 GB/s | 10.4 GB/s |
| 5 | 1 | 1.6 GB/s | 2.6 GB/s | 10.5 GB/s |
| 6 | 1 | 1.7 GB/s | 2.6 GB/s | 10.5 GB/s |
| 7 | 1 | 1.7 GB/s | 2.6 GB/s | 10.5 GB/s |

### 8.2 Test 2: Aggregate bandwidth

开 `1/2/4/8` 张卡并发跑，用 `SpinBarrier` 同步起跑；每轮取最慢卡的 P50，算总带宽，看 scaling。

| GPUs | Memcpy(N) | BatchAsync | GFD Direct |
| --- | ---: | ---: | ---: |
| 1 GPU | 1.66 GB/s | 2.73 GB/s | 7.93 GB/s |
| 2 GPUs | 3.08 GB/s | 4.26 GB/s | 7.82 GB/s |
| 4 GPUs | 4.16 GB/s | 5.57 GB/s | 7.50 GB/s |
| 8 GPUs | 5.59 GB/s | 8.16 GB/s | 9.91 GB/s |

### 8.3 Test 3: NUMA locality analysis

按 NUMA 分组跑 `GFD Direct`，把 `GPU 0-3`、`GPU 4-7`、`All 8` 分开测，确认 locality 是否拖慢聚合吞吐。

| Group | P50 | Aggregate BW |
| --- | ---: | ---: |
| NUMA 0 (GPU 0-3) | 4516.4 us | 7.43 GB/s |
| NUMA 1 (GPU 4-7) | 3371.1 us | 9.95 GB/s |
| All 8 GPUs | 6732.4 us | 9.97 GB/s |

### 8.4 Test 4: Per-GPU BW under full load

8 张卡一起压满，再逐卡记录 P50/P90 和单卡 BW，看满载时谁掉得最厉害。

| GPU | NUMA | P50 | P90 | BW |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 6597.6 us | 6761.5 us | 1.27 GB/s |
| 1 | 0 | 6686.9 us | 6768.1 us | 1.25 GB/s |
| 2 | 0 | 6685.2 us | 6772.9 us | 1.25 GB/s |
| 3 | 0 | 6682.0 us | 6765.8 us | 1.26 GB/s |
| 4 | 1 | 6420.4 us | 6731.8 us | 1.31 GB/s |
| 5 | 1 | 6494.8 us | 6742.8 us | 1.29 GB/s |
| 6 | 1 | 6522.9 us | 8047.4 us | 1.29 GB/s |
| 7 | 1 | 6489.0 us | 6738.3 us | 1.29 GB/s |

汇总：

- 单 GPU baseline：`7.93 GB/s`
- 8 GPU aggregate：`9.91 GB/s`
- 满载时 per-GPU 合计：`10.21 GB/s`

### 8.5 多卡结论

- 多卡场景下，本机 CPU 资源明显比 README 配置紧张
- 8 张 3090 可以跑通，但扩展效率一般
- 绑核修复前结果严重失真；修复后结果恢复为可解释数据

## 9. 多卡 Warp-Spec 结果

命令：

```bash
./build/gfd_multi_gpu_warp_spec
```

配置：

- 每 GPU：`8192 tokens x 16KB = 128 MB`
- 总量：`8 GPU = 1 GB`
- `64 tiles`
- `K=4 chunks/tile`
- block：`5 warps = 160 threads`
- compute：`RMSNorm + sinf`
- Warmup：`5`
- Iterations：`20`

### 9.1 Test 1: Per-GPU sequential

每张卡单独跑 `warp-spec + compute`，看单卡下传输和融合计算的基线。

| GPU | NUMA | P50 | BW |
| --- | ---: | ---: | ---: |
| 0 | 0 | 12.79 ms | 10.50 GB/s |
| 1 | 0 | 12.80 ms | 10.48 GB/s |
| 2 | 0 | 12.79 ms | 10.49 GB/s |
| 3 | 0 | 12.79 ms | 10.50 GB/s |
| 4 | 1 | 12.79 ms | 10.49 GB/s |
| 5 | 1 | 12.89 ms | 10.41 GB/s |
| 6 | 1 | 12.77 ms | 10.51 GB/s |
| 7 | 1 | 12.86 ms | 10.44 GB/s |

### 9.2 Test 2: All 8 GPUs parallel, transfer + compute

8 张卡同时跑 `warp-spec + compute`，看 transfer 和 compute 混在一起时的系统级吞吐。

| GPU | P50 | BW |
| --- | ---: | ---: |
| 0 | 45.08 ms | 2.98 GB/s |
| 1 | 45.06 ms | 2.98 GB/s |
| 2 | 45.08 ms | 2.98 GB/s |
| 3 | 45.07 ms | 2.98 GB/s |
| 4 | 44.85 ms | 2.99 GB/s |
| 5 | 44.85 ms | 2.99 GB/s |
| 6 | 44.85 ms | 2.99 GB/s |
| 7 | 44.83 ms | 2.99 GB/s |

Aggregate：

- P50：`45.08 ms`
- Aggregate BW：`23.82 GB/s`
- Sum BW：`23.88 GB/s`

### 9.3 Test 3: All 8 GPUs pure transfer

把 compute 去掉，只保留 pure transfer，目的是把 `warp-spec` 的搬运上限和计算干扰拆开。

| GPU | P50 | BW |
| --- | ---: | ---: |
| 0 | 44.39 ms | 3.02 GB/s |
| 1 | 44.38 ms | 3.02 GB/s |
| 2 | 44.37 ms | 3.02 GB/s |
| 3 | 44.37 ms | 3.03 GB/s |
| 4 | 44.17 ms | 3.04 GB/s |
| 5 | 44.14 ms | 3.04 GB/s |
| 6 | 44.18 ms | 3.04 GB/s |
| 7 | 44.09 ms | 3.04 GB/s |

Aggregate：

- P50：`44.39 ms`
- Aggregate BW：`24.19 GB/s`
- Sum BW：`24.26 GB/s`

### 9.4 Test 4: Scaling analysis

只改 active GPU 数量，用 `1/2/4/8` 张卡重复 `warp-spec + compute`，看扩展效率和总带宽。

| GPUs | P50 | Agg BW | Eff. |
| --- | ---: | ---: | ---: |
| 1 GPU | 12.77 ms | 10.51 GB/s | 100.0% |
| 2 GPUs | 23.03 ms | 11.65 GB/s | 55.5% |
| 4 GPUs | 44.70 ms | 12.01 GB/s | 28.6% |
| 8 GPUs | 45.15 ms | 23.78 GB/s | 28.3% |

### 9.5 Warp-Spec 结论

- 单卡 `warp-spec` 在本机上约 `10.5 GB/s`
- 8 卡 aggregate 能到 `23.78 GB/s`
- 相比 README 的 Blackwell 结果，扩展效率明显更低
- 但路径完整可运行，说明框架逻辑正确

## 10. SG 测试结果

### 10.1 `gfd_test_sg_e2e`

结果：

- Session 创建成功
- 提交 `4 entries`
- Kernel 正常执行
- `Data correctness: PASS (0 errors)`
- Stats：
  - `desc=4`
  - `bytes=16384`
  - `elapsed=123.77 ms`

### 10.2 `gfd_test_sg_gpu_submit`

结果：

- 启动 `3-warp kernel (4 lists x 8 entries)`
- Poller 统计：
  - `desc=32`
  - `bytes=0.12 MB`
- `Data correctness: PASS (0 errors)`

### 10.3 SG 路径结论

- SG host submit 与 GPU dynamic submit 两条路径都通过
- 至少从正确性与流程角度，本机复现是成功的

## 11. 总结

本机复现的总体结论：

1. 仓库在当前机器上已成功编译并运行。
2. 单卡、多卡、warp-spec、SG 测试路径均已打通。
3. GFD 相对 `cudaMemcpy(N)` 的核心趋势在本机成立。
4. 由于硬件与系统条件差异，本机绝对带宽明显低于 README。
5. 多卡示例原始代码对 CPU 拓扑有强假设；不修复会得到错误数据。

## 12. 后续建议

如果要继续追更高带宽，可优先检查：

- hugepage 是否能真正启用
- poller / gather worker / 控制线程绑核是否还能进一步细化
- CE channel 数是否适合当前 3090 + CPU 条件
- NUMA 分配是否能进一步减少跨节点干扰
- 是否需要专门为 `sm_86` 调整 benchmark 参数

## 13. 复现命令汇总

```bash
cd /data1/lmy/gfd

cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j$(nproc)

CUDA_VISIBLE_DEVICES=0 ./build/gfd_benchmark
./build/gfd_multi_gpu_benchmark
./build/gfd_multi_gpu_warp_spec
./build/gfd_test_sg_e2e
./build/gfd_test_sg_gpu_submit
```

## 14. 文档说明

本文件记录的是**当前机器、本次代码状态、2026-05-25 当天**的实测结果。

若后续继续修改：

- CUDA 版本
- CPU 绑核策略
- hugepage 配置
- benchmark 参数

则结果可能发生变化，应重新记录一版。

## 15. 大尺寸 Sweep

为了回答“从 `64KB` 往上继续增大 token size，GFD 是否还有收益”，额外跑了一个只看大尺寸的 sweep：

```bash
CUDA_VISIBLE_DEVICES=0 GFD_LARGE_SWEEP=1 ./build/gfd_benchmark
```

模式参数：

- 固定 `num_tokens = 128`
- token size：`64KB, 128KB, 256KB, 512KB, 1MB, 2MB`
- Warmup：`5`
- Iterations：`20`

### 15.1 结果

| Config | Total | Memcpy(N) P50 us | BatchAsync P50 us | GFD Queue P50 us | GFD Direct P50 us |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 x 64KB | 8MB | 1016.8 | 790.7 | 1299.0 | 1172.9 |
| 128 x 128KB | 16MB | 1704.0 | 1470.6 | 2509.8 | 2336.8 |
| 128 x 256KB | 32MB | 3048.9 | 2829.6 | 4835.5 | 4637.5 |
| 128 x 512KB | 64MB | 5768.7 | 5547.3 | 10241.5 | 9978.2 |
| 128 x 1MB | 128MB | 11208.6 | 10984.8 | 21478.4 | 20959.4 |
| 128 x 2MB | 256MB | 22085.1 | 21860.8 | 43174.0 | 42554.2 |

对应带宽：

| Config | Total | Memcpy(N) GB/s | BatchAsync GB/s | GFD Queue GB/s | GFD Direct GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 x 64KB | 8MB | 8.25 | 10.61 | 6.46 | 7.15 |
| 128 x 128KB | 16MB | 9.85 | 11.41 | 6.68 | 7.18 |
| 128 x 256KB | 32MB | 11.01 | 11.86 | 6.94 | 7.24 |
| 128 x 512KB | 64MB | 11.63 | 12.10 | 6.55 | 6.73 |
| 128 x 1MB | 128MB | 11.97 | 12.22 | 6.25 | 6.40 |
| 128 x 2MB | 256MB | 12.15 | 12.28 | 6.22 | 6.31 |

### 15.2 结论

这台 3090 上，进入 `64KB+` 后，`Memcpy(N)` 和 `BatchAsync` 已经明显吃到更大的单次搬运收益，而 `GFD` 的吞吐基本稳定在 `6.2-7.2 GB/s`，没有继续扩大 token size 后的额外收益。

如果只看纯吞吐，这组数据里 `GFD` **没有胜出**。它的价值更像是：

- 处理离散地址 / scatter-gather
- 需要 CPU 侧 gather、CE 流水、或 GPU 发起的情况下维持可用路径
- 为更复杂的 overlap / pipeline 结构提供基础
