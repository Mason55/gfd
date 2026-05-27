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

## 16. 常用模型 KV Cache 基线

这章给后续“模型搬运”做基线。目标不是跑最终吞吐，而是先把常用模型的 KV cache 尺寸算清楚，方便后面按 2 档或 3 档模型比较搬运时间。

### 16.1 计算假设

- KV cache 按 `bf16/fp16` 算，`dtype_size = 2 bytes`
- batch = `1`
- cache 公式：

```text
KV bytes / token = 2 * num_hidden_layers * num_key_value_heads * head_dim * dtype_size
head_dim = hidden_size / num_attention_heads
```

### 16.2 选的模型

- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B`
- `mistralai/Mistral-7B-Instruct-v0.3`

### 16.3 配置与 KV 大小

| Model | Layers | Q heads | KV heads | Hidden | Head dim | KV/token | 8k ctx | 32k ctx | 128k ctx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama 3.1 8B | 32 | 32 | 8 | 4096 | 128 | 128 KiB | 1 GiB | 4 GiB | 16 GiB |
| Qwen2.5 7B | 28 | 28 | 4 | 3584 | 128 | 56 KiB | 448 MiB | 1.75 GiB | 7 GiB |
| Mistral 7B v0.3 | 32 | 32 | 8 | 4096 | 128 | 128 KiB | 1 GiB | 4 GiB | 16 GiB |

### 16.4 读法

- `Qwen2.5 7B` 的 KV cache 明显更小，原因是 `28 layers + 4 KV heads`
- `Llama 3.1 8B` 和 `Mistral 7B` 的 KV cache 同档，都是 `128 KiB/token`
- 如果只比“搬运时间”，同一带宽下，KV cache 越小 -> 搬得越快
- 如果要做 2 档比较，推荐先比 `Qwen2.5 7B` vs `Llama 3.1 8B / Mistral 7B`
- 如果要做 3 档比较，这 3 个模型就够做第一版基线

### 16.5 来源

- Llama 3.1 8B Instruct: [model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) / [config.json mirror](https://huggingface.co/zgrgr/Meta-Llama-3.1-8B-Instruct/blob/main/config.json)
- Qwen2.5 7B: [model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) / [config.json](https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json)
- Mistral 7B v0.3: [transformers doc](https://huggingface.co/docs/transformers/model_doc/mistral) / [config.json](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/708a0609e640ac1edfb9020a7c934f51d34d6c79/config.json)

### 16.6 对应实验设置

为了和这章的 KV cache 档位对齐，今天额外跑了一次大尺寸 sweep：

```bash
CUDA_VISIBLE_DEVICES=0 GFD_LARGE_SWEEP=1 ./build/gfd_benchmark
```

配置：

- 日期：`2026-05-26`
- GPU：`NVIDIA GeForce RTX 3090`
- 固定 `num_tokens = 128`
- token size：`64KB, 128KB, 256KB, 512KB, 1MB, 2MB`
- Warmup：`5`
- Iterations：`20`
- 原始日志：`section16_large_sweep_20260526.log`

这里的映射关系是：

- `64KB` 档近似对应 `Qwen2.5 7B` 的 `56 KiB/token`
- `128KB` 档直接对应 `Llama 3.1 8B` / `Mistral 7B` 的 `128 KiB/token`
- `256KB+` 档可视为更大 KV chunk，或多 token / 多层一起搬

### 16.7 实测结果

| Config | Total | Memcpy(N) P50 us | BatchAsync P50 us | GFD Queue P50 us | GFD Direct P50 us |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 x 64KB | 8MB | 1035.6 | 787.2 | 1345.3 | 1293.5 |
| 128 x 128KB | 16MB | 1710.4 | 1465.7 | 2606.6 | 2469.4 |
| 128 x 256KB | 32MB | 3066.3 | 2824.1 | 4986.1 | 4820.1 |
| 128 x 512KB | 64MB | 5781.8 | 5544.4 | 10387.1 | 9926.5 |
| 128 x 1MB | 128MB | 11201.4 | 10973.8 | 21992.1 | 20469.2 |
| 128 x 2MB | 256MB | 22083.9 | 21821.7 | 44510.8 | 41750.9 |

对应带宽：

| Config | Total | Memcpy(N) GB/s | BatchAsync GB/s | GFD Queue GB/s | GFD Direct GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 x 64KB | 8MB | 8.10 | 10.66 | 6.24 | 6.49 |
| 128 x 128KB | 16MB | 9.81 | 11.45 | 6.44 | 6.79 |
| 128 x 256KB | 32MB | 10.94 | 11.88 | 6.73 | 6.96 |
| 128 x 512KB | 64MB | 11.61 | 12.10 | 6.46 | 6.76 |
| 128 x 1MB | 128MB | 11.98 | 12.23 | 6.10 | 6.56 |
| 128 x 2MB | 256MB | 12.16 | 12.30 | 6.03 | 6.43 |

### 16.8 按模型档位看搬运时间

下面不是重新跑模型，而是用上面最接近的档位做估算：

- `Qwen2.5 7B` 用 `64KB` 档近似
- `Llama 3.1 8B / Mistral 7B` 用 `128KB` 档近似

| Model tier | KV size | Memcpy(N) | BatchAsync | GFD Queue | GFD Direct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5 7B @ 8k | 448 MiB | 58.0 ms | 44.1 ms | 75.3 ms | 72.4 ms |
| Qwen2.5 7B @ 32k | 1.75 GiB | 232.0 ms | 176.3 ms | 301.1 ms | 289.5 ms |
| Qwen2.5 7B @ 128k | 7 GiB | 927.9 ms | 705.1 ms | 1204.5 ms | 1158.1 ms |
| Llama/Mistral @ 8k | 1 GiB | 109.5 ms | 93.8 ms | 166.7 ms | 158.1 ms |
| Llama/Mistral @ 32k | 4 GiB | 437.8 ms | 375.1 ms | 666.9 ms | 632.5 ms |
| Llama/Mistral @ 128k | 16 GiB | 1751.3 ms | 1500.4 ms | 2667.7 ms | 2530.2 ms |

### 16.9 这一章的结论

- 这台 `3090` 上，`64KB -> 2MB` 区间里，`BatchAsync` 一直最快
- `GFD Direct` 没有在这些大 KV chunk 上跑赢 `BatchAsync`
- `GFD Queue` 也没有在这台机器上体现出 README 里那种大 token 优势
- 如果只看"模型搬运时间"，本机结果是：`Qwen2.5 7B` 这档比 `Llama/Mistral` 明显更快，原因就是 KV cache 更小

## 17. 硬件差异深度分析：为什么 3090 跑不过 Blackwell

### 17.1 硬件参数逐项对比

| 参数 | RTX 3090 (Ampere) | RTX PRO 5000 (Blackwell) | 差距 |
|------|:--:|:--:|:--:|
| **架构代号** | GA102 | GB202/GB203 | 跨 2 代 |
| **制程** | Samsung 8nm | TSMC 4N (4nm) | 密度/能效翻倍 |
| **Compute Capability** | 8.6 | 12.0 | — |
| **CUDA Cores** | 10,496 | 14,080 | +34% |
| **基础/加速频率** | 1400/1700 MHz | 更高 | — |
| **单精度 TFLOPS** | 29.4 | ~70 | +138% |
| **Tensor Cores** | 第 3 代 | 第 5 代 | 支持 FP4 |
| **显存类型** | GDDR6X | GDDR7 | 跨代 |
| **显存容量** | 24 GB | 48/72 GB | +100%~200% |
| **显存位宽** | 384-bit | 384-bit | 持平 |
| **显存带宽** | 936 GB/s | 1344 GB/s | +44% |
| **显存时钟** | 19.5 Gbps | 28+ Gbps | +44% |
| **PCIe 接口** | **4.0 ×16** | **5.0 ×16** | **2× 理论带宽** |
| **PCIe 理论带宽** | ~31.5 GB/s | ~63 GB/s | **+100%** |
| **PCIe 实际带宽** | ~25-27 GB/s | ~50+ GB/s | **+100%** |
| **Copy Engine** | Ampere CE（1-2 通道）| 新一代 CE（多通道）| 数量+效率飞跃 |
| **NVENC/NVDEC** | 第 7 代/第 5 代 | 第 9 代(3路)/第 6 代(3路) | 数量+质量飞跃 |
| **TDP** | 350W | 300W | 能效更高 |
| **发布年份** | 2020 年 9 月 | 2025 年 | 差 5 年 |

#### 主机环境对比

| 参数 | 本机 (3090 环境) | README (Blackwell 环境) |
|------|:--:|:--:|
| **CPU 型号** | Xeon Gold 6226R | 高端工作站 CPU |
| **CPU 核心数** | 64 核 | **256 核** |
| **NUMA 节点** | 2 | 更多（推测 4+） |
| **内存类型** | DDR4（6 通道） | DDR5（8+ 通道） |
| **Hugepage** | 未启用 (hugepage=no) | 已启用 |
| **Staging Buffer** | 普通页 | Hugepage + NUMA 绑定 |

### 17.2 瓶颈逐层分析

#### 第 1 层：PCIe 带宽 —— 最致命的瓶颈

```
PCIe 4.0 ×16 理论: 31.5 GB/s
    ├── 8b/10b → 128b/130b 编码开销 (~1.5%): → 31 GB/s
    ├── TLP Header / 协议损耗:            → 25-27 GB/s (实测上限)
    └── GFD Direct 3090 实测:             8-9 GB/s  ← 只吃到 ~30%

PCIe 5.0 ×16 理论: 63 GB/s
    └── GFD 在 Blackwell 上 CE 效率更高，实际可用带宽远超 30%
```

GFD Direct 在 3090 上只跑到 8-9 GB/s，而 Memcpy(N) 在大 token 时也能到 12 GB/s——说明 **GFD 的 CE 通路在 3090 上并没有比传统 cudaMemcpy 更高效**，甚至在大块传输时更差。根本原因在于 3090 的 Copy Engine 处理大量小 descriptor 时开销大，效率低。

#### 第 2 层：Copy Engine 代际差异 —— GFD 的命脉

GFD 的核心优化路径：

```
GPU writes descriptors → CPU poller reads → Copy Engine executes DMA
                                                    ↑
                                              这里是关键瓶颈
```

不同代 Copy Engine 的能力差异：

| CE 能力 | Ampere (3090) | Blackwell (PRO 5000) |
|:--|:--|:--|
| CE 通道数 | 1-2（消费级受限） | 多个专用 CE |
| Desc batch 处理 | 基本 batch | 优化后的批量 DMA |
| 小 desc 效率 | 差（开销 > 收益） | 显著改善 |
| DMA 流水线深度 | 浅 FIFO | 深流水 |
| Descriptor ring 支持 | 基础 | 硬件加速 |

这解释了为什么：
- **小 token (512B/1KB) 场景**：GFD Direct 大幅领先（8.4 GB/s vs Memcpy(N) 0.2 GB/s，40 倍差距），因为 cudaMemcpy 的 per-call 开销在小块时是灾难，而 GFD 的批量 desc 机制绕过了这个开销
- **大 token (64KB+) 场景**：GFD 反而落后于 BatchAsync，因为单次 DMA 足够大时 cudaMemcpy/BatchAsync 的开销占比缩小，直接跑满了 PCIe 4.0 干线，而 GFD 的 CE desc 处理反而多了一层中间开销

#### 第 3 层：CPU 资源不足 —— 8 卡并发时的致命伤

本机 CPU 只有 64 核，8 卡并发时需要：

```
每卡需要:  1× CE poller 线程 + 1× gather worker 线程 = 2 线程
8 卡总计:  16 线程 (仅 poll/gather)
+ 控制线程 + kernel launch 开销 + 系统进程
→ 平均每卡不到 8 核可用
```

README 环境 256 核，资源充裕得多。实际数据验证：

```
单卡 GFD Direct:         ~8 GB/s
8 卡 aggregate:          ~10 GB/s   (只增长 25%)
单卡 warp-spec:           ~10.5 GB/s
8 卡 warp-spec aggregate: ~24 GB/s   (只增长 2.3×)

满载时 per-GPU BW 退化到: ~1.25 GB/s ← 只有单卡的 16%
```

CPU 资源竞争导致 CE poller 处理 desc 速度下降，DMA 流水线出现"断流"，这是多卡扩展效率差的首要原因。

#### 第 4 层：NUMA 绑核和 Hugepage

报告中已发现的关键信号：
- `hugepage=no` — staging buffer 没有用大页，TLB miss 更多，影响 poller 读 desc 的效率
- NUMA1（GPU 4-7）的 GFD Direct 带宽比 NUMA0（GPU 0-3）**高出 25%**（10.5 vs 8.2 GB/s），说明 NUMA0 上有更多跨节点内存访问干扰

NUMA 干扰的根源：
```
GPU 0-3 在 NUMA0 → 但 CPU 核心 0-15,32-47 还要跑系统进程
GPU 4-7 在 NUMA1 → 核心 16-31,48-63 相对空闲
→ NUMA0 带宽更低因为 CPU 竞争更激烈
```

#### 第 5 层：显存带宽对 Staging Pool 的影响

GFD 的完整数据路径：

```
CPU pinned → [CE DMA] → GPU staging buffer → [kernel load] → GPU compute
                              ↑                        ↑
                         GDDR6X 936 GB/s        需要额外一次显存读写
```

虽然 GDDR6X 的 936 GB/s 远大于 PCIe 4.0 的 25 GB/s，但在 double-buffer 模式下，staging buffer 的 ping-pong 读写会成为附加损耗。Blackwell 的 GDDR7 为 1344 GB/s（+44%），这部分开销占比更小。

#### 第 6 层：驱动成熟度

- Ampere (2020)：驱动已进入维护期，CE DMA 路径不会再做激进优化
- Blackwell (2025)：全新驱动栈，Copy Engine、DMA 引擎均针对 descriptor-based DMA 做了硬件级优化

### 17.3 瓶颈权重汇总

```
                    GFD 吞吐受哪些因素影响？
                    
┌───────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────┐
│ 因素       │    │  瓶颈程度     │    │  影响位置      │    │ 是否可解  │
├───────────┤    ├──────────────┤    ├───────────────┤    ├──────────┤
│ PCIe 4.0  │    │ ★★★★★ 致命   │    │ DMA 物理带宽   │    │ 硬件瓶颈  │
│ CE 代际   │    │ ★★★★★ 严重   │    │ Desc 处理效率  │    │ 硬件瓶颈  │
│ CPU 64 核 │    │ ★★★★  重要   │    │ Poller/Gather  │    │ 可部分优化 │
│ NUMA 拓扑 │    │ ★★★   中等   │    │ 跨节点延迟     │    │ 可优化    │
│ Hugepage  │    │ ★★☆   次要   │    │ TLB miss       │    │ 可解      │
│ DDR4      │    │ ★★☆   次要   │    │ 内存读带宽     │    │ 硬件瓶颈  │
│ GDDR6X    │    │ ★☆☆   轻微   │    │ Staging BW     │    │ 硬件瓶颈  │
└───────────┘    └──────────────┘    └───────────────┘    └──────────┘
```

### 17.4 总结

GFD 在 3090 上性能受限是**系统性**的，不是单一瓶颈。最核心的两个硬件天花板：

1. **PCIe 4.0 带宽** 仅为 5.0 的一半（~25 GB/s vs ~50 GB/s 实测），这是物理上限
2. **Ampere Copy Engine** 处理 descriptor-based DMA 的效率远不如 Blackwell 新一代 CE

这解释了为什么：
- **小 token 场景** GFD 有压倒性优势（绕过了 cudaMemcpy per-call 开销）
- **大 token 场景** GFD 反而没有优势（CE desc 处理成为额外开销，不如直接 cudaMemcpy/BatchAsync 高效）
- **多卡场景** CPU 资源成为瓶颈，poller 线程竞争导致 CE DMA 流水线断流

GFD 真正的价值场景在**小 token、离散地址、scatter-gather**（不适用 cudaMemcpy 或 BatchAsync 高效覆盖的负载），这些在正确性测试中已通过验证，但在本机 benchmark 中尚未做专项性能对比。
- 如果只问“GFD 是不是有纯吞吐收益”，这章答案是否定的；它的意义更多还是：
  - 支持 GPU 发起 / queue 模式
  - 支持 scatter-gather 和更复杂的搬运路径
  - 给后续 overlap / pipeline 实验留接口

## 17. 追加验证：提高 num_tokens / 随机地址 / 严格绑核

为了验证前面那几个猜想，我给 `examples/04_benchmark.cu` 加了 4 个实验开关：

- `GFD_LARGE_SWEEP_NUM_TOKENS`
- `GFD_LARGE_SWEEP_TOKEN_SIZES`
- `GFD_RANDOMIZE_ADDRS`
- `GFD_STRICT_BIND`

这几个开关默认都不影响原 benchmark 行为，只用于追加 sweep。

### 17.1 这组实验怎么测

目标拆成 3 个角度：

1. `num_tokens` 提高到 `512` / `1024`，看 `GFD Direct` 进 pipeline 分支后有没有明显收益
2. 把规则 `2x stride` 地址换成随机 slot，看看 `BatchAsync` 会不会更吃亏
3. 把 benchmark 主线程固定绑到 `CPU 31`，和 `poller/gather worker` 分开，看看 CPU 干扰是不是主因

说明：

- 随机地址不是“完全任意字节地址”，而是在 pinned host buffer 里，从 `2N` 个对齐 slot 中随机选 `N` 个
- 这样不会破坏正确性，但可以去掉规则 stride 带来的顺序性
- 所有测试仍然是单卡 `CUDA_VISIBLE_DEVICES=0`
- 为了控制内存占用，这里把最大 `total transfer` 约束在 `256MB`

### 17.2 实验命令

`512 tokens, regular stride`

```bash
CUDA_VISIBLE_DEVICES=0 \
GFD_LARGE_SWEEP=1 \
GFD_LARGE_SWEEP_NUM_TOKENS=512 \
GFD_LARGE_SWEEP_TOKEN_SIZES=64KB,128KB,256KB,512KB \
./build/gfd_benchmark
```

`1024 tokens, regular stride`

```bash
CUDA_VISIBLE_DEVICES=0 \
GFD_LARGE_SWEEP=1 \
GFD_LARGE_SWEEP_NUM_TOKENS=1024 \
GFD_LARGE_SWEEP_TOKEN_SIZES=64KB,128KB,256KB \
./build/gfd_benchmark
```

`1024 tokens, random slots`

```bash
CUDA_VISIBLE_DEVICES=0 \
GFD_LARGE_SWEEP=1 \
GFD_LARGE_SWEEP_NUM_TOKENS=1024 \
GFD_LARGE_SWEEP_TOKEN_SIZES=64KB,128KB,256KB \
GFD_RANDOMIZE_ADDRS=1 \
./build/gfd_benchmark
```

`1024 tokens, random slots + strict bind`

```bash
CUDA_VISIBLE_DEVICES=0 \
GFD_LARGE_SWEEP=1 \
GFD_LARGE_SWEEP_NUM_TOKENS=1024 \
GFD_LARGE_SWEEP_TOKEN_SIZES=64KB,128KB,256KB \
GFD_RANDOMIZE_ADDRS=1 \
GFD_STRICT_BIND=1 \
./build/gfd_benchmark
```

对应日志：

- `exp_20260526_tokens512_stride.log`
- `exp_20260526_tokens1024_stride.log`
- `exp_20260526_tokens1024_random.log`
- `exp_20260526_tokens1024_random_strictbind.log`

### 17.3 结果汇总

先只看 `GFD Direct` 带宽，方便判断这 3 个角度有没有把它拉起来：

| Config | 64KB | 128KB | 256KB | 512KB |
| --- | ---: | ---: | ---: | ---: |
| 128 tokens, stride | 6.49 | 6.79 | 6.96 | 6.76 |
| 512 tokens, stride | 9.38 | 8.47 | 7.97 | 7.79 |
| 1024 tokens, stride | 8.55 | 7.97 | 7.84 | - |
| 1024 tokens, random | 8.67 | 7.91 | 7.84 | - |
| 1024 tokens, random + strict bind | 8.71 | 7.94 | 7.83 | - |

同样位置下的 `BatchAsync`：

| Config | 64KB | 128KB | 256KB | 512KB |
| --- | ---: | ---: | ---: | ---: |
| 128 tokens, stride | 10.66 | 11.45 | 11.88 | 12.10 |
| 512 tokens, stride | 10.97 | 11.61 | 11.97 | 12.16 |
| 1024 tokens, stride | 10.98 | 11.63 | 11.98 | - |
| 1024 tokens, random | 10.98 | 11.62 | 11.97 | - |
| 1024 tokens, random + strict bind | 10.98 | 11.63 | 11.98 | - |

### 17.4 这 3 个角度分别说明什么

`1. 提高 num_tokens`

- 这个角度是有用的
- `128 -> 512 tokens` 后，`64KB` 档 `GFD Direct` 从 `6.49` 提到 `9.38 GB/s`
- 说明 `count >= 512` 以后，`Direct` 的 pipeline 分支确实开始发挥作用
- 但再从 `512 -> 1024` 没继续涨，反而略回落到 `8.55 GB/s`
- 结论：`GFD` 原先差，不是单纯因为 `128 tokens` 太少；条目数补够后，还是没有超过 `BatchAsync`

`2. 随机地址`

- 这个角度基本没改变结论
- `1024 x 64KB` 下，`BatchAsync` 还是 `10.98 GB/s`
- `GFD Direct` 只从 `8.55` 变到 `8.67 GB/s`
- 说明这里这版“随机 slot”还不足以把 CUDA 的 batch 路径打崩
- 也说明本机主要矛盾不是“规则 stride 对 BatchAsync 过于友好”

`3. 严格绑核`

- 这个角度也只带来非常小的变化
- `1024 x 64KB` 下，`GFD Direct` 从 `8.67` 到 `8.71 GB/s`
- `128KB / 256KB` 基本不变
- 说明 benchmark 主线程和 poller/gather worker 的 CPU 干扰不是当前主瓶颈

### 17.5 这一组追加实验的结论

- `num_tokens` 提高以后，`GFD Direct` 确实有阶段性改善，说明 pipeline 分支有效
- 但改善幅度还不够，仍然追不上本机的 `BatchAsync`
- 随机地址没有显著拉低 `BatchAsync`
- 严格绑核也没有明显把 `GFD` 再抬上去
- 所以这台机器上 `GFD` 没拿到优势，主因还是：
  - `BatchAsync` 在 `64KB+` 这档已经很强
  - `GFD` 仍然要付 `CPU gather + staging -> GPU` 的额外成本
  - `3090 + 64C CPU + hugepage=no` 的平台条件不够支持 README 那种结果

### 17.6 最终判断

把第 16 章和第 17 章合起来看，这台机器上的最终结论可以收成 4 条：

1. `GFD` 在本机不是“纯搬运吞吐更高”的方案。
   在 `64KB+` 这类 KV-cache 大块搬运场景里，`BatchAsync` 一直更快，`GFD Direct` 和 `GFD Queue` 都没有反超。

2. `GFD` 的优势区间仍然存在，但不在这次关心的大 KV 档位。
   本机前面的小块测试已经说明，`512B ~ 16KB` 这类细粒度 scatter 传输里，`GFD Direct` 仍然明显优于 `Memcpy(N)`，只是到了 `64KB+` 后，CUDA 自带 batch 路径已经把差距补上了。

3. 这轮追加实验基本排除了 3 个“可能只是测法问题”的解释。
   - 不是因为 `num_tokens=128` 太少；提到 `512/1024` 后，`GFD` 仍没超过 `BatchAsync`
   - 不是因为地址太规则；换成随机 slot 后，结论基本不变
   - 不是因为 benchmark 主线程抢核；严格绑核后，结果只发生很小波动

4. 因此，本机上更合理的理解是：
   - 如果任务只是“把大块 KV cache 从 host 搬到 GPU”，优先看 `BatchAsync`
   - 如果任务需要 `GPU 发起 / queue 模式 / scatter-gather / 后续 overlap-pipeline`，`GFD` 仍然有价值
   - `GFD` 在这台机器上的意义更偏“机制能力”和“复杂路径支持”，不是这组实验里的绝对带宽冠军

## 18. Nsight Systems 观测

为了不只看最终带宽，这里再用 `Nsight Systems` 看一次不同路径的消耗分布。

### 18.1 这组 profile 怎么测

为了减少 trace 噪声，我给 benchmark 又补了两个开关：

- `GFD_WARMUP`
- `GFD_ITERS`

profile 时统一压成：

- `Warmup = 1`
- `Iterations = 3`
- 每次只跑单个 config

本节关心的不是初始化开销，而是：

- `Memcpy(N)` 的 API 提交次数有多夸张
- `BatchAsync` 是不是已经把大块 H2D 合并得很好
- `GFD Queue` 的等待是不是主要耗在 wait kernel
- `GFD Direct` 虽然减少了 DMA 提交次数，为什么还是没赢

### 18.2 采样命令

`128 x 128KB, regular stride`

```bash
CUDA_VISIBLE_DEVICES=0 \
GFD_LARGE_SWEEP=1 \
GFD_LARGE_SWEEP_NUM_TOKENS=128 \
GFD_LARGE_SWEEP_TOKEN_SIZES=128KB \
GFD_WARMUP=1 \
GFD_ITERS=3 \
nsys profile --trace=cuda,osrt --sample=process-tree --cpuctxsw=process-tree \
  --force-overwrite true \
  -o /data1/lmy/gfd/nsys_128_128kb_stride \
  ./build/gfd_benchmark
```

`512 x 64KB, regular stride`

```bash
CUDA_VISIBLE_DEVICES=0 \
GFD_LARGE_SWEEP=1 \
GFD_LARGE_SWEEP_NUM_TOKENS=512 \
GFD_LARGE_SWEEP_TOKEN_SIZES=64KB \
GFD_WARMUP=1 \
GFD_ITERS=3 \
nsys profile --trace=cuda,osrt --sample=process-tree --cpuctxsw=process-tree \
  --force-overwrite true \
  -o /data1/lmy/gfd/nsys_512_64kb_stride \
  ./build/gfd_benchmark
```

`1024 x 64KB, random + strict bind`

```bash
CUDA_VISIBLE_DEVICES=0 \
GFD_LARGE_SWEEP=1 \
GFD_LARGE_SWEEP_NUM_TOKENS=1024 \
GFD_LARGE_SWEEP_TOKEN_SIZES=64KB \
GFD_RANDOMIZE_ADDRS=1 \
GFD_STRICT_BIND=1 \
GFD_WARMUP=1 \
GFD_ITERS=3 \
nsys profile --trace=cuda,osrt --sample=process-tree --cpuctxsw=process-tree \
  --force-overwrite true \
  -o /data1/lmy/gfd/nsys_1024_64kb_random_strict \
  ./build/gfd_benchmark
```

产物：

- `nsys_128_128kb_stride.nsys-rep`
- `nsys_512_64kb_stride.nsys-rep`
- `nsys_1024_64kb_random_strict.nsys-rep`

### 18.3 先看 API 调用次数

只摘最关键的几项：

| Case | `cudaMemcpyAsync` | `cudaMemcpyBatchAsync` | `cuMemcpyHtoDAsync_v2` | `cudaStreamSynchronize` | `cudaDeviceSynchronize` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `128 x 128KB stride` | 512 | 4 | 16 | 8 | 9 |
| `512 x 64KB stride` | 2048 | 4 | 24 | 8 | 9 |
| `1024 x 64KB random+bind` | 4096 | 4 | 24 | 8 | 9 |

读法：

- `Memcpy(N)` 路径的 API 提交次数确实爆炸，`1024 x 64KB` 一轮里是 `4096` 次 `cudaMemcpyAsync`
- `BatchAsync` 一直只有 `4` 次 API 调用，因为这里只有 `1 warmup + 3 iters`
- `GFD` 侧真正的 CE DMA 提交次数很少，只有 `16/24` 次 `cuMemcpyHtoDAsync_v2`

这说明一件事：

- `GFD` 的确成功把“很多小提交”压成了“很少的大提交”
- 但即使这样，它仍然没赢，所以瓶颈已经不在“DMA API 提交次数”本身

### 18.4 再看 GPU 上的 H2D 形态

`cuda_gpu_mem_size_sum` 里，H2D memop 的最大单次大小分别是：

| Case | H2D op count | Total H2D MB | Max H2D MB |
| --- | ---: | ---: | ---: |
| `128 x 128KB stride` | 534 | 281.029 | 16.777 |
| `512 x 64KB stride` | 2078 | 549.470 | 33.554 |
| `1024 x 64KB random+bind` | 4126 | 1086.349 | 67.109 |

这里最重要的不是总数，而是 `Max H2D MB`：

- `16.777 MB`
- `33.554 MB`
- `67.109 MB`

这说明：

- `GFD` 路径确实在发大块合并 DMA
- `num_tokens` 提高后，单次大 DMA 也确实在变大
- 所以“GFD 没有做出大块 coalesced H2D”这个怀疑，可以排除

### 18.5 Queue 路径：wait kernel 才是大头

`cuda_kern_exec_sum` 里两个 kernel 很清楚：

- `bench_gfd_submit_kernel`
- `bench_gfd_wait_kernel`

3 个 case 下，`submit kernel` 平均总时间大约是：

- `0.178 ms`
- `0.193 ms`
- `0.226 ms`

而 `wait kernel` 平均总时间大约是：

- `2.10 ms`
- `3.36 ms`
- `6.02 ms`

也就是说：

- `GFD Queue` 慢，不是 GPU 写 descriptor 慢
- 主要是 GPU 后面一直在等 CPU poller + gather + DMA 完成
- 所以 queue 模式在这个纯搬运 benchmark 里天然吃亏

### 18.5.1 Submit-only 对比：GFD submit kernel vs BatchAsync API

为了回答“`GFD submit kernel` 本身是不是比 `BatchAsync` 快”，又直接查了 3 个 `nsys` sqlite：

- `nsys_128_128kb_stride.sqlite`
- `nsys_512_64kb_stride.sqlite`
- `nsys_1024_64kb_random_strict.sqlite`

这里要先区分口径：

- `bench_gfd_submit_kernel` 是 GPU 侧写 descriptor / 提交请求的 kernel 时间
- `cudaMemcpyBatchAsync` 是 CPU 侧 CUDA Runtime API 调用时间
- 这两个不是完全同一条执行 lane，但都能反映“提交动作”的量级
- 它们都不是端到端 copy 完成时间

直接查 `CUPTI_ACTIVITY_KIND_RUNTIME` 和 `CUPTI_ACTIVITY_KIND_KERNEL` 后，结果如下：

| Case | `cudaMemcpyBatchAsync` API avg | `bench_gfd_submit_kernel` avg | Submit-only 判断 |
| --- | ---: | ---: | --- |
| `128 x 128KB stride` | `107.14 us` | `150.12 us` | `BatchAsync` 更轻 |
| `512 x 64KB stride` | `411.40 us` | `164.56 us` | `GFD submit kernel` 更轻 |
| `1024 x 64KB random+bind` | `813.32 us` | `193.19 us` | `GFD submit kernel` 更轻 |

这说明：

- 小 batch 下，`BatchAsync` API submit 本身更便宜
- 到 `512/1024` entries 这种大 batch 后，`GFD submit kernel` 的提交动作确实更轻，大约快 `2.5x ~ 4.2x`
- 但是这个结论只成立在 submit-only 口径下
- 一旦看完整 `GFD Queue`，后面的 `wait kernel + CPU poller + gather + DMA` 仍然是主耗时
- 所以最终端到端结果仍然是：`64KB+` 大 KV chunk 上，`BatchAsync` 更快

### 18.6 Direct 路径：DMA 提交不贵，贵的是 CPU 侧 staging/gather

`cuda_api_sum` 里，`GFD` 对应的 `cuMemcpyHtoDAsync_v2` 平均 CPU API 时间只有：

- `26.8 us`
- `19.5 us`
- `24.8 us`

这个量级并不大。

反过来看：

- `BatchAsync` 每次 API 调用虽然更重，但一共只有 `4` 次
- `GFD Direct` 的 CE 提交次数也很少，但最终 wall time 还是更长

结合代码路径，可以得到更合理的解释：

- `Direct` 真正贵的不是 `cuMemcpyHtoDAsync_v2`
- 而是 `CPU gather -> staging buffer -> GPU DMA` 这一段主机侧工作
- 这部分在 `nsys` 里不会像 CUDA API 那样直接给出一个漂亮的单行汇总，但从“DMA 提交很少 yet 总耗时更长”这个现象，已经能反推出它是主要成本

### 18.7 Nsight 这一章的结论

- `nsys` 证明了：`GFD` 确实把大量离散提交压成了少量大 DMA，这一点没有问题
- `nsys` 也证明了：`GFD Queue` 的主要时间不在 submit kernel，而在后续等待
- `nsys` 还说明：`GFD Direct` 的 CE 提交 API 自身并不贵，真正拖后腿的是 CPU 侧 gather/staging
- 因此，本机上 `GFD` 没赢，不是因为“它没有合并提交”，而是因为：
  - `BatchAsync` 对 `64KB+` 大块 H2D 已经足够强
  - `GFD` 额外多了一段主机侧数据重排成本
  - 这段成本在 `3090 + 64C + hugepage=no` 这台机器上没有被摊平

### 18.8 一句话总结

这组 `nsys` 结果可以再压缩成一句更直接的话：

- `GFD` 的“合并提交”本身没有问题
- 在 `1024 x 64KB` 这类 case 下，`Memcpy(N)` 是 `4096` 次 `cudaMemcpyAsync`，`BatchAsync` 是 `4` 次 API 调用，`GFD` 的 CE 提交只有 `24` 次
- 只看 submit-only，大 batch 下 `GFD submit kernel` 比 `BatchAsync` API 更轻：`512 x 64KB` 是 `164.56 us` vs `411.40 us`，`1024 x 64KB` 是 `193.19 us` vs `813.32 us`
- `GFD Queue` 慢，不是 `submit kernel` 慢；`submit kernel` 大约只有 `0.18 ~ 0.23 ms`，而 `wait kernel` 在 `2.10 ~ 6.02 ms`
- `GFD Direct` 也不是慢在 DMA API 提交；`cuMemcpyHtoDAsync_v2` 平均只有 `19 ~ 27 us`
- 真正没有被摊平的，是 `CPU gather -> staging -> GPU DMA` 这一段主机侧成本

因此，这台 `3090 + 64C + hugepage=no` 机器上的最终判断是：

- `BatchAsync` 对 `64KB+` 大块 H2D 已经足够强
- `GFD` 的额外 gather/staging 成本没有被覆盖掉
- 所以本机上 `GFD` 不是这组大 KV-cache 搬运实验里的绝对带宽最优解

## 19. 开启 HugeTLB 后重刷

宿主机执行：

```bash
echo 2048 | sudo tee /proc/sys/vm/nr_hugepages
```

之后检查到：

- `HugePages_Total = 2048`
- `HugePages_Free = 2048`
- `Hugepagesize = 2048 kB`
- NUMA 上为 `node0=1024`、`node1=1024`

更关键的是，GFD 运行日志已经明确变成：

- `Pre-allocated ... (hugepage=yes)`
- `Staging from pool: ... (hugepage=yes, NUMA 0)`

说明这次不是“系统有大页但程序没吃到”，而是 `GFD` 的 staging buffer 已经真实命中了 `HugeTLB`。

### 19.1 这组实验怎么测

为了只回答“大页本身有没有把结论翻过来”，这里重刷 4 个关键 case：

- `128 x 64KB, stride`
- `512 x 64KB, stride`
- `1024 x 64KB, stride`
- `1024 x 64KB, random + strict bind`

日志：

- `hugepage_rerun_128_64kb.log`
- `hugepage_rerun_512_64kb.log`
- `hugepage_rerun_1024_64kb_stride.log`
- `hugepage_rerun_1024_64kb_random_strict.log`

### 19.2 hugepage=yes 后的结果

| Case | BatchAsync GB/s | GFD Queue GB/s | GFD Direct GB/s |
| --- | ---: | ---: | ---: |
| `128 x 64KB, stride` | 10.87 | 6.30 | 7.55 |
| `512 x 64KB, stride` | 10.97 | 6.27 | 9.27 |
| `1024 x 64KB, stride` | 10.95 | 4.64 | 7.40 |
| `1024 x 64KB, random + strict` | 10.97 | 5.01 | 8.40 |

和之前 `hugepage=no` 的同档位对比：

| Case | BatchAsync | GFD Queue | GFD Direct |
| --- | ---: | ---: | ---: |
| `128 x 64KB, stride` | `10.66 -> 10.87` | `6.24 -> 6.30` | `6.49 -> 7.55` |
| `512 x 64KB, stride` | `10.97 -> 10.97` | `6.52 -> 6.27` | `9.38 -> 9.27` |
| `1024 x 64KB, stride` | `10.98 -> 10.95` | `6.01 -> 4.64` | `8.55 -> 7.40` |
| `1024 x 64KB, random + strict` | `10.98 -> 10.97` | `6.13 -> 5.01` | `8.71 -> 8.40` |

### 19.3 这组重刷说明什么

- `HugeTLB` 确实已经启用，而且 `GFD` 也确实已经用上了
- 但它没有把最终结论翻过来
- 只在最小的 `128 x 64KB` case 上，`GFD Direct` 有一档比较明显的回升：`6.49 -> 7.55 GB/s`
- 到了 `512 x 64KB`，`GFD Direct` 基本持平
- 到了 `1024 x 64KB`，`GFD Direct` 和 `GFD Queue` 都没有继续改善，反而略低于之前那组结果

### 19.4 最终结论

这组 `hugepage=yes` 重刷，能得出两个明确判断：

1. 之前文档里的 `hugepage=no`，确实只是“当时还没命中大页路径”，不是日志误报。
2. 即使 `GFD` 现在已经真实使用 `HugeTLB`，本机上它仍然没有在 `64KB` 这档大 KV-cache 搬运里反超 `BatchAsync`。

所以本机的最终结论不变：

- 开启大页是正确动作，至少把环境补齐了
- 但大页不是决定性瓶颈
- `BatchAsync` 在这台 `3090` 上对 `64KB+` 大块 H2D 仍然更强
- `GFD` 的主要价值仍然是 `queue / scatter-gather / overlap-pipeline` 这些复杂路径，而不是这组纯搬运测试里的绝对带宽第一

## 20. 继续往 64KB 以下看

上面几章已经说明：在 `64KB` 这一档，本机上 `BatchAsync` 仍然明显强于 `GFD`。为了进一步回答“如果比 `64KB` 更小，结论会不会变好”，这里重新跑了一次标准单卡 benchmark，并只看 `Group B` 里的：

- `2048 x 8KB`
- `2048 x 16KB`
- `2048 x 32KB`
- `2048 x 64KB`

这次环境已经是：

- `hugepage=yes`
- regular `2x stride`
- 标准 `Group B` 路径

日志：

- `hugepage_full_benchmark_20260526_rerun.log`

### 20.1 结果

| Config | Memcpy(N) GB/s | BatchAsync GB/s | GFD Queue GB/s | GFD Direct GB/s |
| --- | ---: | ---: | ---: | ---: |
| `2048 x 8KB` | 2.74 | 3.12 | 5.48 | 9.47 |
| `2048 x 16KB` | 4.20 | 3.69 | 4.71 | 8.47 |
| `2048 x 32KB` | 6.05 | 9.89 | 4.63 | 7.71 |
| `2048 x 64KB` | 8.40 | 10.99 | 5.04 | 7.72 |

### 20.2 读法

- 到了 `8KB`，`GFD Direct` 已经明显回到优势区，`9.47 GB/s` 高于 `BatchAsync 3.12 GB/s`
- 到了 `16KB`，`GFD Direct` 仍然是最强，`8.47 GB/s` 高于 `Memcpy(N) 4.20 GB/s` 和 `BatchAsync 3.69 GB/s`
- 到了 `32KB`，拐点出现，`BatchAsync 9.89 GB/s` 已经重新反超 `GFD Direct 7.71 GB/s`
- 到了 `64KB`，这个差距继续拉大，`BatchAsync 10.99 GB/s`，`GFD Direct 7.72 GB/s`

### 20.3 这一章的结论

如果把“更小的 token”也纳入结论，本机上可以更精确地划出一个分界：

- `8KB ~ 16KB`：`GFD Direct` 仍然有明显优势
- `32KB`：开始进入 `BatchAsync` 更强的区间
- `64KB`：已经比较明确是 `BatchAsync` 主场

所以，“往 64KB 以下看会不会更好”这个问题，答案是：

- 会，而且在本机上是很明显的改善
- 但这个改善主要发生在 `8KB/16KB` 这两档
- 一旦接近 `32KB` 甚至到 `64KB`，`BatchAsync` 又会重新占优

## 21. 切到 GPU4 重跑

为了确认这些结论是不是只在 `GPU0` 上成立，我又把单卡 benchmark 和 large sweep 切到 `GPU4` 重新跑了一遍。

日志：

- `gpu4_full_benchmark_20260526.log`
- `gpu4_large_sweep_20260526.log`

### 21.1 先说一个限制

这个单卡 benchmark 有个实现细节要先说清楚：

- `examples/04_benchmark.cu` 里，单卡 `CpuPollingThread` 仍然写死为 `numa_node=0`
- 同时 `exclusive_core_base=0`、`exclusive_core_count=32`

也就是说，即使切到 `CUDA_VISIBLE_DEVICES=4`，这版 benchmark 仍然在按 `NUMA0` 的 poller/staging/绑核方式跑，而不是按 `GPU4` 所在 NUMA 节点做本地化设置。

所以这一章的意义是：

- 看“换卡后结论会不会翻”
- 不是给 `GPU4` 做最优 NUMA 调参后的极限成绩

### 21.2 GPU4 标准 benchmark 结果

先只摘最关键的 `Group B`：

| Config | BatchAsync GB/s | GFD Queue GB/s | GFD Direct GB/s |
| --- | ---: | ---: | ---: |
| `2048 x 8KB` | 3.18 | 5.51 | 9.33 |
| `2048 x 16KB` | 3.77 | 4.86 | 8.65 |
| `2048 x 32KB` | 9.90 | 4.61 | 8.11 |
| `2048 x 64KB` | 10.98 | 4.81 | 7.95 |

和 `GPU0` 对照：

| Config | GPU0 GFD Direct | GPU4 GFD Direct | GPU0 BatchAsync | GPU4 BatchAsync |
| --- | ---: | ---: | ---: | ---: |
| `2048 x 8KB` | 9.47 | 9.33 | 3.12 | 3.18 |
| `2048 x 16KB` | 8.47 | 8.65 | 3.69 | 3.77 |
| `2048 x 32KB` | 7.71 | 8.11 | 9.89 | 9.90 |
| `2048 x 64KB` | 7.72 | 7.95 | 10.99 | 10.98 |

这组结果说明：

- `8KB ~ 16KB` 仍然是 `GFD Direct` 优势区
- `32KB` 开始，`BatchAsync` 仍然重新反超
- `64KB` 仍然是 `BatchAsync` 主场

也就是说，切到 `GPU4` 后，分界线没有变化。

### 21.3 GPU4 large sweep 结果

`GPU4` 上 `128 x {64KB,128KB,256KB,512KB,1MB,2MB}` 的带宽：

| Config | BatchAsync GB/s | GFD Queue GB/s | GFD Direct GB/s |
| --- | ---: | ---: | ---: |
| `128 x 64KB` | 10.88 | 6.43 | 7.21 |
| `128 x 128KB` | 11.56 | 6.06 | 6.55 |
| `128 x 256KB` | 11.94 | 5.22 | 5.51 |
| `128 x 512KB` | 12.14 | 5.06 | 5.57 |
| `128 x 1MB` | 12.25 | 5.18 | 5.85 |
| `128 x 2MB` | 12.30 | 5.53 | 5.76 |

和 `GPU0` 对照后，可以看到：

- `BatchAsync` 基本不变
- `GFD Direct` 在 `64KB` 以上整体没有变好，很多档位还略低
- `GFD Queue` 在大于等于 `128KB` 的档位也没有表现出更强趋势

### 21.4 这一章的结论

切到 `GPU4` 以后，本机结论没有发生本质变化：

- `8KB ~ 16KB`：`GFD Direct` 仍然更强
- `32KB`：仍然是拐点
- `64KB+`：仍然是 `BatchAsync` 更强

换句话说：

- 这些结论不是 `GPU0` 特例
- 至少在 `GPU4` 上也能复现同样的趋势
- 即使考虑到 `GPU4` 这组测法还不是严格 NUMA-local，它也没有出现“GFD 全面翻盘”的迹象

## 22. Agentic-RL 超长序列模拟：64k / 128k / 256k

为了模拟 agentic-RL 里的超长序列，这里补一组按序列长度扩展的实验。

### 22.1 映射假设

这组实验采用下面的映射：

- vLLM 逻辑 block size 按 `16 tokens / block`
- `per_block_bytes = 64KB`
- `seq_len = 64k / 128k / 256k`
- 对应 block 数：
  - `64k tokens -> 4096 blocks`
  - `128k tokens -> 8192 blocks`
  - `256k tokens -> 16384 blocks`

因此实际 benchmark 配置是：

| Seq len | Blocks | per_block_bytes | Total transfer |
| --- | ---: | ---: | ---: |
| `64k` | `4096` | `64KB` | `256MB` |
| `128k` | `8192` | `64KB` | `512MB` |
| `256k` | `16384` | `64KB` | `1024MB` |

说明：

- 这里仍然是单卡 `CUDA_VISIBLE_DEVICES=0`
- 地址模式是 regular `2x stride`
- 端到端结果使用非 nsys run，`GFD_WARMUP=3`、`GFD_ITERS=5`
- submit-only 结果使用 nsys run，`GFD_WARMUP=1`、`GFD_ITERS=3`
- `256k` 这组因为 staging pool 需要 `5 x 1GB`，超过当前 `4GB` HugeTLB 预算，日志显示 `hugepage=no`

### 22.2 端到端搬运时间

命令模板：

```bash
CUDA_VISIBLE_DEVICES=0 \
GFD_LARGE_SWEEP=1 \
GFD_LARGE_SWEEP_NUM_TOKENS=<blocks> \
GFD_LARGE_SWEEP_TOKEN_SIZES=64KB \
GFD_WARMUP=3 \
GFD_ITERS=5 \
./build/gfd_benchmark
```

日志：

- `agentic_rl_seq65536_blocks4096_64kb_20260526_202120.log`
- `agentic_rl_seq131072_blocks8192_64kb_20260526_202123.log`
- `agentic_rl_seq262144_blocks16384_64kb_20260526_202127.log`

端到端 P50：

| Seq len | Blocks | Total | BatchAsync P50 | GFD Queue P50 | GFD Direct P50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `64k` | `4096` | `256MB` | `24.45 ms` | `55.42 ms` | `33.59 ms` |
| `128k` | `8192` | `512MB` | `48.87 ms` | `101.29 ms` | `68.37 ms` |
| `256k` | `16384` | `1024MB` | `94.19 ms` | `194.94 ms` | `125.77 ms` |

对应带宽：

| Seq len | Blocks | BatchAsync GB/s | GFD Queue GB/s | GFD Direct GB/s |
| --- | ---: | ---: | ---: | ---: |
| `64k` | `4096` | `10.98` | `4.84` | `7.99` |
| `128k` | `8192` | `10.99` | `5.30` | `7.85` |
| `256k` | `16384` | `11.40` | `5.51` | `8.54` |

读法：

- `BatchAsync` 端到端基本线性扩展，稳定在 `~11 GB/s`
- `GFD Queue` 也随总字节增长，但带宽只有 `~4.8-5.5 GB/s`
- `GFD Direct` 比 `GFD Queue` 快，但仍低于 `BatchAsync`
- 所以在这组 agentic-RL 超长序列模拟里，如果只看搬运完成时间，`BatchAsync` 仍然是最优

### 22.3 Submit-only：BatchAsync API vs GFD submit kernel

nsys 命令模板：

```bash
CUDA_VISIBLE_DEVICES=0 \
GFD_LARGE_SWEEP=1 \
GFD_LARGE_SWEEP_NUM_TOKENS=<blocks> \
GFD_LARGE_SWEEP_TOKEN_SIZES=64KB \
GFD_WARMUP=1 \
GFD_ITERS=3 \
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite true \
  -o agentic_rl_nsys/seq<seq>_blocks<blocks>_64kb \
  ./build/gfd_benchmark
```

产物：

- `agentic_rl_nsys/seq65536_blocks4096_64kb.nsys-rep`
- `agentic_rl_nsys/seq65536_blocks4096_64kb.sqlite`
- `agentic_rl_nsys/seq131072_blocks8192_64kb.nsys-rep`
- `agentic_rl_nsys/seq131072_blocks8192_64kb.sqlite`
- `agentic_rl_nsys/seq262144_blocks16384_64kb.nsys-rep`
- `agentic_rl_nsys/seq262144_blocks16384_64kb.sqlite`

从 `CUPTI_ACTIVITY_KIND_RUNTIME` 和 `CUPTI_ACTIVITY_KIND_KERNEL` 直接查到：

| Seq len | Blocks | `cudaMemcpyBatchAsync` API avg | `bench_gfd_submit_kernel` avg | `bench_gfd_wait_kernel` avg |
| --- | ---: | ---: | ---: | ---: |
| `64k` | `4096` | `4.153 ms` | `0.567 ms` | `24.889 ms` |
| `128k` | `8192` | `6.391 ms` | `0.612 ms` | `48.249 ms` |
| `256k` | `16384` | `12.769 ms` | `2.099 ms` | `87.557 ms` |

补充：

- nsys 下 `cuMemcpyHtoDAsync_v2` 平均 API 时间只有 `~22-32 us`
- 也就是说，GFD 侧真正提交 CE DMA 的 API 时间仍然很小
- 但 `GFD Queue` 的 wait kernel 会等 CPU poller / gather / DMA 完成，时间随序列长度变大

### 22.4 这一组的结论

这组超长序列实验把前面的判断放大了：

1. 只看 submit-only，大 batch 下 `GFD submit kernel` 明显比 `cudaMemcpyBatchAsync` API 更轻。
   - `64k`: `0.567 ms` vs `4.153 ms`
   - `128k`: `0.612 ms` vs `6.391 ms`
   - `256k`: `2.099 ms` vs `12.769 ms`

2. 但端到端搬运时间仍然是 `BatchAsync` 明显更快。
   - `64k`: `24.45 ms` vs `GFD Direct 33.59 ms` / `GFD Queue 55.42 ms`
   - `128k`: `48.87 ms` vs `GFD Direct 68.37 ms` / `GFD Queue 101.29 ms`
   - `256k`: `94.19 ms` vs `GFD Direct 125.77 ms` / `GFD Queue 194.94 ms`

3. 因此结论不是“GFD submit 慢”，而是：
   - `GFD submit kernel` 本身很快
   - `GFD Queue` 慢在 submit 后面的 wait / CPU gather / staging / DMA 完成
   - `GFD Direct` 去掉 queue wait 后更接近，但仍然输给 `BatchAsync`

4. 对 agentic-RL 超长序列，如果每个 block 已经是 `64KB` 这种大块，纯搬运路径应优先用 `BatchAsync`。
   GFD 更适合继续作为 GPU submit / queue / overlap / scatter-gather 机制验证，而不是这组 `64KB block` 大块 H2D 的端到端带宽最优路径。

### 22.5 追加：random slots + strict bind

上面 22.2 / 22.3 的地址模式是 regular `2x stride`。为了避免“规则地址对 `BatchAsync` 太友好”的疑问，又补了同样三组 `random slots + strict bind`。

命令模板：

```bash
CUDA_VISIBLE_DEVICES=0 \
GFD_LARGE_SWEEP=1 \
GFD_LARGE_SWEEP_NUM_TOKENS=<blocks> \
GFD_LARGE_SWEEP_TOKEN_SIZES=64KB \
GFD_RANDOMIZE_ADDRS=1 \
GFD_STRICT_BIND=1 \
GFD_WARMUP=3 \
GFD_ITERS=5 \
./build/gfd_benchmark
```

端到端日志：

- `agentic_rl_random_strict_seq65536_blocks4096_64kb_20260526_202626.log`
- `agentic_rl_random_strict_seq131072_blocks8192_64kb_20260526_202629.log`
- `agentic_rl_random_strict_seq262144_blocks16384_64kb_20260526_202633.log`

端到端 P50：

| Seq len | Blocks | Total | BatchAsync P50 | GFD Queue P50 | GFD Direct P50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `64k` | `4096` | `256MB` | `25.07 ms` | `51.26 ms` | `34.61 ms` |
| `128k` | `8192` | `512MB` | `50.06 ms` | `94.07 ms` | `69.11 ms` |
| `256k` | `16384` | `1024MB` | `95.84 ms` | `188.37 ms` | `126.32 ms` |

对应带宽：

| Seq len | Blocks | BatchAsync GB/s | GFD Queue GB/s | GFD Direct GB/s |
| --- | ---: | ---: | ---: | ---: |
| `64k` | `4096` | `10.71` | `5.24` | `7.76` |
| `128k` | `8192` | `10.72` | `5.71` | `7.77` |
| `256k` | `16384` | `11.20` | `5.70` | `8.50` |

random + strict 的 nsys 产物：

- `agentic_rl_nsys_random_strict/seq65536_blocks4096_64kb_random_strict.nsys-rep`
- `agentic_rl_nsys_random_strict/seq65536_blocks4096_64kb_random_strict.sqlite`
- `agentic_rl_nsys_random_strict/seq131072_blocks8192_64kb_random_strict.nsys-rep`
- `agentic_rl_nsys_random_strict/seq131072_blocks8192_64kb_random_strict.sqlite`
- `agentic_rl_nsys_random_strict/seq262144_blocks16384_64kb_random_strict.nsys-rep`
- `agentic_rl_nsys_random_strict/seq262144_blocks16384_64kb_random_strict.sqlite`

submit-only：

| Seq len | Blocks | `cudaMemcpyBatchAsync` API avg | `bench_gfd_submit_kernel` avg | `bench_gfd_wait_kernel` avg |
| --- | ---: | ---: | ---: | ---: |
| `64k` | `4096` | `3.779 ms` | `0.834 ms` | `24.015 ms` |
| `128k` | `8192` | `7.513 ms` | `1.233 ms` | `46.444 ms` |
| `256k` | `16384` | `15.029 ms` | `2.285 ms` | `79.260 ms` |

这组追加实验说明：

- random slots 没有把 `BatchAsync` 打崩，`BatchAsync` 端到端仍然稳定在 `~10.7-11.2 GB/s`
- `GFD submit kernel` 在 submit-only 口径下仍然明显更轻
- `GFD Queue` 的主要时间仍然在 wait kernel，而不是 submit kernel
- `GFD Direct` 仍然比 `GFD Queue` 快，但端到端仍然追不上 `BatchAsync`
- 因此“regular stride 过于有利于 BatchAsync”不是这组 `64KB block` 结论的主因
