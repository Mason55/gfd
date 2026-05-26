// ============================================================
// GFD Multi-GPU Benchmark (8 GPUs, NUMA-Optimized)
//
// Measures aggregate H2D transfer bandwidth across multiple GPUs
// comparing:
//   1. cudaMemcpy(N):       Per-token cudaMemcpyAsync (baseline)
//   2. cudaMemcpyBatchAsync: CUDA 12.8+ batch API (single call)
//   3. GFD Direct:          CPU direct-submit (parallel gather + CE DMA)
//
// Optimizations:
//   - Per-GPU staging buffers allocated on local NUMA node
//     (bypasses shared StagingPool to avoid cross-NUMA staging)
//   - CPU source buffers pinned to local NUMA node
//   - Persistent worker threads (eliminates thread spawn overhead)
//   - Spin-barrier synchronization for tight parallel launch
//   - NUMA-aware gather worker core pinning
//
// Topology (auto-detected):
//   GPU 0-3: NUMA node 0, CPUs 0-63 + 128-191
//   GPU 4-7: NUMA node 1, CPUs 64-127 + 192-255
// ============================================================

#include <gfd/gfd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <algorithm>
#include <numeric>
#include <functional>
#include <mutex>
#include <condition_variable>

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#endif

// ---- Configuration ----
static constexpr int MAX_GPUS = 8;
static constexpr int NUM_TOKENS = 2048;
static constexpr int TOKEN_SIZE = 4096;       // 4KB per token
static constexpr size_t TOTAL_SIZE = (size_t)NUM_TOKENS * TOKEN_SIZE;  // 8MB per GPU
static constexpr int SCATTER_STRIDE = 2;
static constexpr int WARMUP = 15;
static constexpr int ITERS = 50;

// Per-GPU NUMA mapping
struct GPUConfig {
    int gpu_id;
    int numa_node;
    int core_base;
    int core_count;
};

// Per-GPU resources
struct GPUContext {
    int             gpu_id;
    CUcontext       cu_ctx;
    char*           cpu_buf;
    char*           gpu_buf;
    gfd::DescriptorQueue* queue;
    gfd::CpuPollingThread* poller;
    gfd::SGEntry*   sg_entries;
    cudaStream_t    stream;
    std::vector<double> latencies;
    // Pre-allocated arrays for cudaMemcpyBatchAsync (avoid heap contention)
    std::vector<void*>       batch_dsts;
    std::vector<const void*> batch_srcs;
    std::vector<size_t>      batch_sizes;
    std::vector<size_t>      batch_attr_idxs;
    cudaMemcpyAttributes     batch_attr;
};

static cudaError_t memcpy_batch_async_compat(
    void** dsts,
    const void** srcs,
    size_t* sizes,
    size_t count,
    cudaMemcpyAttributes* attr,
    size_t* attr_idxs,
    size_t num_attrs,
    cudaStream_t stream)
{
#if CUDART_VERSION >= 12090
    size_t fail_idx = 0;
    return cudaMemcpyBatchAsync(
        dsts, srcs, sizes, count, attr, attr_idxs, num_attrs, &fail_idx, stream);
#else
    return cudaMemcpyBatchAsync(
        dsts, srcs, sizes, count, attr, attr_idxs, num_attrs, stream);
#endif
}

// ---- Benchmark: cudaMemcpy(N) ----
static double bench_memcpy_N(GPUContext& ctx) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < NUM_TOKENS; t++) {
        void* dst = ctx.gpu_buf + (size_t)t * TOKEN_SIZE;
        void* src = ctx.cpu_buf + (size_t)t * TOKEN_SIZE * SCATTER_STRIDE;
        cudaMemcpyAsync(dst, src, TOKEN_SIZE, cudaMemcpyHostToDevice, ctx.stream);
    }
    cudaStreamSynchronize(ctx.stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

// ---- Benchmark: cudaMemcpyBatchAsync (uses pre-allocated arrays) ----
static double bench_memcpy_batch(GPUContext& ctx) {
    auto t0 = std::chrono::high_resolution_clock::now();
    memcpy_batch_async_compat(
        ctx.batch_dsts.data(), ctx.batch_srcs.data(),
        ctx.batch_sizes.data(), NUM_TOKENS,
        &ctx.batch_attr, ctx.batch_attr_idxs.data(), 1, ctx.stream);
    cudaStreamSynchronize(ctx.stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

// ---- Spin barrier (lock-free, reusable) ----
class SpinBarrier {
public:
    SpinBarrier(int count) : count_(count), waiting_(0), generation_(0) {}

    void wait() {
        int gen = generation_.load(std::memory_order_acquire);
        if (waiting_.fetch_add(1, std::memory_order_acq_rel) + 1 == count_) {
            waiting_.store(0, std::memory_order_release);
            generation_.fetch_add(1, std::memory_order_release);
        } else {
            while (generation_.load(std::memory_order_acquire) == gen) {
                __builtin_ia32_pause();
            }
        }
    }

    void reset(int count) {
        count_ = count;
        waiting_.store(0);
    }

private:
    int count_;
    std::atomic<int> waiting_;
    std::atomic<int> generation_;
};

static double percentile(std::vector<double>& v, double p) {
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx;
    size_t hi = lo + 1;
    if (hi >= v.size()) return v.back();
    double frac = idx - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

// Pin current thread to a specific CPU core
static void pin_to_cpu(int cpu) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif
}

int main() {
    cuInit(0);

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > MAX_GPUS) num_gpus = MAX_GPUS;

    printf("============================================================\n");
    printf("  GFD Multi-GPU Benchmark (NUMA-Optimized)\n");
    printf("============================================================\n");
    printf("GPUs detected: %d\n", num_gpus);
    printf("Per-GPU transfer: %d tokens x %d bytes = %zu MB (scattered at %dx stride)\n",
           NUM_TOKENS, TOKEN_SIZE, TOTAL_SIZE / (1024 * 1024), SCATTER_STRIDE);
    printf("Warmup: %d, Iterations: %d\n\n", WARMUP, ITERS);

    // Print GPU names
    for (int i = 0; i < num_gpus; i++) {
        char name[256];
        CUdevice dev;
        cuDeviceGet(&dev, i);
        cuDeviceGetName(name, sizeof(name), dev);
        printf("  GPU %d: %s\n", i, name);
    }
    printf("\n");

    auto topo = gfd::discover_topology(num_gpus);
    gfd::print_topology(topo);
    std::vector<GPUConfig> gpu_configs(num_gpus);
    for (int i = 0; i < num_gpus; i++) {
        int base_cpu = 0, num_cores = 1, stride = 1;
        topo.get_exclusive_cores(i, base_cpu, num_cores, stride);
        gpu_configs[i] = { i, topo.gpus[i].numa_node, base_cpu, num_cores };
    }

    // ---- Do NOT use shared StagingPool ----
    // Each poller self-allocates NUMA-local hugepage staging buffers
    // when pool is not initialized.

    // ---- Initialize all GPU contexts ----
    std::vector<GPUContext> contexts(num_gpus);

    for (int i = 0; i < num_gpus; i++) {
        GPUContext& ctx = contexts[i];
        ctx.gpu_id = i;
        GPUConfig& gcfg = gpu_configs[i];

        // Set GPU
        cudaSetDevice(i);
        CUdevice dev;
        cuDeviceGet(&dev, i);
#if CUDA_VERSION >= 13000
        CUctxCreateParams ctxParams = {};
        cuCtxCreate(&ctx.cu_ctx, &ctxParams, 0, dev);
#else
        cuCtxCreate(&ctx.cu_ctx, 0, dev);
#endif

        // Bind memory allocation to correct NUMA node
#ifdef __linux__
        unsigned long nodemask = 1UL << gcfg.numa_node;
        set_mempolicy(MPOL_BIND, &nodemask, sizeof(nodemask) * 8);
#endif

        // Allocate pinned CPU buffer (on local NUMA node)
        cudaMallocHost(&ctx.cpu_buf, TOTAL_SIZE * SCATTER_STRIDE);
        for (int t = 0; t < NUM_TOKENS; t++) {
            char* ptr = ctx.cpu_buf + (size_t)t * TOKEN_SIZE * SCATTER_STRIDE;
            memset(ptr, (uint8_t)((i * NUM_TOKENS + t) & 0xFF), TOKEN_SIZE);
        }

        // Reset memory policy
#ifdef __linux__
        set_mempolicy(MPOL_DEFAULT, NULL, 0);
#endif

        // Allocate GPU buffer
        cudaMalloc(&ctx.gpu_buf, TOTAL_SIZE);

        // Create stream for memcpy benchmarks
        cudaStreamCreate(&ctx.stream);

        // Allocate descriptor queue (pinned, for poller)
        cudaHostAlloc(&ctx.queue, sizeof(gfd::DescriptorQueue), cudaHostAllocMapped);
        memset(ctx.queue, 0, sizeof(gfd::DescriptorQueue));

        // Build SG entries (pinned host)
        cudaHostAlloc(&ctx.sg_entries, NUM_TOKENS * sizeof(gfd::SGEntry), cudaHostAllocMapped);
        for (int t = 0; t < NUM_TOKENS; t++) {
            ctx.sg_entries[t].dst  = (CUdeviceptr)(ctx.gpu_buf + (size_t)t * TOKEN_SIZE);
            ctx.sg_entries[t].src  = ctx.cpu_buf + (size_t)t * TOKEN_SIZE * SCATTER_STRIDE;
            ctx.sg_entries[t].size = TOKEN_SIZE;
        }

        // Pre-allocate batch arrays (avoid heap contention in parallel runs)
        ctx.batch_dsts.resize(NUM_TOKENS);
        ctx.batch_srcs.resize(NUM_TOKENS);
        ctx.batch_sizes.resize(NUM_TOKENS);
        ctx.batch_attr_idxs.resize(NUM_TOKENS, 0);
        ctx.batch_attr = {};
        ctx.batch_attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        ctx.batch_attr.flags = 0;
        for (int t = 0; t < NUM_TOKENS; t++) {
            ctx.batch_dsts[t] = ctx.gpu_buf + (size_t)t * TOKEN_SIZE;
            ctx.batch_srcs[t] = ctx.cpu_buf + (size_t)t * TOKEN_SIZE * SCATTER_STRIDE;
            ctx.batch_sizes[t] = TOKEN_SIZE;
        }

        // Create polling thread:
        // No shared pool → poller will self-allocate NUMA-local hugepage staging
        int poller_core_count = std::max(1, gcfg.core_count - 1);
        ctx.poller = new gfd::CpuPollingThread(
            ctx.queue, ctx.gpu_buf, ctx.cpu_buf, TOTAL_SIZE,
            /*use_ce=*/true, /*numa_node=*/gcfg.numa_node,
            /*core_offset=*/0, /*num_ce_channels=*/0,
            /*exclusive_core_base=*/gcfg.core_base,
            /*exclusive_core_count=*/poller_core_count);

        if (!ctx.poller->init_copy_engine()) {
            fprintf(stderr, "GPU %d: Failed to init copy engine\n", i);
            return 1;
        }
        ctx.poller->init_direct_ce();
        ctx.poller->start();
    }

    printf("All %d GPUs initialized (per-GPU NUMA-local staging)\n\n", num_gpus);

    // ---- Test 1: Per-GPU sequential bandwidth (3 methods) ----
    printf("────────────────────────────────────────────────────────────\n");
    printf("  Test 1: Per-GPU Bandwidth (sequential, 3 methods)\n");
    printf("────────────────────────────────────────────────────────────\n\n");

    printf("  +------+------+--------------+--------------+--------------+\n");
    printf("  | %4s | %4s | %12s | %12s | %12s |\n",
           "GPU", "NUMA", "Memcpy(N)", "BatchAsync", "GFD Direct");
    printf("  +------+------+--------------+--------------+--------------+\n");

    for (int i = 0; i < num_gpus; i++) {
        GPUContext& ctx = contexts[i];
        cuCtxSetCurrent(ctx.cu_ctx);

        // --- Memcpy(N) ---
        for (int w = 0; w < WARMUP; w++) bench_memcpy_N(ctx);
        std::vector<double> memcpy_lats(ITERS);
        for (int iter = 0; iter < ITERS; iter++) memcpy_lats[iter] = bench_memcpy_N(ctx);
        std::sort(memcpy_lats.begin(), memcpy_lats.end());
        double memcpy_p50 = percentile(memcpy_lats, 50);

        // --- BatchAsync ---
        for (int w = 0; w < WARMUP; w++) bench_memcpy_batch(ctx);
        std::vector<double> batch_lats(ITERS);
        for (int iter = 0; iter < ITERS; iter++) batch_lats[iter] = bench_memcpy_batch(ctx);
        std::sort(batch_lats.begin(), batch_lats.end());
        double batch_p50 = percentile(batch_lats, 50);

        // --- GFD Direct ---
        for (int w = 0; w < WARMUP; w++) ctx.poller->submit_direct(ctx.sg_entries, NUM_TOKENS);
        ctx.latencies.resize(ITERS);
        for (int iter = 0; iter < ITERS; iter++)
            ctx.latencies[iter] = ctx.poller->submit_direct(ctx.sg_entries, NUM_TOKENS);
        std::sort(ctx.latencies.begin(), ctx.latencies.end());
        double gfd_p50 = percentile(ctx.latencies, 50);

        double memcpy_bw = TOTAL_SIZE / (memcpy_p50 * 1e3);
        double batch_bw = TOTAL_SIZE / (batch_p50 * 1e3);
        double gfd_bw = TOTAL_SIZE / (gfd_p50 * 1e3);

        printf("  | %4d | %4d | %5.1f GB/s   | %5.1f GB/s   | %5.1f GB/s   |\n",
               i, gpu_configs[i].numa_node, memcpy_bw, batch_bw, gfd_bw);
    }
    printf("  +------+------+--------------+--------------+--------------+\n");

    // ---- Test 2: Parallel scaling with persistent threads (3 methods) ----
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 2: Aggregate Bandwidth (parallel, 3 methods)\n");
    printf("────────────────────────────────────────────────────────────\n\n");

    std::vector<int> gpu_counts;
    for (int n = 1; n <= num_gpus; n *= 2) {
        gpu_counts.push_back(n);
    }
    if (gpu_counts.back() != num_gpus) {
        gpu_counts.push_back(num_gpus);
    }

    printf("  +-----------+--------------+--------------+--------------+\n");
    printf("  | %9s | %12s | %12s | %12s |\n",
           "GPUs", "Memcpy(N)", "BatchAsync", "GFD Direct");
    printf("  +-----------+--------------+--------------+--------------+\n");

    double single_gpu_bw = 0;

    for (int num_active : gpu_counts) {
        // --- Method 1: Memcpy(N) parallel ---
        {
            std::vector<std::vector<double>> per_gpu_lats(num_active);
            for (auto& v : per_gpu_lats) v.resize(ITERS);
            SpinBarrier barrier(num_active);
            std::vector<std::thread> workers;
            for (int i = 0; i < num_active; i++) {
                workers.emplace_back([&, i]() {
                    GPUContext& ctx = contexts[i];
                    cuCtxSetCurrent(ctx.cu_ctx);
                    pin_to_cpu(gpu_configs[i].core_base + gpu_configs[i].core_count - 1);
                    for (int w = 0; w < WARMUP; w++) { barrier.wait(); bench_memcpy_N(ctx); }
                    for (int iter = 0; iter < ITERS; iter++) { barrier.wait(); per_gpu_lats[i][iter] = bench_memcpy_N(ctx); }
                });
            }
            for (auto& t : workers) t.join();
            std::vector<double> max_lats(ITERS);
            for (int iter = 0; iter < ITERS; iter++) {
                double m = 0;
                for (int i = 0; i < num_active; i++) m = std::max(m, per_gpu_lats[i][iter]);
                max_lats[iter] = m;
            }
            double p50 = percentile(max_lats, 50);
            double memcpy_bw = (double)num_active * TOTAL_SIZE / (p50 * 1e3);

        // --- Method 2: BatchAsync parallel ---
            std::vector<std::vector<double>> per_gpu_lats2(num_active);
            for (auto& v : per_gpu_lats2) v.resize(ITERS);
            SpinBarrier barrier2(num_active);
            std::vector<std::thread> workers2;
            for (int i = 0; i < num_active; i++) {
                workers2.emplace_back([&, i]() {
                    GPUContext& ctx = contexts[i];
                    cuCtxSetCurrent(ctx.cu_ctx);
                    pin_to_cpu(gpu_configs[i].core_base + gpu_configs[i].core_count - 1);
                    for (int w = 0; w < WARMUP; w++) { barrier2.wait(); bench_memcpy_batch(ctx); }
                    for (int iter = 0; iter < ITERS; iter++) { barrier2.wait(); per_gpu_lats2[i][iter] = bench_memcpy_batch(ctx); }
                });
            }
            for (auto& t : workers2) t.join();
            std::vector<double> max_lats2(ITERS);
            for (int iter = 0; iter < ITERS; iter++) {
                double m = 0;
                for (int i = 0; i < num_active; i++) m = std::max(m, per_gpu_lats2[i][iter]);
                max_lats2[iter] = m;
            }
            double batch_p50 = percentile(max_lats2, 50);
            double batch_bw = (double)num_active * TOTAL_SIZE / (batch_p50 * 1e3);

        // --- Method 3: GFD Direct parallel ---
            std::vector<std::vector<double>> per_gpu_lats3(num_active);
            for (auto& v : per_gpu_lats3) v.resize(ITERS);
            SpinBarrier barrier3(num_active);
            std::vector<std::thread> workers3;
            for (int i = 0; i < num_active; i++) {
                workers3.emplace_back([&, i]() {
                    GPUContext& ctx = contexts[i];
                    cuCtxSetCurrent(ctx.cu_ctx);
                    pin_to_cpu(gpu_configs[i].core_base + gpu_configs[i].core_count - 1);
                    for (int w = 0; w < WARMUP; w++) { barrier3.wait(); ctx.poller->submit_direct(ctx.sg_entries, NUM_TOKENS); }
                    for (int iter = 0; iter < ITERS; iter++) { barrier3.wait(); per_gpu_lats3[i][iter] = ctx.poller->submit_direct(ctx.sg_entries, NUM_TOKENS); }
                });
            }
            for (auto& t : workers3) t.join();
            std::vector<double> max_lats3(ITERS);
            for (int iter = 0; iter < ITERS; iter++) {
                double m = 0;
                for (int i = 0; i < num_active; i++) m = std::max(m, per_gpu_lats3[i][iter]);
                max_lats3[iter] = m;
            }
            double gfd_p50 = percentile(max_lats3, 50);
            double gfd_bw = (double)num_active * TOTAL_SIZE / (gfd_p50 * 1e3);

            if (num_active == 1) single_gpu_bw = gfd_bw;

            printf("  | %4d GPU%s | %7.2f GB/s | %7.2f GB/s | %7.2f GB/s |\n",
                   num_active, num_active > 1 ? "s" : " ",
                   memcpy_bw, batch_bw, gfd_bw);
        }
    }
    printf("  +-----------+--------------+--------------+--------------+\n");

    // ---- Test 3: NUMA locality analysis ----
    if (num_gpus >= 8) {
        printf("\n────────────────────────────────────────────────────────────\n");
        printf("  Test 3: NUMA Locality Analysis\n");
        printf("────────────────────────────────────────────────────────────\n\n");

        auto run_group = [&](const char* label, int start, int count) {
            std::vector<std::vector<double>> per_gpu_lats(count);
            for (auto& v : per_gpu_lats) v.resize(ITERS);

            SpinBarrier barrier(count);
            std::vector<std::thread> workers;

            for (int idx = 0; idx < count; idx++) {
                int i = start + idx;
                workers.emplace_back([&, i, idx]() {
                    GPUContext& ctx = contexts[i];
                    cuCtxSetCurrent(ctx.cu_ctx);
                    pin_to_cpu(gpu_configs[i].core_base + gpu_configs[i].core_count - 1);

                    // Warmup
                    for (int w = 0; w < WARMUP; w++) {
                        barrier.wait();
                        ctx.poller->submit_direct(ctx.sg_entries, NUM_TOKENS);
                    }
                    // Timed
                    for (int iter = 0; iter < ITERS; iter++) {
                        barrier.wait();
                        per_gpu_lats[idx][iter] = ctx.poller->submit_direct(ctx.sg_entries, NUM_TOKENS);
                    }
                });
            }
            for (auto& t : workers) t.join();

            std::vector<double> max_lats(ITERS);
            for (int iter = 0; iter < ITERS; iter++) {
                double m = 0;
                for (int idx = 0; idx < count; idx++)
                    m = std::max(m, per_gpu_lats[idx][iter]);
                max_lats[iter] = m;
            }
            double p50 = percentile(max_lats, 50);
            double agg_bw = (double)count * TOTAL_SIZE / (p50 * 1e3);
            printf("  %s: P50 = %7.1f us, Aggregate = %6.2f GB/s\n", label, p50, agg_bw);
        };

        run_group("NUMA 0 (GPU 0-3)", 0, 4);
        run_group("NUMA 1 (GPU 4-7)", 4, 4);
        run_group("All 8 GPUs       ", 0, 8);
    }

    // ---- Test 4: Per-GPU detailed stats in parallel mode ----
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Test 4: Per-GPU Bandwidth Under Full Load (all %d GPUs)\n", num_gpus);
    printf("────────────────────────────────────────────────────────────\n\n");

    {
        std::vector<std::vector<double>> per_gpu_lats(num_gpus);
        for (auto& v : per_gpu_lats) v.resize(ITERS);

        SpinBarrier barrier(num_gpus);
        std::vector<std::thread> workers;

        for (int i = 0; i < num_gpus; i++) {
            workers.emplace_back([&, i]() {
                GPUContext& ctx = contexts[i];
                cuCtxSetCurrent(ctx.cu_ctx);
                pin_to_cpu(gpu_configs[i].core_base + gpu_configs[i].core_count - 1);

                for (int w = 0; w < WARMUP; w++) {
                    barrier.wait();
                    ctx.poller->submit_direct(ctx.sg_entries, NUM_TOKENS);
                }
                for (int iter = 0; iter < ITERS; iter++) {
                    barrier.wait();
                    per_gpu_lats[i][iter] = ctx.poller->submit_direct(ctx.sg_entries, NUM_TOKENS);
                }
            });
        }
        for (auto& t : workers) t.join();

        for (int i = 0; i < num_gpus; i++) {
            double p50 = percentile(per_gpu_lats[i], 50);
            double p90 = percentile(per_gpu_lats[i], 90);
            double bw = TOTAL_SIZE / (p50 * 1e3);
            printf("  GPU %d (NUMA %d): P50 = %7.1f us, P90 = %7.1f us, BW = %6.2f GB/s\n",
                   i, gpu_configs[i].numa_node, p50, p90, bw);
        }

        // Summary
        double total_bw = 0;
        for (int i = 0; i < num_gpus; i++) {
            double p50 = percentile(per_gpu_lats[i], 50);
            total_bw += TOTAL_SIZE / (p50 * 1e3);
        }
        printf("  ─────────────────────────────────────────────────\n");
        printf("  Sum of per-GPU BW: %.2f GB/s\n", total_bw);
    }

    // ---- Summary ----
    printf("\n────────────────────────────────────────────────────────────\n");
    printf("  Summary\n");
    printf("────────────────────────────────────────────────────────────\n");
    printf("  Per-GPU transfer size:  %zu MB (%d x %d bytes)\n",
           TOTAL_SIZE / (1024 * 1024), NUM_TOKENS, TOKEN_SIZE);
    printf("  Single GPU baseline:    %.2f GB/s\n", single_gpu_bw);
    printf("  Theoretical %d-GPU max:  %.2f GB/s\n", num_gpus, single_gpu_bw * num_gpus);
    printf("  Staging: per-GPU NUMA-local hugepages (no shared pool)\n");
    printf("============================================================\n");

    // ---- Cleanup ----
    for (int i = 0; i < num_gpus; i++) {
        GPUContext& ctx = contexts[i];
        cuCtxSetCurrent(ctx.cu_ctx);
        ctx.poller->stop();
        delete ctx.poller;
        cudaStreamDestroy(ctx.stream);
        cudaFreeHost(ctx.sg_entries);
        cudaFreeHost(ctx.queue);
        cudaFree(ctx.gpu_buf);
        cudaFreeHost(ctx.cpu_buf);
    }

    return 0;
}
