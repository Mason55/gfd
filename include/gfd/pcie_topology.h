#pragma once

#include "gfd/log.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#ifdef __linux__
#include <dirent.h>
#include <unistd.h>
#endif

namespace gfd {

struct GpuTopology {
    int gpu_id;
    int numa_node;
    int pcie_bus;
    int cpu_start;
    int cpu_end;
    int ht_offset;
    int num_physical_cores;
};

struct TopologyConfig {
    int num_gpus;
    int total_numa_nodes;
    int cpus_per_numa;
    int physical_cores_per_numa;
    std::vector<GpuTopology> gpus;

    std::vector<int> gpus_per_numa;

    int recommended_ce_channels(int active_gpus_on_same_numa) const {
        if (active_gpus_on_same_numa <= 2) return 3;
        if (active_gpus_on_same_numa <= 4) return 2;
        return 1;
    }

    void get_exclusive_cores(int gpu_id, int& out_base_cpu, int& out_num_cores,
                             int& out_stride) const {
        auto& g = gpus[gpu_id];
        out_stride = 1;
        out_base_cpu = g.cpu_start;
        out_num_cores = g.cpu_end - g.cpu_start + 1;
        if (out_num_cores < 1) out_num_cores = 1;
    }
};

static inline std::vector<std::pair<int, int>> parse_cpulist_ranges(const char* text) {
    std::vector<std::pair<int, int>> ranges;
    if (!text) return ranges;

    std::string cpulist(text);
    size_t start = 0;
    while (start < cpulist.size()) {
        size_t end = cpulist.find(',', start);
        std::string token = cpulist.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!token.empty()) {
            int lo = 0, hi = 0;
            if (sscanf(token.c_str(), "%d-%d", &lo, &hi) == 2) {
                ranges.emplace_back(lo, hi);
            } else if (sscanf(token.c_str(), "%d", &lo) == 1) {
                ranges.emplace_back(lo, lo);
            }
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return ranges;
}

static inline int count_cpulist_ranges(const std::vector<std::pair<int, int>>& ranges) {
    int total = 0;
    for (const auto& range : ranges) {
        total += range.second - range.first + 1;
    }
    return total;
}

static inline TopologyConfig discover_topology(int num_gpus) {
    TopologyConfig topo;
    topo.num_gpus = num_gpus;
    topo.gpus.resize(num_gpus);

#ifdef __linux__
    int max_numa = 0;
    FILE* f = fopen("/sys/devices/system/node/online", "r");
    if (f) {
        char buf[64];
        if (fgets(buf, sizeof(buf), f)) {
            char* dash = strchr(buf, '-');
            if (dash) max_numa = atoi(dash + 1);
        }
        fclose(f);
    }
    topo.total_numa_nodes = max_numa + 1;

    char path[256];
    std::vector<std::vector<std::pair<int, int>>> node_cpu_ranges(topo.total_numa_nodes);
    std::vector<int> node_cpu_counts(topo.total_numa_nodes, 0);
    for (int node = 0; node < topo.total_numa_nodes; node++) {
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
        f = fopen(path, "r");
        if (!f) continue;
        char buf[256] = {};
        if (fgets(buf, sizeof(buf), f)) {
            node_cpu_ranges[node] = parse_cpulist_ranges(buf);
            node_cpu_counts[node] = count_cpulist_ranges(node_cpu_ranges[node]);
        }
        fclose(f);
    }

    topo.cpus_per_numa = node_cpu_counts.empty() ? 0 : node_cpu_counts[0];
    topo.physical_cores_per_numa = topo.cpus_per_numa > 1 ? topo.cpus_per_numa / 2 : topo.cpus_per_numa;

    for (int g = 0; g < num_gpus; g++) {
        topo.gpus[g].gpu_id = g;

        // Query actual NUMA node from PCIe topology via sysfs
        int gpu_numa = -1;
        // Try reading from CUDA device's PCI bus address
        char pci_bus_id[32] = {};
        if (cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), g) == cudaSuccess) {
            // pci_bus_id format: "0000:XX:YY.Z" - convert to sysfs path
            // Lowercase the hex for sysfs lookup
            for (char* p = pci_bus_id; *p; p++) {
                if (*p >= 'A' && *p <= 'F') *p = *p - 'A' + 'a';
            }
            snprintf(path, sizeof(path),
                     "/sys/bus/pci/devices/%s/numa_node", pci_bus_id);
            FILE* nf = fopen(path, "r");
            if (nf) {
                char nbuf[16];
                if (fgets(nbuf, sizeof(nbuf), nf)) {
                    gpu_numa = atoi(nbuf);
                }
                fclose(nf);
            }
        }
        // Fallback: naive heuristic if sysfs query fails
        if (gpu_numa < 0 || gpu_numa >= topo.total_numa_nodes) {
            gpu_numa = (num_gpus > 1 && g >= num_gpus / 2) ? 1 : 0;
        }
        topo.gpus[g].numa_node = gpu_numa;
        topo.gpus[g].cpu_start = 0;
        topo.gpus[g].cpu_end = 0;
        topo.gpus[g].ht_offset = 0;
        topo.gpus[g].num_physical_cores = 1;
        topo.gpus[g].pcie_bus = g;
    }
#else
    topo.total_numa_nodes = 1;
    topo.cpus_per_numa = 16;
    topo.physical_cores_per_numa = 8;
    for (int g = 0; g < num_gpus; g++) {
        topo.gpus[g].gpu_id = g;
        topo.gpus[g].numa_node = 0;
        topo.gpus[g].cpu_start = 0;
        topo.gpus[g].cpu_end = 15;
        topo.gpus[g].ht_offset = 0;
        topo.gpus[g].num_physical_cores = 8;
        topo.gpus[g].pcie_bus = g;
    }
#endif

    topo.gpus_per_numa.resize(topo.total_numa_nodes, 0);
    for (int g = 0; g < num_gpus; g++) {
        topo.gpus_per_numa[topo.gpus[g].numa_node]++;
    }

#ifdef __linux__
    for (int node = 0; node < topo.total_numa_nodes; node++) {
        std::vector<int> gpu_ids;
        for (int g = 0; g < num_gpus; g++) {
            if (topo.gpus[g].numa_node == node) gpu_ids.push_back(g);
        }
        if (gpu_ids.empty()) continue;

        const auto& ranges = node_cpu_ranges[node];
        int remaining_cpus = node_cpu_counts[node];
        size_t range_idx = 0;
        int cursor = ranges.empty() ? 0 : ranges[0].first;

        for (size_t pos = 0; pos < gpu_ids.size(); pos++) {
            int gpu_id = gpu_ids[pos];
            int remaining_gpus = static_cast<int>(gpu_ids.size() - pos);
            int want = remaining_gpus > 0 ? (remaining_cpus + remaining_gpus - 1) / remaining_gpus : 1;

            while (range_idx < ranges.size() && cursor > ranges[range_idx].second) {
                range_idx++;
                if (range_idx < ranges.size()) cursor = ranges[range_idx].first;
            }

            if (range_idx >= ranges.size()) {
                topo.gpus[gpu_id].cpu_start = 0;
                topo.gpus[gpu_id].cpu_end = 0;
                topo.gpus[gpu_id].num_physical_cores = 1;
                continue;
            }

            int take = std::min(want, ranges[range_idx].second - cursor + 1);
            topo.gpus[gpu_id].cpu_start = cursor;
            topo.gpus[gpu_id].cpu_end = cursor + take - 1;
            topo.gpus[gpu_id].num_physical_cores = take;
            cursor += take;
            remaining_cpus -= take;
        }
    }
#endif

    return topo;
}

static inline void print_topology(const TopologyConfig& topo) {
    GFD_LOG_INFO("Topology: %d GPUs, %d NUMA nodes, %d phys cores/node\n",
                 topo.num_gpus, topo.total_numa_nodes, topo.physical_cores_per_numa);
    for (int g = 0; g < topo.num_gpus; g++) {
        auto& gpu = topo.gpus[g];
        int base, ncores, stride;
        topo.get_exclusive_cores(g, base, ncores, stride);
        GFD_LOG_INFO("  GPU %d: NUMA %d, cores %d-%d (stride %d, %d cores)\n",
                     g, gpu.numa_node, base, base + (ncores - 1) * stride, stride, ncores);
    }
    for (int n = 0; n < topo.total_numa_nodes; n++) {
        int ngpu = topo.gpus_per_numa[n];
        int ce_rec = topo.recommended_ce_channels(ngpu);
        GFD_LOG_INFO("  NUMA %d: %d GPUs, recommended %d CE channels/GPU\n",
                     n, ngpu, ce_rec);
    }
}

}  // namespace gfd
