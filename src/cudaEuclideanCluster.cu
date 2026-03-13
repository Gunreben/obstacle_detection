/*
 * CUDA Euclidean Cluster Extraction for x86_64 (RTX desktop GPUs)
 *
 * Replaces the precompiled ARM (Jetson) libcudacluster.so with a source-built
 * implementation using CUDA + Thrust. Implements the same cudaExtractCluster
 * API defined in include/obstacle_detection/cudaCluster.h.
 *
 * Algorithm:
 *   1. GPU kernel: assign each point to a voxel (quantised grid cell)
 *   2. Thrust sort by voxel key
 *   3. Thrust reduce_by_key to count points per voxel
 *   4. CPU BFS over 26-connected voxel graph → clusters
 *   5. Filter clusters by min/max size
 *   6. Write output in the format expected by the node
 */

#include "obstacle_detection/cudaCluster.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>
#include <unordered_map>
#include <vector>
#include <queue>
#include <cstdint>
#include <cstring>
#include <algorithm>

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct ClusterImpl {
    cudaStream_t       stream;
    extractClusterParam_t param;
};

// ---------------------------------------------------------------------------
// CUDA kernel: map each point to its voxel key
// ---------------------------------------------------------------------------
// Voxel key layout: 63 bits, 21 bits per axis (±1 048 575 voxels per axis).
// This covers ±500 km at 0.5 m resolution – more than enough for LiDAR.
// ---------------------------------------------------------------------------

static constexpr int32_t VOXEL_BIAS = 1 << 20;  // 1 048 576

__global__ void assignVoxelKeysKernel(
    const float* __restrict__ points,  // float4 layout: x,y,z,pad
    int64_t*     __restrict__ keys,
    uint32_t*    __restrict__ order,
    unsigned int n,
    float inv_vx, float inv_vy, float inv_vz)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float x = points[i * 4 + 0];
    const float y = points[i * 4 + 1];
    const float z = points[i * 4 + 2];

    int32_t vx = __float2int_rd(x * inv_vx) + VOXEL_BIAS;
    int32_t vy = __float2int_rd(y * inv_vy) + VOXEL_BIAS;
    int32_t vz = __float2int_rd(z * inv_vz) + VOXEL_BIAS;

    // Clamp to valid range
    vx = max(0, min((1 << 21) - 1, vx));
    vy = max(0, min((1 << 21) - 1, vy));
    vz = max(0, min((1 << 21) - 1, vz));

    // Pack three 21-bit values into a single 63-bit key
    keys[i]  = ((int64_t)vx << 42) | ((int64_t)vy << 21) | (int64_t)vz;
    order[i] = i;
}

// ---------------------------------------------------------------------------
// Helper: encode/decode voxel coordinates
// ---------------------------------------------------------------------------

static inline int64_t encodeVoxel(int32_t vx, int32_t vy, int32_t vz)
{
    const auto u = [](int32_t v) { return (int64_t)((v + VOXEL_BIAS) & 0x1FFFFF); };
    return (u(vx) << 42) | (u(vy) << 21) | u(vz);
}

static inline void decodeVoxel(int64_t key, int32_t& vx, int32_t& vy, int32_t& vz)
{
    constexpr int32_t MASK = 0x1FFFFF;
    vx = (int32_t)((key >> 42) & MASK) - VOXEL_BIAS;
    vy = (int32_t)((key >> 21) & MASK) - VOXEL_BIAS;
    vz = (int32_t)( key        & MASK) - VOXEL_BIAS;
}

// ---------------------------------------------------------------------------
// cudaExtractCluster implementation
// ---------------------------------------------------------------------------

cudaExtractCluster::cudaExtractCluster(cudaStream_t stream)
{
    auto* impl    = new ClusterImpl();
    impl->stream  = stream;
    std::memset(&impl->param, 0, sizeof(impl->param));
    m_handle = impl;
}

cudaExtractCluster::~cudaExtractCluster()
{
    delete static_cast<ClusterImpl*>(m_handle);
}

int cudaExtractCluster::set(extractClusterParam_t param)
{
    static_cast<ClusterImpl*>(m_handle)->param = param;
    return 0;
}

int cudaExtractCluster::extract(
    float*        cloud_in,
    int           nCount,
    float*        output,
    unsigned int* index)
{
    if (nCount <= 0) { index[0] = 0; return 0; }

    auto*              impl   = static_cast<ClusterImpl*>(m_handle);
    cudaStream_t       stream = impl->stream;
    const auto&        p      = impl->param;
    const unsigned int n      = static_cast<unsigned int>(nCount);

    const float inv_vx = 1.0f / p.voxelX;
    const float inv_vy = 1.0f / p.voxelY;
    const float inv_vz = 1.0f / p.voxelZ;

    // ── 1. Compute voxel keys on GPU ────────────────────────────────────────
    thrust::device_vector<int64_t>  d_keys(n);
    thrust::device_vector<uint32_t> d_order(n);

    const unsigned int blocks = (n + 255) / 256;
    assignVoxelKeysKernel<<<blocks, 256, 0, stream>>>(
        cloud_in,
        thrust::raw_pointer_cast(d_keys.data()),
        thrust::raw_pointer_cast(d_order.data()),
        n, inv_vx, inv_vy, inv_vz);
    cudaStreamSynchronize(stream);

    // ── 2. Sort (voxel_key, point_index) by key ─────────────────────────────
    thrust::sort_by_key(
        thrust::cuda::par.on(stream),
        d_keys.begin(), d_keys.end(),
        d_order.begin());
    cudaStreamSynchronize(stream);

    // ── 3. Count points per unique voxel ────────────────────────────────────
    thrust::device_vector<int64_t>  d_ukeys(n);
    thrust::device_vector<uint32_t> d_counts(n);

    auto end_it = thrust::reduce_by_key(
        thrust::cuda::par.on(stream),
        d_keys.begin(), d_keys.end(),
        thrust::constant_iterator<uint32_t>(1),
        d_ukeys.begin(),
        d_counts.begin());
    cudaStreamSynchronize(stream);

    const unsigned int num_voxels =
        static_cast<unsigned int>(end_it.first - d_ukeys.begin());

    // ── 4. Transfer to CPU ───────────────────────────────────────────────────
    thrust::host_vector<int64_t>  h_keys(d_ukeys.begin(),
                                          d_ukeys.begin()  + num_voxels);
    thrust::host_vector<uint32_t> h_counts(d_counts.begin(),
                                            d_counts.begin() + num_voxels);
    thrust::host_vector<uint32_t> h_order(d_order.begin(), d_order.end());

    // Voxel start offsets in the sorted h_order array
    std::vector<uint32_t> voxel_start(num_voxels);
    {
        uint32_t off = 0;
        for (unsigned int i = 0; i < num_voxels; ++i) {
            voxel_start[i] = off;
            off += h_counts[i];
        }
    }

    // ── 5. Apply countThreshold; build hash map of valid voxels ─────────────
    std::unordered_map<int64_t, int> key_to_valid;
    key_to_valid.reserve(num_voxels * 2);

    std::vector<uint32_t> valid_vox;  // indices into h_keys / h_counts
    valid_vox.reserve(num_voxels);

    for (unsigned int i = 0; i < num_voxels; ++i) {
        if (static_cast<int>(h_counts[i]) >= p.countThreshold) {
            key_to_valid[h_keys[i]] = static_cast<int>(valid_vox.size());
            valid_vox.push_back(i);
        }
    }

    const unsigned int num_valid = static_cast<unsigned int>(valid_vox.size());
    if (num_valid == 0) { index[0] = 0; return 0; }

    // ── 6. BFS over 26-connected voxel graph ────────────────────────────────
    static constexpr int NDX[26] = {
        -1,0,1,-1,0,1,-1,0,1, -1,0,1,-1,1,-1,0,1, -1,0,1,-1,0,1,-1,0,1};
    static constexpr int NDY[26] = {
        -1,-1,-1,0,0,0,1,1,1, -1,-1,-1,0,0,1,1,1, -1,-1,-1,0,0,0,1,1,1};
    static constexpr int NDZ[26] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1, 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1};

    std::vector<int> voxel_cluster(num_valid, -1);
    int num_raw_clusters = 0;

    for (unsigned int vi = 0; vi < num_valid; ++vi) {
        if (voxel_cluster[vi] != -1) continue;

        const int cid = num_raw_clusters++;
        std::queue<unsigned int> bfs;
        bfs.push(vi);
        voxel_cluster[vi] = cid;

        while (!bfs.empty()) {
            const unsigned int cur = bfs.front(); bfs.pop();
            int32_t cx, cy, cz;
            decodeVoxel(h_keys[valid_vox[cur]], cx, cy, cz);

            for (int nb = 0; nb < 26; ++nb) {
                const int64_t nkey = encodeVoxel(
                    cx + NDX[nb], cy + NDY[nb], cz + NDZ[nb]);
                auto it = key_to_valid.find(nkey);
                if (it == key_to_valid.end()) continue;
                const int nbr = it->second;
                if (voxel_cluster[nbr] != -1) continue;
                voxel_cluster[nbr] = cid;
                bfs.push(static_cast<unsigned int>(nbr));
            }
        }
    }

    // ── 7. Tally point counts per cluster; filter by size ───────────────────
    std::vector<uint32_t> cluster_pts(num_raw_clusters, 0);
    for (unsigned int vi = 0; vi < num_valid; ++vi) {
        cluster_pts[voxel_cluster[vi]] += h_counts[valid_vox[vi]];
    }

    std::vector<int> remap(num_raw_clusters, -1);
    int out_count = 0;
    for (int ci = 0; ci < num_raw_clusters; ++ci) {
        const uint32_t cnt = cluster_pts[ci];
        if (cnt >= p.minClusterSize && cnt <= p.maxClusterSize) {
            remap[ci] = out_count++;
        }
    }

    // ── 8. Write output ──────────────────────────────────────────────────────
    // index[0]      = number of output clusters
    // index[c]      = point count of cluster c  (c = 1 … out_count)
    // output[…*4+…] = x, y, z, 0  for each point, in cluster order

    index[0] = static_cast<unsigned int>(out_count);
    if (out_count == 0) return 0;

    // Per-cluster point counts
    for (int c = 1; c <= out_count; ++c) index[c] = 0;
    for (unsigned int vi = 0; vi < num_valid; ++vi) {
        const int oc = remap[voxel_cluster[vi]];
        if (oc < 0) continue;
        index[oc + 1] += h_counts[valid_vox[vi]];
    }

    // Per-cluster write offsets
    std::vector<uint32_t> clust_off(out_count + 1, 0);
    for (int c = 0; c < out_count; ++c) {
        clust_off[c + 1] = clust_off[c] + index[c + 1];
    }
    std::vector<uint32_t> write_pos(clust_off.begin(), clust_off.begin() + out_count);

    // Copy point data (cloud_in is unified memory, readable from CPU)
    for (unsigned int vi = 0; vi < num_valid; ++vi) {
        const int oc = remap[voxel_cluster[vi]];
        if (oc < 0) continue;
        const uint32_t vox_i = valid_vox[vi];
        const uint32_t start = voxel_start[vox_i];
        const uint32_t cnt   = h_counts[vox_i];

        for (uint32_t k = 0; k < cnt; ++k) {
            const uint32_t src = h_order[start + k];
            const uint32_t dst = write_pos[oc]++;
            output[dst * 4 + 0] = cloud_in[src * 4 + 0];
            output[dst * 4 + 1] = cloud_in[src * 4 + 1];
            output[dst * 4 + 2] = cloud_in[src * 4 + 2];
            output[dst * 4 + 3] = 0.0f;
        }
    }

    return 0;
}
