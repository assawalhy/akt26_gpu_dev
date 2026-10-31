# AKT26 Session 4: CUDA kernel development

My copy of the session 4 GPU assignments. Only `3-ger` and `4-gemm` are graded;
`0-hello-world`, `1-axpy` and `2-dot` are the worked samples from the lecture.

| directory | what it does |
|---|---|
| `0-hello-world` | prints the device properties and launches a trivial kernel |
| `1-axpy` | `y = alpha * x + y`, compared against cuBLAS `SAXPY` |
| `2-dot` | dot product, compared against cuBLAS `Sdot` |
| `3-ger` | rank-1 update `A = A + alpha * x * y^T`, compared against `SGER` |
| `4-gemm` | batched `C_i = A_i * B_i`, compared against `SgemmStridedBatched` |

All matrices are column-major, so `A(i, j)` lives at `A[j * m + i]`.

## Building and running

```sh
cd 3-ger && make && ./ger
cd 4-gemm && make && ./gemm
```

`nvcc` must be on `PATH` and the CUDA toolkit must be reachable. The makefiles
take `CUDADIR`, defaulting to `/usr/local/cuda`. Both programs accept their
problem size on the command line:

```sh
./ger 4000 2000 100     # m, n, repeats
./gemm 16 1000          # n (1..32), batch size
```

Each run checks the result against a CPU reference and against cuBLAS, then
prints `TEST PASS` or `TEST FAIL` and the speedup over cuBLAS.

## Why the makefiles target `sm=75`

This was not the original default, and it is worth writing down.

`sm` means streaming multiprocessor. `sm_75` is the *compute capability* of the
GPU the code is compiled for, `7.5` for the Quadro T2000 in this laptop, which
is Nvidia's Turing generation. It is a hardware generation, unrelated to the
driver version or to the CUDA toolkit version.

`nvcc` can emit two kinds of code for an architecture:

- `code=sm_75` produces SASS, the real machine code. It runs on that
  architecture only, and is not portable to a newer one.
- `code=compute_75` produces PTX, a virtual instruction set that the driver
  compiles for whatever GPU it finds at load time. PTX is forward compatible
  only: `compute_75` runs on 8.0, 8.9, 9.0 and later, but `compute_80` cannot
  run on 7.5.

The makefiles used to say `sm=80`, which is Ampere. That gave the T2000 nothing
it could execute, and every binary died at the first kernel with:

```
no kernel image is available for execution on the device
```

`sm=75` fixes it, and costs nothing on newer hardware: the `compute_75` PTX is
JIT-compiled when the binary runs on an Ampere or newer card. If you need native
Ampere code instead, override it per build without editing anything:

```sh
make sm=80
```

To read the compute capability of any machine:

```sh
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

## 3-ger: rank-1 update

`A = A + alpha * x * y^T` on an `m x n` matrix.

One thread per element of `A`. The block is `32 x 8`, and `threadIdx.x` is the
row index `i`, so consecutive lanes in a warp touch consecutive elements of `A`
and of `x`, which keeps both the load and the store coalesced. `threadIdx.y`
covers eight columns at a time. The grid is `ceil(m / 32) x ceil(n / 8)`, and
the kernel checks `i < m && j < n` because neither dimension is necessarily a
multiple of the block size.

No shared memory: `x` and `y` are tiny and every value is reused across the
whole warp, so they come out of cache. The arithmetic intensity is one
multiply-add per 4-byte store, so this kernel is bandwidth bound, and it lands
at roughly the same throughput as cuBLAS `SGER`.

## 4-gemm: batched matrix multiplication

`C_i = A_i * B_i` for `i < S`, with `n <= 32`, column-major, and the three
batches stored contiguously.

The kernel assigns one thread per element of `C` per matrix, with
`threadIdx.x` as the row and `threadIdx.y` as the column so global accesses stay
coalesced, and `threadIdx.z` selecting which matrix of the block a thread works
on. `n <= 32` means a whole `A_i` and a whole `B_i` fit in shared memory, so
each block stages entire matrices once and every thread then reads them from
shared memory instead of from global memory.

A block covers `mpb` matrices:

```
mpb = min(64, max(1, 256 / (n * n)))
```

The `256 / (n * n)` term keeps the block near 256 threads, since `n * n` threads
are needed per matrix. The `min(64, ...)` is a hardware limit: `blockDim.z`
cannot exceed 64, and for `n = 1` the formula alone would ask for 256. Without
that clamp the launch is rejected, the kernel never runs, and `C` stays zero.
The shared memory footprint is `mpb * 2 * n * n * 4` bytes, allocated
dynamically at launch.

The `k` loop runs in ascending order so the summation matches the CPU reference
and the error stays at the level of normal floating-point reassociation rather
than anything larger.

At the default `n = 16` the formula gives `mpb = 1`, so each block handles a
single matrix with 256 threads and 2 KB of shared memory. The likely reason this
beats cuBLAS by 2.5x is that the entire `A_i` and `B_i` are staged before any
arithmetic starts, so the inner loop touches only shared memory, and cuBLAS has
no specialized path for a 16x16 product. Larger `mpb` kicks in only for small
`n`, where one matrix would otherwise occupy a block of a few threads.

The cuBLAS comparison uses `cublasSgemmStridedBatched` with `CUBLAS_OP_N` on
both operands, `lda = ldb = ldc = n`, stride `n * n`, `alpha = 1`, `beta = 0`.

## Measured results

Quadro T2000 with Max-Q Design, compute capability 7.5, driver 595.71.05, CUDA
12.9, built with `make` (that is, `sm=75`).

`3-ger`, 4000 x 2000, 100 repeats:

| implementation | time (us) | GFLOP/s | error norm |
|---|---|---|---|
| custom kernel | 487.8 | 32.80 | 4.44e-08 |
| cuBLAS `SGER` | 513.7 | 31.15 | 4.44e-08 |

`4-gemm`, n = 16, batch 1000, 100 repeats:

| implementation | time (us) | GFLOP/s | max batch fwd error |
|---|---|---|---|
| custom kernel | 39.1 | 209.75 | 2.44e-08 |
| cuBLAS strided batched | 97.8 | 83.79 | 3.20e-08 |

Both kernels match the reference to within 4.5e-08 relative error, and both beat
cuBLAS at these sizes.

## Correctness beyond the defaults

The default sizes are the easy case. Both programs were also run on shapes that
break assumptions:

- `ger` with `m/n` of 1/1, 33/9 and 1000/7, where neither dimension is a
  multiple of the block, exercising the bounds check.
- `gemm` with `n` of 1, 2, 8, 16, 17 and 32, at batch sizes 1, 3, 7 and 1000,
  exercising `mpb`, the shared memory sizing and the `blockDim.z` clamp.

All pass. The `n = 1` case is what exposed the `blockDim.z` limit above.
