# Lattice: Multithreaded Semiconductor Simulator
A real-time C++ engine that simulates up to 1M charge carriers in a semiconductor crystal lattice, built around a cache-friendly data layout, a custom worker thread pool, and a streaming CPU→GPU render pipeline.

**10.1× faster** physics step after optimization (26.4 → 2.6 ms/frame at 1M particles on Apple Silicon).

https://github.com/user-attachments/assets/67e1e22f-8ecb-4140-aa48-77cb189d9614

## Highlights
* **Data-oriented design**: particles stored as struct-of-arrays so the hot loop streams linearly through memory and auto-vectorizes
* **Concurrency**: persistent worker pool with contention-free range partitioning; each thread owns its slice of particles and its own RNG, so the update needs no locks
* **Zero per-frame allocation**: vertex data is written in place into a staging buffer allocated once, then uploaded to the GPU in a single `glBufferSubData` call and one draw call
* **Built-in observability**: live physics-step timing and FPS in the control panel, plus a headless `--bench` mode for reproducible measurements
* **Adaptive scheduling**: the thread pool only engages above ~25K particles, where parallel work outweighs the cost of waking threads

## Architecture
```
 Main thread, per frame:
   ImGui input ──► Simulator::step() ──► GPU upload ──► Draw
                          │
                          │ dispatch slices
          ┌───────────────┼───────────────┐
      Worker 0        Worker 1   ...   Worker N
      particles       particles        particles
      [0, n/N)        [n/N, 2n/N)      [.., n)
      own xorshift    own xorshift     own xorshift
          └───────────────┼───────────────┘
                          ▼
     Shared vertex buffer (each worker writes only its range)
```
* `ParticleSystem`: struct-of-arrays state + the per-particle physics update
* `WorkerPool`: threads created once at startup; a generation counter and condition variables wake workers each frame and join them before rendering
* `Simulator`: owns the system, pool, per-thread RNGs and the GPU staging buffer; aggregates per-thread measurements (crossings, active carriers)

## Performance
Measured on an Apple M5 (`-O2`), 1M particles:

| Version | Physics step | Speedup |
|---|---|---|
| Original (array-of-structs, `rand()`, per-frame `push_back`) | 26.4 ms | 1.0× |
| Optimized, 1 thread | 9.4 ms | 2.8× |
| Optimized, 4 threads | 3.5 ms | 7.6× |
| Optimized, 10 threads | 2.6 ms | 10.1× |

At the default 1,500-particle grid, the step dropped from 39 µs to 14 µs.

Scaling flattens past 4 threads: the M5 mixes performance and efficiency cores, and at ~2.6 ms/frame the loop is increasingly bound by memory bandwidth (~16 MB of vertex data written per frame).

Run the benchmark yourself (headless, no window):
```
./Electrical_Simulation --bench [particles] [frames]
```

## Physics Model
Each particle is an electron that is either bound in the valence band or free in the conduction band.

1. **Thermal excitation**: bound electrons jump the band gap with a temperature-dependent probability ($P \propto T$)
2. **Recombination & generation**: free carriers fall back into a nearby hole, re-anchoring to a new lattice position
3. **Drift-diffusion**: free electrons accelerate under the applied electric field (drift) with damping from lattice collisions, plus random thermal jitter (diffusion)
4. **Lattice vibration**: bound electrons oscillate around their home position under an electrostatic restoring force
5. **Current measurement**: carriers crossing the right boundary are counted and averaged over 60 frames to report current

Voltage, temperature and time scale are adjustable live. More detail on the physics: [design doc](https://docs.google.com/document/d/1PtdvExCiqH26lgM7H7UgSPyRn9OdLXuFwajHFDokj0U/edit?tab=t.0)

## Tech Stack
* C++17 (STL threads, mutex, condition variables)
* OpenGL 3.3 core, GLSL
* GLFW, GLAD
* Dear ImGui

## Roadmap
1. Move the physics update to a GPU compute shader (or Metal) to remove the CPU→GPU copy
2. Thread pool tuning: pin work to performance cores, double-buffer state to overlap simulation with rendering
3. P-N junction logic to simulate diode rectification
4. Replace the linear $P \propto T$ model with the Boltzmann factor
