# Semiconductor Physics Simulator: Charge Carrier Dynamics
A real-time, low-level C++ simulation of electron behavior in a semiconductor crystal lattice under varying electromagnetic and thermal conditions.

A lot of the project revolved understanding physical fundamentals of the behaviour of electrons to craft an accurate simulation.
Refer Docs for more Info: https://docs.google.com/document/d/1PtdvExCiqH26lgM7H7UgSPyRn9OdLXuFwajHFDokj0U/edit?tab=t.0


https://github.com/user-attachments/assets/67e1e22f-8ecb-4140-aa48-77cb189d9614



## Overview
This project simulates the microscopic movement of electrons within a semiconductor material. By modeling the transition between the Valence Band and the Conduction Band, the simulator visualizes how temperature and voltage influence electrical conductivity.

## Concepts Applied
1. Thermal Excitation:
    - Models the probability of electrons jumping the Band Gap using a temperature-dependent stochastic model ($P \propto T$)
  
2. Recombination & Generation:
    - Simulates the life cycle of charge carriers as they transition from free-moving conduction states back to bound valence states.
  
3. Drift-Diffusion Model:
    - Electrons exhibit "Drift Velocity" in response to an applied Electric Field (Voltage), while "Thermal Jitter" simulates Brownian-like diffusion
  
4. Hardware Acceleration:
     - Utilizes OpenGL for rendering, passing simulation data directly to the GPU via Vertex Buffer Objects (VBOs).

## Performance
The physics step was optimized for cache locality and multi-core CPUs:
* **Struct-of-arrays layout**: each particle field lives in its own contiguous array, so the update loop streams through memory linearly and the compiler can vectorize it
* **Per-thread xorshift RNG**: replaces `rand()`, which is slower and not thread-safe
* **Zero per-frame allocation**: vertices are written in place into a GPU staging buffer allocated once at startup
* **Persistent worker pool**: particles are split across cores; it only kicks in above ~25K particles, where the work outweighs the cost of waking threads

Measured on an Apple M5 (`-O2`), 1M particles:

| Version | Physics step | Speedup |
|---|---|---|
| Original (array-of-structs, `rand()`, per-frame `push_back`) | 26.4 ms | 1.0x |
| Optimized, 1 thread | 9.4 ms | 2.8x |
| Optimized, 4 threads | 3.5 ms | 7.6x |
| Optimized, 10 threads | 2.6 ms | 10.1x |

At the default 1,500-particle grid, the step dropped from 39 µs to 14 µs. The Lab Controls panel shows the live physics step time.

Run the benchmark yourself (headless, no window):
```
./Electrical_Simulation --bench [particles] [frames]
```

## Tech Stack
* C++11/17
* OpenGL 3.3
* GLSL
* GLFW/GLAD
* ImGUI

## Future Enhancements
1. P-N Junction logic to simulate diode rectification
2. Integration of the Boltzmann Factor (instead of the simplified linear relationship of P and T)

