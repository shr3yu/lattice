#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <algorithm>

using namespace std;

// ---------- GLOBAL VARIABLES ----------
const unsigned int SCR_WIDTH = 1200; // Wider for UI
const unsigned int SCR_HEIGHT = 800;

// Simulation Variables
float appliedVoltage = 0.0f;
float temperature = 2.0f;
float simulationSpeed = 1.0f;

// Measurement
int chargeCrossingCount = 0;
float currentMetric = 0.0f;
float measureTimer = 0.0f;

// ---------- RANDOM NUMBERS ----------

// xorshift32: a few cycles per number and no shared state, unlike rand(),
// which is slow and not safe to call from multiple threads
struct XorShift {
    uint32_t state;
    float next() { // uniform in [0, 1)
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state * (1.0f / 4294967296.0f);
    }
};

// ---------- PARTICLE SYSTEM ----------

// Struct-of-arrays layout: each field is its own contiguous array, so the update
// loop streams through memory linearly and the compiler can vectorize it
struct ParticleSystem {
    vector<float> posX, posY, velX, velY, homeX, homeY;
    vector<uint8_t> isFree;

    size_t size() const { return posX.size(); }

    void init(int cols, int rows) {
        size_t n = (size_t)cols * rows;
        for (auto* a : { &posX, &posY, &velX, &velY, &homeX, &homeY }) a->assign(n, 0.0f);
        isFree.assign(n, 0);
        size_t k = 0;
        for (int i = 0; i < cols; i++) { //creates a solid block of material
            for (int j = 0; j < rows; j++, k++) {
                posX[k] = homeX[k] = (float)i / cols * 1.8f - 0.9f;
                posY[k] = homeY[k] = (float)j / rows * 1.6f - 0.8f;
            }
        }
    }

    // Updates particles [begin, end) and writes their vertices straight into the GPU staging buffer.
    // Returns the number of charges that crossed the right edge (for the current measurement).
    int update(size_t begin, size_t end, float electricField, float dt, float temp, XorShift& rng, float* gpuOut, int& freeCount) {
        const float damping = 0.90f;
        const float jumpChance = 0.002f * temp; // the probability of an electron jumping to the conduction band increases with temperature
        const float recombinationRate = 0.01f;
        int crossings = 0;
        int freeHere = 0;

        for (size_t i = begin; i < end; i++) {
            bool free = isFree[i];
            float randomVal = rng.next();

            // 1. STATE TRANSITIONS
            if (!free) {
                if (randomVal < jumpChance) free = true; // Jump to Conduction Band
            }
            else if (randomVal < recombinationRate) { // Recombination
                free = false;
                velX[i] = velY[i] = 0.0f;
                homeX[i] = posX[i]; // falls back to a nearby hole
                homeY[i] = posY[i];
            }

            float rX = rng.next() * 2.0f - 1.0f;
            float rY = rng.next() * 2.0f - 1.0f;

            // 2. MOVEMENT
            if (free) {
                // electrons constantly lose energy by bumping into atoms (prevents electrons from accelerating indefinitely)
                float vx = (velX[i] + electricField * dt) * damping;
                float vy = velY[i] * damping;

                // Thermal Jitter: random unit vector scaled by temperature
                float len = std::sqrt(rX * rX + rY * rY);
                float k = len > 0.0f ? temp * 0.05f / len : 0.0f;
                vx += rX * k;
                vy += rY * k;

                velX[i] = vx;
                velY[i] = vy;
                posX[i] += vx * dt;
                posY[i] += vy * dt;
            }
            else {
                // Valence Vibration plus an electrostatic pull towards the home position (towards lower potential)
                posX[i] += (homeX[i] - posX[i]) * 0.1f + rX * 0.005f * temp;
                posY[i] += (homeY[i] - posY[i]) * 0.1f + rY * 0.005f * temp;
            }

            // 3. BOUNDARIES
            if (posY[i] > 0.9f) { posY[i] = 0.9f; if (free) velY[i] = -velY[i]; }
            if (posY[i] < -0.9f) { posY[i] = -0.9f; if (free) velY[i] = -velY[i]; }

            if (free && posX[i] > 1.0f) { posX[i] = -1.0f; crossings++; }
            else if (free && posX[i] < -1.0f) { posX[i] = 1.0f; }

            isFree[i] = free;
            freeHere += free;

            float* v = gpuOut + i * 4;
            v[0] = posX[i];
            v[1] = posY[i];
            v[2] = 0.0f;
            v[3] = free ? 1.0f : 0.0f;
        }
        freeCount = freeHere;
        return crossings;
    }
};

// ---------- THREAD POOL ----------

// Persistent workers that each own a fixed slice of the particles. Threads are created
// once at startup instead of every frame; run() wakes them and blocks until all finish.
class WorkerPool {
public:
    explicit WorkerPool(int n) : count(max(1, n)) {
        for (int t = 1; t < count; t++) workers.emplace_back([this, t] { loop(t); });
    }
    ~WorkerPool() {
        { lock_guard<mutex> lk(m); quitting = true; generation++; }
        wake.notify_all();
        for (auto& w : workers) w.join();
    }
    int size() const { return count; }

    // Runs job(threadIndex) on every worker; the calling thread acts as worker 0
    void run(const function<void(int)>& fn) {
        if (count == 1) { fn(0); return; }
        { lock_guard<mutex> lk(m); job = &fn; pending = count - 1; generation++; }
        wake.notify_all();
        fn(0);
        unique_lock<mutex> lk(m);
        done.wait(lk, [this] { return pending == 0; });
    }

private:
    void loop(int t) {
        uint64_t seen = 0;
        for (;;) {
            const function<void(int)>* fn;
            {
                unique_lock<mutex> lk(m);
                wake.wait(lk, [&] { return generation != seen; });
                seen = generation;
                if (quitting) return;
                fn = job;
            }
            (*fn)(t);
            { lock_guard<mutex> lk(m); if (--pending == 0) done.notify_one(); }
        }
    }

    int count;
    vector<thread> workers;
    mutex m;
    condition_variable wake, done;
    const function<void(int)>* job = nullptr;
    int pending = 0;
    uint64_t generation = 0;
    bool quitting = false;
};

// One physics step for every particle, split evenly across the pool
struct Simulator {
    ParticleSystem ps;
    WorkerPool pool;
    vector<XorShift> rngs;
    vector<int> crossings, freeCounts;
    vector<float> gpuData; // allocated once, rewritten in place every frame

    Simulator(int cols, int rows, int threads, uint32_t seed) : pool(threads) {
        ps.init(cols, rows);
        gpuData.assign(ps.size() * 4, 0.0f);
        for (int t = 0; t < pool.size(); t++) rngs.push_back({ (seed + 1) * 0x9E3779B9u * (t + 1) | 1u });
        crossings.assign(pool.size(), 0);
        freeCounts.assign(pool.size(), 0);
    }

    // Returns charges that crossed the right edge this step
    int step(float electricField, float dt, float temp) {
        size_t n = ps.size();
        int workers = pool.size();
        pool.run([&](int t) {
            size_t begin = n * t / workers, end = n * (t + 1) / workers;
            crossings[t] = ps.update(begin, end, electricField, dt, temp, rngs[t], gpuData.data(), freeCounts[t]);
        });
        int total = 0;
        for (int c : crossings) total += c;
        return total;
    }

    int activeCarriers() const {
        int total = 0;
        for (int c : freeCounts) total += c;
        return total;
    }
};

// ---------- BENCHMARK ----------

// Headless timing of the physics step (no window, no GPU):  --bench [particles] [frames]
int runBenchmark(size_t particles, int frames) {
    int cols = max(1, (int)std::sqrt(particles * 5.0 / 3.0)); // keep the default 5:3 grid shape
    int rows = max(1, (int)(particles / cols));
    int hw = max(1, (int)thread::hardware_concurrency());
    printf("Benchmark: %d particles, %d frames, V=10 T=2.0\n", cols * rows, frames);

    vector<int> threadCounts = { 1, 2, 4 };
    if (hw > 4) threadCounts.push_back(hw);
    double baseline = 0.0;
    for (int threads : threadCounts) {
        Simulator sim(cols, rows, threads, 1);
        for (int f = 0; f < 20; f++) sim.step(10.0f * 0.3f, 0.01f, 2.0f); // warm up
        auto t0 = chrono::steady_clock::now();
        for (int f = 0; f < frames; f++) sim.step(10.0f * 0.3f, 0.01f, 2.0f);
        double ms = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count() / frames;
        if (threads == 1) baseline = ms;
        printf("  %2d thread(s): %8.3f ms/frame  (%.2fx)  free carriers %.1f%%\n",
            threads, ms, baseline / ms, 100.0 * sim.activeCarriers() / sim.ps.size());
    }
    return 0;
}

const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 1) in float aIsFree;\n"
"out float isFree;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"   isFree = aIsFree;\n"
"}\0";

const char* fragmentShaderSource = "#version 330 core\n"
"in float isFree;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"   if(isFree > 0.5) \n"
"       FragColor = vec4(1.0f, 0.9f, 0.2f, 1.0f); // Yellow\n"
"   else \n"
"       FragColor = vec4(0.3f, 0.4f, 0.5f, 1.0f); // Gray\n"
"}\n\0";

void framebuffer_size_callback(GLFWwindow* window, int width, int height);

int main(int argc, char** argv)
{
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        size_t particles = argc > 2 ? strtoull(argv[2], nullptr, 10) : 1000000;
        int frames = argc > 3 ? atoi(argv[3]) : 200;
        return runBenchmark(particles, frames);
    }

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "NVIDIA/Apple Demo - Semiconductor Physics", NULL, NULL);
    if (window == NULL) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { return -1; }

    // ------------------------------------
    // SETUP IMGUI
    // ------------------------------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark(); // Cool dark theme
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Shader Setup
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Particle Setup
    int rows = 30;
    int cols = 50;
    // Waking a thread costs more than updating a few thousand particles, so only
    // spread the work out when each thread gets a meaningful slice (~25K particles)
    int threads = max(1, min(rows * cols / 25000, (int)thread::hardware_concurrency()));
    Simulator sim(cols, rows, threads, (uint32_t)time(0));

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sim.gpuData.size() * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    float physicsMs = 0.0f;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // IMGUI FRAME START
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // UI DESIGN
        ImGui::Begin("Lab Controls");

        ImGui::Text("Device Parameters");
        ImGui::SliderFloat("Voltage (V)", &appliedVoltage, 0.0f, 10.0f);
        ImGui::SliderFloat("Temperature (T)", &temperature, 0.0f, 5.0f);
        ImGui::SliderFloat("Time Scale", &simulationSpeed, 0.0f, 2.0f);

        ImGui::Separator();

        ImGui::Text("Measurements");
        // Simple color change for current text
        if (currentMetric > 5.0f) ImGui::TextColored(ImVec4(1, 1, 0, 1), "Current: %.2f mA", currentMetric);
        else ImGui::Text("Current: %.2f mA", currentMetric);

        ImGui::Text("Active Carriers: %d", sim.activeCarriers());

        ImGui::Text("Physics step %.3f ms on %d threads", physicsMs, sim.pool.size());
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();

        // PHYSICS UPDATE

        // Calculate current metric
        measureTimer += 1.0f;
        if (measureTimer > 60.0f) {
            currentMetric = (float)chargeCrossingCount / 6.0f; // Scale for display
            chargeCrossingCount = 0;
            measureTimer = 0.0f;
        }

        float dt = 0.01f * simulationSpeed;
        float electricField = appliedVoltage * 0.3f;

        auto physicsStart = chrono::steady_clock::now();
        chargeCrossingCount += sim.step(electricField, dt, temperature);
        physicsMs = chrono::duration<float, milli>(chrono::steady_clock::now() - physicsStart).count();

        // RENDER
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sim.gpuData.size() * sizeof(float), sim.gpuData.data());

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glPointSize(4.0f);
        glDrawArrays(GL_POINTS, 0, (GLsizei)sim.ps.size());

        // Render ImGui over the top
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}
