#ifndef HELPER_TIMER_H
#define HELPER_TIMER_H

#include <chrono>

class StopWatchInterface {
public:
    virtual ~StopWatchInterface() {}
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void reset() = 0;
    virtual float getTime() = 0;
};

class StopWatchSimple : public StopWatchInterface {
private:
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point stopTime;
    bool running;
    double elapsedMs;

public:
    StopWatchSimple() : running(false), elapsedMs(0.0) {}

    virtual void start() override {
        startTime = std::chrono::high_resolution_clock::now();
        running = true;
    }

    virtual void stop() override {
        if (running) {
            stopTime = std::chrono::high_resolution_clock::now();
            elapsedMs += std::chrono::duration<double, std::milli>(stopTime - startTime).count();
            running = false;
        }
    }

    virtual void reset() override {
        elapsedMs = 0.0;
        running = false;
    }

    virtual float getTime() override {
        if (running) {
            auto now = std::chrono::high_resolution_clock::now();
            return static_cast<float>(elapsedMs + std::chrono::duration<double, std::milli>(now - startTime).count());
        }
        return static_cast<float>(elapsedMs);
    }
};

inline void sdkCreateTimer(StopWatchInterface** timer) {
    *timer = new StopWatchSimple();
}

inline void sdkDeleteTimer(StopWatchInterface** timer) {
    if (timer && *timer) {
        delete *timer;
        *timer = nullptr;
    }
}

inline void sdkStartTimer(StopWatchInterface** timer) {
    if (timer && *timer) (*timer)->start();
}

inline void sdkStopTimer(StopWatchInterface** timer) {
    if (timer && *timer) (*timer)->stop();
}

inline void sdkResetTimer(StopWatchInterface** timer) {
    if (timer && *timer) (*timer)->reset();
}

inline float sdkGetTimerValue(StopWatchInterface** timer) {
    if (timer && *timer) return (*timer)->getTime();
    return 0.0f;
}

#endif
