//
// Created by Junkang on 2023/11/15.
//

#include <iostream>
#include <chrono>

#include <gtest/gtest.h>
#include <Common/Math/Simd.h>
#include <Common/Math/Vector.h>
#include <Common/Math/Matrix.h>

using namespace std::chrono;
using namespace Common;

// It seems that passing func by lambada may also cause some overload, and we can't get the accurate execute time
void MeasureFuncTime(const std::function<void()>& func, std::string name) {
    auto t1 = high_resolution_clock::now();
    for (uint32_t i = 0; i < 100; i++) {
        func();
    }
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms = t2 - t1;

    std::cout << "Function " << name << " took time: " << ms.count() / 100 << " ms\n";
}

TEST(SimdTest, BasicOperatorTest)
{
    FVec4 v1 {1, 2, 3, 4};
    FVec4 v2 {5, 6, 7, 8};
    FVec4 v3 {0, 0, 0, 0};

    // Add
    {
        auto t1 = high_resolution_clock::now();
        for (uint32_t i = 0; i < 100; i++) {
            SimdAdd<float>(v3.data, v1.data, v2.data);
        }
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms = t2 - t1;

        std::cout << "SimdAdd took time: " << ms.count() / 100 << " ms\n";
    }

    {
        auto t1 = high_resolution_clock::now();
        for (uint32_t k = 0; k < 100; k++) {
            for (uint32_t i = 0; i < 4; i++) {
                v3[i] = v1[i] + v2[i];
            }
        }
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms = t2 - t1;

        std::cout << "LoopAdd took time: " << ms.count() / 100 << " ms\n";
    }

    // Sub
    {
        auto t1 = high_resolution_clock::now();
        for (uint32_t i = 0; i < 100; i++) {
            SimdSub<float>(v3.data, v1.data, v2.data);
        }
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms = t2 - t1;

        std::cout << "SimdSub took time: " << ms.count() / 100 << " ms\n";
    }

    {
        auto t1 = high_resolution_clock::now();
        for (uint32_t k = 0; k < 100; k++) {
            for (uint32_t i = 0; i < 4; i++) {
                v3[i] = v1[i] - v2[i];
            }
        }
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms = t2 - t1;

        std::cout << "LoopSub took time: " << ms.count() / 100 << " ms\n";
    }

    // Mul
    {
        auto t1 = high_resolution_clock::now();
        for (uint32_t i = 0; i < 100; i++) {
            SimdMul<float>(v3.data, v1.data, v2.data);
        }
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms = t2 - t1;

        std::cout << "SimdMul took time: " << ms.count() / 100 << " ms\n";
    }

    {
        auto t1 = high_resolution_clock::now();
        for (uint32_t k = 0; k < 100; k++) {
            for (uint32_t i = 0; i < 4; i++) {
                v3[i] = v1[i] * v2[i];
            }
        }
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms = t2 - t1;

        std::cout << "LoopMul took time: " << ms.count() / 100 << " ms\n";
    }

    // Div
    {
        auto t1 = high_resolution_clock::now();
        for (uint32_t i = 0; i < 100; i++) {
            SimdDiv<float>(v3.data, v1.data, v2.data);
        }
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms = t2 - t1;

        std::cout << "SimdDiv took time: " << ms.count() / 100 << " ms\n";
    }

    {
        auto t1 = high_resolution_clock::now();
        for (uint32_t k = 0; k < 100; k++) {
            for (uint32_t i = 0; i < 4; i++) {
                v3[i] = v1[i] / v2[i];
            }
        }
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms = t2 - t1;

        std::cout << "LoopDiv took time: " << ms.count() / 100 << " ms\n";
    }

}

TEST(SimdTest, DotCrossTest)
{
    FVec4 v1 {1, 2, 3, 4};
    FVec4 v2 {5, 6, 7, 8};
    FVec4 v3 {0, 0, 0, 0};

    // Dot
    {
        auto t1 = high_resolution_clock::now();
        for (uint32_t k = 0; k < 100; k++) {
            float dotVal;
            SimdDot(v3.data, v1.data, v2.data);
            dotVal = v3[0];
        }
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms = t2 - t1;

        std::cout << "SimdDot took time: " << ms.count() / 100 << " ms\n";
    }

    {
        auto t1 = high_resolution_clock::now();
        for (uint32_t k = 0; k < 100; k++) {
            float dotVal = 0;
            for (uint32_t i = 0; i < 4; i++) {
                dotVal += v1[i] * v2[i];
            }
        }
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms = t2 - t1;

        std::cout << "LoopDot took time: " << ms.count() / 100 << " ms\n";
    }

    // Cross
    {
        auto t1 = high_resolution_clock::now();
        for (uint32_t k = 0; k < 100; k++) {
            SimdCross<float>(v3.data, v1.data, v2.data);
        }
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms = t2 - t1;

        std::cout << "SimdCross took time: " << ms.count() / 100 << " ms\n";
    }

    {
        auto t1 = high_resolution_clock::now();
        for (uint32_t k = 0; k < 100; k++) {
            v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
            v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
            v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
        }
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms = t2 - t1;

        std::cout << "LoopCross took time: " << ms.count() / 100 << " ms\n";
    }
}