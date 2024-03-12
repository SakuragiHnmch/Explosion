//
// Created by Junkang on 2023/11/6.
//

#pragma once

#ifdef __x86_64__
#include <emmintrin.h> // we use sse2
#else
#include <sse2neon.h>
#endif

#include <Common/Math/Half.h>

namespace Common {
    /**
    * @param A0	Selects which element (0-3) from 'A' into 1st slot in the result
    * @param A1	Selects which element (0-3) from 'A' into 2nd slot in the result
    * @param B2	Selects which element (0-3) from 'B' into 3rd slot in the result
    * @param B3	Selects which element (0-3) from 'B' into 4th slot in the result
    */
    #define SHUFFLEMASK(A0,A1,B2,B3) ( (A0) | ((A1)<<2) | ((B2)<<4) | ((B3)<<6) )

    #define SHUFFLEMASK2(A0,A1) ((A0) | ((A1)<<1))

    // Only Support 4-component vector, the lhs and rhs must be packed(aligned)
    template<typename T>
    inline void SimdAdd(T* dst, const T* lhs, const T* rhs);

    template<typename T>
    inline void SimdSub(T* dst, const T* lhs, const T* rhs);

    template<typename T>
    inline void SimdMul(T* dst, const T* lhs, const T* rhs);

    template<typename T>
    inline void SimdDiv(T* dst, const T* lhs, const T* rhs);

    template<typename T>
    inline void SimdDot(T* dst, const T* lhs, const T* rhs);

    template<typename T>
    inline void SimdCross(T* dst, const T* lhs, const T* rhs);
}

namespace Common {
    template<typename T>
    inline void SimdAdd(T* dst, const T* lhs, const T* rhs)
    {
        for (int32_t i = 0; i < 4; i++) {
            dst[i] = lhs[i] + rhs[i];
        }
    }

    template<>
    inline void SimdAdd<float>(float* dst, const float* lhs, const float* rhs)
    {
        __m128 a = _mm_load_ps(lhs);
        __m128 b = _mm_load_ps(rhs);
        __m128 v = _mm_add_ps(a, b);

        _mm_store_ps(dst, v);
    }

    template<>
    inline void SimdAdd<double>(double* dst, const double* lhs, const double* rhs)
    {
        __m128d a = _mm_load_pd(lhs);
        __m128d b = _mm_load_pd(&lhs[2]);
        __m128d c = _mm_load_pd(rhs);
        __m128d d = _mm_load_pd(&rhs[2]);

        __m128d v1 = _mm_add_pd(a, c);
        __m128d v2 = _mm_add_pd(b, d);

        _mm_store_pd(dst, v1);
        _mm_store_pd(&dst[2], v2);
    }

    template<>
    inline void SimdAdd<HFloat>(HFloat* dst, const HFloat* lhs, const HFloat* rhs)
    {
        float fLhs[4], fRhs[4], fDst[4];
        for (int32_t i = 0; i < 4; i++) {
            fLhs[i] = lhs[i].AsFloat();
            fRhs[i] = rhs[i].AsFloat();
        }

        SimdAdd<float>(fDst, fLhs, fRhs);

        for (int32_t i = 0; i < 4; i++) {
            dst[i] = HFloat(fDst[i]);
        }
    }

    template<>
    inline void SimdAdd<int32_t>(int32_t* dst, const int32_t* lhs, const int32_t* rhs)
    {
        __m128i a = _mm_load_si128((__m128i*)lhs);
        __m128i b = _mm_load_si128((__m128i*)rhs);
        __m128i v = _mm_add_epi32(a, b);

        _mm_store_si128((__m128i*)dst, v);
    }

    template<>
    inline void SimdAdd<uint32_t>(uint32_t* dst, const uint32_t* lhs, const uint32_t* rhs)
    {
        __m128i a = _mm_load_si128((__m128i*)lhs);
        __m128i b = _mm_load_si128((__m128i*)rhs);
        __m128i v = _mm_add_epi32(a, b);

        _mm_store_si128((__m128i*)dst, v);
    }

    template<typename T>
    inline void SimdSub(T* dst, const T* lhs, const T* rhs)
    {
        for (int32_t i = 0; i < 4; i++) {
            dst[i] = lhs[i] - rhs[i];
        }
    }

    template<>
    inline void SimdSub<float>(float* dst, const float* lhs, const float* rhs)
    {
        __m128 a = _mm_load_ps(lhs);
        __m128 b = _mm_load_ps(rhs);
        __m128 v = _mm_sub_ps(a, b);

        _mm_store_ps(dst, v);
    }

    template<>
    inline void SimdSub<double>(double* dst, const double* lhs, const double* rhs)
    {
        __m128d a = _mm_load_pd(lhs);
        __m128d b = _mm_load_pd(&lhs[2]);
        __m128d c = _mm_load_pd(rhs);
        __m128d d = _mm_load_pd(&rhs[2]);

        __m128d v1 = _mm_sub_pd(a, c);
        __m128d v2 = _mm_sub_pd(b, d);

        _mm_store_pd(dst, v1);
        _mm_store_pd(&dst[2], v2);
    }

    template<>
    inline void SimdSub<HFloat>(HFloat* dst, const HFloat* lhs, const HFloat* rhs)
    {
        float fLhs[4], fRhs[4], fDst[4];
        for (int32_t i = 0; i < 4; i++) {
            fLhs[i] = lhs[i].AsFloat();
            fRhs[i] = rhs[i].AsFloat();
        }

        SimdSub<float>(fDst, fLhs, fRhs);

        for (int32_t i = 0; i < 4; i++) {
            dst[i] = HFloat(fDst[i]);
        }
    }

    template<>
    inline void SimdSub<int32_t>(int32_t* dst, const int32_t* lhs, const int32_t* rhs)
    {
        __m128i a = _mm_load_si128((__m128i*)lhs);
        __m128i b = _mm_load_si128((__m128i*)rhs);
        __m128i v = _mm_sub_epi32(a, b);

        _mm_store_si128((__m128i*)dst, v);
    }

    template<>
    inline void SimdSub<uint32_t>(uint32_t* dst, const uint32_t* lhs, const uint32_t* rhs)
    {
        __m128i a = _mm_load_si128((__m128i*)lhs);
        __m128i b = _mm_load_si128((__m128i*)rhs);
        __m128i v = _mm_sub_epi32(a, b);

        _mm_store_si128((__m128i*)dst, v);
    }

    template<typename T>
    inline void SimdMul(T* dst, const T* lhs, const T* rhs)
    {
        for (int32_t i = 0; i < 4; i++) {
            dst[i] = lhs[i] * rhs[i];
        }
    }

    template<>
    inline void SimdMul<float>(float* dst, const float* lhs, const float* rhs)
    {
        __m128 a = _mm_load_ps(lhs);
        __m128 b = _mm_load_ps(rhs);
        __m128 v = _mm_mul_ps(a, b);

        _mm_store_ps(dst, v);
    }

    template<>
    inline void SimdMul<double>(double* dst, const double* lhs, const double* rhs)
    {
        __m128d a = _mm_load_pd(lhs);
        __m128d b = _mm_load_pd(&lhs[2]);
        __m128d c = _mm_load_pd(rhs);
        __m128d d = _mm_load_pd(&rhs[2]);

        __m128d v1 = _mm_mul_pd(a, c);
        __m128d v2 = _mm_mul_pd(b, d);

        _mm_store_pd(dst, v1);
        _mm_store_pd(&dst[2], v2);
    }

    template<>
    inline void SimdMul<HFloat>(HFloat* dst, const HFloat* lhs, const HFloat* rhs)
    {
        float fLhs[4], fRhs[4], fDst[4];
        for (int32_t i = 0; i < 4; i++) {
            fLhs[i] = lhs[i].AsFloat();
            fRhs[i] = rhs[i].AsFloat();
        }

        SimdMul<float>(fDst, fLhs, fRhs);

        for (int32_t i = 0; i < 4; i++) {
            dst[i] = HFloat(fDst[i]);
        }
    }

    template<>
    inline void SimdMul<int32_t>(int32_t* dst, const int32_t* lhs, const int32_t* rhs)
    {
        __m128i a = _mm_load_si128((__m128i*)lhs);
        __m128i b = _mm_load_si128((__m128i*)rhs);
        __m128i v = _mm_mullo_epi32(a, b);

        _mm_store_si128((__m128i*)dst, v);
    }

    template<>
    inline void SimdMul<uint32_t>(uint32_t* dst, const uint32_t* lhs, const uint32_t* rhs)
    {
        __m128i a = _mm_load_si128((__m128i*)lhs);
        __m128i b = _mm_load_si128((__m128i*)rhs);
        __m128i v = _mm_mullo_epi32(a, b);

        _mm_store_si128((__m128i*)dst, v);
    }

    template<typename T>
    inline void SimdDiv(T* dst, const T* lhs, const T* rhs)
    {
        for (int32_t i = 0; i < 4; i++) {
            dst[i] = lhs[i] / rhs[i];
        }
    }

    template<>
    inline void SimdDiv<float>(float* dst, const float* lhs, const float* rhs)
    {
        __m128 a = _mm_load_ps(lhs);
        __m128 b = _mm_load_ps(rhs);
        __m128 v = _mm_div_ps(a, b);

        _mm_store_ps(dst, v);
    }

    template<>
    inline void SimdDiv<double>(double* dst, const double* lhs, const double* rhs)
    {
        __m128d a = _mm_load_pd(lhs);
        __m128d b = _mm_load_pd(&lhs[2]);
        __m128d c = _mm_load_pd(rhs);
        __m128d d = _mm_load_pd(&rhs[2]);

        __m128d v1 = _mm_div_pd(a, c);
        __m128d v2 = _mm_div_pd(b, d);

        _mm_store_pd(dst, v1);
        _mm_store_pd(&dst[2], v2);
    }

    template<>
    inline void SimdDiv<HFloat>(HFloat* dst, const HFloat* lhs, const HFloat* rhs)
    {
        float fLhs[4], fRhs[4], fDst[4];
        for (int32_t i = 0; i < 4; i++) {
            fLhs[i] = lhs[i].AsFloat();
            fRhs[i] = rhs[i].AsFloat();
        }

        SimdDiv<float>(fDst, fLhs, fRhs);

        for (int32_t i = 0; i < 4; i++) {
            dst[i] = HFloat(fDst[i]);
        }
    }

    //TODO Integer Division seems not worth to implement now
    // https://stackoverflow.com/questions/16822757/sse-integer-division

    template<typename T>
    inline void SimdDot(T* dst, const T* lhs, const T* rhs)
    {
        T temp = 0;
        for (int32_t i = 0; i < 4; i++) {
            temp += lhs[i] * rhs[i];
        }

        memcpy(dst, &temp, sizeof(T));
    }

    template<>
    inline void SimdDot<float>(float* dst, const float* lhs, const float* rhs)
    {
        __m128 a = _mm_load_ps(lhs);
        __m128 b = _mm_load_ps(rhs);

        __m128 v = _mm_mul_ps(a, b);                               // (XX, YY, ZZ, WW)
        __m128 t = _mm_shuffle_ps(v, v, SHUFFLEMASK(2, 3, 0, 1));  // (WW, ZZ, XX, YY)
        v = _mm_add_ps(v, t);                                 // (XX + WW, YY + ZZ, ZZ + XX, WW + YY);
        t = _mm_shuffle_ps(v, v, SHUFFLEMASK(1, 0, 3, 2));         // (YY + ZZ, XX + WW, WW + YY, ZZ + XX);
        v = _mm_add_ps(v, t);                                 // (XX + WW + YY + ZZ, ...)

        _mm_store_ps(dst, v);
    }

    template<>
    inline void SimdDot<double>(double* dst, const double* lhs, const double* rhs)
    {
        __m128d a = _mm_load_pd(lhs);
        __m128d b = _mm_load_pd(&lhs[2]);
        __m128d c = _mm_load_pd(rhs);
        __m128d d = _mm_load_pd(&rhs[2]);

        __m128d v1 = _mm_mul_pd(a, c);                      // (XX, YY)
        __m128d v2 = _mm_mul_pd(b, d);                   // (ZZ, WW)
        __m128d v3 = _mm_add_pd(v1, v2);                 // (XX + ZZ, YY + WW)
        v1 = _mm_shuffle_pd(v3, v3, SHUFFLEMASK2(1, 0)); // (YY + WW, XX + ZZ)
        v3 = _mm_add_pd(v1, v3);                         // (XX + ZZ + YY + WW, ...)

        _mm_store_pd(dst, v3);
    }

    template<>
    inline void SimdDot<HFloat>(HFloat* dst, const HFloat* lhs, const HFloat* rhs)
    {
        float fLhs[4], fRhs[4], fDst[4];
        for (int32_t i = 0; i < 4; i++) {
            fLhs[i] = lhs[i].AsFloat();
            fRhs[i] = rhs[i].AsFloat();
        }

        SimdDot<float>(fDst, fLhs, fRhs);

        for (int32_t i = 0; i < 4; i++) {
            dst[i] = HFloat(fDst[i]);
        }
    }

    template<typename T>
    inline void SimdCross(T* dst, const T* lhs, const T* rhs)
    {
        dst[0] = lhs[1] * rhs[2] - lhs[2] * rhs[1];
        dst[1] = lhs[2] * rhs[0] - lhs[0] * rhs[2];
        dst[2] = lhs[0] * rhs[1] - lhs[1] * rhs[0];
    }

    template<>
    inline void SimdCross<float>(float* dst, const float* lhs, const float* rhs)
    {
        __m128 a = _mm_load_ps(lhs);
        __m128 b = _mm_load_ps(rhs);

        __m128 c = _mm_mul_ps(a, _mm_shuffle_ps(b, b, SHUFFLEMASK(1, 2, 0, 3)));
        __m128 d = _mm_mul_ps(_mm_shuffle_ps(a, a, SHUFFLEMASK(1, 2, 0, 3)), b);

        __m128 v = _mm_sub_ps(c, d);
        v = _mm_shuffle_ps(v, v, SHUFFLEMASK(1, 2, 0, 3));

        _mm_store_ps(dst, v);
    }

    template<>
    inline void SimdCross<HFloat>(HFloat* dst, const HFloat* lhs, const HFloat* rhs)
    {
        float fLhs[4], fRhs[4], fDst[4];
        for (int32_t i = 0; i < 4; i++) {
            fLhs[i] = lhs[i].AsFloat();
            fRhs[i] = rhs[i].AsFloat();
        }

        SimdCross<float>(fDst, fLhs, fRhs);

        for (int32_t i = 0; i < 4; i++) {
            dst[i] = HFloat(fDst[i]);
        }
    }
}


