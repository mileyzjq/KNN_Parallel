#include <x86intrin.h>
#include <assert.h>

float fvec_L2sqr_base(const float *x, const float *y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

// Following code from Faiss
static inline __m128 masked_read (int d, const float *x) {
    assert (0 <= d && d < 4);
    __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
    switch (d) {
        case 3:
            buf[2] = x[2];
        case 2:
            buf[1] = x[1];
        case 1:
            buf[0] = x[0];
    }
    return _mm_load_ps (buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}


// SSE implementation
float fvec_L2sqr_sse (const float *x, const float *y, size_t d) {
    __m128 msum1 = _mm_setzero_ps();

    while (d >= 4) {
        __m128 mx = _mm_loadu_ps (x); x += 4;
        __m128 my = _mm_loadu_ps (y); y += 4;
        const __m128 a_m_b1 = mx - my;
        msum1 += a_m_b1 * a_m_b1;
        d -= 4;
    }

    if (d > 0) {
        // add the last 1, 2 or 3 values
        __m128 mx = masked_read (d, x);
        __m128 my = masked_read (d, y);
        __m128 a_m_b1 = mx - my;
        msum1 += a_m_b1 * a_m_b1;
    }

    msum1 = _mm_hadd_ps (msum1, msum1);
    msum1 = _mm_hadd_ps (msum1, msum1);
    return  _mm_cvtss_f32 (msum1);
}