#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define X86_DISPATCH 1
#else
#define X86_DISPATCH 0
#endif

#define AREA 4096
#define HEIGHT 16
#define MAX_COLUMNS 16
#define HEADER 22
#define COLUMN_STRIDE 10
#define SHAPE_STRIDE (HEADER + MAX_COLUMNS * COLUMN_STRIDE)
#define MAX_BATCH 512

int t4_abi(void) { return 1; }

int t4_capabilities(void) {
#if X86_DISPATCH
    __builtin_cpu_init();
    return (!!__builtin_cpu_supports("avx2")) | ((!!__builtin_cpu_supports("bmi2")) << 1);
#else
    return 0;
#endif
}

int t4_shape_stride(void) { return SHAPE_STRIDE; }

int t4_validate_shapes(const int32_t *shapes, int count) {
    if (!shapes || count < 1 || count > 1024) return -1;
    for (int r = 0; r < count; ++r) {
        const int32_t *s = shapes + r * SHAPE_STRIDE;
        for (int axis = 0; axis < 4; ++axis) if (s[axis] < 1 || s[axis] > 16) return -2;
        if (s[4] < 1 || s[4] > MAX_COLUMNS || s[5] < 1 || s[5] > 16) return -3;
        for (int i = 0; i < s[4]; ++i) {
            const int32_t *c = s + HEADER + i * COLUMN_STRIDE;
            if (c[0] < 0 || c[0] >= AREA || (c[0] & 15) >= s[0]
                || ((c[0] >> 4) & 15) >= s[1] || (c[0] >> 8) >= s[3]) return -4;
            if (c[1] < 1 || c[1] > 65535 || c[2] < 0 || c[3] <= c[2] || c[3] > s[2]) return -5;
            for (int d = 0; d < 6; ++d) if (c[4+d] < -1 || c[4+d] >= s[4]) return -6;
        }
    }
    return 0;
}

int t4_features(const uint16_t *board, int32_t *heights, int32_t *counts, int32_t *totals) {
    if (!board || !heights || !counts || !totals) return -1;
    memset(counts, 0, 16 * sizeof(*counts));
    int height_sum = 0, mass = 0;
    for (int i = 0; i < AREA; ++i) {
        unsigned m = board[i];
        int h = m ? 32 - __builtin_clz(m) : 0;
        heights[i] = h;
        height_sum += h;
        unsigned v = m - ((m >> 1) & 0x5555u);
        v = (v & 0x3333u) + ((v >> 2) & 0x3333u);
        v = (v + (v >> 4)) & 0x0f0fu;
        mass += ((v * 0x0101u) >> 8) & 0x1fu;
        while (m) {
            counts[__builtin_ctz(m)]++;
            m &= m - 1;
        }
    }
    totals[0] = height_sum;
    totals[1] = height_sum - mass;
    return 0;
}

int t4_buried(const uint16_t *board, int target) {
    if (!board || target < 0 || target >= HEIGHT) return -1;
    int count = 0;
    unsigned bit = 1u << target;
    for (int i = 0; i < AREA; ++i) count += !(board[i] & bit) && ((unsigned)board[i] >> (target+1));
    return count;
}

static int landing_scalar(const int32_t *heights, const int32_t *s, int base) {
    int z = -HEIGHT;
    for (int i = 0; i < s[4]; ++i) {
        const int32_t *c = s + HEADER + i * COLUMN_STRIDE;
        int value = heights[base+c[0]] - c[2];
        if (value > z) z = value;
    }
    return z;
}

#if X86_DISPATCH
__attribute__((target("avx2")))
static void landing_avx2(const int32_t *heights, const int32_t *s, const int32_t *bases, int32_t *z) {
    __m256i origins = _mm256_loadu_si256((const __m256i *)bases);
    __m256i best = _mm256_set1_epi32(-HEIGHT);
    for (int i = 0; i < s[4]; ++i) {
        const int32_t *c = s + HEADER + i * COLUMN_STRIDE;
        __m256i index = _mm256_add_epi32(origins, _mm256_set1_epi32(c[0]));
        __m256i old = _mm256_i32gather_epi32(heights, index, 4);
        best = _mm256_max_epi32(best, _mm256_sub_epi32(old, _mm256_set1_epi32(c[2])));
    }
    _mm256_storeu_si256((__m256i *)z, best);
}

__attribute__((target("bmi2")))
static void compact_bmi2(const uint16_t *input, uint16_t *output, unsigned keep) {
    for (int i = 0; i < AREA; ++i) output[i] = (uint16_t)_pext_u32(input[i], keep);
}
#endif

int t4_compact(const uint16_t *input, uint16_t *output, unsigned cleared, int use_bmi2) {
    if (!input || !output || cleared > 65535u) return -1;
#if X86_DISPATCH
    if (use_bmi2 && (t4_capabilities() & 2)) {
        compact_bmi2(input, output, (~cleared) & 65535u);
        return 0;
    }
#else
    (void)use_bmi2;
#endif
    for (int i = 0; i < AREA; ++i) {
        unsigned v = input[i];
        for (int z = 15; z >= 0; --z) if (cleared & (1u << z)) {
            unsigned low = (1u << z) - 1u;
            v = (v & low) | ((v >> (z+1)) << z);
        }
        output[i] = (uint16_t)v;
    }
    return 0;
}

int t4_scores(const uint16_t *board, const int32_t *heights, const int32_t *counts,
              const int32_t *shapes, int shape_count, const int32_t *specs, int n,
              int target, double board_score, double height_weight, double holes_weight,
              double bump_weight, const double *rough_delta, double *scores,
              int32_t *z, int32_t *gains, uint8_t *clears, int use_avx2) {
    if (!board || !heights || !counts || !shapes || !specs || !rough_delta
        || !scores || !z || !gains || !clears || n < 1 || n > MAX_BATCH
        || target < -1 || target >= HEIGHT || shape_count < 1 || shape_count > 1024) return -1;
    int32_t bases[MAX_BATCH];
    for (int row = 0; row < n; ++row) {
        const int32_t *p = specs + row*4;
        if (p[0] < 0 || p[0] >= shape_count) return -2;
        const int32_t *s = shapes + p[0]*SHAPE_STRIDE;
        if (p[1] < 0 || p[1] > 16-s[0] || p[2] < 0 || p[2] > 16-s[1]
            || p[3] < 0 || p[3] > 16-s[3]) return -3;
        bases[row] = p[1] + p[2]*16 + p[3]*256;
        scores[row] = -INFINITY;
        gains[row] = -1;
        clears[row] = 0;
    }
    int simd = use_avx2 && (t4_capabilities() & 1);
    for (int row = 0; row < n;) {
        int r = specs[row*4], end = row+1;
        while (end < n && specs[end*4] == r) ++end;
        const int32_t *s = shapes + r*SHAPE_STRIDE;
#if X86_DISPATCH
        if (simd) for (; row+8 <= end; row+=8) landing_avx2(heights,s,bases+row,z+row);
#else
        (void)simd;
#endif
        for (; row < end; ++row) z[row] = landing_scalar(heights,s,bases[row]);
    }
    int high = -1;
    for (int row = 0; row < n; ++row) {
        const int32_t *s = shapes + specs[row*4]*SHAPE_STRIDE;
        if (z[row] < 0 || z[row]+s[2] > HEIGHT) continue;
        int gain = 0, blocked = 0;
        if (target >= 0) {
            unsigned bit = 1u << target;
            for (int i = 0; i < s[4]; ++i) {
                const int32_t *c = s + HEADER + i*COLUMN_STRIDE;
                unsigned old = board[bases[row]+c[0]];
                unsigned added = (unsigned)c[1] << z[row];
                if (!(old & bit)) {
                    gain += !!(added & bit);
                    blocked |= !(added & bit) && ((old|added) >> (target+1));
                }
            }
        }
        if (blocked) continue;
        gains[row] = gain;
        if (gain > high) high = gain;
    }
    static const int steps[6] = {-1,1,-16,16,-256,256};
    for (int row = 0; row < n; ++row) {
        if (gains[row] < 0 || (target >= 0 && gains[row] < high)) continue;
        const int32_t *s = shapes + specs[row*4]*SHAPE_STRIDE;
        int base = bases[row], landing = z[row];
        for (int dz = 0; dz < s[2]; ++dz) if (s[6+dz] && counts[landing+dz]+s[6+dz] == AREA) {
            clears[row] = 1;
            scores[row] = 0.0;
            break;
        }
        if (clears[row]) continue;
        int delta = 0;
        double rough = 0.0;
        for (int i = 0; i < s[4]; ++i) {
            const int32_t *c = s + HEADER + i*COLUMN_STRIDE;
            int index = base+c[0], old = heights[index], value = landing+c[3];
            delta += value-old;
            int coordinates[3] = {index & 15, (index >> 4) & 15, index >> 8};
            for (int d = 0; d < 6; ++d) {
                int coord = coordinates[d/2];
                if ((!(d&1) && coord == 0) || ((d&1) && coord == 15)) continue;
                int j = index+steps[d], partner = c[4+d];
                if (partner >= 0 && j < index) continue;
                int neighbor_old = heights[j];
                int neighbor_new = partner >= 0 ? landing+s[HEADER+partner*COLUMN_STRIDE+3] : neighbor_old;
                int before = neighbor_old-old, after = neighbor_new-value;
                if (before < 0) before = -before;
                if (after < 0) after = -after;
                rough += rough_delta[after*17+before];
            }
        }
        double value = board_score + (height_weight+holes_weight)*delta;
        value = value - s[5]*holes_weight;
        scores[row] = value + bump_weight*rough;
    }
    return 0;
}
