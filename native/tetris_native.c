#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

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

/* Preserve every row, including illegal/blocked drops: the neural shortlist
 * applies its own legality filter after combining these integer metrics. */
int t4_sustained_metrics(const int32_t *heights, const int32_t *shapes, int shape_count,
                         const int32_t *specs, const int32_t *z, int n, int target,
                         int64_t *gains, uint8_t *blocked, int64_t *future_holes) {
    if (!heights || !shapes || !specs || !z || !gains || !blocked || !future_holes
        || n < 0 || target < 0 || target >= HEIGHT || shape_count < 1 || shape_count > 1024) return -1;
    for (int row = 0; row < n; ++row) {
        const int32_t *p = specs + row*4;
        if (p[0] < 0 || p[0] >= shape_count) return -2;
        const int32_t *s = shapes + p[0]*SHAPE_STRIDE;
        if (p[1] < 0 || p[1] > 16-s[0] || p[2] < 0 || p[2] > 16-s[1]
            || p[3] < 0 || p[3] > 16-s[3] || z[row] < -HEIGHT || z[row] > HEIGHT) return -3;
        int base = p[1] + p[2]*16 + p[3]*256;
        int relative = target-z[row], above = relative+1;
        if (above < 0) above = 0;
        if (above > HEIGHT+1) above = HEIGHT+1;
        int64_t gain = 0, future = 0;
        int buried = 0;
        for (int i = 0; i < s[4]; ++i) {
            const int32_t *c = s + HEADER + i*COLUMN_STRIDE;
            int old = heights[base+c[0]], top = z[row]+c[3];
            int on_target = relative >= 0 && relative < HEIGHT && ((unsigned)c[1] & (1u << relative));
            int gap = old <= target;
            gain += gap && on_target;
            buried |= gap && top > target && !on_target;
            int floor = old > target+1 ? old : target+1;
            if (top > floor) future += top-floor;
            future -= __builtin_popcount((unsigned)c[1] >> above);
        }
        gains[row] = gain;
        blocked[row] = (uint8_t)buried;
        future_holes[row] = future;
    }
    return 0;
}

/* A CROSS can land without holes if its six neighbors are one cell taller than
   its center. Reserve disjoint completed wells and build partial wells while
   keeping the skyline low enough for the other expanded pieces. */
enum { WELL_ARMS = 6, READY_WELLS = 12, PARTIAL_WELLS = 48, INTERIOR = 14*14*14 };
static const int well_steps[WELL_ARMS] = {-1, 1, -16, 16, -256, 256};

typedef struct {
    int center, height, matched, freedom;
} CrossWell;

typedef struct {
    uint8_t height[AREA];
    int8_t role[AREA]; /* -1 reserved, 0 unassigned, or one-based partial well */
    unsigned buried_levels;
    CrossWell partial[PARTIAL_WELLS];
    int partial_count, ready_count;
} CrossField;

typedef struct {
    int rotation, base, z, holes, levels;
    int64_t score, trials;
    const double *weights; /* learned linear scorer, or NULL for the fixed rules */
    double dscore;
} WellChoice;

enum { WELL_FEATURES = 7 };

static int has_neighbor(int column, int direction) {
    int coordinate = (column >> (4*(direction/2))) & 15;
    return (direction & 1) ? coordinate < 15 : coordinate > 0;
}

static int well_is_free(const CrossField *field, int center) {
    if (field->role[center]) return 0;
    for (int d = 0; d < WELL_ARMS; ++d)
        if (field->role[center + well_steps[d]]) return 0;
    return 1;
}

static void mark_well(CrossField *field, int center, int8_t role) {
    field->role[center] = role;
    for (int d = 0; d < WELL_ARMS; ++d) field->role[center + well_steps[d]] = role;
}

static int compare_wells(const void *a, const void *b) {
    const CrossWell *left = a, *right = b;
    if (left->matched != right->matched) return right->matched - left->matched;
    if (left->freedom != right->freedom) return right->freedom - left->freedom;
    if (left->height != right->height) return left->height - right->height;
    return left->center - right->center;
}

static void find_cross_wells(CrossField *field) {
    CrossWell available[INTERIOR];
    size_t available_count = 0;
    for (int w = 1; w < 15; ++w) for (int y = 1; y < 15; ++y) for (int x = 1; x < 15; ++x) {
        int center = w*256 + y*16 + x, target = field->height[center] + 1;
        if (target > 14) continue;
        int matched = 0, freedom = 0;
        for (int d = 0; d < WELL_ARMS; ++d) {
            int arm = center + well_steps[d], height = field->height[arm];
            if (height > target) { matched = -1; break; }
            if (height == target) { ++matched; continue; }
            int accessible = 0;
            for (int k = 0; k < WELL_ARMS; ++k) {
                int adjacent = arm + well_steps[k];
                if (has_neighbor(arm, k) && adjacent != center && field->height[adjacent] <= height)
                    ++accessible;
            }
            int deficit = target - height;
            freedom += 4*(accessible < 3 ? accessible : 3) - 5*(deficit < 5 ? deficit : 5);
        }
        if (matched < 0) continue;
        if (matched == WELL_ARMS) {
            if (well_is_free(field, center)) {
                mark_well(field, center, -1);
                if (++field->ready_count == READY_WELLS) return;
            }
        } else {
            available[available_count++] = (CrossWell){center, target-1, matched, freedom};
        }
    }
    qsort(available, available_count, sizeof(*available), compare_wells);
    for (size_t i = 0; i < available_count; ++i) {
        const CrossWell *well = available + i;
        if (!well_is_free(field, well->center)) continue;
        field->partial[field->partial_count++] = *well;
        mark_well(field, well->center, (int8_t)field->partial_count);
        if (field->partial_count == PARTIAL_WELLS) break;
    }
}

/* Only the columns touched by this placement can change its wells' merit. */
static int well_progress(const CrossField *field, const int32_t *shape, int base,
                         int z, int *covers_ready) {
    static const int merit[] = {0, 1, 3, 8, 22, 66, 660};
    int touched[MAX_COLUMNS], remaining[MAX_COLUMNS], used = 0, benefit = 0;
    *covers_ready = 0;
    for (int j = 0; j < shape[4]; ++j) {
        const int32_t *column = shape + HEADER + j*COLUMN_STRIDE;
        int index = base + column[0], role = field->role[index];
        if (role < 0) { *covers_ready = 1; continue; }
        if (!role) continue;
        const CrossWell *well = &field->partial[role-1];
        int k = 0;
        while (k < used && touched[k] != role) ++k;
        if (k == used) {
            remaining[k] = well->matched;
            touched[used++] = role;
        }
        int target = well->height + 1, top = z + column[3];
        if (index == well->center || top > target) remaining[k] = -1000;
        else if (top == target && field->height[index] < target) ++remaining[k];
    }
    for (int k = 0; k < used; ++k) {
        int matched = remaining[k] > 0 ? remaining[k] : 0;
        int delta = merit[matched] - merit[field->partial[touched[k]-1].matched];
        benefit += delta > 0 ? delta : delta/8;
    }
    return benefit;
}

static void consider_well_move(const CrossField *field, const int32_t *shapes,
                               int rotation, int base, WellChoice *best) {
    ++best->trials;
    const int32_t *shape = shapes + rotation*SHAPE_STRIDE;
    int z = 0;
    for (int j = 0; j < shape[4]; ++j) {
        const int32_t *column = shape + HEADER + j*COLUMN_STRIDE;
        int landing = field->height[base+column[0]] - column[2];
        if (landing > z) z = landing;
    }
    if (z + shape[2] > HEIGHT) return;

    int holes = 0, cost = 0, high = 0;
    unsigned gaps = 0;
    for (int j = 0; j < shape[4]; ++j) {
        const int32_t *column = shape + HEADER + j*COLUMN_STRIDE;
        int old = field->height[base+column[0]], low = z + column[2], top = z + column[3];
        holes += low - old;
        gaps |= ((1u << low)-1u) ^ ((1u << old)-1u);
        cost += top*top - old*old;
        if (top > high) high = top;
    }
    int levels = __builtin_popcount(gaps & ~field->buried_levels);
    if (best->weights) {
        int covers_ready, roughness = 0;
        int benefit = well_progress(field, shape, base, z, &covers_ready);
        for (int j = 0; j < shape[4]; ++j) {
            const int32_t *column = shape + HEADER + j*COLUMN_STRIDE;
            int index = base + column[0], old = field->height[index], top = z + column[3];
            for (int k = 0; k < WELL_ARMS; ++k) if (has_neighbor(index, k)) {
                int neighbor = field->height[index+well_steps[k]];
                roughness += (top-neighbor)*(top-neighbor) - (old-neighbor)*(old-neighbor);
            }
        }
        const double *v = best->weights;
        double score = v[0]*levels + v[1]*holes + v[2]*covers_ready - v[3]*benefit
                     + v[4]*cost + v[5]*roughness + v[6]*high;
        if (score < best->dscore) {
            best->rotation = rotation; best->base = base; best->z = z;
            best->holes = holes; best->levels = levels; best->dscore = score;
        }
        return;
    }
    if (levels > best->levels || (levels == best->levels && holes > best->holes)) return;

    int covers_ready;
    int benefit = well_progress(field, shape, base, z, &covers_ready);
    int64_t core = levels*INT64_C(1000000000000000) + holes*INT64_C(5000000000000)
                 + covers_ready*INT64_C(200000000000) - benefit*INT64_C(2000000) + cost*INT64_C(5000);
    /* This cutoff and the first-visited tie rule are part of the policy. */
    if (core > best->score + 10000) return;
    int roughness = 0;
    for (int j = 0; j < shape[4]; ++j) {
        const int32_t *column = shape + HEADER + j*COLUMN_STRIDE;
        int index = base + column[0], old = field->height[index], top = z + column[3];
        for (int k = 0; k < WELL_ARMS; ++k) if (has_neighbor(index, k)) {
            int neighbor = field->height[index+well_steps[k]];
            roughness += (top-neighbor)*(top-neighbor) - (old-neighbor)*(old-neighbor);
        }
    }
    int64_t score = core + 8*roughness + high;
    if (score < best->score) {
        best->rotation = rotation;
        best->base = base;
        best->z = z;
        best->holes = holes;
        best->levels = levels;
        best->score = score;
    }
}

static int well_random_index(uint32_t *state, int size) {
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return (int)(((uint64_t)value * (unsigned)size) >> 32);
}

static void consider_well_arms(const CrossField *field, const int32_t *shapes,
                               int rotations, WellChoice *best) {
    int limit = field->ready_count >= 3 ? 4 : 14;
    for (int i = 0; i < field->partial_count && i < limit; ++i) {
        const CrossWell *well = &field->partial[i];
        int target = well->height + 1;
        for (int d = 0; d < WELL_ARMS; ++d) {
            int arm = well->center + well_steps[d], deficit = target - field->height[arm];
            if (deficit <= 0) continue;
            for (int r = 0; r < rotations; ++r) {
                const int32_t *shape = shapes + r*SHAPE_STRIDE;
                for (int j = 0; j < shape[4]; ++j) {
                    const int32_t *column = shape + HEADER + j*COLUMN_STRIDE;
                    if (column[3] - column[2] != deficit) continue;
                    int offset = column[0];
                    int x = (arm & 15) - (offset & 15);
                    int y = ((arm >> 4) & 15) - ((offset >> 4) & 15);
                    int w = (arm >> 8) - (offset >> 8);
                    if (x >= 0 && x <= 16-shape[0] && y >= 0 && y <= 16-shape[1]
                            && w >= 0 && w <= 16-shape[3])
                        consider_well_move(field, shapes, r, arm-offset, best);
                }
            }
        }
    }
}

/* board is the canonical 4096-column bitmask. piece_index follows expanded
   PieceStream order (I=0, CROSS=13), turn is the number already placed.
   out: rotation, x, y, w, landing, new holes, newly buried levels, trials. */
static int cross_wells_impl(const uint16_t *board, const int32_t *shapes, int rotations,
                            int piece_index, uint32_t turn, int budget,
                            const double *weights, int32_t *out);

int t4_cross_wells(const uint16_t *board, const int32_t *shapes, int rotations,
                   int piece_index, uint32_t turn, int budget, int32_t *out) {
    return cross_wells_impl(board, shapes, rotations, piece_index, turn, budget, NULL, out);
}

/* Same candidate generator, but each candidate is scored by a learned linear
   function of the seven well features (lower is better). */
int t4_cross_wells_weighted(const uint16_t *board, const int32_t *shapes, int rotations,
                            int piece_index, uint32_t turn, int budget,
                            const double *weights, int32_t *out) {
    if (!weights) return -1;
    for (int i = 0; i < WELL_FEATURES; ++i) if (!isfinite(weights[i])) return -1;
    return cross_wells_impl(board, shapes, rotations, piece_index, turn, budget, weights, out);
}

static int cross_wells_impl(const uint16_t *board, const int32_t *shapes, int rotations,
                            int piece_index, uint32_t turn, int budget,
                            const double *weights, int32_t *out) {
    if (!board || !out || piece_index < 0 || piece_index >= 15 || budget < 0) return -1;
    int valid = t4_validate_shapes(shapes, rotations);
    if (valid < 0) return valid;
    /* The expanded pieces have solid, unique columns. The gap count and well
       merit both rely on this. Guard the public native entry point as well. */
    for (int r = 0; r < rotations; ++r) {
        const int32_t *shape = shapes + r*SHAPE_STRIDE;
        for (int i = 0; i < shape[4]; ++i) {
            const int32_t *column = shape + HEADER + i*COLUMN_STRIDE;
            if ((uint32_t)column[1] != (1u << column[3]) - (1u << column[2])) return -7;
            for (int j = 0; j < i; ++j) if (shape[HEADER+j*COLUMN_STRIDE] == column[0]) return -7;
        }
    }

    CrossField field = {0};
    for (int i = 0; i < AREA; ++i) {
        unsigned mask = board[i];
        int height = mask ? 32 - __builtin_clz(mask) : 0;
        field.height[i] = (uint8_t)height;
        field.buried_levels |= ((1u << height) - 1u) ^ mask;
    }
    if (piece_index != 13) find_cross_wells(&field);

    WellChoice best = {.rotation = -1, .holes = 999, .levels = 99, .score = INT64_C(1)<<61,
                      .weights = weights, .dscore = HUGE_VAL};
    uint32_t state = (UINT32_C(0x9e3779b9)*(turn+1u)) ^ UINT32_C(0x84937923)
                   ^ (UINT32_C(0xa3324543)*(uint32_t)piece_index);
    if (!state) state = 3;
    if (rotations > 1) for (int n = 0; n < budget; ++n) {
        int r = well_random_index(&state, rotations);
        const int32_t *shape = shapes + r*SHAPE_STRIDE;
        int x = well_random_index(&state, 17-shape[0]);
        int y = well_random_index(&state, 17-shape[1]);
        int w = well_random_index(&state, 17-shape[3]);
        consider_well_move(&field, shapes, r, w*256+y*16+x, &best);
    }
    consider_well_arms(&field, shapes, rotations, &best);
    if (piece_index == 0) for (int r = 0; r < rotations; ++r) {
        const int32_t *shape = shapes + r*SHAPE_STRIDE;
        if (shape[4] == 1 && shape[HEADER] == 0 && shape[0] == 1 && shape[1] == 1 && shape[3] == 1)
            for (int i = 0; i < AREA; ++i) consider_well_move(&field, shapes, r, i, &best);
    }
    if (!budget || best.holes != 0 || rotations == 1) {
        for (int r = 0; r < rotations; ++r) {
            const int32_t *shape = shapes + r*SHAPE_STRIDE;
            for (int w = 0; w <= 16-shape[3]; ++w) for (int y = 0; y <= 16-shape[1]; ++y)
                for (int x = 0; x <= 16-shape[0]; ++x)
                    consider_well_move(&field, shapes, r, w*256+y*16+x, &best);
        }
    }
    if (best.rotation < 0) return 0;
    out[0] = best.rotation; out[1] = best.base & 15; out[2] = (best.base >> 4) & 15;
    out[3] = best.base >> 8; out[4] = best.z; out[5] = best.holes; out[6] = best.levels;
    out[7] = best.trials < INT32_MAX ? (int32_t)best.trials : INT32_MAX;
    return 1;
}

/* Play the trained-wells policy over a fixed piece sequence without leaving C.
   board: 4096 column masks in engine order, updated in place.
   shapes: the expanded pieces' shape records back to back; piece p uses
   shape_count[p] records starting at record shape_offset[p].
   pieces: expanded piece index of each placement; turn0 is the number of
   placements already made, so the policy sees the same turn as in Python.
   moves: 7 bytes per placement (piece, rotation, x, y, w, landing z, cleared
   layers). Returns the placements made; fewer than n means the next piece had
   no legal move (physical top-out). Negative values report invalid input. */
int t4_play_trained_wells(uint16_t *board, const int32_t *shapes, const int32_t *shape_offset,
                          const int32_t *shape_count, const uint8_t *pieces, int n,
                          uint32_t turn0, int budget, const double *weights, uint8_t *moves) {
    if (!board || !shapes || !shape_offset || !shape_count || !pieces || !moves || !weights
            || n < 0 || budget < 0) return -1;
    for (int i = 0; i < WELL_FEATURES; ++i) if (!isfinite(weights[i])) return -1;
    for (int p = 0; p < 15; ++p) {
        if (shape_count[p] < 1 || shape_count[p] > 255 || shape_offset[p] < 0) return -2;
        int valid = t4_validate_shapes(shapes + (size_t)shape_offset[p]*SHAPE_STRIDE, shape_count[p]);
        if (valid < 0) return valid;
    }
    for (int t = 0; t < n; ++t) {
        int piece = pieces[t];
        if (piece >= 15) return -8;
        const int32_t *rotations = shapes + (size_t)shape_offset[piece]*SHAPE_STRIDE;
        int32_t out[8];
        int found = cross_wells_impl(board, rotations, shape_count[piece], piece, turn0 + (uint32_t)t,
                                     budget, weights, out);
        if (found < 0) return found;
        if (!found) return t;
        const int32_t *shape = rotations + (size_t)out[0]*SHAPE_STRIDE;
        int base = out[3]*256 + out[2]*16 + out[1], z = out[4];
        if (z < 0 || z + shape[2] > HEIGHT) return -9;
        for (int j = 0; j < shape[4]; ++j) {
            const int32_t *column = shape + HEADER + j*COLUMN_STRIDE;
            unsigned cells = (unsigned)column[1] << z;
            if (board[base + column[0]] & cells) return -10;
            board[base + column[0]] |= (uint16_t)cells;
        }
        unsigned full = 0xffffu;
        for (int i = 0; i < AREA && full; ++i) full &= board[i];
        if (full) {
            for (int i = 0; i < AREA; ++i) {
                unsigned mask = board[i], kept = 0;
                for (int level = 0, k = 0; level < HEIGHT; ++level)
                    if (!((full >> level) & 1u)) kept |= ((mask >> level) & 1u) << k++;
                board[i] = (uint16_t)kept;
            }
        }
        uint8_t *record = moves + (size_t)t*7;
        record[0] = (uint8_t)piece; record[1] = (uint8_t)out[0]; record[2] = (uint8_t)out[1];
        record[3] = (uint8_t)out[2]; record[4] = (uint8_t)out[3]; record[5] = (uint8_t)z;
        record[6] = (uint8_t)__builtin_popcount(full);
    }
    return n;
}
