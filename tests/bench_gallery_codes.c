#include <stdint.h>
#include <stddef.h>

/* Each sum is bounded by 512 * 127 * 32767 < INT32_MAX. */
void scan_codes(const uint8_t *rows, const int16_t *query, size_t n,
                int bits, int32_t *scores) {
    for (size_t i = 0; i < n; ++i) {
        int32_t sum = 0;
        if (bits == 8) {
            const int8_t *row = (const int8_t *)rows + 512 * i;
            #pragma omp simd reduction(+:sum)
            for (int j = 0; j < 512; ++j)
                sum += row[j] * query[j];
        } else {
            const uint8_t *row = rows + 256 * i;
            #pragma omp simd reduction(+:sum)
            for (int j = 0; j < 256; ++j) {
                sum += ((row[j] & 15) - 7) * query[2*j];
                sum += ((row[j] >> 4) - 7) * query[2*j + 1];
            }
        }
        scores[i] = sum;
    }
}
