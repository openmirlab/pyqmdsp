/* Test-process clock only: preserve original srand(time(0)) with a fixed seed.
 * Reads: libc time ABI. This library is never linked into the package. */
#include <time.h>
time_t time(time_t *result) {
    const time_t fixed = 1780000000;
    if (result) *result = fixed;
    return fixed;
}
