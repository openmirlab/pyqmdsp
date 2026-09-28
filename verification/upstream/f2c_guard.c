/* Link-only guard for original dlamch's unused legacy warning output.
 * Reads: f2c.h. Fail loudly if this diagnostic-only path is ever reached. */
#include <stdio.h>
#include <stdlib.h>
#include "f2c.h"
int s_wsfe(cilist *unused) {
    (void)unused;
    fputs("Original dlamch requested unsupported Fortran diagnostic output\n", stderr);
    abort();
}
int do_fio(integer *count, char *text, ftnlen length) {
    (void)count; (void)text; (void)length; abort();
}
int e_wsfe(void) { abort(); }
