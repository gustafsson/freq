#include "detectgdb.h"

#include <stdio.h>

#if defined(__GNUC__)
    #include <unistd.h>
#else
    #define fileno _fileno
#endif

static bool was_started_through_gdb_ = DetectGdb::is_running_through_gdb ();

// gdb apparently opens FD(s) 3,4,5 (whereas a typical program uses only stdin=0, stdout=1, stderr=2)
bool DetectGdb::
        is_running_through_gdb()
{
    bool gdb = false;
    FILE *fd = fopen("/tmp", "r");

    if (fileno(fd) >= 5)
    {
        gdb = true;
    }

    fclose(fd);
    return gdb;
}


bool DetectGdb::
        was_started_through_gdb()
{
    return was_started_through_gdb_;
}
