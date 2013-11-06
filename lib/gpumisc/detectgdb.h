#ifndef DETECTGDB_H
#define DETECTGDB_H

class DetectGdb
{
public:
    static bool is_running_through_gdb();
    static bool was_started_through_gdb();
};

#endif // DETECTGDB_H
