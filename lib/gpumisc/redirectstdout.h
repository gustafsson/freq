#ifndef REDIRECTSTDOUT_H
#define REDIRECTSTDOUT_H

#include <stdio.h>

#include <boost/noncopyable.hpp>

/**
Redirects stdout and optionally stderr to a file. The redirection is restored
in the destructor such that following stream operations on stdout (or stderr)
apply to the original stdout and stderr.

In Windows a program might not initially have stdout nor stderr. In that case
they are initialized in the RedirectStdout constructor and pointed to the 
given file, and the redirection is NOT restored in the destructor. The first
redirection in a Windows application without stdout and stderr will stay and
stream operations to stdout and stderr after the destructor will act on the
same file. The file is not closed until application exit.
*/
class RedirectStdout: public boost::noncopyable
{
public:
    RedirectStdout(const char* output_filename, bool redirect_stderr_to_same_file=true);

    ~RedirectStdout();
private:
    FILE* target_file, *target_file2;

    int    stdout_fd, stderr_fd;
    fpos_t stdout_pos, stderr_pos;
};

#endif // REDIRECTSTDOUT_H
