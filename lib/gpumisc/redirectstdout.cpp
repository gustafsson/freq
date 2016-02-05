#include "redirectstdout.h"

#ifdef _WIN32
#include <io.h>
#define dup _dup
#define dup2 _dup2
#define fileno _fileno
#else
#include <stdio.h>
#include <unistd.h>
#endif

#include <string.h> // memset

RedirectStdout::
        RedirectStdout(const char* output_filename, bool redirect_stderr_to_same_file)
        :
        target_file(0),
        target_file2(0),
        stdout_fd(-1),
        stderr_fd(-1)
{
    ::memset(&stdout_pos, 0, sizeof(stdout_pos));
    ::memset(&stderr_pos, 0, sizeof(stderr_pos));

    // Don't know how restore stdout once redirected with freopen
    // us fopen/dup2 instead, dup2 has to be used for stderr anyways
    int e = 0;

    if (-2 == fileno(stdout))
    {
        // Windows application without console window, no stdout state to copy
        
        // This doesn't work if stdout has been used since the program started.

        // Direct stdout to 'output_filename'
        target_file = freopen(output_filename, "a", stdout);
    }
    else
    {
        if ( 0 < fileno(stdout)) // if stdout is open
        {
            // Copy the state of stdout
            fflush(stdout);
            fgetpos(stdout, &stdout_pos);
            stdout_fd = dup(fileno(stdout));
        }

        // Direct stdout to 'output_filename'
        target_file = fopen(output_filename, "a");
        e = dup2( fileno(target_file), fileno(stdout));
    }

    if (redirect_stderr_to_same_file)
    {
        if (-2 == fileno(stderr)) // Windows application without console window
        {
            target_file2 = freopen(output_filename, "a", stderr);
            e = dup2( fileno(target_file), fileno(stderr));
        }
        else
        {
            if (0<fileno(stderr))
            {
                // Copy the state of stderr
                fflush(stderr);
                fgetpos(stderr, &stderr_pos);
                stderr_fd = dup(fileno(stderr));
            }

            // Direct stderr to the same file
            e = dup2( fileno(target_file), fileno(stderr));
        }
    }

    if (-1 == e)
    {
        if (redirect_stderr_to_same_file)
            fprintf(stderr, "Failed to redirect stdout and stderr to '%s'\n", output_filename);
        else
            fprintf(stderr, "Failed to redirect stdout to '%s'\n", output_filename);
    }
}


// Thanks to http://stackoverflow.com/questions/1673764/freopen-reverting-back-to-original-stream 
// about reverting back to original stream using dup2
RedirectStdout::
        ~RedirectStdout()
{
    if (!target_file)
        return;


    // restore stderr
    if (stderr_fd != -1)
    {
        if (target_file2)
            fclose(target_file2);

        dup2(stderr_fd, fileno(stderr));
        fsetpos(stderr, &stderr_pos);
    }


    // restore stdout
    if (stdout_fd != -1) 
        // if the program did have a valid stdout to start with. 
        // I.e a windows application without a console window 
        // doesn't have a valid stdout to start with.
    {
        dup2(stdout_fd, fileno(stdout));
        fsetpos(stdout, &stdout_pos);

        // close target file
        fclose(target_file);
    }
    else
    {
        // closing target_file would leave stdout in an invalid state
        // TODO reset stdout to the invalid -2 file descriptor somehow
        // _dup2(-2, _fileno(stdout)); doesn't do it
    }}
