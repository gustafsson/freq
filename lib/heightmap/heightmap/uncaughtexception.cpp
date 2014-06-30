#include "uncaughtexception.h"
#include "tasktimer.h"

namespace Heightmap {

std::function<void(boost::exception_ptr)> UncaughtException::handle_exception =
        [](boost::exception_ptr x)
        {
            fflush(stdout);
            fprintf(stderr, "%s",
                    str(boost::format("\nUncaught exception\n"
                                      "%s\n\n")
                        % boost::current_exception_diagnostic_information ()).c_str());
            fflush(stderr);

            boost::rethrow_exception (x);
        };

} // namespace Heightmap
