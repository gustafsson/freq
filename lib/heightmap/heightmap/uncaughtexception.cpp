#include "uncaughtexception.h"
#include "log.h"

namespace Heightmap {

std::function<void(boost::exception_ptr)> UncaughtException::handle_exception =
        [](boost::exception_ptr x)
        {
            std::string info = boost::current_exception_diagnostic_information ();
            Log("!!! %s") % info;

            fflush(stdout);
            fprintf(stderr, "%s",
                    str(boost::format("\nUncaught exception\n"
                                      "%s\n\n")
                        % info).c_str());
            fflush(stderr);

            boost::rethrow_exception (x);
        };

} // namespace Heightmap
