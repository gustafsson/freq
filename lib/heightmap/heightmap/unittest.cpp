#include "unittest.h"

#include "heightmap/freqaxis.h"
#include "heightmap/blockmanagement/merge/mergertexture.h"
#include "heightmap/blockmanagement/mipmapbuilder.h"
#include "heightmap/blockmanagement/blockfactory.h"
#include "heightmap/blockmanagement/blockinitializer.h"
#include "heightmap/render/renderset.h"
#include "heightmap/render/blocktextures.h"

// common backtrace tools
#include "timer.h"
#include "tasktimer.h"
#include "trace_perf.h"
#include "demangle.h"

#include <stdio.h>
#include <exception>

#include <boost/exception/diagnostic_information.hpp>

using namespace std;

namespace Heightmap {

string lastname;

#define RUNTEST(x) do { \
        TaskTimer tt("%s", #x); \
        lastname = #x; \
        x::test (); \
    } while(false)

int UnitTest::
        test(bool rethrow_exceptions)
{
    try {
        Timer(); // Init performance counting
        TaskTimer tt("Running tests");

        RUNTEST(Heightmap::FreqAxis);
        RUNTEST(Heightmap::Block);
        RUNTEST(Heightmap::BlockManagement::MipmapBuilder);
        RUNTEST(Heightmap::BlockManagement::Merge::MergerTexture);
        RUNTEST(Heightmap::BlockManagement::BlockFactory);
        RUNTEST(Heightmap::BlockManagement::BlockInitializer);
        RUNTEST(Heightmap::BlockLayout);
        RUNTEST(Heightmap::Render::BlockTextures);
        RUNTEST(Heightmap::Render::RenderSet);
        RUNTEST(Heightmap::VisualizationParams);

    } catch (const ExceptionAssert& x) {
        if (rethrow_exceptions)
            throw;

        char const * const * f = boost::get_error_info<boost::throw_file>(x);
        int const * l = boost::get_error_info<boost::throw_line>(x);
        char const * const * c = boost::get_error_info<ExceptionAssert::ExceptionAssert_condition>(x);
        std::string const * m = boost::get_error_info<ExceptionAssert::ExceptionAssert_message>(x);

        fflush(stdout);
        fprintf(stderr, "%s",
                str(boost::format("%s:%d: %s. %s\n"
                                  "%s\n"
                                  " FAILED in %s::test()\n\n")
                    % (f?*f:0) % (l?*l:-1) % (c?*c:0) % (m?*m:0) % boost::diagnostic_information(x) % lastname ).c_str());
        fflush(stderr);
        return 1;
    } catch (const exception& x) {
        if (rethrow_exceptions)
            throw;

        fflush(stdout);
        fprintf(stderr, "%s",
                str(boost::format("%s\n"
                                  "%s\n"
                                  " FAILED in %s::test()\n\n")
                    % vartype(x) % boost::diagnostic_information(x) % lastname ).c_str());
        fflush(stderr);
        return 1;
    } catch (...) {
        if (rethrow_exceptions)
            throw;

        fflush(stdout);
        fprintf(stderr, "%s",
                str(boost::format("Not an std::exception\n"
                                  "%s\n"
                                  " FAILED in %s::test()\n\n")
                    % boost::current_exception_diagnostic_information () % lastname ).c_str());
        fflush(stderr);
        return 1;
    }

    printf("\n OK\n\n");
    return 0;
}

} // namespace BacktraceTest
