#include "unittest.h"

// gpumisc units
#include "datastoragestring.h"
#include "factor.h"
#include "geometricalgebra.h"
#include "glframebuffer.h"
#include "glinfo.h"
#include "glprojection.h"
#include "glsyncobjectmutex.h"
#include "gltextureread.h"
#include "neat_math.h"
#include "resampletexture.h"
#include "float16.h"

// common backtrace tools
#include "timer.h"
#include "tasktimer.h"
#include "demangle.h"

#include <stdio.h>
#include <exception>

#include <boost/exception/diagnostic_information.hpp>

using namespace std;

namespace gpumisc {

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

        RUNTEST(DataStorageString);
        RUNTEST(Factor);
        RUNTEST(GeometricAlgebra);
        RUNTEST(GlFrameBuffer);
        RUNTEST(glinfo);
        RUNTEST(glProjection);
#if defined(LEGACY_OPENGL) && !defined(_WIN32)
        RUNTEST(GlSyncObjectMutex);
#endif
        RUNTEST(GlTextureRead);
        RUNTEST(neat_math);
        RUNTEST(ResampleTexture);
        RUNTEST(Float16Compressor);

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
