#include "setuplocktimewarning.h"
#include "shared_state_traits_backtrace.h"
#include "applicationerrorlogcontroller.h"
#include "exceptionassert.h"
#include "tasktimer.h"

namespace Tools {

typedef boost::error_info<struct type_info_tag,std::string> error_type_info;

SetupLockTimeWarning::SetupLockTimeWarning()
{
    shared_state_traits_backtrace::default_warning =
            [](double lock_time, double limit, void*, const std::type_info& t)
            {
                TaskTimer tt("shared_state_traits_backtrace::default_warning");

                try {
                    EXCEPTION_ASSERTX( lock_time < limit, boost::format("%s was locked for %s > %s")
                                           % demangle(t)
                                           % TaskTimer::timeToString (lock_time)
                                           % TaskTimer::timeToString (limit));
                    EXCEPTION_ASSERTX(false, "shouldn't reach here");
                } catch (boost::exception& x) {
                    x << error_type_info(demangle(t));
                    ApplicationErrorLogController::registerException (boost::current_exception());
                }
            };
}

} // namespace Tools
