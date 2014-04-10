#include "setuplocktimewarning.h"
#include "shared_state_traits_backtrace.h"
#include "applicationerrorlogcontroller.h"
#include "exceptionassert.h"

namespace Tools {

typedef boost::error_info<struct type_info_tag,std::string> error_type_info;

SetupLockTimeWarning::SetupLockTimeWarning()
{
    shared_state_traits_backtrace::default_warning =
            [](double lock_time, double limit, void*, const std::type_info& t)
            {
                try {
                    EXCEPTION_ASSERT_LESS (lock_time, limit);
                } catch (boost::exception& x) {
                    x << error_type_info(demangle(t));
                    ApplicationErrorLogController::registerException (boost::current_exception());
                }
            };
}

} // namespace Tools
