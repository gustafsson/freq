#include "workers.h"
#include "signal/processing/targetschedule.h"
#include "timer.h"
#include "signal/processing/bedroomsignaladapter.h"
#include "demangle.h"
#include "tasktimer.h"

namespace Signal {
namespace Processing {


Workers::Workers(Processing::ISchedule::ptr schedule, Processing::Bedroom::ptr bedroom)
    :
      schedule_(schedule),
      bedroom_(bedroom)
{

}


void Workers::
        print(const DeadEngines& engines)
{
    if (engines.empty ())
        return;

    TaskInfo ti("Dead engines");

    for (const DeadEngines::value_type& e : engines) {
        Signal::ComputingEngine::ptr engine = e.first;
        std::exception_ptr x = e.second;
        std::string enginename = engine ? vartype(*engine.get ()) : (vartype(engine.get ())+"==0");

        if (x)
        {
            std::string details;
            try {
                std::rethrow_exception(x);
            } catch(...) {
                details = boost::current_exception_diagnostic_information();
            }

            TaskInfo(boost::format("engine %1% failed.\n%2%")
                     % enginename
                     % details);
        }
        else
        {
            TaskInfo(boost::format("engine %1% stopped")
                     % enginename);
        }
    }
}


} // namespace Processing
} // namespace Signal
