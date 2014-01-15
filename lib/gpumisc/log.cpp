#include "log.h"
#include "TaskTimer.h"

Log::
        Log(const std::string& s)
    :
      boost::format(s)
{}


Log::
        ~Log()
{
    TaskInfo((const boost::format&)*this);
}
