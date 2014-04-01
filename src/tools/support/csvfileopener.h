#ifndef TOOLS_SUPPORT_CSVFILEOPENER_H
#define TOOLS_SUPPORT_CSVFILEOPENER_H

#include "tools/openfilecontroller.h"

namespace Tools {
namespace Support {

/**
 * @brief The CsvfileOpener class should open files supported by Adapters::CsvTimeseries.
 */
class CsvfileOpener : public OpenfileController::OpenfileInterface
{
public:
    virtual Patterns patterns();
    virtual Signal::OperationDesc::ptr reopen(QString url, Signal::OperationDesc::ptr);

private:

public:
    static void test();
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_CSVFILEOPENER_H
