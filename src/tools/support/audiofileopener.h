#ifndef TOOLS_AUDIOFILECONTROLLER_H
#define TOOLS_AUDIOFILECONTROLLER_H

#include "tools/openfilecontroller.h"

namespace Tools {
namespace Support {

/**
 * @brief The AudiofileController class should open files supported by libsndfile.
 */
class AudiofileOpener : public OpenfileController::OpenfileInterface
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

#endif // TOOLS_AUDIOFILECONTROLLER_H
