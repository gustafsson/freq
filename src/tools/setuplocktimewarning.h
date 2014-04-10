#ifndef TOOLS_SETUPLOCKTIMEWARNING_H
#define TOOLS_SETUPLOCKTIMEWARNING_H

#include <QObject>

namespace Tools {

/**
 * @brief The SetupLockTimeWarning class should make shared_state_traits_backtrace
 * register exceptions in ApplicationErrorLogController.
 */
class SetupLockTimeWarning : public QObject
{
public:
    SetupLockTimeWarning();
};

} // namespace Tools

#endif // TOOLS_SETUPLOCKTIMEWARNING_H
