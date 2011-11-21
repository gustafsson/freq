#include "openfileerror.h"

namespace Sawe {

OpenFileError::
        OpenFileError(const std::string& message)
        :
        std::runtime_error(message)
{}

} // namespace Sawe
