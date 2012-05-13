#ifndef OPENFILEERROR_H
#define OPENFILEERROR_H

#include <stdexcept>

namespace Sawe {

class OpenFileError : public std::runtime_error
{
public:
    OpenFileError(const std::string& message);
};

} // namespace Sawe

#endif // OPENFILEERROR_H
