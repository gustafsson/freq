#ifndef HEIGHTMAP_UNCAUGHTEXCEPTION_H
#define HEIGHTMAP_UNCAUGHTEXCEPTION_H

#include <functional>
#include <boost/exception_ptr.hpp>

namespace Heightmap {

class UncaughtException
{
public:
    // This function may throw or return
    static std::function<void(boost::exception_ptr)> handle_exception;
};

} // namespace Heightmap

#endif // HEIGHTMAP_UNCAUGHTEXCEPTION_H
