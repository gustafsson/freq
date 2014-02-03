#ifndef TEST_PRINTBUFFER_H
#define TEST_PRINTBUFFER_H

#include <string>
#include "signal/buffer.h"
#include "TaskTimer.h"

namespace Test {

/**
 * @brief The PrintBuffer class should print buffer contents for debugging purposes.
 */
class PrintBuffer
{
public:
    static std::string printBuffer(Signal::pBuffer);
    static std::string printBufferStats(Signal::pBuffer);

public:
    static void test();
};


#define PRINT_BUFFER(b, arg) \
    do { \
        TaskInfo ti(boost::format("%s(%s): %s(%s = %s) -> %s = %s") % __FILE__ % __LINE__ % __FUNCTION__ % (#arg) % (arg) % (#b) % b->getInterval ()); \
        TaskInfo(boost::format("%s") % Test::PrintBuffer::printBuffer (b)); \
    } while(false)

#define PRINT_BUFFER_STATS(b, arg) \
    do { \
        TaskInfo ti(boost::format("%s(%s): %s(%s = %s) -> %s = %s") % __FILE__ % __LINE__ % __FUNCTION__ % (#arg) % (arg) % (#b) % b->getInterval ()); \
        TaskInfo(boost::format("%s") % Test::PrintBuffer::printBufferStats (b)); \
    } while(false)

} // namespace Test

#endif // TEST_PRINTBUFFER_H
