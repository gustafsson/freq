#ifndef TEST_PRINTDATASTORAGE_H
#define TEST_PRINTDATASTORAGE_H

#include "datastorage.h"

namespace Test {


/**
 * @brief The PrintDataStorage class should print contents of a DataStorage for debugging purposes.
 */
class PrintDataStorage
{
public:
    static std::string printDataStorage(DataStorage<float>::Ptr data);
    static std::string printDataStorageStats(DataStorage<float>::Ptr data);

public:
    static void test();
};


#define COMPARE_DATASTORAGE(expected, sizeof_expected, data) \
    do { \
        EXCEPTION_ASSERT(expected); \
        EXCEPTION_ASSERT(data); \
        EXCEPTION_ASSERT_EQUALS(sizeof_expected, data->numberOfBytes ()); \
        \
        float *p = data->getCpuMemory (); \
        \
        if (0 != memcmp(p, expected, sizeof_expected)) \
        { \
            Log("%s") % (DataStorageSize)data->size (); \
            for (size_t i=0; i<data->numberOfElements (); i++) \
                Log("%s: %s\t%s\t%s") % i % p[i] % expected[i] % (p[i] - expected[i]); \
        \
            EXCEPTION_ASSERT_EQUALS(0, memcmp(p, expected, sizeof_expected)); \
        } \
    } while(false)

#define PRINT_DATASTORAGE(data, arg) \
    do { \
        TaskInfo ti(format("%s(%s): %s(%s = %s) -> %s = %s") % __FILE__ % __LINE__ % __FUNCTION__ % (#arg) % (arg) % (#data) % b->getInterval ()); \
        TaskInfo(boost::format("%s"), PrintDataStorage::printDataStorage (data)); \
    } while(false)

#define PRINT_DATASTORAGE_STATS(data, arg) \
    do { \
        TaskInfo ti(format("%s(%s): %s(%s = %s) -> %s = %s") % __FILE__ % __LINE__ % __FUNCTION__ % (#arg) % (arg) % (#data) % b->getInterval ()); \
        TaskInfo(boost::format("%s"), PrintDataStorage::printDataStorageStats (data)); \
    } while(false)

} // namespace Test

#endif // TEST_PRINTDATASTORAGE_H
