#ifndef DATASTORAGESTRING_H
#define DATASTORAGESTRING_H

#include "datastorage.h"
#include "log.h"

/**
 * @brief The DataStorageString class should print contents of a DataStorage for debugging purposes.
 */
class DataStorageString
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
            Log("%s = %s vs %s") % (#data) % data->size () % (#expected); \
            for (size_t i=0; i<data->numberOfElements (); i++) \
                Log("%s: %s\t%s\t%s") % i % p[i] % expected[i] % (p[i] - expected[i]); \
        \
            EXCEPTION_ASSERT_EQUALS(0, memcmp(p, expected, sizeof_expected)); \
        } \
    } while(false)

#define PRINT_DATASTORAGE(data, arg) \
    do { \
        Log("%s(%s): %s(%s = %s) -> %s = %s") % __FILE__ % __LINE__ % __FUNCTION__ % (#arg) % (arg) % (#data) \
                    % DataStorageString::printDataStorage (data); \
    } while(false)

#define PRINT_DATASTORAGE_STATS(data, arg) \
    do { \
        Log("%s(%s): %s(%s = %s) -> %s = %s") % __FILE__ % __LINE__ % __FUNCTION__ % (#arg) % (arg) % (#data) \
                    % DataStorageString::printDataStorageStats (data); \
    } while(false)

#endif // DATASTORAGESTRING_H
