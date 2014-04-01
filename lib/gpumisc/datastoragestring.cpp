#include "datastoragestring.h"
#include "Statistics.h"

std::string DataStorageString::
        printDataStorage(DataStorage<float>::ptr data)
{
    EXCEPTION_ASSERT(data);

    std::stringstream ss;
    DataStorageSize sz = data->size ();
    ss << sz;
    float *p = sz.width ? data->getCpuMemory () : 0;

    if (sz.depth>1) {
        for (int z=0; z<sz.depth; ++z) {
            for (int y=0; y<sz.height; ++y) {
                ss << std::endl << "[z:" << z << ", y:" << y << "] = { ";

                float *q = p + y*sz.width + z*sz.height*sz.width;

                if (sz.width)
                    ss << q[0];

                for (int x=1; x<sz.width; ++x)
                    ss << ", " << q[x];

                ss << " }";
            }
        }
    } else if (sz.height>1) {
        for (int y=0; y<sz.height; ++y) {
            ss << std::endl << "[y:" << y << "] = { ";

            float *q = p + y*sz.width;

            if (sz.width)
                ss << q[0];

            for (int x=1; x<sz.width; ++x)
                ss << ", " << q[x];

            ss << " }";
        }
    } else {
        ss << " = { ";

        if (sz.width)
            ss << p[0];

        for (int x=1; x<sz.width; ++x)
            ss << ", " << p[x];

        ss << " }";
    }

    return ss.str ();
}


std::string DataStorageString::
        printDataStorageStats(DataStorage<float>::ptr data)
{
    std::stringstream ss;
    Statistics<float> s(data, false, true);
    ss << "size = " << data->size () << ", min = " << *s.getMin () << ", max = " << *s.getMax ()
       << ", mean = " << s.getMean () << ", std = " << s.getStd ();

    return ss.str();
}


void DataStorageString::
        test()
{
    // It should print contents of a DataStorage for debugging purposes.
    {
        float srcdata[] = { 1, 2, 3, 4 };
        DataStorage<float>::ptr data;
        {
            data = CpuMemoryStorage::BorrowPtr( DataStorageSize(2,2), srcdata, false );

            std::string s = DataStorageString::printDataStorage (data);
            std::string expected = "[2, 2]\n[y:0] = { 1, 2 }\n[y:1] = { 3, 4 }";
            EXCEPTION_ASSERT_EQUALS(s, expected);

            std::string stats = DataStorageString::printDataStorageStats (data);
            expected = "size = [2, 2], min = 1, max = 4, mean = 2.5, std = 1.11803";
            EXCEPTION_ASSERT_EQUALS(stats, expected);
        }

        {
            data = CpuMemoryStorage::BorrowPtr( DataStorageSize(3), srcdata, false );
            std::string s = DataStorageString::printDataStorage (data);
            std::string expected = "[3] = { 1, 2, 3 }";
            EXCEPTION_ASSERT_EQUALS(s, expected);

            std::string stats = DataStorageString::printDataStorageStats (data);
            expected = "size = [3], min = 1, max = 3, mean = 2, std = 0.816497";
            EXCEPTION_ASSERT_EQUALS(stats, expected);
        }

        {
            data = CpuMemoryStorage::BorrowPtr( DataStorageSize(2,1,2), srcdata, false );
            std::string s = DataStorageString::printDataStorage (data);
            std::string expected = "[2, 1, 2]\n[z:0, y:0] = { 1, 2 }\n[z:1, y:0] = { 3, 4 }";
            EXCEPTION_ASSERT_EQUALS(s, expected);

            std::string stats = DataStorageString::printDataStorageStats (data);
            expected = "size = [2, 1, 2], min = 1, max = 4, mean = 2.5, std = 1.11803";
            EXCEPTION_ASSERT_EQUALS(stats, expected);
        }

        {
            data = CpuMemoryStorage::BorrowPtr( DataStorageSize(1,2,2), srcdata, false );
            std::string s = DataStorageString::printDataStorage (data);
            std::string expected = "[1, 2, 2]\n[z:0, y:0] = { 1 }\n[z:0, y:1] = { 2 }\n[z:1, y:0] = { 3 }\n[z:1, y:1] = { 4 }";
            EXCEPTION_ASSERT_EQUALS(s, expected);

            std::string stats = DataStorageString::printDataStorageStats (data);
            expected = "size = [1, 2, 2], min = 1, max = 4, mean = 2.5, std = 1.11803";
            EXCEPTION_ASSERT_EQUALS(stats, expected);
        }
    }
}
