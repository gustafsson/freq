/// @see class DataStorage
#ifndef DATASTORAGE_H
#define DATASTORAGE_H

#include <set>
#include <string>

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>

#include "exceptionassert.h"
#include "datastorageaccess.h"


class DataStorageImplementation;

/// @see class DataStorage
class DataStorageVoid
{
protected:
    DataStorageVoid(DataStorageSize size, size_t bytesPerelement);
    DataStorageVoid( const DataStorageVoid& );
    virtual ~DataStorageVoid();

public:
    typedef std::set<DataStorageImplementation*> StorageImplementations;

    const StorageImplementations& validContent() const { return validContent_; }


    DataStorageSize size() const { return size_; }
    DataStorageSize sizeInBytes() const;


    int bytesPerElement() const { return bytesPerElement_; }


    size_t numberOfElements() const
    {
        return (size_t)size_.width
                * size_.height
                * size_.depth;
    }
    size_t numberOfBytes() const
    {
        return numberOfElements() * bytesPerElement_;
    }


    // TODO Make private
    template<typename T>
    T* FindStorage() const
    {
        T* r = 0;

        // boost/foreach.hpp seems to not work with nvcc and boost/assert.hpp in windows
        for( StorageImplementations::const_iterator itr = storage_.begin();
             itr != storage_.end(); ++itr )
        {
            DataStorageImplementation* p = *itr;
            if (T* t = dynamic_cast<T*>( p ))
            {
                // There must be at most one of each type
                EXCEPTION_ASSERT( r == 0 );
                r = t;
            }
        }

        return r;
    }


    template<typename T>
    bool HasValidContent() const
    {
        bool r = false;

        // boost/foreach.hpp seems to not work with nvcc and boost/assert.hpp in windows
        for( StorageImplementations::const_iterator itr = validContent_.begin();
             itr != validContent_.end(); ++itr )
        {
            DataStorageImplementation* p = *itr;
            if (dynamic_cast<T*>( p ))
            {
                // There must be at most one of each type
                EXCEPTION_ASSERT( r == false );
                r = true;
            }
        }

        return r;
    }


    template<typename StorageType>
    StorageType* AccessStorage(bool read, bool write)
    {
        StorageType* t = FindStorage<StorageType>();
        if (!t)
            t = new StorageType(this, read, write);

        t = dynamic_cast<StorageType*>(UpdateValidContent( t, read, write ));
        return t;
    }


    template<typename StorageType>
    void OnlyKeepOneStorage()
    {
        if (!validContent_.empty ())
        {
            DataStorageImplementation* t = AccessStorage<StorageType>(true, true);
            OnlyKeepOneStorage( t );
        }
    }


    /**
      Set all updated contents to zero
      */
    void ClearContents();


    /**
      Discards all memory allocated by 'this' instance.
      */
    void DiscardAllData();


    /**
      Perform a deep copy of the data in another instance.
      Tries to match storage implementation in 'b' with existing allocations in 'this'.
      */
    void DeepCopy(const DataStorageVoid& b);


    /**
      Setup a lazy cow of another instance. Implies calling 'DiscardAllData()'.
      */
    DataStorageVoid& operator=( const DataStorageVoid& );


    static std::string getMemorySizeText( unsigned long long size, char decimals=-1, char type='g' );


protected:
    friend class DataStorageImplementation;

    StorageImplementations storage_;
    StorageImplementations validContent_;


private:
    /// size of data in x, y and z
    DataStorageSize size_;
    int bytesPerElement_;


    void OnlyKeepOneStorage(DataStorageImplementation* t);
    DataStorageImplementation* UpdateValidContent( DataStorageImplementation* t, bool read, bool write );
    DataStorageImplementation* CopyStorage(const DataStorageVoid& b);

};


// findImplementationType<DataStorageImplementation> is an invalid template argument
template<>
DataStorageImplementation* DataStorageVoid::
        FindStorage<DataStorageImplementation>() const;


void* getCpuMemory(DataStorageVoid*);


/**
 * @brief The DataStorage class moves and tracks data between memory types.
 * The DataStorage class keeps track of data stored concurrently in multiple
 * memory types such as in Cuda device memory, OpenCL buffer, CPU RAM,
 * OpenGL VBOs, explicit hard disk swap file, etc. Memory types are referenced
 * by classes that representes storage of data, such as CpuMemoryStorage,
 * CudaGlobalStorage and OpenClMemoryStorage. They all implement the interface
 * DataStorageImplementation.
 *
 *
 * Motivation:
 * When a chunk of data is used in various algorithms on different devices with
 * _loose couplings_ it is hard to know wheter another algorithm already has
 * made a copy of the data from, say CPU to Cuda.
 *
 *
 * Usage:
 * DataStorage keeps track of which types of memory that has been allocated to
 * store a copy of some data, and where that data is currently flagged as
 * up-to-date.
 *
 * - Requesting read access from a memory type that is not flagged as up-to-date
 *   causes DataStorage to issue a copy (and allocation if necessary) from a
 *   type that is up-to-date.
 * - Requesting write access in one memory type flags copies in other memory
 *   types as not up-to-date.
 * - Requesting read-write access essentially performs a read request followed
 *   by a write request but memory type specific optimizations may apply.
 *
 * Instances of DataStorage are typically shuffled around using the type
 * DataStorage::Ptr, often together with typedefs such as:
 *
 * typedef DataStorage<float> MyData;
 * MyData::Ptr myData(new MyData(...) );
 *
 *
 * Example:
 * 1) read source data from wherever to CPU RAM
 * 2) run algorithm in Cuda kernel without affecting the original data
 * 3) run algorithm in CPU function on the same original data
 * 4) run algorithm in Cuda kernel on the output of 2 and 3 together
 *
 * The lines concerning memory management for each step follows:
 * 1) MyData::Ptr myData = []
 *    {
 *        size_t number_of_elements = 12345;
 *        MyData::Ptr myData(new MyData(number_of_elements) );
 *
 *        float* p = CpuMemoryStorage::WriteAll<1>( myData ); // [1], [2]
 *
 *        readSourceDataFromWhereverToCpuRam( p );
 *
 *        return myData;
 *    }
 *
 * 2) MyData::Ptr intermediateData = [](MyData::Ptr myData)
 *    {
 *        // Prepare intermediate data
 *        MyData::Ptr intermediateData(new MyData(myData.size()) )
 *
 *        // copy data from CPU RAM to Cuda
 *        float* cuda_input = CudaGlobalStorage::ReadOnly<1>( myData ).ptr();
 *
 *        // allocate 'intermediateData' in Cuda
 *        float* cuda_output = CudaGlobalStorage::WriteAll<1>( intermediateData ).ptr();
 *
 *        runAlgorithmInCudaKernel( cuda_input, cuda_output );
 *
 *        return intermediateData;
 *    }(myData);
 *
 * 3) [](MyData::Ptr myData) {
 *        // this flags the data that was pointed to by 'cuda_input' above as
 *        // not up-to-date but doesn't deallocate it and doesn't perform any
 *        // memory copy. Nethier the value of 'p' nor the contents of p[] has
 *        // been changed since (1).
 *        p = CpuMemoryStorage::ReadWrite<1>( myData );
 *
 *        // Change the contents of p[]
 *        runAlgorithmInCpu( p );
 *    }(myData);
 *
 * 4) [](MyData::Ptr myData, MyData::Ptr intermediateData) {
 *        // replaces the old data in cuda memory with an updated copy from CPU
 *        // RAM since that's the only place where myData is currently
 *        // up-to-date. The Cuda memory is allocated since before and reused.
 *        // The data in CPU RAM is then flaged as not up-to-date.
 *        float* cuda_input_and_output = CudaGlobalStorage::ReadWrite<1>( myData ).ptr();
 *
 *        // returns the same pointer as 'cuda_output' had before and.
 *        float* cuda_input = CudaGlobalStorage::ReadOnly<1>( intermediateData ).ptr();
 *
 *        runAnotherAlgorithmInCudaKernel( cuda_input, cuda_input_and_output );
 *    }(myData, intermediateData);
 *
 * [1] To just have simple access to cpu memory there is also a short version as
 * well myData->getCpuMemory().
 * [2] The template argument 1 refers to a one-dimensional access pattern, the
 * actual data reference is implementation dependent. The dimensionality
 * argument matters for implementations that uses a pitch > width for memory
 * efficiency.
 *
 *
 * Advanced usage:
 * myData->AccessStorage<'StorageType'>(...)->Access(...) can be used
 * instead of the helpers 'StorageType'::ReadWrite, etc.
 *
 * myData->HasValidContent<'StorageType'>() can be used to check if a read
 * request would cause a copy or not.
 */
template<typename T=char>
class DataStorage: public DataStorageVoid
{
public:
    typedef boost::shared_ptr<DataStorage<T> > ptr;
    typedef T element_type;

    DataStorage(DataAccessPosition_t size_x)
        :DataStorageVoid(DataStorageSize(size_x), sizeof(T))
    {
    }

    DataStorage(DataAccessPosition_t size_x, DataAccessPosition_t size_y)
        :DataStorageVoid(DataStorageSize(size_x, size_y), sizeof(T))
    {
    }

    DataStorage(DataAccessPosition_t size_x, DataAccessPosition_t size_y, DataAccessPosition_t size_z)
        :DataStorageVoid(DataStorageSize(size_x, size_y, size_z), sizeof(T))
    {
    }

    DataStorage(DataStorageSize size)
        :DataStorageVoid(size, sizeof(T))
    {
    }


    /// Effectively the equivalent of CpuMemoryStorage::ReadWrite<3>( shared_ptr_to_this )
    T* getCpuMemory()
    {
        return (T*)::getCpuMemory(this);
    }

};


/**
 * @brief The DataStorageImplementation class
 * @see class DataStorage
 */
class DataStorageImplementation: boost::noncopyable
{
protected:
    DataStorageImplementation( DataStorageVoid* p );
public:
    virtual ~DataStorageImplementation();


    DataStorageSize size() const { return dataStorage()->size(); }


    /**
      Returns the 'DataStorageVoid' instance for this storage implementation. In case
      of present Cow copies this returns the first instance, with which 'this' was created.
      */
    DataStorageVoid* dataStorage() const { return dataStorage_; }


    /**
      Adds a Cow reference to 'parent'.
      */
    void addCowCopy(DataStorageVoid*parent);


    /**
      Removes 'this' from 'parent'. 'this' must be a datastorage of 'parent'.

      Returns true if 'parent' was a CowCopy.

      Or,
      if 'parent' is dataStorage_ (the owner) and at least one cow copy
      exists, then a randomly choosen cow copy (the first in dataStorageCow_)
      is made the new owner. Returns true.

      Otherwise returns false.
      */
    bool removeCowCopy(DataStorageVoid*parent);


    /**
      Performs a deep copy of this to all Cow clones. Returns the storage for 'parent'.
      */
    DataStorageImplementation* copyOnWrite(DataStorageVoid*parent);


    /**
      Tries to update 'this' from the list of valid storages in dataStorage().
      */
    bool updateThis();


    /**
      Tries to update 'this' with the data from 'p'.
      */
    bool updateThis(DataStorageImplementation*p);


    /**
      Set all elements to zero
      */
    virtual void clear() = 0;


    /**
      Creates a new instance of a subtype. Used to copy data.
      */
    virtual DataStorageImplementation* newInstance( DataStorageVoid* p ) = 0;


    /**
      Cow breaks in instances of DataStorageImplementation that doesn't take
      ownership of its data. Disallow Cow in that case.
      */
    virtual bool allowCow() = 0;

private:
    /**
      Return true if p was updated from this
      */
    virtual bool updateOther(DataStorageImplementation*p) = 0;


    /**
      Return true if this could be updated from p
      */
    virtual bool updateFromOther(DataStorageImplementation*p) = 0;


private:
    DataStorageVoid* dataStorage_;
    std::set<DataStorageVoid*> dataStorageCow_;
};


#endif // DATASTORAGE_H
