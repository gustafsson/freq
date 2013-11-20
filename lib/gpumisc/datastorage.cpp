#include "datastorage.h"

#include "computationkernel.h"
#include "cpumemorystorage.h"
#include "TaskTimer.h"
#include "demangle.h"

#include <algorithm>

#include <boost/assert.hpp>
#include <boost/foreach.hpp>

//#define TIME_DataStorageImplementation
#define TIME_DataStorageImplementation if(0)

DataStorageImplementation::
        DataStorageImplementation( DataStorageVoid* dataStorage )
            :
            dataStorage_(dataStorage)
{
    TIME_DataStorageImplementation
            TaskInfo tt("%s %p ( %u, %u, %u ), elemsize %u",
                         vartype(*this).c_str(), dataStorage_,
                         size().width, size().height, size().depth,
                         dataStorage_->bytesPerElement());

    EXCEPTION_ASSERT( dataStorage_ );

    dataStorage_->storage_.insert( this );
}


DataStorageImplementation::
        ~DataStorageImplementation()
{
    bool dataStorageHasThis = (1 == dataStorage_->storage_.count( this ));
    EXCEPTION_ASSERT( dataStorageHasThis );
    EXCEPTION_ASSERT( dataStorageCow_.empty() );

    dataStorage_->storage_.erase( this );
    dataStorage_->validContent_.erase( this );
}


void DataStorageImplementation::
        addCowCopy(DataStorageVoid *parent)
{
    EXCEPTION_ASSERT( dataStorage_->numberOfBytes() == parent->numberOfBytes() );

    dataStorageCow_.insert( parent );
    parent->storage_.insert( this );
    if (dataStorage_->validContent_.count(this))
        parent->validContent_.insert( this );
}


bool DataStorageImplementation::
        removeCowCopy(DataStorageVoid*parent)
{
    if (dataStorageCow_.count(parent))
    {
        dataStorageCow_.erase(parent);
        parent->storage_.erase(this);
        parent->validContent_.erase(this);
        return true;
    }

    EXCEPTION_ASSERT( parent == dataStorage_ );

    if (parent == dataStorage_ && 1 <= dataStorageCow_.size ())
    {
        DataStorageVoid* q = *dataStorageCow_.begin ();
        dataStorageCow_.erase (q);
        dataStorage_->validContent_.erase (this);
        dataStorage_->storage_.erase (this);
        dataStorage_ = q;
        return true;
    }

    return false;
}


DataStorageImplementation* DataStorageImplementation::
        copyOnWrite(DataStorageVoid* parent)
{
    DataStorageImplementation* r = 0;

    std::set<DataStorageVoid*> cows = dataStorageCow_;
    BOOST_FOREACH(DataStorageVoid* p, cows)
    {
        TIME_DataStorageImplementation
                TaskTimer tt("copyOnWrite");

        p->DiscardAllData();

        DataStorageImplementation* q = newInstance( p );
        bool updated = q->updateThis( this );
        EXCEPTION_ASSERT( updated );
        p->validContent_.insert( q );

        if (parent == p)
            r = q;
    }

    if (parent == dataStorage_)
        r = this;

    EXCEPTION_ASSERT( r != 0 );
    return r;
}


bool DataStorageImplementation::
        updateThis()
{
    DataStorageVoid::StorageImplementations validContent = dataStorage()->validContent();

    if (validContent.empty())
    {
        TIME_DataStorageImplementation
                TaskTimer tt("%s %p clear( %u, %u, %u ), elemsize %u",
                             vartype(*this).c_str(), dataStorage_,
                             size().width, size().height, size().depth,
                             dataStorage()->bytesPerElement());

        this->clear();

        return true;
    }

    if (1 == validContent.count(this))
        return true;

    BOOST_FOREACH( DataStorageImplementation* p, validContent )
    {
        if (updateThis(p))
            return true;
    }

    return false;
}


bool DataStorageImplementation::
        updateThis(DataStorageImplementation*p)
{
    TIME_DataStorageImplementation TaskTimer tt("Updating %s %p (%u, %u, %u), elemsize %u",
            demangle( typeid(*this) ).c_str(), dataStorage_,
            size().width, size().height, size().depth, dataStorage_->bytesPerElement() );

    if (p->updateOther( this ))
    {
        TIME_DataStorageImplementation TaskInfo(
                "Updated from %s %p updateOther",
                demangle( typeid(*p) ).c_str(), p->dataStorage() );
        return true;
    }

    if (this->updateFromOther(p))
    {
        TIME_DataStorageImplementation TaskInfo(
                "Updated from %s updateFromOther",
                demangle( typeid(*p) ).c_str() );
        return true;
    }

    return false;
}


DataStorageVoid::
        DataStorageVoid(DataStorageSize size, size_t bytesPerElement)
    :
    size_( size ),
    bytesPerElement_( bytesPerElement )
{
}


DataStorageVoid::
        DataStorageVoid( const DataStorageVoid& b )
    :
    size_( b.size_ ),
    bytesPerElement_( b.bytesPerElement_ )
{
    *this = b;
}


DataStorageVoid::
        ~DataStorageVoid()
{
    DiscardAllData();
}


DataStorageSize DataStorageVoid::
        sizeInBytes() const
{
    DataStorageSize sz = size();
    sz.width *= bytesPerElement();
    return sz;
}


void DataStorageVoid::
        ClearContents()
{
    BOOST_FOREACH( DataStorageImplementation* p, validContent_ )
    {
        p->clear();
    }
}


void DataStorageVoid::
        DiscardAllData()
{
    TIME_DataStorageImplementation
            TaskTimer tt("%s.DiscardAllData %p #%d ( %u, %u, %u ), elemsize %u",
                         vartype(*this).c_str(), this, storage_.size(),
                         size().width, size().height, size().depth,
                         bytesPerElement());

    validContent_.clear();

    while (!storage_.empty())
    {
        if (!(**storage_.begin()).removeCowCopy(this))
        {
            (**storage_.begin()).copyOnWrite(this);
            size_t s = storage_.size();
            delete *storage_.begin();
            EXCEPTION_ASSERT( storage_.size() + 1 == s );
        }
    }
}


DataStorageVoid& DataStorageVoid::
        operator=(const DataStorageVoid& b)
{
    bool allowCow = false;
    BOOST_FOREACH( DataStorageImplementation* p, b.validContent_ )
    {
        // There has to be one instances that permits Cow copies
        allowCow |= p->allowCow();
    }
    BOOST_FOREACH( DataStorageImplementation* p, this->storage_ )
    {
        // All allocated instances in 'this' must support being replaced by 'Cow' copies.
        allowCow &= p->allowCow();
    }

    if (allowCow)
    {
        TIME_DataStorageImplementation
                TaskTimer tt("Adding a COW copy");

        // COW, copy on write, postpones copying of data chunks until 'UpdateValidContent'
        DiscardAllData();

        BOOST_FOREACH( DataStorageImplementation* p, b.storage_ )
        {
            if (p->allowCow())
                p->addCowCopy( this );
        }
    }
    else
    {
        DeepCopy(b);
    }

    return *this;
}


void DataStorageVoid::
        DeepCopy(const DataStorageVoid&b)
{
    validContent_.clear();

    DataStorageImplementation* p = CopyStorage(b);
    if (p)
        validContent_.insert( p );
}


/*static*/ std::string DataStorageVoid::
       getMemorySizeText( unsigned long long size, char decimals, char type )
{
    if (type != 'g' && type != 'f')
        type = 'g';

    float value = (float)size;
    std::string format;
    if (size>>40 > 4) {
        format = "TB";
        value = size/(float)(((unsigned long long )1)<<40);

    } else if (size>>30 > 5) {
        format = "GB";
        value = size/(float)(1<<30);

    } else if (size>>20 > 5) {
        format = "MB";
        value = size/(float)(1<<20);

    } else if (size>>10 > 5) {
        format = "KB";
        value = size/(float)(1<<10);

    } else {
        return (boost::format("%u B") % size).str();
    }

    if (decimals < 0)
        format = (boost::format("%%%c ") % type).str();
    else
        format = (boost::format("%%.%u%c ") % decimals % type).str();

    return (boost::format(format) % value).str();
}


void DataStorageVoid::
        OnlyKeepOneStorage(DataStorageImplementation* t)
{
    EXCEPTION_ASSERT(storage_.count(t) && validContent_.count(t));

    StorageImplementations toRemove = storage_;
    toRemove.erase( t );

    BOOST_FOREACH( DataStorageImplementation* p, toRemove )
    {
        delete p; // removes from storage_
    }

    EXCEPTION_ASSERT( storage_.size() == 1 );
    EXCEPTION_ASSERT( storage_.count(t) == 1 );
    validContent_ = storage_;
}


DataStorageImplementation* DataStorageVoid::
        UpdateValidContent( DataStorageImplementation* t, bool read, bool write )
{
    if (read)
    {
        bool storageIsUpToDate = 0!=validContent_.count(t);

        if (!storageIsUpToDate)
        {
            t = t->copyOnWrite(this);
            bool success = t->updateThis();
            EXCEPTION_ASSERT( success );
            validContent_.insert( t );
        }
    }

    if (write)
    {
        t = t->copyOnWrite(this);
        validContent_.clear();
        validContent_.insert( t );
    }

    EXCEPTION_ASSERT(read || write);

    return t;
}


DataStorageImplementation* DataStorageVoid::
        CopyStorage(const DataStorageVoid& b)
{
    // Look for anything we have allocated that can use valid content from 'b'
    // Note. if b.validContent_ is empty this operator doesn't do anything.
    StorageImplementations oldValidContent = validContent_;
    validContent_.clear();

    BOOST_FOREACH( DataStorageImplementation* p, oldValidContent )
    {
        BOOST_FOREACH( DataStorageImplementation* bp, b.validContent_ )
        {
            if (p->updateThis( bp ))
                return p;
        }
    }
    StorageImplementations oldAllocations = storage_;

    // skip storages that has already been tried above
    BOOST_FOREACH( DataStorageImplementation* p, oldValidContent )
        oldAllocations.erase( p );

    BOOST_FOREACH( DataStorageImplementation* p, oldAllocations )
    {
        BOOST_FOREACH( DataStorageImplementation* bp, b.validContent_ )
        {
            if (p->updateThis( bp ))
                return p;
        }
    }

    BOOST_FOREACH( DataStorageImplementation* bp, b.validContent_ )
    {
        DataStorageImplementation* p = bp->newInstance( this );

        EXCEPTION_ASSERT( p );

        if (p->updateThis( bp ))
            return p;

        return p;
    }

    return 0;
}


void* getCpuMemory(DataStorageVoid*p)
{
    return p->FindCreateStorage<CpuMemoryStorage>( true, true )->AccessBytes<3>().ptr();
}
