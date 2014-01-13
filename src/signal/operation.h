#ifndef SIGNALOPERATION_H
#define SIGNALOPERATION_H

//signal
#include "buffer.h"
#include "intervals.h"
#include "processing/iinvalidator.h"

// gpumisc
#include "volatileptr.h"

// QString
#include <QString>

namespace Signal {

namespace Processing { class Step; }

class ComputingEngine;

class OperationDesc;

/**
 * @brief The Operation class should describe the interface for performing signal processing on signal data.
 *
 * 'process' should only be called from one thread. But use VolatilePtr anyways. The overhead is low.
 */
class SaweDll Operation: public VolatilePtr<Operation>
{
public:
    /**
      Virtual housekeeping.
      */
    virtual ~Operation() {}


    /**
     * @brief process computes the operation
     * @param A buffer with data to process. The interval of the buffer will
     * be equal to a value returned by 'OperationDesc::requiredInterval(...)' param 'I'.
     * @return processed data. Returned buffer interval must be equal to expectedOutput in:
     * 'OperationDesc::requiredInterval(b->getInterval(), &expectedOutput)'
     */
    virtual Signal::pBuffer process(Signal::pBuffer b) = 0;


    static void test(Ptr o, OperationDesc*);
};


/**
 * @brief The OperationDesc class should describe the interface for creating instances of the Operation interface.
 *
 * It should invalidate caches if the operation parameters change.
 *
 * All methods except one have const access to make it more likely that there are none or few side-effects.
 */
class SaweDll OperationDesc: public VolatilePtr<OperationDesc>
{
public:
    /**
      Virtual housekeeping.
      */
    virtual ~OperationDesc() {}


    /**
     * @brief requiredInterval should figure out which input interval that is
     * needed for a given output interval.
     * @param I describes an interval in the output.
     * @param expectedOutput describes which interval that will be computed
     * when 'requiredInterval' is processed. This will overlap 'I.first'.
     * expectedOutput may be null to be ignored.
     * @return the interval that is required to compute a valid chunk
     * representing interval I. If the operation can not compute a valid chunk
     * representing the at least interval I at once the operation can request
     * a smaller chunk for processing instead by setting 'expectedOutput'.
     */
    virtual Interval requiredInterval( const Interval& I, Interval* expectedOutput ) const = 0;


    /**
     * @brief affectedInterval
     * @param I
     * @return which interval of samples that needs to be recomputed if 'I'
     * changes in the input.
     */
    virtual Interval affectedInterval( const Interval& I ) const = 0;


    /**
     * @brief copy creates a copy of 'this'.
     * @return a copy.
     */
    virtual OperationDesc::Ptr copy() const = 0;


    /**
     * @brief createOperation instantiates an operation that uses this description.
     * Different computing engines could be used to instantiate different types.
     *
     * May return an empty Operation::Ptr if an operation isn't supported by a
     * given ComputingEngine in which case some other thread will have to
     * populate the cache instead and call invalidate samples when they are
     * ready. This thread will then have to wait, or do something else. It will
     * be marked as complete without computing anything until invalidated.
     * @return a newly created operation.
     */
    virtual Operation::Ptr createOperation(ComputingEngine* engine=0) const = 0;


    /**
     * @brief The SignalExtent struct is returned by OperationDesc::extent ()
     */
    struct Extent {
        boost::optional<Interval> interval;
        boost::optional<float> sample_rate;
        boost::optional<int> number_of_channels;
    };


    /**
     * @brief extent describes the extent of this operation. Extent::interval is allowed
     * to change during the lifetime of an OperationDesc.
     * @return Does not initialize anything unless this operation creates any special extent.
     */
    virtual Extent extent() const;


    /**
     * Returns a string representation of this operation. Mainly used for debugging.
     */
    virtual QString toString() const;


    /**
     * @brief setInvalidator sets an functor to be used by deprecateCache.
     * @param invalidator
     */
    void setInvalidator(Signal::Processing::IInvalidator::Ptr invalidator);


    /**
     * @brief operator == checks if two instances of OperationDesc would generate
     * identical instances of Operation. The default behaviour is to just check
     * the type of the argument.
     * @param d
     * @return
     */
    virtual bool operator==(const OperationDesc& d) const;
    bool operator!=(const OperationDesc& b) const { return !(*this == b); }


    /**
     * @brief operator << makes OperationDesc support common printing routines
     * using the stream operator.
     * @param os
     * @param d
     * @return os
     */
    friend std::ostream& operator << (std::ostream& os, const OperationDesc& d);


protected:
    friend class Signal::Processing::Step;

    /**
     * @brief deprecateCache should be called when parameters change.
     * @param what If what is Signal::Intervals::Intervals_ALL then Step will
     * recreate operations for computing engines as needed.
     *
     * deprecateCache without 'volatile' will release the lock while calling IInvalidator.
     */
    void deprecateCache(Signal::Intervals what=Signal::Intervals::Intervals_ALL);
    void deprecateCache(Signal::Intervals what=Signal::Intervals::Intervals_ALL) const volatile;


private:
    /**
     * @brief invalidator_ is used by deprecateCache.
     *
     * Could be a list<IInvalidator::Ptr> to support adding the same OperationDesc
     * at multiple locations in the Dag.
     */
    Signal::Processing::IInvalidator::Ptr invalidator_;
};

} // namespace Signal

#endif // SIGNALOPERATION_H
