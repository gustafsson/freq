#ifndef SIGNALOPERATION_H
#define SIGNALOPERATION_H

//signal
#include "buffer.h"
#include "intervals.h"
#include "processing/iinvalidator.h"

// gpumisc
#include "shared_state.h"
#include "shared_state_traits_backtrace.h"

// QString
#include <QString>

namespace Signal {

namespace Processing { class Step; }

class ComputingEngine;

class OperationDesc;

/**
 * @brief The Operation class should describe the interface for performing signal processing on signal data.
 *
 * 'process' should only be called from one thread.
 */
class SaweDll Operation
{
public:
    typedef std::shared_ptr<Operation> ptr;

    /**
      Virtual housekeeping.
      */
    virtual ~Operation() {}


    /**
     * @brief process computes the operation
     * @param A buffer with data to process. The interval of the buffer must
     * be equal to a value returned by 'OperationDesc::requiredInterval(...)' param 'I'.
     * @return processed data. Returned buffer interval must be equal to expectedOutput in:
     * 'OperationDesc::requiredInterval(b->getInterval(), &expectedOutput)'
     */
    virtual Signal::pBuffer process(Signal::pBuffer b) = 0;


    static void test(ptr o, OperationDesc*);
};


/**
 * @brief The OperationDesc class should describe the interface for creating instances of the Operation interface.
 *
 * It should invalidate caches if the operation parameters change.
 *
 * All methods except one have const access to make it more likely that there are none or few side-effects.
 */
class SaweDll OperationDesc
{
public:
    typedef shared_state<OperationDesc> ptr;
    typedef shared_state_traits_backtrace shared_state_traits;

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
    virtual OperationDesc::ptr copy() const = 0;


    /**
     * @brief createOperation instantiates an operation that uses this description.
     * Different computing engines could be used to instantiate different types.
     *
     * May return an empty Operation::Ptr if an operation isn't supported by a
     * given ComputingEngine.
     *
     * Note that 'requiredInterval' must be called within the same lock of 'this'
     * to guarantee that the created operation maches the Intervals given by
     * 'requiredInterval'. Any changes of to 'this' may create new intervals.
     * Therefore 'createOperation' should be called for each new call to
     * process to take any new settings into account.
     *
     * Changes to 'this' after 'createOperation' should not change the state
     * of the 'Operation'. But any results obtained therefrom are obsolete.
     *
     * @return a newly created operation.
     */
    virtual Operation::ptr createOperation(ComputingEngine* engine=0) const = 0;


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
     *
     * Could be a list<IInvalidator::Ptr> to support adding the same OperationDesc
     * at multiple locations in the Dag.
     */
    void setInvalidator(Signal::Processing::IInvalidator::ptr invalidator);
    Signal::Processing::IInvalidator::ptr getInvalidator() const;


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


private:
    /**
     * @brief invalidator_ is used by deprecateCache.
     *
     * Could be a list<IInvalidator::Ptr> to support adding the same OperationDesc
     * at multiple locations in the Dag.
     */
    Signal::Processing::IInvalidator::ptr invalidator_;
};

} // namespace Signal

#endif // SIGNALOPERATION_H
