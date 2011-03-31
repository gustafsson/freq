#ifndef SIGNALOPERATIONCOMPOSITE_H
#define SIGNALOPERATIONCOMPOSITE_H

// TODO these belong to their respective tool

#include "signal/operation.h"
#include "tfr/cwtfilter.h"

namespace Tools {
    namespace Support {

/**
  OperationSubOperations is used by complex Operations that are built
  by combining sequences of several other Operations.

  _sourceSubOperation is an Operation that gets updated on changes of source
  by calls to void source(pOperation).

  _readSubOperation is read from by pBuffer read(unsigned,unsigned).

  _sourceSubOperation is initially set up as a dummy operation which only
  reads from _source. _readSubOperation is supposed to be created by another
  class subclassing OperationSubOperations. Hence the protected constructor.
  */
class OperationSubOperations : public Signal::Operation {
public:
    Signal::pOperation subSource() { return Operation::source(); }

    /// this skips all contained suboperations
    virtual Signal::pOperation source() const { return source_sub_operation_->source(); }
    virtual void source(Signal::pOperation v) { source_sub_operation_->source(v); }

    /**
        affected_samples needs to take subSource into account.
        If samples are moved by a sub operation affected_samples might have to
        be overloaded.
    */
    virtual Signal::Intervals affected_samples();

    std::string name() { return name_; }

protected:
    OperationSubOperations(Signal::pOperation source, std::string name);

    Signal::pOperation source_sub_operation_;
    std::string name_;
};


/**
  OperationContainer contains exactly 1 other Operation (as opposed to
  OperationSubOperations which contains an arbitrary number of operations in
  sequence). This is useful when you want to pass around an Operation but the
  Operation implementation might change afterwards.

  This happens for instance with selection tools. The selection filter has a
  specific location in the Operation tree but when the user changes the
  selection the implementation might change from a Rectangle to a
  OperationOtherSilent and back again.
 */
class OperationContainer: public OperationSubOperations
{
public:
    OperationContainer(Signal::pOperation source, std::string name );

    void setContent(Signal::pOperation content)
    {
        if (!content)
            Operation::source( source_sub_operation_ );
        else
        {
            Operation::source( content );
            Operation::source()->source( source_sub_operation_ );
        }
    }
    Signal::pOperation content() { return subSource(); }
};


/**
  Example 1:
  start:  1234567
  OperationSetSilent( start, 1, 2 );
  result: 1004567
*/
class OperationSetSilent: public OperationSubOperations {
public:
    OperationSetSilent( Signal::pOperation source, const Signal::Interval& section );

    void reset( const Signal::Interval& section );

    virtual Signal::Intervals zeroed_samples() { return affected_samples(); }
    virtual Signal::Intervals affected_samples() { return section_; }
private:
    Signal::Interval section_;
};

/**
  Example 1:
  start:  1234567
  OperationOtherSilent( start, 1, 2 );
  result: 0230000
*/
class OperationOtherSilent: public OperationSubOperations {
public:
    OperationOtherSilent( Signal::pOperation source, const Signal::Interval& section );
    OperationOtherSilent( float fs, const Signal::Interval& section );

    void reset( const Signal::Interval& section, float fs=0 );

    Signal::Interval section() { return section_; }
private:
    Signal::Interval section_;
};

/**
  Example 1:
  start:  1234567
  OperationCrop( start, 1, 2 );
  result: 23
*/
class OperationCrop: public OperationSubOperations {
public:
    OperationCrop( Signal::pOperation source, const Signal::Interval& section );

    void reset( const Signal::Interval& section );
};

/**
  Example 1:
  start:  1234567
  OperationMove( start, 1, 2, 4 );
  result: 1004237

  explanation of example 1:
  intermediate 1: 1004007
  intermediate 2: 0000230 (2 samples, starting from 1 are moved to position 4)
  result = sum of intermediate 1 and intermediate 2

  Example 2:
  start:  1234567
  OperationMove( start, 1, 2, 1 );
  result: 1234567 ( same as start )

  Example 3:
  start:  1234567
  OperationMove( start, 1, 2, 0 );
  result: 1204567 ("10" + "23" = "33")
  */
class OperationMove: public OperationSubOperations {
public:
    OperationMove( Signal::pOperation source, const Signal::Interval& section, unsigned newFirstSample );

    void reset( const Signal::Interval& section, unsigned newFirstSample );
};

/**
  Example 1:
  start:  1234567
  OperationMoveMerge( start, 1, 2, 3 );
  result: 1006867 ("45" + "23" = "68")

  explanation of example 1:
  intermediate 1: 1004567
  intermediate 2: 0002300 (2 samples, starting from 1 are moved to position 3)
  result = sum of intermediate 1 and intermediate 2

  Example 2:
  start:  1234567
  OperationMoveMerge( start, 1, 2, 1 );
  result: 1234567 ( same as start )

  Example 3:
  start:  1234567
  OperationMoveMerge( start, 1, 2, 0 );
  result: 3304567 ("10" + "23" = "33")
  */
class OperationMoveMerge: public OperationSubOperations {
public:
    OperationMoveMerge( Signal::pOperation source, const Signal::Interval& section, unsigned newFirstSample );

    void reset( const Signal::Interval& section, unsigned newFirstSample );
};

/**
  Example 1:
  start:  1234567
  OperationShift( start, 1 );
  result: 01234567

  Example 2:
  start:  1234567
  OperationShift( start, 0 );
  result: 1234567 ( same as start )

  Example 3:
  start:  1234567
  OperationShift( start, -1 );
  result: 234567
  */
class OperationShift: public OperationSubOperations {
public:
    OperationShift( Signal::pOperation source, long sampleShift );

    void reset( long sampleShift );
};

class OperationMoveSelection: public OperationSubOperations {
public:
    OperationMoveSelection( Signal::pOperation source, Signal::pOperation selectionFilter, long sampleShift, float freqDelta );

    void reset( Signal::pOperation selectionFilter, long sampleShift, float freqDelta );
};


class OperationOnSelection: public OperationSubOperations {
public:
    OperationOnSelection( Signal::pOperation source, Signal::pOperation insideSelection, Signal::pOperation outsideSelection, Signal::pOperation operation );

    virtual std::string name();

    void reset( Signal::pOperation insideSelection, Signal::pOperation outsideSelection, Signal::pOperation operation );

    Signal::pOperation selection() { return insideSelection_; }

private:
    Signal::pOperation insideSelection_;
    Signal::pOperation operation_;
};


} // namespace Support
} // namespace Tools

#endif // SIGNALOPERATIONCOMPOSITE_H
