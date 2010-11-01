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
    Signal::pOperation subSource() { return _readSubOperation; }

	std::string name() { return _name; }

protected:
    OperationSubOperations(Signal::pOperation source, std::string name = "");

    virtual Signal::pBuffer read( const Signal::Interval&  I);
    virtual Signal::pOperation source() { return Signal::Operation::source(); }
    virtual void source(Signal::pOperation v);

    Signal::pOperation _sourceSubOperation;
    Signal::pOperation _readSubOperation;
	std::string _name;
};

/**
  Example 1:
  start:  1234567
  OperationCrop( start, 1, 2 );
  result: 23
*/
class OperationSetSilent: public OperationSubOperations {
public:
    OperationSetSilent( Signal::pOperation source, unsigned firstSample, unsigned numberOfSamples );

    void reset( unsigned firstSample, unsigned numberOfSamples );
};

/**
  Example 1:
  start:  1234567
  OperationCrop( start, 1, 2 );
  result: 23
*/
class OperationCrop: public OperationSubOperations {
public:
    OperationCrop( Signal::pOperation source, unsigned firstSample, unsigned numberOfSamples );

    void reset( unsigned firstSample, unsigned numberOfSamples );
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
    OperationMove( Signal::pOperation source, unsigned firstSample, unsigned numberOfSamples, unsigned newFirstSample );

    void reset( unsigned firstSample, unsigned numberOfSamples, unsigned newFirstSample );
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
    OperationMoveMerge( Signal::pOperation source, unsigned firstSample, unsigned numberOfSamples, unsigned newFirstSample );

    void reset( unsigned firstSample, unsigned numberOfSamples, unsigned newFirstSample );
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
    OperationShift( Signal::pOperation source, int sampleShift );

    void reset( int sampleShift );
};

class OperationMoveSelection: public OperationSubOperations {
public:
    OperationMoveSelection( Signal::pOperation source, Signal::pOperation selectionFilter, int sampleShift, float freqDelta );

    void reset( Signal::pOperation selectionFilter, int sampleShift, float freqDelta );
};

} // namespace Support
} // namespace Tools

#endif // SIGNALOPERATIONCOMPOSITE_H
