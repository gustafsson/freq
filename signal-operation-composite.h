#ifndef SIGNALOPERATIONCOMPOSITE_H
#define SIGNALOPERATIONCOMPOSITE_H

#include "signal-operation.h"
#include "signal-filteroperation.h"

namespace Signal {

/**
  OperationSubOperations is used by complex Operations that are built
  by combining sequences of several other Operations.

  _sourceSubOperation is an Operation that gets updated on changes of source
  by calls to void source(pSource).

  _readSubOperation is read from by pBuffer read(unsigned,unsigned).

  _sourceSubOperation is initially set up as a dummy operation which only
  reads from _source. _readSubOperation is supposed to be created by another
  class subclassing OperationSubOperations. Hence the protected constructor.
  */
class OperationSubOperations : public Operation {
public:
    pSource subSource() { return _readSubOperation; }
protected:
    OperationSubOperations(pSource source);

    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples );
    virtual void source(pSource v);

    pSource _sourceSubOperation;
    pSource _readSubOperation;
};

/**
  Example 1:
  start:  1234567
  OperationCrop( start, 1, 2 );
  result: 23
*/
class OperationSetSilent: public OperationSubOperations {
public:
    OperationSetSilent( pSource source, unsigned firstSample, unsigned numberOfSamples );

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
    OperationCrop( pSource source, unsigned firstSample, unsigned numberOfSamples );

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
    OperationMove( pSource source, unsigned firstSample, unsigned numberOfSamples, unsigned newFirstSample );

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
    OperationMoveMerge( pSource source, unsigned firstSample, unsigned numberOfSamples, unsigned newFirstSample );

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
    OperationShift( pSource source, int sampleShift );

    void reset( int sampleShift );
};

} // namespace Signal

#endif // SIGNALOPERATIONCOMPOSITE_H
