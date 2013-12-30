#ifndef SIGNALOPERATIONCOMPOSITE_H
#define SIGNALOPERATIONCOMPOSITE_H

// TODO these belong to their respective tool

#include "signal/operation.h"
#include "tfr/cwtfilter.h"

namespace Tools {
    namespace Support {


/**
  Example 1:
  start:  1234567
  OperationOtherSilent( start, 1, 2 );
  result: 0230000
*/
class OperationOtherSilent: public Signal::OperationDesc {
public:
    class Operation: public Signal::Operation {
    public:
        Operation( const Signal::Interval& section );

        Signal::pBuffer process(Signal::pBuffer b);

    private:
        Signal::Interval section_;
    };

    OperationOtherSilent( const Signal::Interval& section );

    // OperationDesc
    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    Signal::Interval affectedInterval( const Signal::Interval& I ) const;
    Signal::OperationDesc::Ptr copy() const;
    Signal::Operation::Ptr createOperation(Signal::ComputingEngine* engine=0) const;

    Signal::Intervals zeroed_samples();

    void reset( const Signal::Interval& section );

    Signal::Interval section() { return section_; }
private:
    Signal::Interval section_;

    friend class boost::serialization::access;
    OperationOtherSilent():section_(Signal::Interval()) {} // only used by deserialization

    template<class archive>
    void serialize(archive& ar, const unsigned int /*version*/)
    {
        using boost::serialization::make_nvp;

        ar & BOOST_SERIALIZATION_NVP(section_.first)
           & BOOST_SERIALIZATION_NVP(section_.last);
    }

public:
    static void test();
};


/**
  Example 1:
  start:  1234567
  OperationCrop( start, 1, 2 );
  result: 23
*/
class OperationCrop: public OperationOtherSilent {
public:
    OperationCrop( const Signal::Interval& section );

    // OperationDesc
    Extent extent() const;
    QString toString() const;

private:
    Signal::Interval section_;

    friend class boost::serialization::access;
    OperationCrop():OperationOtherSilent(Signal::Interval()) {} // only used by deserialization

    template<class archive>
    void serialize(archive& ar, const unsigned int /*version*/)
    {
        using boost::serialization::make_nvp;

        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(OperationOtherSilent);
    }

public:
    static void test();
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
#if 0 // TODO implement using branching in the Dag because this operation involves a merge of two or more different signals
class OperationMove: public OperationSubOperations {
public:
    OperationMove( Signal::pOperation source, const Signal::Interval& section, unsigned newFirstSample );

    void reset( const Signal::Interval& section, unsigned newFirstSample );
};
#endif

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
#if 0 // TODO implement using branching in the Dag because this operation involves a merge of two or more different signals
class OperationMoveMerge: public OperationSubOperations {
public:
    OperationMoveMerge( Signal::pOperation source, const Signal::Interval& section, unsigned newFirstSample );

    void reset( const Signal::Interval& section, unsigned newFirstSample );
};
#endif

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
class OperationShift: public Signal::OperationDesc {
public:
    OperationShift( long sampleShift );

    // OperationDesc
    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    Signal::Interval affectedInterval( const Signal::Interval& I ) const;
    Signal::OperationDesc::Ptr copy() const;
    Signal::Operation::Ptr createOperation(Signal::ComputingEngine* engine=0) const;

private:
    long sampleShift_;
};

#ifdef USE_CUDA
class OperationMoveSelection: public OperationSubOperations {
public:
    OperationMoveSelection( Signal::pOperation source, Signal::pOperation selectionFilter, long sampleShift, float freqDelta );

    void reset( Signal::pOperation selectionFilter, long sampleShift, float freqDelta );
};
#endif

#if 0 // TODO implement using branching in the Dag because this operation involves a merge of two or more different signals
class OperationOnSelection: public OperationSubOperations {
public:
    OperationOnSelection( Signal::pOperation source, Signal::pOperation insideSelection, Signal::pOperation outsideSelection, Signal::pOperation operation );

    virtual std::string name();

    void reset( Signal::pOperation insideSelection, Signal::pOperation outsideSelection, Signal::pOperation operation );

    Signal::pOperation selection() { return insideSelection_; }
    Signal::pOperation operation() { return operation_; }

private:
    Signal::pOperation insideSelection_;
    Signal::pOperation operation_;


    friend class boost::serialization::access;
    OperationOnSelection():OperationSubOperations(Signal::pOperation(),"") {} // only used by deserialization

    template<class archive>
    void serialize(archive& ar, const unsigned int /*version*/)
    {
        using boost::serialization::make_nvp;

        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(OperationSubOperations)
           & BOOST_SERIALIZATION_NVP(insideSelection_)
           & BOOST_SERIALIZATION_NVP(operation_);
    }
};
#endif

} // namespace Support
} // namespace Tools

#endif // SIGNALOPERATIONCOMPOSITE_H
