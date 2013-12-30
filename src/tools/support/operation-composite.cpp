#include "operation-composite.h"

#include "signal/operation-basic.h"
#include "filters/move.h"
#include "filters/ellipse.h"
#include "demangle.h"
#include "signal/computingengine.h"

using namespace Signal;

namespace Tools {
    namespace Support {

    // OperationSubOperations  /////////////////////////////////////////////////////////////////

OperationSubOperations::
        OperationSubOperations(Signal::pOperation source, std::string name)
:   DeprecatedOperation(pOperation()),
    source_sub_operation_( new DummyOperation(source)),
    name_(name)
{
//    enabled(false);
//    source_sub_operation_->enabled(false);
    DeprecatedOperation::source( source_sub_operation_ );
}


Intervals affected_samples_recursive_until(pOperation o, pOperation stop)
{
    Intervals r;
    if (o)
    {
        r = o->affected_samples();
        if (o!=stop)
            r |= o->translate_interval( affected_samples_recursive_until(o->source(), stop) );
    }
    return r;
}


Intervals OperationSubOperations::
        affected_samples()
{
    return affected_samples_recursive_until( subSource(), source_sub_operation_);
}


Intervals zeroed_samples_recursive_until(pOperation o, pOperation stop)
{
    Intervals r;
    if (o)
    {
        r = o->zeroed_samples();
        if (o!=stop)
            r |= o->translate_interval( zeroed_samples_recursive_until(o->source(), stop) );
    }
    return r;
}


Signal::Intervals OperationSubOperations::
        zeroed_samples()
{
    return zeroed_samples_recursive_until( subSource(), source_sub_operation_);
}


    // OperationCrop  /////////////////////////////////////////////////////////////////

OperationCrop::Extent OperationCrop::
        extent() const
{
    Extent x;
    x.interval = section_;
    return x;
}

QString OperationCrop::
        toString() const
{
    std::stringstream ss;
    ss << "Crop " << section_;
    return ss.str().c_str ();
}

void OperationCrop::
        test()
{
    /**
      Example 1:
      start:  1234567
      OperationCrop( start, 1, 2 );
      result: 23
    */
    {
        EXCEPTION_ASSERTX(false, "not implemented");
    }
}

    // OperationOtherSilent  /////////////////////////////////////////////////////////////////



OperationOtherSilent::Operation::
        Operation(const Interval &section)
    :
      section_(section)
{}


pBuffer OperationOtherSilent::Operation::
        process (pBuffer b)
{
    Signal::Intervals I = b->getInterval ();
    I -= section_;

    foreach (Signal::Interval i, I) {
        Buffer zero(i, b->sample_rate(), b->number_of_channels ());
        *b |= zero;
    }

    return b;
}


OperationOtherSilent ::OperationOtherSilent(const Interval &section)
    :
      section_(section)
{
}


Interval OperationOtherSilent::
        requiredInterval( const Interval& I, Interval* expectedOutput ) const
{
    if (expectedOutput)
        *expectedOutput = I;
    return I;
}


Interval OperationOtherSilent::
        affectedInterval( const Interval& I ) const
{
    return I;
}


OperationDesc::Ptr OperationOtherSilent::
        copy() const
{
    return OperationDesc::Ptr(new OperationOtherSilent(section_));
}


Signal::Operation::Ptr OperationOtherSilent::
        createOperation(ComputingEngine* engine) const
{
    if (0==engine || dynamic_cast<Signal::ComputingCpu*>(engine))
        return Signal::Operation::Ptr(new OperationOtherSilent::Operation(section_));
    return Signal::Operation::Ptr();
}


void OperationOtherSilent::
        test()
{
    /**
      Example 1:
      start:  1234567
      OperationOtherSilent( start, 1, 2 );
      result: 0230000
    */
    {
        EXCEPTION_ASSERTX(false, "not implemented");
    }
}

    // OperationMove  /////////////////////////////////////////////////////////////////
#if 0 // TODO implement using branching in the Dag

OperationMove::
        OperationMove( pOperation source, const Signal::Interval& section, unsigned newFirstSample )
:   OperationSubOperations( source, "Move" )
{
    reset(section, newFirstSample);
}

void OperationMove::
        reset( const Signal::Interval& section, unsigned newFirstSample )
{
    // Note: difference to OperationMoveMerge is that OperationMove has the silenceTarget step

    Intervals newSection = section;
    if (newFirstSample<section.first)
        newSection >>= (section.first-newFirstSample);
    else
        newSection <<= (newFirstSample-section.first);

    pOperation silenceTarget( new OperationSetSilent(source_sub_operation_, newSection.spannedInterval() ));
    pOperation silence( new OperationSetSilent(silenceTarget, section ));

    pOperation crop( new OperationCrop( source_sub_operation_, section ));
    pOperation moveToNewPos( new OperationInsertSilence( crop, Interval(0, newFirstSample)));

    pOperation addition( new OperationSuperposition (moveToNewPos, silence ));

    DeprecatedOperation::source( addition );
}
#endif

    // OperationMoveMerge  /////////////////////////////////////////////////////////////////
#if 0 // TODO implement using branching in the Dag

OperationMoveMerge::
        OperationMoveMerge( pOperation source, const Signal::Interval& section, unsigned newFirstSample )
:   OperationSubOperations( source, "Move and merge" )
{
    reset(section, newFirstSample);
}

void OperationMoveMerge::
        reset( const Signal::Interval& section, unsigned newFirstSample )
{
    pOperation silence( new OperationSetSilent (source_sub_operation_, section ));

    pOperation crop( new OperationCrop( source_sub_operation_, section ));
    pOperation moveToNewPos( new OperationInsertSilence( crop, Interval(0, newFirstSample)));

    pOperation addition( new OperationSuperposition (moveToNewPos, silence ));

    DeprecatedOperation::source( addition );
}

#endif
    // OperationShift  /////////////////////////////////////////////////////////////////

class OperationShiftOperation: public Signal::Operation
{
public:
    OperationShiftOperation( long sampleShift )
        :
          sampleShift_(sampleShift)
    {

    }


    Signal::pBuffer process(Signal::pBuffer b)
    {
        UnsignedF o = b->sample_offset () + sampleShift_;
        b->set_sample_offset (o);
        return b;
    }

private:
    long long sampleShift_;
};

OperationShift::
        OperationShift( long sampleShift )
    :
      sampleShift_(sampleShift)
{
}

Signal::Interval OperationShift::
        requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const
{
    if (expectedOutput)
        *expectedOutput = I;

    return (Signal::Intervals(I) >>= sampleShift_).spannedInterval ();
}

Signal::Interval OperationShift::
        affectedInterval( const Signal::Interval& I ) const
{
    return (Signal::Intervals(I) <<= sampleShift_).spannedInterval ();
}

Signal::OperationDesc::Ptr OperationShift::
        copy() const
{
    return Signal::OperationDesc::Ptr(new OperationShift(sampleShift_));
}

Signal::Operation::Ptr OperationShift::
        createOperation(Signal::ComputingEngine* engine) const
{
    if (0==engine || dynamic_cast<Signal::ComputingCpu*>(engine))
        return Signal::Operation::Ptr(new OperationShiftOperation(sampleShift_));
    return Signal::Operation::Ptr();
}


#ifdef USE_CUDA
    // OperationMoveSelection  /////////////////////////////////////////////////////////////////

OperationMoveSelection::
        OperationMoveSelection( pOperation source, pOperation selectionFilter, long sampleShift, float freqDelta )
:	OperationSubOperations( source, "OperationMoveSelection" )
{
	reset(selectionFilter, sampleShift, freqDelta );
}


void OperationMoveSelection::
    reset( pOperation selectionFilter, long sampleShift, float freqDelta )
{
    // Take out the samples affected by selectionFilter and move them
    // 'sampleShift' in time and 'freqDelta' in frequency

    pOperation  extract, remove;
    if (Filters::Ellipse* f = dynamic_cast<Filters::Ellipse*>(selectionFilter.get())) {

        // Create filter for extracting selection
        extract.reset( new Filters::Ellipse(*f) );
        dynamic_cast<Filters::Ellipse*>(extract.get())->_save_inside = true;
        extract->source( source() );

        // Create filter for removing selection
        remove.reset( new Filters::Ellipse(*f) );
        dynamic_cast<Filters::Ellipse*>(remove.get())->_save_inside = false;
        remove->source( source() );

	} else {
		throw std::invalid_argument(std::string(__FUNCTION__) + " only supports Tfr::EllipseFilter as selectionFilter");
	}

    pOperation extractAndMove = extract;
    {
        // Create operation for moving extracted selection in time
        if (0!=sampleShift)
        extractAndMove.reset( new OperationShift( extractAndMove, sampleShift ));

        // Create operation for moving extracted selection in frequency
        if (0!=freqDelta)
        {
            pOperation t( new Filters::Move( freqDelta ));
            t->source( extractAndMove );
            extractAndMove = t;
        }

	}

    pOperation mergeSelection( new OperationSuperposition( remove, extractAndMove ));

    Operation::source( mergeSelection );
}
#endif



    // OperationFilterSelection  /////////////////////////////////////////////////////////////////

#if 0 // TODO implement using branching in the Dag because this operation involves a merge of two or more different signals
OperationOnSelection::
        OperationOnSelection( pOperation source, pOperation insideSelection, pOperation outsideSelection, Signal::pOperation operation )
:   OperationSubOperations( source, "OperationOnSelection" )
{
    reset( insideSelection, outsideSelection, operation );
}


std::string OperationOnSelection::
        name()
{
    return (operation_?operation_->name():"(null)") + " in " + (insideSelection_?insideSelection_->name():"(null)");
}


void OperationOnSelection::
        reset( pOperation insideSelection, pOperation outsideSelection, Signal::pOperation operation )
{
    EXCEPTION_ASSERT(insideSelection);
    EXCEPTION_ASSERT(outsideSelection);
    EXCEPTION_ASSERT(operation);

    insideSelection_ = insideSelection;
    operation_ = operation;

    // Take out the samples affected by selectionFilter

    outsideSelection->source( source_sub_operation_ );
    insideSelection->source( source_sub_operation_ );
    operation->source( insideSelection );

    pOperation mergeSelection( new OperationSuperposition( operation, outsideSelection ));

    // Makes reads read from 'mergeSelection'
    DeprecatedOperation::source( mergeSelection );
}
#endif

    } // namespace Support
} // namespace Tools
