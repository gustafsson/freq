#include "operation-composite.h"

#include "signal/operation-basic.h"
#include "filters/move.h"
#include "filters/ellipse.h"
#include <demangle.h>

using namespace Signal;

namespace Tools {
    namespace Support {


    // OperationSubOperations  /////////////////////////////////////////////////////////////////

OperationSubOperations::
        OperationSubOperations(pOperation source, std::string name)
:   Operation(source),
    source_sub_operation_( new Operation(source)),
    name_(name)
{
    enabled(false);
    source_sub_operation_->enabled(false);
}


    // OperationContainer  /////////////////////////////////////////////////////////////////

OperationContainer::
        OperationContainer(Signal::pOperation source, std::string name )
            :
            OperationSubOperations(source, name)
{
}


    // OperationCrop  /////////////////////////////////////////////////////////////////

OperationCrop::
        OperationCrop( pOperation source, unsigned firstSample, unsigned numberOfSamples )
:   OperationSubOperations( source )
{
    reset(firstSample, numberOfSamples);
}

void OperationCrop::
        reset( unsigned firstSample, unsigned numberOfSamples )
{
    pOperation cropBefore( new OperationRemoveSection( source_sub_operation_, 0, firstSample ));
    pOperation cropAfter( new OperationRemoveSection( cropBefore, numberOfSamples,
                                                   cropBefore->number_of_samples() - numberOfSamples ));

    _source = cropAfter;
}


    // OperationSetSilent  /////////////////////////////////////////////////////////////////
OperationSetSilent::
        OperationSetSilent( pOperation source, unsigned firstSample, unsigned numberOfSamples )
:   OperationSubOperations( source )
{
    reset(firstSample, numberOfSamples);
}

void OperationSetSilent::
        reset( unsigned firstSample, unsigned numberOfSamples )
{
    firstSample_ = firstSample;
    numberOfSamples_ = numberOfSamples;

    pOperation remove( new OperationRemoveSection( source_sub_operation_, firstSample, numberOfSamples ));
    pOperation addSilence( new OperationInsertSilence (remove, firstSample, numberOfSamples ));

    _source = addSilence;
}

Signal::Intervals OperationSetSilent::
        affected_samples()
{
    return Signal::Interval(firstSample_,firstSample_+ numberOfSamples_);
}


    // OperationOtherSilent  /////////////////////////////////////////////////////////////////
OperationOtherSilent::
        OperationOtherSilent( Signal::pOperation source, unsigned firstSample, unsigned numberOfSamples )
:   OperationSubOperations( source )
{
    reset(firstSample, numberOfSamples);
}

void OperationOtherSilent::
        reset( unsigned firstSample, unsigned numberOfSamples )
{
    pOperation silentBefore( new OperationSetSilent( source_sub_operation_, 0, firstSample ));
    pOperation silentAfter( new OperationSetSilent( silentBefore, firstSample+numberOfSamples, Interval::IntervalType_MAX ));

    _source = silentAfter;
}

    // OperationMove  /////////////////////////////////////////////////////////////////

OperationMove::
        OperationMove( pOperation source, unsigned firstSample, unsigned numberOfSamples, unsigned newFirstSample )
:   OperationSubOperations( source )
{
    reset(firstSample, numberOfSamples, newFirstSample);
}

void OperationMove::
        reset( unsigned firstSample, unsigned numberOfSamples, unsigned newFirstSample )
{
    // Note: difference to OperationMoveMerge is that OperationMove has the silenceTarget step
    pOperation silenceTarget( new OperationSetSilent(source_sub_operation_, newFirstSample, numberOfSamples ));
    pOperation silence( new OperationSetSilent(silenceTarget, firstSample, numberOfSamples ));

    pOperation crop( new OperationCrop( source_sub_operation_, firstSample, numberOfSamples ));
    pOperation moveToNewPos( new OperationInsertSilence( crop, 0, newFirstSample));

    pOperation addition( new OperationSuperposition (moveToNewPos, silence ));

    _source = addition;
}


    // OperationMoveMerge  /////////////////////////////////////////////////////////////////

OperationMoveMerge::
        OperationMoveMerge( pOperation source, unsigned firstSample, unsigned numberOfSamples, unsigned newFirstSample )
:   OperationSubOperations( source )
{
    reset(firstSample, numberOfSamples, newFirstSample);
}

void OperationMoveMerge::
        reset( unsigned firstSample, unsigned numberOfSamples, unsigned newFirstSample )
{
    pOperation silence( new OperationSetSilent (source_sub_operation_, firstSample, numberOfSamples ));

    pOperation crop( new OperationCrop( source_sub_operation_, firstSample, numberOfSamples ));
    pOperation moveToNewPos( new OperationInsertSilence( crop, 0, newFirstSample));

    pOperation addition( new OperationSuperposition (moveToNewPos, silence ));

    _source = addition;
}


    // OperationShift  /////////////////////////////////////////////////////////////////

OperationShift::
        OperationShift( pOperation source, int sampleShift )
:   OperationSubOperations( source )
{
    reset(sampleShift);
}

void OperationShift::
        reset( int sampleShift )
{
    if ( 0 < sampleShift )
    {
        pOperation addSilence( new OperationInsertSilence( source_sub_operation_, 0u, (unsigned)sampleShift ));
        _source = addSilence;
    } else if (0 > sampleShift ){
        pOperation removeStart( new OperationRemoveSection( source_sub_operation_, 0u, (unsigned)-sampleShift ));
        _source = removeStart;
	} else {
        _source = source_sub_operation_;
	}
}


    // OperationShift  /////////////////////////////////////////////////////////////////

OperationMoveSelection::
        OperationMoveSelection( pOperation source, pOperation selectionFilter, int sampleShift, float freqDelta )
:	OperationSubOperations( source, "OperationMoveSelection" )
{
	reset(selectionFilter, sampleShift, freqDelta );
}


void OperationMoveSelection::
    reset( pOperation selectionFilter, int sampleShift, float freqDelta )
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

    _source = mergeSelection;
}

    } // namespace Support
} // namespace Tools
