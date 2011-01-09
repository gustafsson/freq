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
        OperationSubOperations(Signal::pOperation source, std::string name)
:   Operation(pOperation()),
    source_sub_operation_( new Operation(source)),
    name_(name)
{
    enabled(false);
    source_sub_operation_->enabled(false);
    _source = source_sub_operation_;
}


Intervals OperationSubOperations::
        affected_samples()
{
    return _source->affected_samples_until(source_sub_operation_);
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
        OperationCrop( pOperation source, const Signal::Interval& section )
:   OperationSubOperations( source, "Crop" )
{
    reset(section);
}

void OperationCrop::
        reset( const Signal::Interval& section )
{
    _source = source_sub_operation_;
    // remove before section
    if (section.first)
        _source = pOperation( new OperationRemoveSection( _source, Signal::Interval(0, section.first) ));

    // remove after section
    if (section.count()<Signal::Interval::IntervalType_MAX)
        _source = pOperation( new OperationRemoveSection( _source, Signal::Interval( section.count(), Signal::Interval::IntervalType_MAX)));
}


    // OperationSetSilent  /////////////////////////////////////////////////////////////////
OperationSetSilent::
        OperationSetSilent( pOperation source, const Signal::Interval& section )
:   OperationSubOperations( source, "Clear section" ),
    section_( section )
{
    reset(section);
}

void OperationSetSilent::
        reset( const Signal::Interval& section )
{
    section_ = section;

    pOperation remove( new OperationRemoveSection( source_sub_operation_, section ));
    pOperation addSilence( new OperationInsertSilence (remove, section ));

    _source = addSilence;
}


    // OperationOtherSilent  /////////////////////////////////////////////////////////////////
OperationOtherSilent::
        OperationOtherSilent( Signal::pOperation source, const Signal::Interval& section )
:   OperationSubOperations( source, "Clear all but section" ),
    section_(section)
{
    reset(section);
}

void OperationOtherSilent::
        reset( const Signal::Interval& section )
{
    section_ = section;
    pOperation p = source_sub_operation_;
    if (section.first)
        // silent before section
        p = pOperation( new OperationSetSilent( p, Signal::Interval(0, section.first) ));
    if (section.last < Interval::IntervalType_MAX)
        // silent after section
        p = pOperation( new OperationSetSilent( p, Signal::Interval(section.last, Interval::IntervalType_MAX) ));

    _source = p;
}

    // OperationMove  /////////////////////////////////////////////////////////////////

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

    pOperation silenceTarget( new OperationSetSilent(source_sub_operation_, newSection ));
    pOperation silence( new OperationSetSilent(silenceTarget, section ));

    pOperation crop( new OperationCrop( source_sub_operation_, section ));
    pOperation moveToNewPos( new OperationInsertSilence( crop, Interval(0, newFirstSample)));

    pOperation addition( new OperationSuperposition (moveToNewPos, silence ));

    _source = addition;
}


    // OperationMoveMerge  /////////////////////////////////////////////////////////////////

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

    _source = addition;
}


    // OperationShift  /////////////////////////////////////////////////////////////////

OperationShift::
        OperationShift( pOperation source, long sampleShift )
:   OperationSubOperations( source, "Shift" )
{
    reset(sampleShift);
}

void OperationShift::
        reset( long sampleShift )
{
    if ( 0 < sampleShift )
    {
        pOperation addSilence( new OperationInsertSilence( source_sub_operation_, Interval( 0, sampleShift) ));
        _source = addSilence;
    } else if (0 > sampleShift ){
        pOperation removeStart( new OperationRemoveSection( source_sub_operation_, Interval( 0, -sampleShift) ));
        _source = removeStart;
	} else {
        _source = source_sub_operation_;
	}
}


    // OperationShift  /////////////////////////////////////////////////////////////////

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

    _source = mergeSelection;
}

    } // namespace Support
} // namespace Tools
