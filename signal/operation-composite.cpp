#include "signal/operation-composite.h"
#include "signal/operation-basic.h"
#include "filters/filters.h"
#include <demangle.h>

namespace Signal {


    // OperationSubOperations  /////////////////////////////////////////////////////////////////

OperationSubOperations::
	OperationSubOperations(pOperation source, std::string name)
:   Operation(source),
    _sourceSubOperation( new Operation(source)),
	_name(name)
{}

pBuffer OperationSubOperations ::
        read( const Interval &I )
{
    return _readSubOperation->read( I );
}

void OperationSubOperations ::
        source(pOperation v)
{
    Operation *o = dynamic_cast<Operation*>(_sourceSubOperation.get());
    if (0 == o) throw std::runtime_error("0==o");
    o->source(v);

    Operation::source(v);
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
    pOperation cropBefore( new OperationRemoveSection( _source, 0, firstSample ));
    pOperation cropAfter( new OperationRemoveSection( cropBefore, numberOfSamples,
                                                   cropBefore->number_of_samples() - numberOfSamples ));

    _sourceSubOperation = cropBefore;
    _readSubOperation = cropAfter;
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
    pOperation remove( new OperationRemoveSection( _sourceSubOperation, firstSample, numberOfSamples ));
    pOperation addSilence( new OperationInsertSilence (remove, firstSample, numberOfSamples ));

    _readSubOperation = addSilence;
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
    pOperation silenceTarget( new OperationSetSilent(_sourceSubOperation, newFirstSample, numberOfSamples ));
    pOperation silence( new OperationSetSilent(silenceTarget, firstSample, numberOfSamples ));
    pOperation crop( new OperationCrop( _sourceSubOperation, firstSample, numberOfSamples ));
    pOperation moveToNewPos( new OperationInsertSilence( crop, 0, newFirstSample));
    pOperation addition( new OperationSuperposition (moveToNewPos, silence ));

    _readSubOperation = addition;
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
    pOperation silence( new OperationSetSilent (_sourceSubOperation, firstSample, numberOfSamples ));
    pOperation crop( new OperationCrop( _sourceSubOperation, firstSample, numberOfSamples ));
    pOperation moveToNewPos( new OperationInsertSilence( crop, 0, newFirstSample));
    pOperation addition( new OperationSuperposition (moveToNewPos, silence ));

    _readSubOperation = addition;
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
        pOperation addSilence( new OperationInsertSilence( Operation::source(), 0u, (unsigned)sampleShift ));
        _sourceSubOperation = _readSubOperation = addSilence;
    } else if (0 > sampleShift ){
        pOperation removeStart( new OperationRemoveSection( Operation::source(), 0u, (unsigned)-sampleShift ));
        _sourceSubOperation = _readSubOperation = removeStart;
	} else {
        _sourceSubOperation = _readSubOperation = _source;
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
    if (Filters::EllipsFilter* f = dynamic_cast<Filters::EllipsFilter*>(selectionFilter.get())) {

        // Create filter for extracting selection
        extract.reset( new Filters::EllipsFilter(*f) );
        dynamic_cast<Filters::EllipsFilter*>(extract.get())->_save_inside = true;
        extract->source( source() );

        // Create filter for removing selection
        remove.reset( new Filters::EllipsFilter(*f) );
        dynamic_cast<Filters::EllipsFilter*>(remove.get())->_save_inside = false;
        remove->source( source() );

	} else {
		throw std::invalid_argument(std::string(__FUNCTION__) + " only supports Tfr::EllipsFilter as selectionFilter");
	}

    Signal::pOperation extractAndMove = extract;
    {
        // Create operation for moving extracted selection in time
        if (0!=sampleShift)
        extractAndMove.reset( new OperationShift( extractAndMove, sampleShift ));

        // Create operation for moving extracted selection in frequency
        if (0!=freqDelta)
        {
            Signal::pOperation t( new Filters::MoveFilter( freqDelta ));
            t->source( extractAndMove );
            extractAndMove = t;
        }

	}

    Signal::pOperation mergeSelection( new Signal::OperationSuperposition( remove, extractAndMove ));

	_readSubOperation = mergeSelection;
}

} // namespace Signal

