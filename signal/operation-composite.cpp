#include "signal/operation-composite.h"
#include "signal/operation-basic.h"
#include <demangle.h>

namespace Signal {


    // OperationSubOperations  /////////////////////////////////////////////////////////////////

OperationSubOperations::
	OperationSubOperations(pSource source, std::string name)
:   Operation(source),
    _sourceSubOperation( new Operation(source)),
	_name(name)
{}

pBuffer OperationSubOperations ::
        read( unsigned firstSample, unsigned numberOfSamples )
{
    return _readSubOperation->read( firstSample, numberOfSamples );
}

void OperationSubOperations ::
        source(pSource v)
{
    Operation *o = dynamic_cast<Operation*>(_sourceSubOperation.get());
    if (0 == o) throw std::runtime_error("0==o");
    o->source(v);

    Operation::source(v);
}


    // OperationCrop  /////////////////////////////////////////////////////////////////

OperationCrop::
        OperationCrop( pSource source, unsigned firstSample, unsigned numberOfSamples )
:   OperationSubOperations( source )
{
    reset(firstSample, numberOfSamples);
}

void OperationCrop::
        reset( unsigned firstSample, unsigned numberOfSamples )
{
    pSource cropBefore( new OperationRemoveSection( _source, 0, firstSample ));
    pSource cropAfter( new OperationRemoveSection( cropBefore, numberOfSamples,
                                                   cropBefore->number_of_samples() - numberOfSamples ));

    _sourceSubOperation = cropBefore;
    _readSubOperation = cropAfter;
}


    // OperationSetSilent  /////////////////////////////////////////////////////////////////
OperationSetSilent::
        OperationSetSilent( pSource source, unsigned firstSample, unsigned numberOfSamples )
:   OperationSubOperations( source )
{
    reset(firstSample, numberOfSamples);
}

void OperationSetSilent::
        reset( unsigned firstSample, unsigned numberOfSamples )
{
    pSource remove( new OperationRemoveSection( _sourceSubOperation, firstSample, numberOfSamples ));
    pSource addSilence( new OperationInsertSilence (remove, firstSample, numberOfSamples ));

    _readSubOperation = addSilence;
}


    // OperationMove  /////////////////////////////////////////////////////////////////

OperationMove::
        OperationMove( pSource source, unsigned firstSample, unsigned numberOfSamples, unsigned newFirstSample )
:   OperationSubOperations( source )
{
    reset(firstSample, numberOfSamples, newFirstSample);
}

void OperationMove::
        reset( unsigned firstSample, unsigned numberOfSamples, unsigned newFirstSample )
{
    // Note: difference to OperationMoveMerge is that OperationMove has the silenceTarget step
    pSource silenceTarget( new OperationSetSilent(_sourceSubOperation, newFirstSample, numberOfSamples ));
    pSource silence( new OperationSetSilent(silenceTarget, firstSample, numberOfSamples ));
    pSource crop( new OperationCrop( _sourceSubOperation, firstSample, numberOfSamples ));
    pSource moveToNewPos( new OperationInsertSilence( crop, 0, newFirstSample));
    pSource addition( new OperationSuperposition (moveToNewPos, silence ));

    _readSubOperation = addition;
}


    // OperationMoveMerge  /////////////////////////////////////////////////////////////////

OperationMoveMerge::
        OperationMoveMerge( pSource source, unsigned firstSample, unsigned numberOfSamples, unsigned newFirstSample )
:   OperationSubOperations( source )
{
    reset(firstSample, numberOfSamples, newFirstSample);
}

void OperationMoveMerge::
        reset( unsigned firstSample, unsigned numberOfSamples, unsigned newFirstSample )
{
    pSource silence( new OperationSetSilent (_sourceSubOperation, firstSample, numberOfSamples ));
    pSource crop( new OperationCrop( _sourceSubOperation, firstSample, numberOfSamples ));
    pSource moveToNewPos( new OperationInsertSilence( crop, 0, newFirstSample));
    pSource addition( new OperationSuperposition (moveToNewPos, silence ));

    _readSubOperation = addition;
}


    // OperationShift  /////////////////////////////////////////////////////////////////

OperationShift::
        OperationShift( pSource source, int sampleShift )
:   OperationSubOperations( source )
{
    reset(sampleShift);
}

void OperationShift::
        reset( int sampleShift )
{
    if ( 0 < sampleShift )
    {
        pSource addSilence( new OperationInsertSilence( Operation::source(), 0u, (unsigned)sampleShift ));
        _sourceSubOperation = _readSubOperation = addSilence;
    } else if (0 > sampleShift ){
        pSource removeStart( new OperationRemoveSection( Operation::source(), 0u, (unsigned)-sampleShift ));
        _sourceSubOperation = _readSubOperation = removeStart;
	} else {
        _sourceSubOperation = _readSubOperation = _source;
	}
}


    // OperationShift  /////////////////////////////////////////////////////////////////

OperationMoveSelection::
		OperationMoveSelection( pSource source, Tfr::pFilter selectionFilter, int sampleShift, float freqDelta )
:	OperationSubOperations( source, "OperationMoveSelection" )
{
	reset(selectionFilter, sampleShift, freqDelta );
}

void OperationMoveSelection::
	reset( Tfr::pFilter selectionFilter, int sampleShift, float freqDelta )
{
	Tfr::pFilter extract, remove;
	if (Tfr::EllipsFilter* f = dynamic_cast<Tfr::EllipsFilter*>(selectionFilter.get())) {
		bool v = f->_save_inside;
		f->_save_inside = false;
		remove.reset( new Tfr::EllipsFilter(*f) );
		f->_save_inside = true;
                extract.reset( new Tfr::EllipsFilter(*f) );
		f->_save_inside = v;
	} else {
		throw std::invalid_argument(std::string(__FUNCTION__) + " only supports Tfr::EllipsFilter as selectionFilter");
	}

	Signal::pSource extractAndMoveSelection;
	{
		// Create filter for extracting selection
		Tfr::pFilter moveFilter(new Tfr::MoveFilter( freqDelta ));
		Tfr::FilterChain* pchain(new Tfr::FilterChain);
		Tfr::pFilter chain(pchain);
		pchain->push_back(extract);
		pchain->push_back(moveFilter);

		// Create CwtFilter for applying filters
                extractAndMoveSelection.reset(new Tfr::CwtFilter( Operation::source(), Tfr::Cwt::SingletonP(), chain ));

		// Create operation to move selection
		if (0!=sampleShift)
			extractAndMoveSelection.reset( new OperationShift( extractAndMoveSelection, sampleShift ));
	}

        Signal::pSource removeSelection( new Tfr::CwtFilter( Operation::source(), Tfr::Cwt::SingletonP(), remove ));

	Signal::pSource mergeSelection( new Signal::OperationSuperposition( removeSelection, extractAndMoveSelection ));

	_readSubOperation = mergeSelection;
}

} // namespace Signal

