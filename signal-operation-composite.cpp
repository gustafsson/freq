#include "signal-operation-composite.h"
#include "signal-operation-basic.h"

namespace Signal {


    // OperationSubOperations  /////////////////////////////////////////////////////////////////

OperationSubOperations::
        OperationSubOperations(pSource source)
:   Operation(source),
    _sourceSubOperation( new Operation(source))
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

} // namespace Signal

