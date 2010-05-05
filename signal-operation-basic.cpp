#include "signal-operation-basic.h"

namespace Signal {

    // OperationRemoveSection ///////////////////////////////////////////////////////////

OperationRemoveSection::
        OperationRemoveSection( pSource source, unsigned firstSample, unsigned numberOfRemovedSamples )
:   Operation( source ),
    _firstSample( firstSample ),
    _numberOfRemovedSamples( numberOfRemovedSamples )
{}

pBuffer OperationRemoveSection::
        read( unsigned firstSample, unsigned numberOfSamples )
{
    if (firstSample + numberOfSamples <= _firstSample )
        return _source->read( firstSample, numberOfSamples );

    if (firstSample < _firstSample)
        return _source->read( firstSample, _firstSample - firstSample );

    return _source->read( firstSample + _numberOfRemovedSamples, numberOfSamples );
}

    // OperationInsertSilence ///////////////////////////////////////////////////////////

OperationInsertSilence::
        OperationInsertSilence( pSource source, unsigned firstSample, unsigned numberOfSilentSamples )
:   Operation( source ),
    _firstSample( firstSample ),
    _numberOfSilentSamples( numberOfSilentSamples )
{}


pBuffer OperationInsertSilence::
        read( unsigned firstSample, unsigned numberOfSamples )
{
    if (firstSample + numberOfSamples <= _firstSample )
        return _source->read( firstSample, numberOfSamples );

    if (firstSample < _firstSample)
        return _source->read( firstSample, _firstSample - firstSample );

    if (firstSample > _firstSample +  _numberOfSilentSamples)
        return _source->read( firstSample - _numberOfSilentSamples, numberOfSamples );

    // Create silence
    unsigned length = _numberOfSilentSamples - (firstSample - _firstSample);
    if ( length > numberOfSamples )
        length = numberOfSamples;

    pBuffer r(new Buffer );
    r->sample_offset = firstSample;
    r->sample_rate = _source->sample_rate();
    r->waveform_data.reset( new GpuCpuData<float>( 0, make_cudaExtent(length,1,1) ));
    memset(r->waveform_data->getCpuMemory(), 0, r->waveform_data->getSizeInBytes1D());
    return r;
}

// OperationSuperposition ///////////////////////////////////////////////////////////

OperationSuperposition::
        OperationSuperposition( pSource source, pSource source2 )
:   _source( source ),
    _source2( source2 )
{
    if (_source->sample_rate() != _source2->sample_rate())
        throw std::invalid_argument("_source->sample_rate() != _source2->sample_rate()");
}

pBuffer OperationSuperposition::
        read( unsigned firstSample, unsigned numberOfSamples )
{
    pBuffer a = _source->read(firstSample, numberOfSamples );
    pBuffer b = _source->read(firstSample, numberOfSamples );
    if (a->interleaved()!=Buffer::Only_Real) a = a->getInterleaved(Buffer::Only_Real);
    if (b->interleaved()!=Buffer::Only_Real) b = b->getInterleaved(Buffer::Only_Real);

    pBuffer r(new Buffer);
    r->sample_rate = sample_rate();
    r->sample_offset = std::max( a->sample_offset, b->sample_offset );
    unsigned l = std::min( a->sample_offset + a->number_of_samples(), b->sample_offset + b->number_of_samples() );
    l -= r->sample_offset;

    r->waveform_data.reset( new GpuCpuData<float>(0, make_cudaExtent(l,1,1)));
    float *pa = a->waveform_data->getCpuMemory();
    float *pb = b->waveform_data->getCpuMemory();
    float *pr = r->waveform_data->getCpuMemory();
    pa += r->sample_offset-a->sample_offset;
    pb += r->sample_offset-b->sample_offset;
    for (unsigned i=0; i<r->number_of_samples(); i++)
        pr[i] = pa[i] + pb[i];

    return r;
}

} // namespace Signal
