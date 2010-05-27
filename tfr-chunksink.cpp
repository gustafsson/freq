#include "tfr-chunksink.h"
#include "signal-filteroperation.h"
#include "tfr-wavelet.cu.h"

namespace Tfr {

pChunk ChunkSink::
        getChunk( Signal::pBuffer b, Signal::pSource s )
{
    // Get a chunk for this block
    Tfr::pChunk chunk;

    // If buffer comes directly from a Signal::FilterOperation
    Signal::FilterOperation* filterOp = dynamic_cast<Signal::FilterOperation*>(s.get());

    Signal::pSource s2;

    if (s)
    {
        s2.reset( filterOp = new Signal::FilterOperation( s, Tfr::pFilter()));
    }

    if (filterOp) {
        // use the Cwt chunk still stored in FilterOperation
        chunk = filterOp->pick_previous_chunk();

        if (0 == chunk) {
            // try again
            filterOp->read( b->sample_offset, b->number_of_samples() );
            chunk = filterOp->pick_previous_chunk();
        }
    }

    if (0 == chunk) {
        // otherwise compute the Cwt of this block
        chunk = Tfr::CwtSingleton::operate( b );

        // Don't know anything aboout the nearby data, so assume its all valid
        chunk->transform_data.reset();
        chunk->n_valid_samples = chunk->transform_data->getNumberOfElements().width;
        chunk->first_valid_sample = 0;
    }

    return chunk;
}

pChunk ChunkSink::
        cleanChunk( pChunk c )
{
    if ( 0 == c->first_valid_sample && c->nSamples() == c->n_valid_samples )
        return c;

    pChunk clamped( new Chunk );
    clamped->transform_data.reset( new GpuCpuData<float2>(0, make_cudaExtent( c->n_valid_samples, c->nScales(), c->nChannels() )));

    ::wtClamp( c->transform_data->getCudaGlobal(), c->first_valid_sample, clamped->transform_data->getCudaGlobal() );
    clamped->chunk_offset = c->chunk_offset + c->first_valid_sample;
    clamped->first_valid_sample = 0;
    clamped->max_hz = c->max_hz;
    clamped->min_hz = c->min_hz;
    clamped->n_valid_samples = c->n_valid_samples;

    return clamped;
}

} // namespace Tfr
