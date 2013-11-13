#include "cwtchunk.h"
#include "waveletkernel.h"

namespace Tfr {

pChunk CwtChunkPart::
        cleanChunk() const
{
    const Chunk* c = this;

    pChunk clamped( new CwtChunk );
    clamped->transform_data.reset( new ChunkData( c->n_valid_samples, c->nScales(), c->nChannels() ));

    ::wtClamp( c->transform_data, c->first_valid_sample, clamped->transform_data );
    clamped->freqAxis = c->freqAxis;

    clamped->chunk_offset = c->chunk_offset + c->first_valid_sample;
    clamped->sample_rate = c->sample_rate;
    clamped->first_valid_sample = 0;
    clamped->n_valid_samples = c->n_valid_samples;

    return clamped;
}


} // namespace Tfr
