#include "supersample.h"
#include "stft.h"

// std
#include <stdexcept>
#include <string.h> //memset

namespace Tfr {

Signal::pBuffer SuperSample::
        supersample( Signal::pBuffer b, float requested_sample_rate )
{
    if (b->sample_rate == requested_sample_rate)
        return b;

    float bsample_rate = b->sample_rate;
    float fmultiple = log2f( requested_sample_rate/bsample_rate );
    int multiple = (int)fmultiple;
    if (fmultiple < 0 )
        throw std::invalid_argument("requested_sample_rate must be bigger than "
                                    "b->sample_rate");
    if (fmultiple != multiple)
        throw std::invalid_argument("requested_sample_rate must be an exact "
                                    "multiple of b->sample_rate");

    Tfr::pChunk chunk = Tfr::Fft()( b );

    unsigned src_window_size = ((Tfr::StftChunk*)chunk.get())->window_size;
    Tfr::pChunk biggerchunk( new Tfr::StftChunk( src_window_size << multiple ));
    biggerchunk->freqAxis = chunk->freqAxis;
    biggerchunk->chunk_offset = chunk->chunk_offset << multiple;
    biggerchunk->first_valid_sample = chunk->first_valid_sample << multiple;
    biggerchunk->n_valid_samples = chunk->n_valid_samples << multiple;

    DataStorageSize src_sz = chunk->transform_data->getNumberOfElements();
    DataStorageSize dest_sz( ((src_sz.width - 1) << multiple) + 1, 1, 1);
    biggerchunk->transform_data.reset( new ChunkData(dest_sz) );

    biggerchunk->sample_rate = requested_sample_rate / src_window_size;
    biggerchunk->original_sample_rate = requested_sample_rate;

    ChunkElement* src = chunk->transform_data->getCpuMemory();
    ChunkElement* dest = biggerchunk->transform_data->getCpuMemory();


    float normalize = 1.f/src_window_size;
    (dest[0] = src[0])*=(0.5*normalize);
    for (unsigned i=1; i<src_sz.width; ++i)
    {
        (dest[i] = src[i])*=normalize;
    }


    memset( dest + src_sz.width, 0, (dest_sz.width - src_sz.width) * sizeof(ChunkElement) );


    Signal::pBuffer r = Tfr::Fft().inverse( biggerchunk );

    BOOST_ASSERT( r->sample_rate == requested_sample_rate );

    return r;
}

} // namespace Tfr
