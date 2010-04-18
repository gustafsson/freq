#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "signal-source.h"
#include "signal-sink.h"
#include "transform-inverse.h"
#include <boost/shared_ptr.hpp>
#include <map>
#include "filter.h"

typedef unsigned int cufftHandle; /* from cufft.h */

class Transform
{
public:
    typedef unsigned ChunkIndex;

    Transform( Signal::pSource source,
               unsigned channel,
               unsigned samples_per_chunk,
               unsigned scales_per_octave,
               float wavelet_std_t,
               int temp_to_remove_playback_device=-1);
    ~Transform();

    ChunkIndex             getChunkIndex( unsigned including_sample ) const;
    pTransform_chunk       getChunk( ChunkIndex n, cudaStream_t stream=0 );

    /* discard cached data, releases all GPU memory */
    void     gc();

    /* properties, altering ANY of these is equivalent to creating a new
       instance of class Transform */
    unsigned  samples_per_chunk() const { return _samples_per_chunk; }
    void      samples_per_chunk( unsigned );
    pTransform_chunk previous_chunk( unsigned &out_chunk_index );
    boost::shared_ptr<GpuCpuData<float2> >  stft( ChunkIndex n, cudaStream_t stream=0 );
    Signal::pBuffer stft( float start, float end, unsigned* chunkSize=0, cudaStream_t stream=0);
    void      saveCsv(ChunkIndex chunk_number=(ChunkIndex)-1);
    void      recompute_filter(pFilter f);

    boost::shared_ptr<Transform_inverse> inverse();
    FilterChain filter_chain;

    int       _temp_to_remove_playback;

private:
#ifdef _USE_CHUNK_CACHE
    pTransform_chunk allocateChunk( ChunkIndex n );
    pTransform_chunk releaseChunkFurthestAwayFrom( ChunkIndex n );
    void             clampTransform( pTransform_chunk out_chunk, pTransform_chunk in_transform, cudaStream_t stream );
#endif // #ifdef _USE_CHUNK_CACHE
    pTransform_chunk computeTransform( ChunkIndex n, cudaStream_t stream );
    void             merge_chunk(Signal::pBuffer r, pTransform_chunk transform);
    Signal::pBuffer  prepare_inverse(float start, float end);

    boost::shared_ptr<TaskTimer> filterTimer;

};

typedef boost::shared_ptr<class Transform> pTransform;

#endif // TRANSFORM_H
