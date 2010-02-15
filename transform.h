#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "transform-chunk.h"
#include "waveform.h"
#include <boost/shared_ptr.hpp>
#include <map>

typedef unsigned int cufftHandle; /* from cufft.h */

typedef boost::shared_ptr<class Transform> pTransform;

class Transform
{
public:
    typedef unsigned ChunkIndex;

    Transform( pWaveform waveform,
               unsigned channel,
               unsigned samples_per_chunk,
               unsigned scales_per_octave,
               float wavelet_std_t );
    ~Transform();

    ChunkIndex             getChunkIndex( unsigned including_sample );
    pTransform_chunk       getChunk( ChunkIndex n, cudaStream_t stream=0 );
    /*static*/ pWaveform_chunk computeInverse( pTransform_chunk chunk, cudaStream_t stream=0 );

    /* discard cached data, releases all GPU memory */
    void     gc();

    /* properties, altering ANY of these is equivalent to creating a new
       instance of class Transform */
    unsigned  channel() const { return _channel; }
    void      channel( unsigned );
    unsigned  scales_per_octave() const { return _scales_per_octave; }
    void      scales_per_octave( unsigned );
    unsigned  samples_per_chunk() const { return _samples_per_chunk; }
    void      samples_per_chunk( unsigned );
    float     wavelet_std_t() const { return _wavelet_std_t; }
    void      wavelet_std_t( float );
    pWaveform original_waveform() { return _original_waveform; }
    void      original_waveform( pWaveform );
    float     number_of_octaves() const;
    float     min_hz() const { return _min_hz; }
    void      min_hz(float f);
    float     max_hz() const { return _max_hz; }
    void      max_hz(float f);

private:
    pTransform_chunk allocateChunk( ChunkIndex n );
    pTransform_chunk releaseChunkFurthestAwayFrom( ChunkIndex n );
    pTransform_chunk computeTransform( pWaveform_chunk chunk, cudaStream_t stream );
    void             clampTransform( pTransform_chunk out_chunk, pTransform_chunk in_transform, cudaStream_t stream );

    /* caches */
    typedef std::map<ChunkIndex, pTransform_chunk> ChunkMap;
    ChunkMap                                _oldChunks;
    pTransform_chunk                        _intermediate_wt;
    boost::shared_ptr<GpuCpuData<float2> >  _intermediate_ft;

    /* property values */
    pWaveform _original_waveform;
    unsigned  _channel;
    unsigned  _scales_per_octave;
    unsigned  _samples_per_chunk;
    float     _wavelet_std_t;
    float     _min_hz;
    float     _max_hz;
};

#endif // TRANSFORM_H
