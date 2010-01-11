#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "transform-chunk.h"
#include "waveform.h"
#include <boost/shared_ptr.hpp>
#include <map>

typedef unsigned int cufftHandle; /* from cufft.h */

class Transform
{
public:
    typedef unsigned ChunkNumber;

    Transform( pWaveform waveform,
               unsigned channel=0,
               unsigned scales_per_octave = 50,
               unsigned samples_per_chunk = 1<<14,
               float wavelet_std_t = .1 );

    ChunkNumber            getChunkNumber( float including_time_t );
    pTransform_chunk       getChunk( ChunkNumber n, cudaStream_t stream=0 );
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
    pWaveform originalWaveform() { return _originalWaveform; }
    void      originalWaveform( pWaveform );

private:
    pTransform_chunk allocateChunk( ChunkNumber n ) const;
    pTransform_chunk releaseChunk( ChunkNumber furthest_away_from_n );
    pTransform_chunk computeTransform( pWaveform_chunk chunk, cudaStream_t stream );
    void             clampTransform( pTransform_chunk out_chunk, pTransform_chunk in_transform, cudaStream_t stream );

    /* caches */
    typedef std::map<ChunkNumber. pTransform_chunk> ChunkMap;
    ChunkMap                                _oldChunks;
    cufftHandle                             _fft_single;
    cufftHandle                             _fft_many;
    pTransform_chunk                        _intermediate_wt;
    boost::shared_ptr<GpuCpuData<float> >   _intermediate_ft;

    /* property values */
    unsigned  _channel;
    float     _wavelet_std_t;
    unsigned  _scales_per_octave;
    unsigned  _samples_per_chunk;
    pWaveform _originalWaveform;
};

#endif // TRANSFORM_H
