#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "transform-chunk.h"
#include "waveform.h"
#include <boost/shared_ptr.hpp>
#include <map>
#include "filter.h"

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

    ChunkIndex             getChunkIndex( unsigned including_sample ) const;
    pTransform_chunk       getChunk( ChunkIndex n, cudaStream_t stream=0 );
    /*static*/ pWaveform_chunk computeInverse( pTransform_chunk chunk, cudaStream_t stream=0 );
    pWaveform_chunk        computeInverse( float start=0, float end=-1);

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
    float     wavelet_std_t() const { return _wavelet_std_samples/(float)_original_waveform->sample_rate(); }
    void      wavelet_std_t( float );
    pWaveform original_waveform() const { return _original_waveform; }
    void      original_waveform( pWaveform );
    void      play_inverse();
    pWaveform get_inverse_waveform();
    float     number_of_octaves() const;
    unsigned  nScales() { return (unsigned)(number_of_octaves() * scales_per_octave()); }
    float     min_hz() const { return _min_hz; }
    void      min_hz(float f);
    float     max_hz() const { return _max_hz; }
    void      max_hz(float f);
    void      setInverseArea(float t1, float f1, float t2, float f2);
    pTransform_chunk previous_chunk( unsigned &out_chunk_index );

    boost::shared_ptr<GpuCpuData<float2> >  stft( ChunkIndex n, cudaStream_t stream=0 );
    pWaveform_chunk                         stft( float start, float end, unsigned* chunkSize=0, cudaStream_t stream=0);

    FilterChain filter_chain;
    EllipsFilter built_in_filter;
    void      recompute_filter(pFilter);
private:
#ifdef _USE_CHUNK_CACHE
    pTransform_chunk allocateChunk( ChunkIndex n );
    pTransform_chunk releaseChunkFurthestAwayFrom( ChunkIndex n );
    void             clampTransform( pTransform_chunk out_chunk, pTransform_chunk in_transform, cudaStream_t stream );
#endif // #ifdef _USE_CHUNK_CACHE
    pTransform_chunk computeTransform( ChunkIndex n, cudaStream_t stream );
    void             merge_chunk(pWaveform_chunk r, pTransform_chunk transform);
    pWaveform_chunk  prepare_inverse(float start, float end);

    /* caches */
#ifdef _USE_CHUNK_CACHE
    typedef std::map<ChunkIndex, pTransform_chunk> ChunkMap;
    ChunkMap                                _oldChunks;
#endif // #ifdef _USE_CHUNK_CACHE
    pTransform_chunk                        _intermediate_wt;
    boost::shared_ptr<GpuCpuData<float2> >  _intermediate_ft;

    /* property values */
    pWaveform _original_waveform;
    pWaveform _inverse_waveform;
    unsigned  _channel;
    unsigned  _scales_per_octave;
    unsigned  _samples_per_chunk;
    unsigned  _wavelet_std_samples;
    float     _min_hz;
    float     _max_hz;
    cufftHandle _fft_many;
    cufftHandle _fft_single;
    unsigned _fft_width;

    boost::shared_ptr<TaskTimer> filterTimer;

};

#endif // TRANSFORM_H
