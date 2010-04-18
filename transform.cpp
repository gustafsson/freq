#include "transform.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include "CudaException.h"
#include <math.h>
#include "CudaProperties.h"
#include "throwInvalidArgument.h"
#include <iostream>
#include <fstream>
#include "Statistics.h"
#include "StatisticsRandom.h"
#include <string.h>
#include "wavelet.cu.h"
#include <msc_stdc.h>
#include "signal-audiofile.h"

using namespace std;

static bool someWhatAccurateTiming = false;

static void cufftSafeCall( cufftResult_t cufftResult) {
    if (cufftResult != CUFFT_SUCCESS) {
        ThrowInvalidArgument( cufftResult );
    }
}


Transform::Transform( Signal::pSource waveform, unsigned channel, unsigned samples_per_chunk, unsigned scales_per_octave, float wavelet_std_t, int playback )
:   _temp_to_remove_playback(playback),
    _original_waveform( waveform ),
    _channel( channel ),
    _scales_per_octave( scales_per_octave ),
    _samples_per_chunk( samples_per_chunk ),
    _wavelet_std_samples( 0 ),
    _min_hz(20.f),
    _max_hz( waveform->sample_rate()/2.f ),
    _fft_many(-1),
    _fft_width(0)
{
    /*
    Each chunk will then have the size
        chunk_byte_size = sizeof(float)*2 * _scales_per_octave
                        * _samples_per_chunk * number_of_octaves
    Which for default values
        scales_per_octave = 50,
        samples_per_chunk = 1<<14
        number_of_octaves=log2(20000)-log2(20) ~ 10
    is
        chunk_byte_size = 62.5 MB
    Which in turn means that a graphics card with 512 MB of memory will have
    space for roughly 6 chunks (leaving about 137 MB for other purposes).
    */

    this->wavelet_std_t( wavelet_std_t );

    CudaProperties::printInfo(CudaProperties::getCudaDeviceProp());
}

Transform::~Transform()
{

    gc();
}


Transform::ChunkIndex Transform::getChunkIndex( unsigned including_sample ) const {
    return including_sample / _samples_per_chunk;
}


/*
getChunk initializes asynchronous computation if 0!=stream (inherited
behaviour from cuda). Also note, from NVIDIA CUDA Programming Guide section
3.2.6.1, v2.3 Stream:

"Two commands from different streams cannot run concurrently if either a page-
locked host memory allocation, a device memory allocation, a device memory set,
a device ↔ device memory copy, or any CUDA command to stream 0 is called in-
between them by the host thread."

That is, a call to getChunk shall never result in any of the operations
mentioned above, which basically leaves kernel execution and page-locked
host ↔ device memory copy. For this purpose, a page-locked chunk of memory
is allocated beforehand and reserved for the Waveform_chunk. Previous copies
with the page-locked chunk is synchronized on beforehand.
*/
pTransform_chunk Transform::getChunk( ChunkIndex n, cudaStream_t stream ) {
#ifndef _USE_CHUNK_CACHE
    if (_intermediate_wt &&
        n * _samples_per_chunk >=_intermediate_wt->first_valid_sample + _intermediate_wt->chunk_offset &&
        (n+1) * _samples_per_chunk <= _intermediate_wt->first_valid_sample + _intermediate_wt->chunk_offset + _intermediate_wt->n_valid_samples)
    {
        if (!_intermediate_wt->modified && _intermediate_wt.unique())
            /* only return the previous chunk if it is unused and not modified */
            return _intermediate_wt;
    }
#else // _USE_CHUNK_CACHE
    if (_oldChunks.find(n) != _oldChunks.end()) {
        if (!_oldChunks[n]->modified && _oldChunks[n].unique())
            /* only use old chunks if they're not modified and unused */
            return _oldChunks[n];
    }
#endif // _USE_CHUNK_CACHE

    TaskTimer tt(TaskTimer::LogVerbose, "computing transform");



    /*
    static cudaStream_t previousStream = -1;
    if (0 < previousStream) {
        cudaStreamSynchronize( previousStream );
    }
    previousStream = stream;
    */

    /* performing the transform might require allocating memory, */
    pTransform_chunk wt = computeTransform( n, stream );

    filter_chain(*wt);
#ifndef _USE_CHUNK_CACHE
    return wt;
#else
    /* allocateChunk is more prone to reuse old caches than to actually
       allocate any memory */
    pTransform_chunk clamped = allocateChunk( n );
    clampTransform( clamped, wt, stream );

    _oldChunks[n] = clamped;
    return clamped;
#endif
}

stft {
    /* The waveform resides in CPU memory, beacuse in total we might have hours
       of data which would instantly waste the entire GPU memory. In those
       cases it is likely though that the OS chooses to page parts of the
       waveform. Therefore this takes in general an undefined amount of time
       before it returns.

       However, we optimize for the assumption that it does reside in readily
       available cpu memory.
    */
    unsigned
        first_valid = _samples_per_chunk*n,
        offs,
        n_samples;

    if (first_valid > _wavelet_std_samples)
        offs = first_valid - _wavelet_std_samples;
    else
        offs = 0;

    first_valid -= offs;
    // necessary size: n_samples = first_valid + _samples_per_chunk + _wavelet_std_samples;
    // somewhat bigger for the first chunk is ok
    n_samples = _samples_per_chunk + _wavelet_std_samples*2;

    TaskTimer(TaskTimer::LogVerbose, "[%g - %g] %g s, total including redundant %g s",
                 (offs+first_valid)/(float)_original_waveform->sample_rate(),
                 (offs+first_valid+_samples_per_chunk)/(float)_original_waveform->sample_rate(),
                 _samples_per_chunk/(float)_original_waveform->sample_rate(),
                 n_samples/(float)_original_waveform->sample_rate()).suppressTiming();

    // Signal::Audiofile* a = dynamic_cast<Signal::Audiofile*>(_original_waveform.get());
    Signal::pBuffer waveform_chunk = _original_waveform->read(
            offs, n_samples)->getInterleaved( Signal::Buffer::Interleaved_Complex );
}

Signal::pBuffer Transform::stft(float startt, float endt, unsigned* chunkSize, cudaStream_t stream)
{
    BOOST_ASSERT(startt<=endt);

    const unsigned ftChunk = 1 << 11;
    unsigned first_chunk = startt*_original_waveform->sample_rate();
    first_chunk = first_chunk>ftChunk/2 ? first_chunk - ftChunk/2: 0;
    first_chunk/=ftChunk;//(
    unsigned after_last_chunk = 2 + endt*_original_waveform->sample_rate()/ftChunk;
    cudaExtent requiredFtSz = make_cudaExtent( (after_last_chunk-first_chunk)*ftChunk, 1, 1 );
    // The in-signal is be padded to a power of 2 for faster calculations (or rather, "power of a small prime")
    requiredFtSz.width = (1 << ((unsigned)ceil(log2((float)requiredFtSz.width))));
    requiredFtSz.width = max(requiredFtSz.width, (size_t)ftChunk);


    Signal::pBuffer complete_stft = _original_waveform->read(
            first_chunk*ftChunk, requiredFtSz.width)->getInterleaved( Signal::Buffer::Interleaved_Complex );

/*    Signal::Audiofile* a = dynamic_cast<Signal::Audiofile*>(_original_waveform.get());
    Signal::pBuffer complete_stft = a->getChunk(
            first_chunk*ftChunk, requiredFtSz.width, 0,
            Signal::Buffer::Interleaved_Complex);
*/
    cufftComplex* d = (cufftComplex*)complete_stft->waveform_data->getCudaGlobal().ptr();

    // Transform signal
    cufftHandle fft_all;
    cufftSafeCall(cufftPlan1d(&fft_all, ftChunk, CUFFT_C2C, requiredFtSz.width/ftChunk));

    cufftSafeCall(cufftSetStream(fft_all, stream));
    cufftSafeCall(cufftExecC2C(fft_all, d, d, CUFFT_FORWARD));
    cufftDestroy(fft_all);

    if (chunkSize)
        *chunkSize = ftChunk;

    return complete_stft;
}

void Transform::recompute_filter(pFilter f) {
    inverse()->recompute_filter(f);

    if (f.get()) {
        float a,b;
        f->range(a,b);
        if (_intermediate_wt->startTime()>a && _intermediate_wt->endTime()<b)
            _intermediate_wt->chunk_offset = (unsigned)-1;
    } else {
        Signal::Audiofile* af = dynamic_cast<Signal::Audiofile*>(inverse()->get_inverse_waveform().get());
        filter_chain.invalidateWaveform(*this, *af->getChunkBehind());
        _intermediate_wt->chunk_offset = (unsigned)-1;
    }
}

string csv_number()
{
    string basename = "sonicawe-";
    for (unsigned c = 1; c<10; c++)
    {
        stringstream filename;
        filename << basename << c << ".csv";
        fstream csv(filename.str().c_str());
        if (!csv.is_open())
            return filename.str();
    }
    return basename+"0.csv";
}

void Transform::saveCsv(ChunkIndex chunk_number)
{
    string filename = csv_number();
    TaskTimer tt("Saving CSV-file %s", filename.c_str());
    ofstream csv(filename.c_str());

    ChunkIndex n = chunk_number;
    if (n == (ChunkIndex)-1)
        n = getChunkIndex( inverse()->built_in_filter._t1 * _original_waveform->sample_rate() );

    pTransform_chunk chunk = getChunk( n );
    float2* p = chunk->transform_data->getCpuMemory();
    cudaExtent s = chunk->transform_data->getNumberOfElements();

    for (unsigned y = 0; y<s.height; y++) {
        stringstream ss;
        for (unsigned x = 0; x<s.width; x++) {
            float2& v = p[x + y*s.width];
            ss << v.x << " " << v.y << " ";
        }
        csv << ss.str() << endl;
    }
}

void Transform::merge_chunk(Signal::pBuffer r, pTransform_chunk transform)
{
    Signal::pBuffer chunk = inverse()->computeInverse( transform, 0 );

    // merge
    unsigned out_offs = (chunk->sample_offset > r->sample_offset)? chunk->sample_offset - r->sample_offset : 0;
    unsigned in_offs = (r->sample_offset > chunk->sample_offset)? r->sample_offset - chunk->sample_offset : 0;
    unsigned count = chunk->waveform_data->getNumberOfElements().width;
    if (count>in_offs)
        count -= in_offs;
    else
        return;

    cudaMemcpy( &r->waveform_data->getCudaGlobal().ptr()[ out_offs ],
                &chunk->waveform_data->getCudaGlobal().ptr()[ in_offs ],
                count*sizeof(float), cudaMemcpyDeviceToDevice );
}

void Transform::channel( unsigned value ) {
    if (value==_channel) return;
    gc(); // invalidate allocated memory chunks
    _channel=value;
}

void Transform::samples_per_chunk( unsigned value ) {
    if (value==_samples_per_chunk) return;
    gc();
    _samples_per_chunk=value;
}


void Transform::wavelet_std_t( float value ) {
    if (value*_original_waveform->sample_rate()==_wavelet_std_samples) return;
    gc();
    _wavelet_std_samples=value*_original_waveform->sample_rate();
    _wavelet_std_samples = (_wavelet_std_samples+31)/32*32;
}


void Transform::original_waveform( Signal::pSource value ) {
    if (value==_original_waveform) return;
    gc();
    _original_waveform=value;
}


pTransform_chunk Transform::previous_chunk( unsigned &out_chunk_index ) {
    if (_intermediate_wt && _intermediate_wt->chunk_offset != (unsigned)-1)
    {
        out_chunk_index = getChunkIndex(_intermediate_wt->chunk_offset + _intermediate_wt->first_valid_sample);
        if (out_chunk_index >= getChunkIndex(_intermediate_wt->chunk_offset + _intermediate_wt->first_valid_sample + _intermediate_wt->n_valid_samples))
            out_chunk_index = (unsigned)-1;
    }

    return _intermediate_wt;
}

boost::shared_ptr<Transform_inverse> Transform::inverse()
{
    static boost::shared_ptr<Transform_inverse> p ( new Transform_inverse( this->_original_waveform, 0, this ) );
    return p;
}

#ifdef _USE_CHUNK_CACHE
pTransform_chunk Transform::allocateChunk( ChunkIndex n )
{
    if (_oldChunks.find(n) != _oldChunks.end()) {
        if (!_oldChunks[n].unique())
            _oldChunks.erase(_oldChunks.find(n));
        else
            return _oldChunks[n];
    }

    // look for caches no longer in use
    pTransform_chunk chunk = releaseChunkFurthestAwayFrom( n );

    if (!chunk) {
        // or allocate a new chunk
        chunk = pTransform_chunk ( new Transform_chunk());
        chunk->transform_data.reset(new GpuCpuData<float2>(0,
                make_uint3(_samples_per_chunk, number_of_octaves()*_scales_per_octave, 1), GpuCpuVoidData::CudaGlobal ));
        chunk->min_hz = 20;
        chunk->max_hz = chunk->sample_rate/2;
        chunk->sample_rate = _original_waveform->sample_rate();
    }
    chunk->sample_offset = _samples_per_chunk*n;

    _oldChunks[n] = chunk;
    return chunk;
}


pTransform_chunk Transform::releaseChunkFurthestAwayFrom( ChunkIndex n )
{
    ChunkMap::iterator proposal;
    while (!_oldChunks.empty()) {
		ChunkMap::iterator last = _oldChunks.end();
		last--;
        proposal = abs((int)n-(int)_oldChunks.begin()->first)
                 < abs((int)n-(int)last->first)
                 ? _oldChunks.begin()
                 : last;

        if ( proposal->second.unique()) {
            pTransform_chunk r = proposal->second;
            _oldChunks.erase( proposal );
            return r;

        } else {
            _oldChunks.erase( proposal );
        }
    }
    return pTransform_chunk();
}
#endif

#ifdef _USE_CHUNK_CACHE
void Transform::clampTransform( pTransform_chunk out_chunk, pTransform_chunk in_transform, cudaStream_t stream )
{
    BOOST_ASSERT( 0 && "deprecated" );
    // update this function to use proper offsets

    out_chunk->sample_rate = in_transform->sample_rate;
    out_chunk->min_hz = in_transform->min_hz;
    out_chunk->max_hz = in_transform->max_hz;
    size_t in_offset = out_chunk->sample_offset - in_transform->sample_offset;
    BOOST_ASSERT( out_chunk->sample_offset >= in_transform->sample_offset );
    BOOST_ASSERT( in_transform->transform_data->getNumberOfElements().width >= out_chunk->transform_data->getNumberOfElements().width );
    BOOST_ASSERT( in_transform->transform_data->getNumberOfElements().height >= out_chunk->transform_data->getNumberOfElements().height );
    BOOST_ASSERT( in_transform->transform_data->getNumberOfElements().depth >= out_chunk->transform_data->getNumberOfElements().depth );
    cudaMemset( out_chunk->transform_data->getCudaGlobal().ptr(), 0, out_chunk->transform_data->getSizeInBytes1D());
    size_t last_sample = _original_waveform->number_of_samples()>out_chunk->sample_offset?
                         _original_waveform->number_of_samples()-out_chunk->sample_offset:0;

    ::wtClamp( in_transform->transform_data->getCudaGlobal().ptr(),
               in_transform->transform_data->getNumberOfElements(),
               in_offset,
               last_sample,
               out_chunk->transform_data->getCudaGlobal().ptr(),
               out_chunk->transform_data->getNumberOfElements(),
               stream );
}
#endif
