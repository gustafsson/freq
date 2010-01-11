#include "wavelettransform.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include "CudaException.h"
#include <math.h>
#include "CudaProperties.h"
#include "throwInvalidArgument.h"
#include <iostream>
#include "Statistics.h"
#include "StatisticsRandom.h"
#include <string.h>

#include "wavelet.cu.h"

Transform::Transform( pWaveform waveform, unsigned scales_per_octave, unsigned samples_per_chunk, float wavelet_std_t )
:   _originalWaveform(waveform),
    _scales_per_octave( scales_per_octave ),
    _samples_per_chunk( samples_per_chunk ),
    _wavelet_std_t( wavelet_std_t )
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

    CudaProperties::printInfo(CudaProperties::getCudaDeviceProp());
}

Transform::~Transform()
{

    gc();
}

ChunkNumber Transform::getChunkNumber( float including_time_t ) {
    return t*_originalWaveform->_sample_rate / samples_per_chunk;
}


/*
getChunk initializes asynchronous computation if 0!=stream (inherited
behaviour from cuda). Also from NVIDIA CUDA Programming Guide section
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
pTransform_chunk Transform::getChunk( ChunkNumber n, cudaStream_t stream ) {
    if (_oldChunks.find(n) != _oldChunks.end()) {
        if (!_oldChunks[n]->modified() && _oldChunks[n].unique())
            /* only use old chunks if they're not modified and unused */
            return _oldChunks[n];
    }

    /* The waveform resides in CPU memory. In total we might have hours of data
       which would instantly waste the entire GPU memory. In those cases it is
       likely that the OS chooses to page parts of the waveform. Therefore this
       takes in general an undefined amount of time before it returns.

       However, we optimize for the assumption that it does reside in readily
       available cpu memory.
    */
    Waveform_chunk wave = _originalWaveform->getChunk(
            samples_per_chunk*n - wavelet_std_t*_originalWaveform->sampleRate,
            samples_per_chunk + 2*wavelet_std_t*_originalWaveform->sampleRate);

    /*
    static cudaStream_t previousStream = -1;
    if (0 < previousStream) {
        cudaStreamSynchronize( previousStream );
    }
    previousStream = stream;
    */

    /* performing the transform might require allocating memory, */
    pTransform_chunk wt = computeTransform( wave, stream );

    /* allocateChunk is more prone to reuse old caches than to actually
       allocate any memory */
    pTransform_chunk clamped = allocateChunk( n );
    clampTransform( clamped, wt, stream );

    _oldChunks[n] = clamped;
}


pWaveform_chunk Transform::computeInverse( pTransform_chunk chunk, cudaStream_t stream ) {
    cudaExtent sz = makeCudaExtent( chunk->nSamples(), 1, 1);

    pWaveform_chunk r( new Waveform_chunk());
    r->sampleOffset = chunk->sampleOffset;
    r->sampleRate = chunk->sampleRate;
    r->waveform_data.reset( new GpuCpuData<float>(0, sz, GpuCpuVoidData::CudaGlobal) );

    {
        TaskTimer tt(__FUNCTION__);

        // summarize them all
        ::wtInverse( chunk->transform_data->getCudaGlobal().ptr(),
                     chunk->transform_data->getNumberOfElements(),
                     r->waveform_data->getCudaGlobal().ptr(),
                     r->waveform_data->getNumberOfElements(),
                     stream );
    }

    {
        TaskTimer tt("inverse corollary");

        size_t n = r->waveform_data->getNumberOfElements1D();
        float* data = r->waveform_data->getCpuMemory();
        pWaveform_chunk originalChunk = _originalWaveform->getChunk(chunk->sampleOffset, chunk->nSamples(), _channel);
        float* orgdata = originalChunk->waveform_data->getCpuMemory();

        double sum = 0, orgsum=0;
        for (size_t i=0; i<n; i++) {
            sum += fabsf(data[i]);
        }
        for (size_t i=0; i<n; i++) {
            orgsum += fabsf(orgdata[i]);
        }
        float scale = orgsum/sum;
        for (size_t i=0; i<n; i++)
            data[i] *= scale;
        tt.info("scales %g, %g, %g", sum, orgsum, scale);

        r->writeFile("outtest.wav");
    }
    return r;
}


void Transform::gc() {
    _oldChunks.clear();
    _intermediate_wt.clear();
}


void Transform::channel( unsigned value ) {
    if (value==_channel) return;
    gc(); // invalidate allocated memory chunks
    _channel=value;
}


void Transform::scales_per_octave( unsigned value ) {
    if (value==_scales_per_octave) return;
    gc();
    _scales_per_octave=value;
}


void Transform::samples_per_chunk( unsigned value ) {
    if (value==_samples_per_chunk) return;
    gc();
    _samples_per_chunk=value;
}


void Transform::wavelet_std_t( float value ) {
    if (value==_wavelet_std_t) return;
    gc();
    _wavelet_std_t=value;
}


void Transform::originalWaveform( pWaveform value ) {
    if (value==_originalWaveform) return;
    gc();
    _originalWaveform=value;
}


pTransform_chunk Transform::allocateChunk( ChunkNumber n )
{
    if (_oldChunks.find(n) != _oldChunks.end()) {
        if (!_oldChunks[n].unique())
            _oldChunks.remove(_oldChunks.find(n));
        else
            return _oldChunks[n];
    }

    // look for caches no longer in use
    pTransform_chunk chunk = releaseChunkFurthestAwayFrom( n );

    if (!chunk) {
        // or allocate a new chunk
        chunk = pTransform_chunk ( new Transform_chunk());
        chunk->transform_data.reset(new GpuCpuData<float>(
                makeCudaExtent(_samples_per_chunk, _scales_per_octave, 1), 0, GpuCpuDataVoid::CudaGlobal ));
        chunk->minHz = 20;
        chunk->maxHz = chunk->sampleRate/2;
        chunk->sampleRate =  _originalWaveform->sampleRate;
    }
    chunk->sampleOffset = _samples_per_chunk*n;

    _oldChunks[chunk_n] = chunk;
    return chunk;
}


pTransform_chunk Transform::releaseChunkFurthestAwayFrom( ChunkNumber n )
{
    ChunkMap::iterator proposal;

    while (!_oldChunks.empty()) {
        proposal = abs(n-_oldChunks.first().first)
                 < abs(n-_oldChunks.last().first)
                 ? _oldChunks.first()
                 : --_oldChunks.end();
        _oldChunks.remove( proposal );

        if ( proposal.second.unique())
            return proposal.second;
    }
    return pTransform_chunk();
}


static void cufftSafeCall( cufftResult_t cufftResult) {
    if (cufftResult != CUFFT_SUCCESS) {
        ThrowInvalidArgument( cufftResult );
    }
}


pTransform_chunk Transform::computeTransform( pWaveform_chunk waveform_chunk, cudaStream_t stream )
{
    // Compute required size of transform
    {
        // Compute number of scales to use
        float octaves = log2(_intermediate_wt->maxHz)-log2(_intermediate_wt->minHz);
        unsigned nFrequencies = _scales_per_octave*octaves;
        unsigned n = waveform_chunk->sampleOffset/_samples_per_chunk;

        // The waveform must be complex interleaved
        if (Waveform_chunk::Interleaved_Complex != waveform_chunk->interleaved())
            waveform_chunk = waveform_chunk->getInterleaved();

        cudaExtent requiredFtSz( waveform_chunk->data->getNumberOfElements().width, nFrequencies, 1 );
        // The in-signal is be padded to a power of 2 for faster calculations (or rather, "power of a small prime")
        // TODO: measure time differences with other sizes
        // TODO: test these three different sizes
        //requiredFtSz.width = (1 << ((unsigned)ceil(log2(requiredFtSz.width))));
        //requiredFtSz.width = (1 << ((unsigned)ceil(log2(requiredFtSz.width+1))));
        requiredFtSz.width = (1 << ((unsigned)ceil(log2(requiredFtSz.width)+1)));
        cudaExtent requiredWtSz( requiredFtSz, nFrequencies, 1 );

        if (_intermediate_wt && _intermediate_wt->getNumberOfElements()!=requiredWtSz)
            _intermediate_wt.reset();

        if (_intermediate_ft && _intermediate_ft->getNumberOfElements()!=requiredFtSz)
            _intermediate_ft.reset();

        if (!_intermediate_wt) {
            // allocate a new chunk
            pTransform_chunk chunk = pTransform_chunk ( new Transform_chunk());

            while (!_oldChunks.empty()) try {
                chunk->transform_data.reset(new GpuCpuData<float>( 0, requiredWtSz, GpuCpuDataVoid::CudaGlobal ));
                break;
            } catch ( const CudaException& x ) {
                if (x.type() == MemoryException)
                    releaseChunkFurthestAwayFrom( n );
                else
                    throw;
            }

            chunk->sampleRate =  _originalWaveform->sampleRate;
            chunk->sampleOffset = _samples_per_chunk*n - _wavelet_std_t*_originalWaveform->sampleRate;
            chunk->minHz = 20;
            chunk->maxHz = chunk->sampleRate/2;
            if (0>chunk->sampleOffset) chunk->sampleOffset=0;

            _intermediate_wt = chunk;
        }

        if (!_intermediate_ft) {
            while (!_oldChunks.empty()) try {
                _intermediate_ft.reset(new GpuCpuData<float>( 0, requiredFtSz, GpuCpuDataVoid::CudaGlobal ));
                break;
            } catch ( const CudaException& x ) {
                if (x.type() == MemoryException)
                    releaseChunkFurthestAwayFrom( n );
                else
                    throw;
            }
        }
    }  // Compute required size of transform


    TaskTimer tt(__FUNCTION__);
    {
        TaskTimer tt2("start");

        cudaMemset( _intermediate_wt->getCudaGlobal().ptr(), 0, _intermediate_wt->getSizeInBytes1D() );
        cudaMemset( _intermediate_ft->getCudaGlobal().ptr(), 0, _intermediate_ft->getSizeInBytes1D() );
        cudaMemcpy( _intermediate_ft->getCudaGlobal().ptr()+2,
                    waveform_chunk->data->getCudaGlobal().ptr(),
                    waveform_chunk->data->getsizeInBytes().width,
                    cudaMemcpyDeviceToDevice);

        // Transform signal
        cufftSafeCall(cufftPlan1d(&_fft_single, n, CUFFT_C2C, 1));
        cufftSafeCall(cufftSetStream(_fft_single, stream));
        cufftSafeCall(cufftExecC2C(_fft_single,
                                   (cufftComplex *)_intermediate_ft->getCudaGlobal().ptr(),
                                   (cufftComplex *)_intermediate_ft->getCudaGlobal().ptr(),
                                   CUFFT_FORWARD));

        //Destroy CUFFT context
        cufftDestroy(_fft_single);

        CudaException_ThreadSynchronize();
    }

    {
        TaskTimer tt("computing");
        ::wtCompute( _intermediate_ft->getCudaGlobal().ptr(),
                     _intermediate_wt->data->getCudaGlobal().ptr(),
                     _intermediate_wt->sampleRate,
                     _intermediate_wt->minHz,
                     _intermediate_wt->maxHz,
                     _intermediate_wt->data->getNumberOfElements() );

        CudaException_ThreadSynchronize();
    }

    {
        TaskTimer tt("inverse fft");

        // Transform signal back
        cufftSafeCall(cufftPlan1d(&_fft_many, n, CUFFT_C2C, nFrequencies));
        cufftSafeCall(cufftExecC2C(_fft_many,
                                   (cufftComplex *)_intermediate_wt->data->getCudaGlobal().ptr(),
                                   (cufftComplex *)_intermediate_wt->data->getCudaGlobal().ptr(),
                                   CUFFT_INVERSE));

        // Destroy CUFFT context
        cufftDestroy(_fft_many);
        CudaException_ThreadSynchronize();
    }

    CudaException_ThreadSynchronize();

    return _transform;
}


void Transform::clampTransform( pTransform_chunk out_chunk, pTransform_chunk in_transform, cudaStream_t stream )
{
    cudaExtent offset( _wavelet_std_t*_originalWaveform->sample_rate, 0, 0 );
    ::wtClamp( in_transform->transform_data.getGpuMemory(),
               in_transform->transform_data.getNumberOfElements(),
               out_chunk->transform_data.getGpuMemory(),
               out_chunk->transform_data.getNumberOfElements(),
               offset,
               stream );
}
