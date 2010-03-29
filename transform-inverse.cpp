#include "transform.h"
#include "signal-audiofile.h"
#include <stdio.h>
#include <string> // min/max
#include "CudaException.h"
#include "wavelet.cu.h"

using namespace std;

Transform_inverse::Transform_inverse(Signal::pSource waveform, Callback* callback, Transform* temp_to_remove)
:   _original_waveform(waveform),
    _callback(callback),
    _temp_to_remove(temp_to_remove)
{
    _t1 = _f1 = _t2 = _f2 = 0;
}

void Transform_inverse::compute_inverse( float startt, float endt, pFilter filter )
{
    float chunksize=.1; // seconds
    float waveletsize=.05; // extra

    filter.get();

    for (float t=startt; t<endt; t+=chunksize) {
        float t1 = t-waveletsize;
        if (t1<0) t1=0;
        float t2 = t+chunksize+waveletsize;
        if (t2>endt) t2=endt;

        /*
        pWaveform_chunk waveform = _waveform->getChunk(t1,t2);
        pTransform_chunk transform = morlet( waveform );
        transform->getTransform_chunk(t1, t2);

        filter(transform);

        Waveform_chunk inverse = morlet_inverse( transform );

        _callback->transform_inverse_callback( inverse );
        */
    }
}

void Transform_inverse::setInverseArea(float t1, float f1, float t2, float f2) {
    float mint, maxt;
    switch(2)
    {
    case 1: // square
        mint = min(min(_t1, _t2), min(t1, t2));
        maxt = max(max(_t1, _t2), max(t1, t2));
        _t1 = min(t1, t2);
        _f1 = min(f1, f2);
        _t2 = max(t1, t2);
        _f2 = max(f1, f2);
        break;
    case 2: // circle
        mint = min(_t1 - fabs(_t1-_t2), t1 - fabs(t1-t2));
        maxt = max(_t1 + fabs(_t1-_t2), t1 + fabs(t1-t2));
        _t1 = t1;
        _f1 = f1;
        _t2 = t2;
        _f2 = f2;
        break;
    }

    Signal::Audiofile* a = dynamic_cast<Signal::Audiofile*>(_inverse_waveform.get());
    if (_inverse_waveform && a->getChunkBehind())
    {
        for (unsigned n = _temp_to_remove->getChunkIndex( max(0.f,mint)*_original_waveform->sample_rate());
             n <= _temp_to_remove->getChunkIndex( max(0.f,maxt)*_original_waveform->sample_rate());
             n++)
        {
            a->getChunkBehind()->valid_transform_chunks.erase(n);
        }
    }
    filterTimer.reset(new TaskTimer("Computing inverse [%g,%g], %g s", mint, maxt, maxt-mint));
}

Signal::pSource Transform_inverse::get_inverse_waveform()
{
    if (0 == _inverse_waveform) {
        Signal::Audiofile* a = new Signal::Audiofile();
        _inverse_waveform.reset(a);
        a->setChunk(prepare_inverse(0, _original_waveform->length()));
    }

    unsigned n, last_index=this->_temp_to_remove->getChunkIndex( _original_waveform->number_of_samples() );

    unsigned previous_chunk_index = (unsigned)-1;
    pTransform_chunk transform = this->_temp_to_remove->previous_chunk(previous_chunk_index);

    Signal::Audiofile* a = dynamic_cast<Signal::Audiofile*>(_inverse_waveform.get());
    if (transform &&
        previous_chunk_index <= last_index &&
        0==a->getChunkBehind()->valid_transform_chunks.count(previous_chunk_index))
    {
        n = previous_chunk_index;
    } else {
        transform.reset();
        for (n=0; n<=last_index; n++) {
            if (0==a->getChunkBehind()->valid_transform_chunks.count(n)) {
                transform = _temp_to_remove->getChunk(n);
                break;
            }
        }
    }

    if (transform) {
        Signal::Audiofile* a = dynamic_cast<Signal::Audiofile*>(_inverse_waveform.get());
        merge_chunk(a->getChunkBehind(), transform);

        a->getChunkBehind()->valid_transform_chunks.insert( n );
        a->getChunkBehind()->modified = true;


        for (n=0; n<=last_index; n++) {
            if (0==a->getChunkBehind()->valid_transform_chunks.count(n))
                break;
        }
        if (n>last_index) {
            if(filterTimer)
                filterTimer->info("Computed entire inverse");
            filterTimer.reset();
            a->play();
        }
    }

    return _inverse_waveform;
}

Signal::pBuffer Transform_inverse::computeInverse( float start, float end)
{
    Signal::pBuffer r = prepare_inverse(start, end);

    for(Transform::ChunkIndex n = _temp_to_remove->getChunkIndex(start);
         n*_temp_to_remove->samples_per_chunk() < end*r->sample_rate;
         n++)
    {
        merge_chunk(r, _temp_to_remove->getChunk(n));
    }
    return r;
}

Signal::pBuffer Transform_inverse::prepare_inverse(float start, float end)
{
    unsigned n = _temp_to_remove->original_waveform()->number_of_samples();

    if(start<0) start=0;
    Signal::pBuffer r( new Signal::Buffer());
    r->sample_rate = _temp_to_remove->original_waveform()->sample_rate();
    r->sample_offset = min((float)n, r->sample_rate*start);
    n -= r->sample_offset;
    if (start<=end)
        n = min((float)n, r->sample_rate*(end-start));
    fprintf(stdout, "rate = %d, offs = %d, n = %d, orgn = %d\n", r->sample_rate, r->sample_offset, n, _temp_to_remove->original_waveform()->number_of_samples());
    fflush(stdout);

    r->waveform_data.reset( new GpuCpuData<float>(0, make_cudaExtent(n, 1, 1), GpuCpuVoidData::CudaGlobal) );
    cudaMemset(r->waveform_data->getCudaGlobal().ptr(), 0, r->waveform_data->getSizeInBytes1D());

    return r;
}

void Transform_inverse::merge_chunk(Signal::pBuffer r, pTransform_chunk transform)
{
    Signal::pBuffer chunk = computeInverse( transform, 0 );

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

Signal::pBuffer Transform_inverse::computeInverse( pTransform_chunk chunk, cudaStream_t stream ) {
    cudaExtent sz = make_cudaExtent( chunk->n_valid_samples, 1, 1);

    Signal::pBuffer r( new Signal::Buffer());
    r->sample_offset = chunk->chunk_offset + chunk->first_valid_sample;
    r->sample_rate = chunk->sample_rate;
    r->waveform_data.reset( new GpuCpuData<float>(0, sz, GpuCpuVoidData::CudaGlobal) );

    float4 area = make_float4(
            _t1 * _original_waveform->sample_rate() - r->sample_offset,
            _f1 * _temp_to_remove->nScales(),
            _t2 * _original_waveform->sample_rate() - r->sample_offset,
            _f2 * _temp_to_remove->nScales());
    {
        TaskTimer tt(__FUNCTION__);

        // summarize them all
        ::wtInverse( chunk->transform_data->getCudaGlobal().ptr() + chunk->first_valid_sample,
                     r->waveform_data->getCudaGlobal().ptr(),
                     chunk->transform_data->getNumberOfElements(),
                     area,
                     chunk->n_valid_samples,
                     stream );

        CudaException_ThreadSynchronize();
    }

/*    {
        TaskTimer tt("inverse corollary");

        size_t n = r->waveform_data->getNumberOfElements1D();
        float* data = r->waveform_data->getCpuMemory();
        pWaveform_chunk originalChunk = _original_waveform->getChunk(chunk->sample_offset, chunk->nSamples(), _channel);
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
    }*/
    return r;
}
