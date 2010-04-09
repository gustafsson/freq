#include "transform.h"
#include "signal-audiofile.h"
#include <stdio.h>
#include <string> // min/max
#include "CudaException.h"
#include "wavelet.cu.h"

using namespace std;

Transform_inverse::Transform_inverse(Signal::pSource waveform, Callback* callback, Transform* temp_to_remove)
:   built_in_filter(0,0,0,0,false),
    _original_waveform(waveform),
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
    // clear inverse
    _inverse_waveform.reset();

    built_in_filter = EllipsFilter(t1,f1,t2,f2,true);
    Signal::Audiofile* a = dynamic_cast<Signal::Audiofile*>(get_inverse_waveform().get());
    built_in_filter.invalidateWaveform(*_temp_to_remove, *a->getChunkBehind());
}

void Transform_inverse::recompute_filter(pFilter f) {
    if (f.get()) {
        Signal::Audiofile* af = dynamic_cast<Signal::Audiofile*>(get_inverse_waveform().get());
        f->invalidateWaveform(*_temp_to_remove, *af->getChunkBehind());
    }
}

void Transform_inverse::play_inverse()
{
    Signal::Audiofile* af = dynamic_cast<Signal::Audiofile*>(_inverse_waveform.get());
    if ( _inverse_waveform ) if (af->getChunkBehind() )
    {
        float start_time, end_time;
        built_in_filter.range(start_time, end_time);
        filterTimer.reset(new TaskTimer("Computing inverse [%g,%g], %g s", start_time, end_time, end_time-start_time));

        af->getChunkBehind()->play_when_done = true;
    }

    get_inverse_waveform();
}


Signal::pSource Transform_inverse::get_inverse_waveform()
{
    if (0 == _inverse_waveform) {
        Signal::Audiofile* a = new Signal::Audiofile(this->_temp_to_remove->_temp_to_remove_playback);
        _inverse_waveform.reset(a);
        a->setChunk(prepare_inverse(0, _original_waveform->length()));

        for (unsigned n = 0;
             n <= _temp_to_remove->getChunkIndex( a->number_of_samples() );
             n++)
        {
            a->getChunkBehind()->valid_transform_chunks.insert(n);
        }
    }

    float start_time, end_time;

    built_in_filter.range(start_time, end_time);

    unsigned n,
//        first_index = _temp_to_remove->getChunkIndex( (unsigned)(max(start_time,0.f)*_original_waveform->sample_rate()) ),
        last_index = _temp_to_remove->getChunkIndex( min((unsigned)(end_time*_original_waveform->sample_rate()), _original_waveform->number_of_samples() ) );

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
    }

    Signal::Audiofile* af = dynamic_cast<Signal::Audiofile*>(_inverse_waveform.get());
    Signal::pBuffer cb = af->getChunkBehind();
    if (cb->play_when_done)
    {
        for (n=0; n<=last_index; n++) {
            if (0==cb->valid_transform_chunks.count(n))
                break;
        }
        if (n>last_index) {
            if(filterTimer)
                filterTimer->info("Computed entire inverse");
            filterTimer.reset();
            af->play();

            af->getChunkBehind()->play_when_done = false;
        }
    }

    return _inverse_waveform;
}

/*pWaveform_chunk Transform::computeInverse( float start, float end)
{
    pWaveform_chunk r = prepare_inverse(start, end);

    for(Transform::ChunkIndex n = getChunkIndex(start);
         n*samples_per_chunk() < end*r->sample_rate;
         n++)
    {
        merge_chunk(r, getChunk(n));
    }
    return r;
}*/

Signal::pBuffer Transform_inverse::prepare_inverse(float start, float end)
{
    unsigned n = _temp_to_remove->original_waveform()->number_of_samples();

    if(start<0) start=0;
    Signal::pBuffer r( new Signal::Buffer());
    r->sample_rate = _temp_to_remove->original_waveform()->sample_rate();
    r->sample_offset = (unsigned)min((float)n, r->sample_rate*start);
    n -= r->sample_offset;
    if (start<=end)
        n = max(1.f, min((float)n, r->sample_rate*(end-start)));
    //fprintf(stdout, "rate = %d, offs = %d, n = %d, orgn = %d\n", r->sample_rate, r->sample_offset, n, _temp_to_remove->original_waveform()->number_of_samples());
    //fflush(stdout);

    r->waveform_data.reset( new GpuCpuData<float>(0, make_cudaExtent(n, 1, 1), GpuCpuVoidData::CudaGlobal) );
    cudaMemset(r->waveform_data->getCudaGlobal().ptr(), 0, r->waveform_data->getSizeInBytes1D());

    return r;
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

Signal::pBuffer Transform_inverse::computeInverse( pTransform_chunk chunk, cudaStream_t stream ) {
    cudaExtent sz = make_cudaExtent( chunk->n_valid_samples, 1, 1);

    Signal::pBuffer r( new Signal::Buffer());
    r->sample_offset = chunk->chunk_offset + chunk->first_valid_sample;
    r->sample_rate = chunk->sample_rate;
    r->waveform_data.reset( new GpuCpuData<float>(0, sz, GpuCpuVoidData::CudaGlobal) );

    float4 area = make_float4(
            built_in_filter._t1 * _original_waveform->sample_rate() - r->sample_offset,
            built_in_filter._f1 * _temp_to_remove->nScales(),
            built_in_filter._t2 * _original_waveform->sample_rate() - r->sample_offset,
            built_in_filter._f2 * _temp_to_remove->nScales());
    {
        TaskTimer tt(TaskTimer::LogVerbose, __FUNCTION__);

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
