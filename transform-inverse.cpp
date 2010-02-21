#include "transform-inverse.h"
#include "transform-chunk.h"

Transform_inverse::Transform_inverse(pWaveform waveform, Callback* callback)
:   _waveform(waveform),
    _callback(callback)
{
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

    if (_inverse_waveform && _inverse_waveform->getChunkBehind())
    {
        for (unsigned n = getChunkIndex( max(0.f,mint)*_original_waveform->sample_rate());
             n <= getChunkIndex( max(0.f,maxt)*_original_waveform->sample_rate());
             n++)
        {
            _inverse_waveform->getChunkBehind()->valid_transform_chunks.erase(n);
        }
    }
    filterTimer.reset(new TaskTimer("Computing inverse [%g,%g], %g s", mint, maxt, maxt-mint));
}

pWaveform Transform_inverse::get_inverse_waveform()
{
    if (0 == _inverse_waveform) {
        _inverse_waveform.reset(new Waveform());
        _inverse_waveform->setChunk(prepare_inverse(0, _original_waveform->length()));
    }

    unsigned n, last_index=getChunkIndex( _original_waveform->number_of_samples() );


    unsigned previous_chunk_index = (unsigned)-1;
    pTransform_chunk transform = previous_chunk(previous_chunk_index);
    if (transform &&
        previous_chunk_index <= last_index &&
        0==_inverse_waveform->getChunkBehind()->valid_transform_chunks.count(previous_chunk_index))
    {
        n = previous_chunk_index;
    } else {
        transform.reset();
        for (n=0; n<=last_index; n++) {
            if (0==_inverse_waveform->getChunkBehind()->valid_transform_chunks.count(n)) {
                transform = getChunk(n);
                break;
            }
        }
    }

    if (transform) {
        merge_chunk(_inverse_waveform->getChunkBehind(), transform);

        _inverse_waveform->getChunkBehind()->valid_transform_chunks.insert( n );
        _inverse_waveform->getChunkBehind()->modified = true;


        for (n=0; n<=last_index; n++) {
            if (0==_inverse_waveform->getChunkBehind()->valid_transform_chunks.count(n))
                break;
        }
        if (n>last_index) {
            if(filterTimer)
                filterTimer->info("Computed entire inverse");
            filterTimer.reset();
            _inverse_waveform->play();
        }
    }

    return _inverse_waveform;
}

pWaveform_chunk Transform_inverse::computeInverse( float start, float end)
{
    pWaveform_chunk r = prepare_inverse(start, end);

    for(Transform::ChunkIndex n = getChunkIndex(start);
         n*samples_per_chunk() < end*r->sample_rate;
         n++)
    {
        merge_chunk(r, getChunk(n));
    }
    return r;
}

pWaveform_chunk Transform_inverse::prepare_inverse(float start, float end)
{
    unsigned n = original_waveform()->number_of_samples();

    if(start<0) start=0;
    pWaveform_chunk r( new Waveform_chunk());
    r->sample_rate = original_waveform()->sample_rate();
    r->sample_offset = min((float)n, r->sample_rate*start);
    n -= r->sample_offset;
    if (start<=end)
        n = min((float)n, r->sample_rate*(end-start));
    fprintf(stdout, "rate = %d, offs = %d, n = %d, orgn = %d\n", r->sample_rate, r->sample_offset, n, original_waveform()->number_of_samples());
    fflush(stdout);

    r->waveform_data.reset( new GpuCpuData<float>(0, make_cudaExtent(n, 1, 1), GpuCpuVoidData::CudaGlobal) );
    cudaMemset(r->waveform_data->getCudaGlobal().ptr(), 0, r->waveform_data->getSizeInBytes1D());

    return r;
}

void Transform_inverse::merge_chunk(pWaveform_chunk r, pTransform_chunk transform)
{
    pWaveform_chunk chunk = computeInverse( transform, 0 );

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

