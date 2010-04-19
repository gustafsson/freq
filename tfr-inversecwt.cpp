#include "tfr-inversecwt.h"
/**
  #include "transform.h"
#include "signal-audiofile.h"
#include <stdio.h>
#include <string> // min/max
#include "CudaException.h"
#include "wavelet.cu.h"
  */

namespace Tfr {

InverseCwt::InverseCwt(cudaStream_t stream)
:   _stream(stream)
{
}

Signal::pBuffer operator()(Tfr::Chunk& chunk)
{
    EllipsFilter* e;
    SquareFilter* s;
    if (filter.get()) {
        e = dynamic_cast<EllipsFilter*>(filter.get());
        s = dynamic_cast<SquareFilter*>(filter.get());

        if (!e && !s) {
            (*filter)( chunk );
        }
    }
    cudaExtent sz = make_cudaExtent( chunk.n_valid_samples, 1, 1);

    Signal::pBuffer r( new Signal::Buffer());
    r->sample_offset = chunk.chunk_offset + chunk.first_valid_sample;
    r->sample_rate = chunk.sample_rate;
    r->waveform_data.reset( new GpuCpuData<float>(0, sz, GpuCpuVoidData::CudaGlobal) );

    {
        TaskTimer tt(TaskTimer::LogVerbose, __FUNCTION__);

        if (e) {
            float4 area = make_float4(
                    e->_t1 * _original_waveform->sample_rate() - r->sample_offset,
                    e->_f1 * chunk.nScales(),
                    e->_t2 * _original_waveform->sample_rate() - r->sample_offset,
                    e->_f2 * chunk.nScales());

            ::wtInverseEllips( chunk.transform_data->getCudaGlobal().ptr() + chunk.first_valid_sample,
                         r->waveform_data->getCudaGlobal().ptr(),
                         chunk.transform_data->getNumberOfElements(),
                         area,
                         chunk.n_valid_samples,
                         _stream );
        } else if (s) {
            float4 area = make_float4(
                    s->_t1 * _original_waveform->sample_rate() - r->sample_offset,
                    s->_f1 * chunk.nScales(),
                    s->_t2 * _original_waveform->sample_rate() - r->sample_offset,
                    s->_f2 * chunk.nScales());

            ::wtInverseBox( chunk.transform_data->getCudaGlobal().ptr() + chunk.first_valid_sample,
                         r->waveform_data->getCudaGlobal().ptr(),
                         chunk.transform_data->getNumberOfElements(),
                         area,
                         chunk.n_valid_samples,
                         _stream );
        } else {
            ::wtInverse( chunk.transform_data->getCudaGlobal().ptr() + chunk.first_valid_sample,
                         r->waveform_data->getCudaGlobal().ptr(),
                         chunk.transform_data->getNumberOfElements(),
                         chunk.n_valid_samples,
                         _stream );
        }

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

} // namespace Tfr
