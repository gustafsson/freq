#include "inversecwt.h"
#include "wavelet.cu.h"

#include <CudaException.h>

//#define TIME_ICWT
#define TIME_ICWT if(0)

namespace Tfr {

InverseCwt::InverseCwt(cudaStream_t stream)
:   _stream(stream)
{
}

Signal::pBuffer InverseCwt::
operator()(Tfr::Chunk& chunk)
{
    TIME_ICWT TaskTimer tt("InverseCwt");

    cudaExtent sz = make_cudaExtent( chunk.n_valid_samples, 1, 1);
    Signal::pBuffer r( new Signal::Buffer());
    r->sample_offset = chunk.chunk_offset + chunk.first_valid_sample;
    r->sample_rate = chunk.sample_rate;
    r->waveform_data.reset( new GpuCpuData<float>(0, sz, GpuCpuVoidData::CudaGlobal) );

    EllipsFilter* e = 0;
    SquareFilter* s = 0;
    if (filter.get()) {
        //e = dynamic_cast<EllipsFilter*>(filter.get());
        //s = dynamic_cast<SquareFilter*>(filter.get());

        if (!e && !s) {
            Signal::SamplesIntervalDescriptor
                    chunkSid = chunk.getInterval(),
                    chunkCopy = chunkSid;
            if ((chunkSid -= filter->getUntouchedSamples( chunk.sample_rate )).isEmpty())
            {
                // Filter won't do anything
            }
            else if ((chunkCopy -= filter->getZeroSamples( chunk.sample_rate )).isEmpty())
            {
                // Filter will set everything to 0
                cudaMemset( chunk.transform_data->getCudaGlobal().ptr(),
                            0, chunk.transform_data->getSizeInBytes1D());
                cudaMemset( r->waveform_data->getCudaGlobal().ptr(), 0, r->waveform_data->getSizeInBytes1D() );
                return r;
            }
            else
            {
                // Filter would actually do something, do it
                (*filter)( chunk );
            }
        }
    }

    {
        TIME_ICWT TaskTimer tt(TaskTimer::LogVerbose, __FUNCTION__);

        if (e) {
            float4 area = make_float4(
                    e->_t1 * chunk.sample_rate - r->sample_offset,
                    e->_f1 * chunk.nScales(),
                    e->_t2 * chunk.sample_rate - r->sample_offset,
                    e->_f2 * chunk.nScales());

            ::wtInverseEllips( chunk.transform_data->getCudaGlobal().ptr() + chunk.first_valid_sample,
                         r->waveform_data->getCudaGlobal().ptr(),
                         chunk.transform_data->getNumberOfElements(),
                         area,
                         chunk.n_valid_samples,
                         _stream );
        } else if (s) {
            float4 area = make_float4(
                    s->_t1 * chunk.sample_rate - r->sample_offset,
                    s->_f1 * chunk.nScales(),
                    s->_t2 * chunk.sample_rate - r->sample_offset,
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

        TIME_ICWT CudaException_ThreadSynchronize();
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
