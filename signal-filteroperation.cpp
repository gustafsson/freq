#include "signal-filteroperation.h"
#include <stringprintf.h>
#include <CudaException.h>
#include <memory.h>

//#define TIME_FILTEROPERATION
#define TIME_FILTEROPERATION if(0)

namespace Signal {

FilterOperation::
        FilterOperation(pSource source, Tfr::pFilter filter)
:   OperationCache(source),
    _filter( filter ),
    _save_previous_chunk( false )
{
}

pBuffer FilterOperation::
        readRaw( unsigned firstSample, unsigned numberOfSamples )
{
    TIME_FILTEROPERATION TaskTimer tt("FilterOperation::readRaw ( %u, %u )", firstSample, numberOfSamples);
    unsigned wavelet_std_samples = Tfr::CwtSingleton::instance()->wavelet_std_samples( _source->sample_rate());

    // wavelet_std_samples gets stored in cwt so that inverse_cwt can take it
    // into account and create an inverse that is of the desired size.
    unsigned redundant_samples = 0;
    if (firstSample < wavelet_std_samples) redundant_samples = firstSample;
    else redundant_samples = wavelet_std_samples;

    unsigned first_valid_sample = firstSample;
    firstSample -= redundant_samples;

    if (numberOfSamples<10)
        numberOfSamples=10;

    _previous_chunk.reset();

    // If we're not asked to compute a chunk, try to take shortcuts
    if (!_save_previous_chunk)
    {
        if (!_filter) {
            TIME_FILTEROPERATION TaskTimer("No filter, moving on").suppressTiming();
            return _source->read(first_valid_sample, numberOfSamples);
        }

        SamplesIntervalDescriptor work(first_valid_sample, first_valid_sample + numberOfSamples);

        // If filter would make all these samples zero, return immediately
        if ((work - _filter->getZeroSamples( _source->sample_rate() )).isEmpty()) {
            pBuffer b( new Buffer( first_valid_sample, numberOfSamples, _source->sample_rate() ));
            ::memset( b->waveform_data->getCpuMemory(), 0, b->waveform_data->getSizeInBytes1D());
            TIME_FILTEROPERATION SamplesIntervalDescriptor(b->getInterval()).print("FilterOp silent");
            return b;
        }

        // If filter would leave all these samples unchanged, return immediately
        if ((work-_filter->getUntouchedSamples( _source->sample_rate() )).isEmpty()) {
            // Attempt a regular simple read
            pBuffer b = _source->read(first_valid_sample, numberOfSamples);
            work = b->getInterval();
            work -= _filter->getUntouchedSamples( _source->sample_rate() );
            if (work.isEmpty()) {
                TIME_FILTEROPERATION SamplesIntervalDescriptor(b->getInterval()).print("FilterOp unaffected");
                return b;
            }
            TIME_FILTEROPERATION SamplesIntervalDescriptor(b->getInterval()).print("FilterOp fixed unaffected");
            // Failed, return the exact samples validated as untouched
            return _source->readFixedLength(first_valid_sample, numberOfSamples);
        }
    }

    // If we've reached this far, the transform will have to be computed

    // These computations require a lot of memory allocations
    // If we encounter out of cuda memory, we decrease the required
    // memory in this while loop.
    pBuffer r;
    while(true) try {
        unsigned L = redundant_samples + numberOfSamples + wavelet_std_samples;

        pBuffer b = _source->readFixedLength( firstSample, L );
        TIME_FILTEROPERATION SamplesIntervalDescriptor(b->getInterval()).print("FilterOp subread");

        // Compute the continous wavelet transform
        Tfr::pChunk c = Tfr::CwtSingleton::operate( b );

        // Apply filter
        if (_filter)
        {
            SamplesIntervalDescriptor work(c->getInterval());
            work -= _filter->getUntouchedSamples( _source->sample_rate() );
            // Only apply filter if it would affect these samples
            if (!work.isEmpty())
            {
                (*_filter)( *c );
            }
        }

        // Don't compute the inverse if the chunk was requested.
        if (_save_previous_chunk)
        {
            // Just make the buffer smaller
            SamplesIntervalDescriptor::Interval i = c->getInterval();
            r.reset(new Buffer(i.first, i.last-i.first,b->sample_rate));
            if (b->interleaved() != Signal::Buffer::Only_Real)
                b = b->getInterleaved(Signal::Buffer::Only_Real);
            memcpy(r->waveform_data->getCpuMemory(),
                   b->waveform_data->getCpuMemory() + (i.first - b->sample_offset),
                   r->waveform_data->getSizeInBytes1D());
        }
        else
        {
            r = Tfr::InverseCwt()( *c );
        }

        TIME_FILTEROPERATION SamplesIntervalDescriptor(r->getInterval()).print("FilterOp after inverse");

        if (_save_previous_chunk)
            _previous_chunk = c;

        break;
    } catch (const CufftException &x) {
        switch (x.getCufftError())
        {
            case CUFFT_EXEC_FAILED:
            case CUFFT_ALLOC_FAILED:
                break;
            default:
                throw;
        }

        Tfr::pCwt cwt = Tfr::CwtSingleton::instance();
        unsigned newL = cwt->prev_good_size( numberOfSamples, sample_rate());
        if (newL < numberOfSamples ) {
            numberOfSamples = newL;

            TaskTimer("FilterOperation reducing chunk size to readRaw( %u, %u )\n%s", first_valid_sample, numberOfSamples, x.what() ).suppressTiming();
            continue;
        }

        // Try to decrease tf_resolution
        if (cwt->tf_resolution() > 1)
            cwt->tf_resolution(1/0.8f);

        if (cwt->tf_resolution() > exp(-2.f ))
        {
            cwt->tf_resolution( cwt->tf_resolution()*0.8f );
            float std_t = cwt->morlet_std_t(0, sample_rate());
            cwt->wavelet_std_t( std_t );

            TaskTimer("FilterOperation reducing tf_resolution to %g\n%s", cwt->tf_resolution(), x.what() ).suppressTiming();
            continue;
        }
        throw std::invalid_argument(printfstring("Not enough memory. Parameter 'wavelet_std_t=%g, tf_resolution=%g' yields a chunk size of %u MB.\n\n%s)",
                             cwt->wavelet_std_t(), cwt->tf_resolution(), cwt->wavelet_std_samples(sample_rate())*cwt->nScales(sample_rate())*sizeof(float)*2>>20, x.what()));
    } catch (const CudaException &x) {
        if (cudaErrorMemoryAllocation != x.getCudaError() )
            throw;

        Tfr::pCwt cwt = Tfr::CwtSingleton::instance();
        unsigned newL = cwt->prev_good_size( numberOfSamples, sample_rate());
        if (newL < numberOfSamples ) {
            numberOfSamples = newL;

            TaskTimer("FilterOperation reducing chunk size to readRaw( %u, %u )\n%s", first_valid_sample, numberOfSamples, x.what() ).suppressTiming();
            continue;
        }

        // Try to decrease tf_resolution
        if (cwt->tf_resolution() > 1)
            cwt->tf_resolution(1/0.8f);

        if (cwt->tf_resolution() > exp(-2.f ))
        {
            cwt->tf_resolution( cwt->tf_resolution()*0.8f );
            float std_t = cwt->morlet_std_t(0, sample_rate());
            cwt->wavelet_std_t( std_t );

            TaskTimer("FilterOperation reducing tf_resolution to %g\n%s", cwt->tf_resolution(), x.what() ).suppressTiming();
            continue;
        }

        throw std::invalid_argument(printfstring("Not enough memory. Parameter 'wavelet_std_t=%g, tf_resolution=%g' yields a chunk size of %u MB.\n\n%s)",
                             cwt->wavelet_std_t(), cwt->tf_resolution(), cwt->wavelet_std_samples(sample_rate())*cwt->nScales(sample_rate())*sizeof(float)*2>>20, x.what()));
    }

    _save_previous_chunk = false;

    return r;
}

bool FilterOperation::
        cacheMiss(unsigned firstSample, unsigned numberOfSamples)
{
    if (_save_previous_chunk)
        return true;

    return OperationCache::cacheMiss(firstSample, numberOfSamples);
}

void FilterOperation::
        meldFilters()
{
    FilterOperation* f = dynamic_cast<FilterOperation*>( _source.get());
    if (0==f) return;

    f->meldFilters();

    Tfr::FilterChain* c = dynamic_cast<Tfr::FilterChain*>(_filter.get());
    if (0==c) {
        if (_filter) {
            c = new Tfr::FilterChain;
            c->push_back( _filter );
            _filter = Tfr::pFilter( c );
        } else {
            _filter = f->filter();
        }
    }

    if (0!=c) {
        Tfr::FilterChain* c2 = dynamic_cast<Tfr::FilterChain*>(f->filter().get());
        if (0==c2) {
            if(f->filter())
                c->push_back( f->filter() );
        } else {
            c->insert(c->end(), c2->begin(), c2->end());
        }
    }

    // Remove _source (this effectively prevents two subsequent FilterOperation to
    // have different parameters for Cwt and InverseCwt)
    _source = f->source();
}

Tfr::pChunk FilterOperation::
        previous_chunk()
{
    _save_previous_chunk = true;
    return _previous_chunk;
}

void FilterOperation::
        release_previous_chunk()
{
    _previous_chunk.reset();
}

void FilterOperation::
        filter( Tfr::pFilter f )
{
    unsigned FS = sample_rate();

    // Start with the assumtion that all touched samples will have to be recomputed
    SamplesIntervalDescriptor a = f->getTouchedSamples(FS);

    if (_filter)
    {
        // Don't recompute what would still be zero
        a -= (_filter->getZeroSamples( sample_rate() ) & f->getZeroSamples( sample_rate() ));
    }

    // The samples that were previously invalid are still invalid, merge
    _invalid_samples |= a;

    // Change filter
    _filter = f;
}

} // namespace Signal
