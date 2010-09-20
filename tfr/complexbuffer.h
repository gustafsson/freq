#ifndef TFR_COMPLEXBUFFER_H
#define TFR_COMPLEXBUFFER_H

#include "signal/source.h"

namespace Tfr
{
class ComplexBuffer
{
public:
    ComplexBuffer(unsigned firstSample,
           unsigned numberOfSamples,
           unsigned FS,
           unsigned numberOfChannels=1);

    /**
        Create a complex waveform out of a real waveform.
    */
    ComplexBuffer(const Signal::Buffer& b);


    /**
        Really inefficient, don't do this. Will recompute get_real for each
        call. Instead, call get_real(), store that pBuffer and then call
        waveform_data().

        The pointer is valid for the lifetime of this class, or as long as the
        pBuffer returned from get_real() isn't deleted.
    */
    virtual GpuCpuData<float>* waveform_data();


    /**
        Overloaded from buffer
    */
    virtual unsigned number_of_samples() const { return _complex_waveform_data->getNumberOfElements().width/2; }


    unsigned        sample_offset;
    unsigned        sample_rate;

    /**
        Used to convert back to real data, will discard imaginary part.
    */
    Signal::pBuffer get_real();


    /**
        Access the complex waveform
    */
    GpuCpuData<float2>* complex_waveform_data() const {
        return _complex_waveform_data.get();
    }

protected:
    Signal::pBuffer _my_real;

    boost::scoped_ptr<GpuCpuData<float2> >
                    _complex_waveform_data;
};

} // namespace Tfr
#endif // TFR_COMPLEXBUFFER_H
