#include "ellipse.h"
#include "ellipsekernel.h"

#include "tfr/cwt.h"
#include "tfr/chunk.h"

// gpumisc
#include "TaskTimer.h"
#include "computationkernel.h"

// std
#include <iomanip>

//#define TIME_FILTER
#define TIME_FILTER if(0)

using namespace Tfr;

namespace Filters {

EllipseKernel::
        EllipseKernel(float t1, float f1, float t2, float f2, bool save_inside)
{
    _centre_t = t1;
    _centre_f = f1;
    _centre_plus_radius_t = t2;
    _centre_plus_radius_f = f2;
    _save_inside = save_inside;
}


void EllipseKernel::
        operator()( Tfr::ChunkAndInverse& c )
{
    Chunk& chunk = *c.chunk;
    TIME_FILTER TaskTimer tt("Ellipse");

    Area area = {
            (float)(_centre_t * chunk.sample_rate - chunk.chunk_offset.asFloat()),
            chunk.freqAxis.getFrequencyScalarNotClamped( _centre_f ),
            (float)(_centre_plus_radius_t * chunk.sample_rate - chunk.chunk_offset.asFloat()),
            chunk.freqAxis.getFrequencyScalarNotClamped( _centre_plus_radius_f )};

    ::removeDisc( chunk.transform_data,
                  area, _save_inside, chunk.sample_rate );

    TIME_FILTER ComputationSynchronize();
}



Ellipse::
        Ellipse(float t1, float f1, float t2, float f2, bool save_inside)
    :
      CwtFilterDesc(Tfr::pChunkFilter()),
      _centre_t(t1),
      _centre_f(f1),
      _centre_plus_radius_t(t2),
      _centre_plus_radius_f(f2),
      _save_inside(save_inside)
{
    updateChunkFilter ();
}


std::string Ellipse::
        name()
{
    std::stringstream ss;
    ss << std::setiosflags(std::ios::fixed)
       << std::setprecision(1)
       << "Ellipse at " <<  _centre_t << " s, "
       << std::setprecision(0) << _centre_f << " Hz, size " << std::log(std::fabs((_centre_plus_radius_t-_centre_t)*(_centre_plus_radius_f-_centre_f)*M_PI));

    if (!this->_save_inside)
        ss << ", save outside";

    return ss.str();
}


Signal::Intervals Ellipse::
        zeroed_samples(float FS)
{
    return _save_inside ? outside_samples(FS) : Signal::Intervals();
}


Signal::Intervals Ellipse::
        affected_samples(float FS)
{
    return (_save_inside ? Signal::Intervals() : outside_samples(FS)).inverse();
}


Signal::Intervals Ellipse::
        outside_samples(float FS)
{
    float r = fabsf(_centre_t - _centre_plus_radius_t);
    r += ((Tfr::Cwt*)transform_desc_.get())->wavelet_time_support_samples(FS)/FS;

    long double
        start_time_d = std::max(0.f, _centre_t - r)*FS,
        end_time_d = std::max(0.f, _centre_t + r)*FS;

    Signal::IntervalType
        start_time = std::min((long double)Signal::Interval::IntervalType_MAX, start_time_d),
        end_time = std::min((long double)Signal::Interval::IntervalType_MAX, end_time_d);

    Signal::Intervals sid;
    if (start_time < end_time)
        sid = Signal::Intervals(start_time, end_time);

    return ~sid;
}


void Ellipse::
        updateChunkFilter()
{
    Tfr::pChunkFilter cf(new EllipseKernel(_centre_t, _centre_f, _centre_plus_radius_t, _centre_plus_radius_f, _save_inside));
    chunk_filter_ = Tfr::FilterKernelDesc::Ptr(new Tfr::CwtKernelDesc(cf));
}


void Ellipse::
        test()
{
    // It should filter out an ellipse selection from the signal.
    EXCEPTION_ASSERT(false);
}


} // namespace Filters
