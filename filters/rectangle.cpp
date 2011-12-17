#include "rectangle.h"
#include "rectanglekernel.h"
#include "tfr/chunk.h"

// gpumisc
#include <computationkernel.h>

// std
#include <iomanip>
#include <float.h> // FLT_MAX

using namespace Tfr;

//#define TIME_FILTER
#define TIME_FILTER if(0)

#ifdef min
#undef min
#undef max
#endif

namespace Filters {

Rectangle::Rectangle(float t1, float f1, float t2, float f2, bool save_inside) {
    _t1 = std::min(t1, t2);
    _f1 = std::min(f1, f2);
    _t2 = std::max(t1, t2);
    _f2 = std::max(f1, f2);
    _save_inside = save_inside;
}


std::string Rectangle::
        name()
{
    std::stringstream ss;
    ss << std::setiosflags(std::ios::fixed);
    if (_t2 == FLT_MAX)
        ss << "Bandpass from ";
    else
        ss << "Rectangle from ";

    if (_t2 != FLT_MAX)
       ss << std::setprecision(1) << _t1 << " s, ";
    ss << std::setprecision(0) << _f1 << " Hz to ";
    if (_t2 != FLT_MAX)
       ss << std::setprecision(1) << _t2 << " s, ";
    ss << std::setprecision(0) << _f2 << " Hz";

    if (!this->_save_inside)
        ss << ", save outside";

    return ss.str();
}


void Rectangle::operator()( Chunk& chunk) {
    TIME_FILTER TaskTimer tt("Rectangle");

    Area area = {
            _t1 * chunk.sample_rate - chunk.chunk_offset.asFloat(),
            chunk.freqAxis.getFrequencyScalarNotClamped( _f1 ),
            _t2 * chunk.sample_rate - chunk.chunk_offset.asFloat(),
            chunk.freqAxis.getFrequencyScalarNotClamped( _f2 ) };

    ::removeRect( chunk.transform_data,
                  area,
                  _save_inside);

    TIME_FILTER ComputationSynchronize();
}


Signal::Intervals Rectangle::
        zeroed_samples()
{
    return _save_inside ? outside_samples() : Signal::Intervals();
}


Signal::Intervals Rectangle::
        affected_samples()
{
    return (_save_inside ? Signal::Intervals() : outside_samples()).inverse();
}


Signal::Intervals Rectangle::
        outside_samples()
{
    long double FS = sample_rate();

    long double
        start_time_d = std::max(0.f, _t1)*FS,
        end_time_d = std::max(0.f, _t2)*FS;

    Signal::IntervalType
        start_time = std::min((long double)Signal::Interval::IntervalType_MAX, start_time_d),
        end_time = std::min((long double)Signal::Interval::IntervalType_MAX, end_time_d);

    Signal::Intervals sid;
    if (start_time < end_time)
        sid = Signal::Intervals(start_time, end_time);

    return ~sid;
}

} // namespace Filters
