#include "visualizationparams.h"

#include "exceptionassert.h"

namespace Heightmap {

VisualizationParams::
        VisualizationParams()
    :
      details_(new details)
{
    // by default there is no transform_desc, and nothing will be drawn

    // display_scale is also left to its default value

    details::WritePtr(details_)->amplitude_axis_ = AmplitudeAxis_5thRoot;
}


bool VisualizationParams::
        operator==(const volatile VisualizationParams& b) const volatile
{
    if (&b == this)
        return true;

    return (transform_desc() && b.transform_desc()
            ? *transform_desc() == *b.transform_desc()
            : transform_desc() == b.transform_desc()) &&
        display_scale() == b.display_scale() &&
        amplitude_axis() == b.amplitude_axis();
}


bool VisualizationParams::
        operator!=(const volatile VisualizationParams& b) const volatile
{
    return !(*this == b);
}


Tfr::TransformDesc::Ptr VisualizationParams::
        transform_desc() const volatile
{
    const VisualizationParams* self = const_cast<const VisualizationParams*>(this);
    return details::ReadPtr(self->details_)->transform_desc_;
}


void VisualizationParams::
        transform_desc(Tfr::TransformDesc::Ptr v) volatile
{
    const VisualizationParams* self = const_cast<const VisualizationParams*>(this);
    details::WritePtr(self->details_)->transform_desc_ = v;
}


Tfr::FreqAxis VisualizationParams::
        display_scale() const volatile
{
    const VisualizationParams* self = const_cast<const VisualizationParams*>(this);
    return details::ReadPtr(self->details_)->display_scale_;
}


void VisualizationParams::
        display_scale(Tfr::FreqAxis v) volatile
{
    const VisualizationParams* self = const_cast<const VisualizationParams*>(this);
    details::WritePtr(self->details_)->display_scale_ = v;
}


AmplitudeAxis VisualizationParams::
        amplitude_axis() const volatile
{
    const VisualizationParams* self = const_cast<const VisualizationParams*>(this);
    return details::ReadPtr(self->details_)->amplitude_axis_;
}


void VisualizationParams::
        amplitude_axis(AmplitudeAxis v) volatile
{
    const VisualizationParams* self = const_cast<const VisualizationParams*>(this);
    details::WritePtr(self->details_)->amplitude_axis_ = v;
}


void VisualizationParams::
        test()
{
    // It should describe all parameters that define how waveform data turns
    // into pixels on a heightmap
    {
        // This class has requirements on how other classes should use it.

        Tfr::FreqAxis f; f.setLinear (1);
        VisualizationParams::Ptr v(new VisualizationParams);
        v->transform_desc(Tfr::TransformDesc::Ptr());
        v->amplitude_axis(AmplitudeAxis_Linear);

        EXCEPTION_ASSERT(*v == *v);

        VisualizationParams::Ptr v2(new VisualizationParams);
        v2->transform_desc(Tfr::TransformDesc::Ptr());
        v2->amplitude_axis(AmplitudeAxis_Linear);

        // FreqAxis compares not equal to an uninitialized instance
        EXCEPTION_ASSERT(*v != *v2);

        v->display_scale(f);
        v2->display_scale(f);

        EXCEPTION_ASSERT(*v == *v2);

        v2->amplitude_axis(AmplitudeAxis_Logarithmic);
        EXCEPTION_ASSERT(*v != *v2);
    }
}

} // namespace Heightmap
