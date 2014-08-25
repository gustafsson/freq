#include "renderviewaxes.h"
#include "tfr/cwt.h"

RenderViewAxes::RenderViewAxes(Tools::RenderModel& render_model)
    :
      render_model(render_model)
{
}


void RenderViewAxes::
        linearFreqScale()
{
    auto x = render_model.recompute_extent ();
    float fs = x.sample_rate.get();

    Heightmap::FreqAxis fa;
    fa.setLinear( fs );

//    if (currentTransform() && fa.min_hz < currentTransformMinHz())
//    {
//        fa.min_hz = currentTransformMinHz();
//        fa.f_step = fs/2 - fa.min_hz;
//    }

    render_model.display_scale ( fa );
}


void RenderViewAxes::
        logFreqScale()
{
    auto x = render_model.recompute_extent ();
    float fs = x.sample_rate.get();

    Heightmap::FreqAxis fa;

    {
        auto td = render_model.transform_descs ().write ();
        Tfr::Cwt& cwt = td->getParam<Tfr::Cwt>();
        fa.setLogarithmic(
                cwt.get_wanted_min_hz (fs),
                cwt.get_max_hz (fs) );

//        if (currentTransform() && fa.min_hz < currentTransformMinHz())
//        {
//            // Happens typically when currentTransform is a cepstrum transform with a short window size
//            fa.setLogarithmic(
//                    currentTransformMinHz(),
//                    cwt.get_max_hz(fs) );
//        }
    }

    render_model.display_scale( fa );
}


void RenderViewAxes::
        linearYScale()
{
    render_model.render_settings.y_offset = 0.0;
    render_model.render_settings.y_scale = 0.5;
    render_model.render_settings.log_scale.reset (0);
    render_model.render_settings.color_mode = Heightmap::Render::RenderSettings::ColorMode_FixedColor;
}