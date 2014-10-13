#include "optimaltimefrequencyresolution.h"
#include "tfr/stftdesc.h"
#include "tfr/waveformrepresentation.h"
#include "log.h"
#include "largememorybank.h"

//#define LOG_TRANSFORM
#define LOG_TRANSFORM if(0)

OptimalTimeFrequencyResolution::OptimalTimeFrequencyResolution(QQuickItem *parent) :
    QQuickItem(parent)
{
}


void OptimalTimeFrequencyResolution::
        setSquircle(Squircle* s)
{
    if (squircle_)
        disconnect (squircle_, SIGNAL(cameraChanged()), this, SLOT(onCameraChanged()));

    this->squircle_ = s;

    connect (squircle_, SIGNAL(cameraChanged()), this, SLOT(onCameraChanged()));
}


void OptimalTimeFrequencyResolution::
        onCameraChanged()
{
    if (!squircle_)
        return;

    Tools::RenderModel& render_model = *squircle_->renderModel ();
    const auto c = *render_model.camera.read();
    const vectord& q = c.q;

    // match zmin/zmax with TouchNavigation
    auto viewport = render_model.gl_projection.read ()->viewport;
    float aspect = viewport[2]/(float)viewport[3];
    float zmin = std::min(0.5,0.4/(c.zscale/-c.p[2]*aspect));
    float zmax = 1.0-zmin;
    float zfocus = (q[2]-zmin)/(zmax-zmin);
    if (zmin==zmax) zfocus=0.5;

    auto tm = render_model.tfr_mapping ().read ();
    float ds = 0.1;
    float fs = tm->targetSampleRate();
    float hz = tm->display_scale().getFrequency(float(zfocus)),
          hz2 = tm->display_scale().getFrequency(float(zfocus < 0.5 ? zfocus + ds : zfocus - ds));
    tm.unlock ();

    // read info about current transform
    Tfr::TransformDesc::ptr t = render_model.transform_desc();
    float time_res = 1/t->displayedTimeResolution(fs,hz); // samples per 1 time at camera focus
    float s = t->freqAxis(fs).getFrequencyScalarNotClamped(hz);
    float s2 = t->freqAxis(fs).getFrequencyScalarNotClamped(hz2);
    float bin_res = std::fabs (s2-s)/ds; // bins per 1 scale at camera focus

    // compute wanted and current current ratio of time resolution versus frequency resolution
    float ratio = c.zscale/c.xscale; // bins/samples
    float current_ratio = bin_res/time_res;

    Tfr::TransformDesc::ptr newt = t->copy ();
    if (Tfr::StftDesc* stft = dynamic_cast<Tfr::StftDesc*>(newt.get ()))
    {
        int w = stft->chunk_size ();
        float k = std::sqrt(ratio/current_ratio);
        stft->set_approximate_chunk_size(w*k);
    }
//        else if (Tfr::Cwt* cwt = dynamic_cast<Tfr::Cwt*>(newt.get ()))
//        {
//            float s = stft->scales_per_octave ();
//            float k = std::sqrt(ratio/current_ratio);
//            cwt->scales_per_octave(s/k);
//            if (*newt != *t)
//                render_model()->set_transform_desc (newt);
//        }
    else if (dynamic_cast<Tfr::WaveformRepresentationDesc*>(newt.get ()))
    {
        // do nothing
    }
    else
    {
        Log("touchnavigation: unknown transform");
    }

    if (*newt != *t)
    {
        LOG_TRANSFORM Log("OptimalTimeFrequencyResolution: %s") % newt->toString ();
        render_model.set_transform_desc (newt);
        emit squircle_->displayedTransformDetailsChanged();

        lmb_gc (16 << 20); // 16 MB, could use a lower value if it should be more aggressive
    }
}
