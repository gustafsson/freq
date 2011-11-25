#include "zoomcameracommand.h"

#include "tools/rendermodel.h"
#include "navigationcommand.h"

#include "sawe/project.h"

namespace Tools {
namespace Commands {

ZoomCameraCommand::
        ZoomCameraCommand(Tools::RenderModel* model, float dt, float ds, float dz)
            :
            NavigationCommand(model),
            dt(dt),
            ds(ds),
            dz(dz)
{
}


std::string ZoomCameraCommand::
        toString()
{
    return "Zoom camera";
}


void ZoomCameraCommand::
        executeFirst()
{
    if (dt) if (!zoom( 1500*dt*model->xscale, ScaleX ))
    {
        float L = model->project()->worker.source()->length();
        float d = std::min( 0.5f * fabsf(dt), fabsf(model->_qx - L/2));
        model->_qx += model->_qx>L*.5f ? -d : d;
    }

    if (ds) if (!zoom( 1500*ds*model->zscale, ScaleZ ))
    {
        float d = std::min( 0.5f * fabsf(ds), fabsf(model->_qz - .5f));
        model->_qz += model->_qz>.5f ? -d : d;
    }

    if (dz) zoom(dz, Zoom );
}


bool ZoomCameraCommand::
        zoom(int delta, ZoomMode mode)
{
    float L = model->project()->worker.length();
    float fs = model->project()->head->head_source()->sample_rate();
    float min_xscale = 4.f/std::max(L,10/fs);
    float max_xscale = 0.05f*fs;


    const Tfr::FreqAxis& tfa = model->collections[0]->transform()->freqAxis(fs);
    unsigned maxi = tfa.getFrequencyScalar(fs/2);

    float hza = tfa.getFrequency(0u);
    float hza2 = tfa.getFrequency(1u);
    float hzb = tfa.getFrequency(maxi - 1);
    float hzb2 = tfa.getFrequency(maxi - 2);

    const Tfr::FreqAxis& ds = model->display_scale();
    float scalara = ds.getFrequencyScalar( hza );
    float scalara2 = ds.getFrequencyScalar( hza2 );
    float scalarb = ds.getFrequencyScalar( hzb );
    float scalarb2 = ds.getFrequencyScalar( hzb2 );

    float minydist = std::min(fabsf(scalara2 - scalara), fabsf(scalarb2 - scalarb));

    float min_yscale = 4.f;
    float max_yscale = 1.f/minydist;

    if (delta > 0)
    {
        switch(mode)
        {
        case ScaleX: if (model->xscale == min_xscale)
                return false;
            break;
        case ScaleZ: if (model->zscale == min_yscale)
                return false;
            break;
        default:
            break;
        }
    }

    switch(mode)
    {
    case Zoom: doZoom( delta, 0, 0, 0 ); break;
    case ScaleX: doZoom( delta, &model->xscale, &min_xscale, &max_xscale); break;
    case ScaleZ: doZoom( delta, &model->zscale, &min_yscale, &max_yscale ); break;
    }

    return true;
}


void ZoomCameraCommand::
        doZoom(int delta, float* scale, float* min_scale, float* max_scale)
{
    //TaskTimer("NavigationController wheelEvent %s %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
    float ps = 0.0005;
    if(scale)
    {
        float d = ps * delta;
        if (d>0.1)
            d=0.1;
        if (d<-0.1)
            d=-0.1;

        *scale *= (1-d);

        if (d > 0 )
        {
            if (min_scale && *scale<*min_scale)
                *scale = *min_scale;
        }
        else
        {
            if (max_scale && *scale>*max_scale)
                *scale = *max_scale;
        }
    }
    else
    {
        model->_pz *= (1+ps * delta);

        if (model->_pz<-40) model->_pz = -40;
        if (model->_pz>-.1) model->_pz = -.1;
    }
}


} // namespace Commands
} // namespace Tools