#ifndef RENDERVIEWAXES_H
#define RENDERVIEWAXES_H

#include "tools/rendermodel.h"

class RenderViewAxes
{
public:
    RenderViewAxes(Tools::RenderModel& render_model);

    void linearFreqScale();
    void waveformScale();
    void logYScale();
    void linearYScale();
    void cameraOnFront();
    void logFreqAxis();

private:
    Tools::RenderModel& render_model;
};

#endif // RENDERVIEWAXES_H
