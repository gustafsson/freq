#ifndef RENDERVIEWAXES_H
#define RENDERVIEWAXES_H

#include "tools/rendermodel.h"

class RenderViewAxes
{
public:
    RenderViewAxes(Tools::RenderModel& render_model);

    void linearFreqScale();
    void logFreqScale();
    void linearYScale();
    void logYScale();
    void cameraOnFront();
    void logZScale();

private:
    Tools::RenderModel& render_model;
};

#endif // RENDERVIEWAXES_H
