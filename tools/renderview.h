#ifndef RENDERVIEW_H
#define RENDERVIEW_H

#include "rendermodel.h"
#include "heightmap/renderer.h"

namespace Tools
{
    class RenderView
    {
    public:
        RenderView(RenderModel* model);

        Heightmap::pRenderer renderer;

        void setPosition( float time, float f );

        double _qx, _qy, _qz; // position

        // TODO need to be able to update a QWidget, signal?
        // is this data/function model or view?
    private:
        RenderModel* model;
    };
} // namespace Tools

#endif // RENDERVIEW_H
