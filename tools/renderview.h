#ifndef RENDERVIEW_H
#define RENDERVIEW_H

#include "rendermodel.h"
#include "heightmap/renderer.h"

#include <QGLWidget>

namespace Tools
{
    class RenderView: public QWidget
    {
        Q_OBJECT
    public:
        RenderView(RenderModel* model);
        virtual ~RenderView();

        Heightmap::pRenderer renderer;

        void setPosition( float time, float f );

        double _qx, _qy, _qz; // position

        // TODO need to be able to update a QWidget, signal?
        // is this data/function model or view?

        RenderModel* model;

        QGLWidget* displayWidget;

    signals:
        void destroyingRenderView();

    };
} // namespace Tools

#endif // RENDERVIEW_H
