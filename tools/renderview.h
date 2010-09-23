#ifndef RENDERVIEW_H
#define RENDERVIEW_H

#include "rendermodel.h"

#include <QGLWidget>

namespace Ui { class DisplayWidget; }

namespace Tools
{
    class RenderView: public QGLWidget
    {
        Q_OBJECT
    public:
        RenderView(RenderModel* model);
        virtual ~RenderView();

        void setPosition( float time, float f );

        // TODO remove position and use renderer->camera instead
        double _qx, _qy, _qz; // position
        float _px, _py, _pz, // TODO beautify
            _rx, _ry, _rz;
        float xscale;

        // TODO need to be able to update a QWidget, signal?
        // is this data/function model or view?

        RenderModel* model;

        QWidget* displayWidget;

    signals:
        void destroyingRenderView();
        void paintedView();


    private:
        /// @overload QGLWidget::initializeGL()
        virtual void initializeGL();

        /// @overload QGLWidget::resizeGL()
        virtual void resizeGL( int width, int height );

        /// @overload QGLWidget::paintGL()
        virtual void paintGL();


        void setupCamera();

        boost::scoped_ptr<TaskTimer> _work_timer;
        boost::scoped_ptr<TaskTimer> _render_timer;

        float _prevLimit;

        Ui::DisplayWidget* dw();
    };
} // namespace Tools

#endif // RENDERVIEW_H
