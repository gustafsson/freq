#ifndef RENDERVIEW_H
#define RENDERVIEW_H

#pragma once

#include "rendermodel.h"
//#include "tools/commands/viewstate.h"
#include "glprojection.h"

// gpumisc
#include "timer.h"

// boost
#include <boost/scoped_ptr.hpp>

// Qt
#include <QGraphicsScene>
#include <QTransform>

class GlFrameBuffer;
class QGLWidget;

namespace Heightmap { class Reference; }

namespace Tools
{
    namespace Support {
    // TODO remove
    class ToolSelector;
    }

    class GraphicsView;

    class RenderView: public QObject
    {
        Q_OBJECT
    public:
        RenderView(RenderModel* model);
        virtual ~RenderView();

        /// Similiar to QGLWidget::initializeGL()
        void initializeGL();
        void setStates();
        void defaultStates();

        /// Similiar to QGLWidget::resizeGL()
        void resizeGL( QRect rect, int device_height );

        /// Similiar to QGLWidget::paintGL()
        void paintGL();

        // Owned by commandInvoker
//        QPointer<Tools::Commands::ViewState> viewstate;

        RenderModel* model;

        QGLWidget *glwidget;

        // graphicsview belongs in rendercontroller but this simplifies access from other classes
        GraphicsView* graphicsview;
        Support::ToolSelector* tool_selector;

        glProjection gl_projection;

        /**
         * @brief rect describes the viewport in QWidget pixels
         * gl_projection->viewport and rect() describe the same area. However,
         * gl_projection->viewport has the bottom of the drawable area as y=0,
         * rect() has the top of the drawable area as y=0.
         * @return QRect in pixels, high-dpi pixels if applicable
         */
        QRect rect();

        void emitTransformChanged();
        void emitAxisChanged();

    public slots:
        void redraw();

    signals:
        /**
         * @brief redraw
         */
        void redrawSignal();

        /**
         * @brief destroying. Use 'Qt::DirectConnection'
          Emitted in the destructor, before the OpenGL context is destroyed.
          QObject::destroyed() is emitted shortly after, but after the OpenGL
          context is destroyed.
          */
        void destroying();

        /**
         * @brief prePaint. Use 'Qt::DirectConnection'
          Emitted right before camera setup. A tool have the option to affect
          the renderview camera for this frame.
          */
        void prePaint();

        /**
         * @brief painting. Use 'Qt::DirectConnection'
          Emitted during painting, but after the heightmap has been rendered.
          Tool specific stuff is rendered here.
          */
        void painting();

        /**
         * @brief postPaint. Use 'Qt::DirectConnection'
         */
        void postPaint();

        /**
         * @brief finishedWorkSection. Use 'Qt::AutoConnection'
         */
        void finishedWorkSection();


        /**
         * @brief transformChanged is emitted through emitTransformChanged.
         * emitTransformChanged should be called whenever the state of the
         * transform description has changed. This signal might be issued
         * several times during a frame. Use 'Qt::QueuedConnection'.
         */
        void transformChanged();

        /**
         * @brief axisChanged. Use 'Qt::QueuedConnection'
         */
        void axisChanged();

    private slots:
        void clearCaches();
        void finishedWorkSectionSlot();

    private:
        void setupCamera();
        void setRotationForAxes(bool);

        unsigned rect_y_;
        boost::scoped_ptr<TaskTimer> _render_timer;
        boost::scoped_ptr<GlFrameBuffer> _renderview_fbo;
    };
} // namespace Tools

#endif // RENDERVIEW_H
