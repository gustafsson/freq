#ifndef RENDERVIEW_H
#define RENDERVIEW_H

#include "rendermodel.h"
#include "support/toolselector.h"

// gpumisc
#include <TAni.h>

// Qt
#include <QGLWidget>
#include <QGraphicsScene>
#include <QTransform>

namespace Tools
{
    class RenderView: public QGraphicsScene
    {
        Q_OBJECT
    public:
        RenderView(RenderModel* model);
        virtual ~RenderView();

        virtual void drawBackground(QPainter *painter, const QRectF &);
        QPointF getScreenPos( Heightmap::Position pos, double* dist );
        Heightmap::Position getHeightmapPos( QPointF viewport_coordinates );
        Heightmap::Position getPlanePos( QPointF pos, bool* success );
        float getHeightmapValue( Heightmap::Position pos );

        virtual bool event( QEvent * e );
        virtual bool eventFilter(QObject* o, QEvent* e);
        virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
        virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
        virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);

        void setPosition( float time, float f );
        void makeCurrent();

        // TODO remove position and use renderer->camera instead
        double _qx, _qy, _qz; // position
        float _px, _py, _pz, // TODO beautify
            _rx, _ry, _rz;
        float xscale;
        float last_ysize;
        floatAni orthoview;
        //QTransform projectionTransform;
        QTransform modelviewTransform;
        QTransform viewTransform;

        // TODO need to be able to update a QWidget, signal?
        // is this data/function model or view?

        RenderModel* model;
        QGLWidget *glwidget;

        Support::ToolSelector* toolSelector();

        boost::scoped_ptr<Support::ToolSelector> tool_selector;

        unsigned last_width() { return _last_width; }
        unsigned last_height() { return _last_height; }

    public slots:
        void userinput_update();

    signals:
        /**
          Emitted in the destructor, before the OpenGL context is destroyed.
          QObject::destroyed() is emitted shortly after, but after the OpenGL
          context is destroyed.
          */
        void destroying();

        /**
          Emitted right before camera setup. A tool have the option to affect
          the renderview camera for this frame.
          */
        void prePaint();

        /**
          Emitted during painting, but after the heightmap has been rendered.
          Tool specific stuff is rendered here.
          */
        void painting();

        /**
          */
        void postPaint();

    private:
        /// @overload QGLWidget::initializeGL()
        virtual void initializeGL();

        /// @overload QGLWidget::resizeGL()
        virtual void resizeGL( int width, int height );

        /// @overload QGLWidget::paintGL()
        virtual void paintGL();


        void setStates();
        void setLights();
        void defaultStates();
        void setupCamera();

        boost::scoped_ptr<TaskTimer> _work_timer;
        boost::scoped_ptr<TaskTimer> _render_timer;

        bool _inited;
        float _prevLimit;
        unsigned _last_width;
        unsigned _last_height;
    };
} // namespace Tools

#endif // RENDERVIEW_H
