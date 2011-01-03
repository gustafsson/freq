#ifndef RENDERVIEW_H
#define RENDERVIEW_H

#pragma once

#include "rendermodel.h"
#include "support/toolselector.h"
#include "commentview.h"

// gpumisc
#include <TAni.h>

// Qt
#include <QGLWidget>
#include <QGraphicsScene>
#include <QTransform>

class GlFrameBuffer;

namespace Heightmap
{
    class Reference;
}

namespace Tools
{
    class RenderView: public QGraphicsScene
    {
        Q_OBJECT
    public:
        RenderView(RenderModel* model);
        virtual ~RenderView();

        virtual void drawBackground(QPainter *painter, const QRectF &);
        void drawCollections(GlFrameBuffer* fbo);
        QPointF getScreenPos( Heightmap::Position pos, double* dist );
        Heightmap::Position getHeightmapPos( QPointF viewport_coordinates, bool useRenderViewContext = true );
        Heightmap::Position getPlanePos( QPointF pos, bool* success, bool useRenderViewContext = true );
        float getHeightmapValue( Heightmap::Position pos, Heightmap::Reference* ref = 0, float* find_local_max = 0, bool fetch_interpolation = false );

        /**
          You might want to use Heightmap::Reference::containsPoint(p) to se
          if the returned reference actually is a valid reference for the point
          given. It will not be valid if 'p' lies outside the spectrogram.
          */
        Heightmap::Reference findRefAtCurrentZoomLevel(Heightmap::Position p);

        virtual bool event( QEvent * e );
        virtual bool eventFilter(QObject* o, QEvent* e);
        virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
        virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
        virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);

        void setPosition( float time, float f );
        void makeCurrent();

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
        float last_length() { return _last_length; }

		template<class Archive> void serialize_items(Archive& ar, const unsigned int version) {
			QList<QGraphicsItem *> itms = items();
			/*foreach( QGraphicsItem * itm, itms ) {
				if (dynamic_cast<
			}
			this->*/
		}

    public slots:
        void userinput_update();

    private slots:
        void clearCaches();

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


        void drawCollection(int, Signal::FinalSource*);

        void setStates();
        void setLights();
        void defaultStates();
        void setupCamera();
        void computeChannelColors();

        boost::scoped_ptr<TaskTimer> _work_timer;
        boost::scoped_ptr<TaskTimer> _render_timer;
        boost::scoped_ptr<GlFrameBuffer> _renderview_fbo;

        bool _inited;
        float _prevLimit;
        unsigned _last_width;
        unsigned _last_height;
        int _try_gc;

        float _last_length;
        GLdouble modelview_matrix[16], projection_matrix[16];
        GLint viewport_matrix[4];

        std::vector<float4> channel_colors;
    };
} // namespace Tools

#endif // RENDERVIEW_H
