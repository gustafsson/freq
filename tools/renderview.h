#ifndef RENDERVIEW_H
#define RENDERVIEW_H

#pragma once

#include "rendermodel.h"
#include "support/toolselector.h"
#include "commentview.h"

// gpumisc
#include <TAni.h>

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
    class GraphicsView;

    class RenderView: public QGraphicsScene
    {
        Q_OBJECT
    public:
        RenderView(RenderModel* model);
        virtual ~RenderView();

        virtual void drawBackground(QPainter *painter, const QRectF &);
        virtual void drawForeground(QPainter *painter, const QRectF &);
        void drawCollections(GlFrameBuffer* fbo, float yscale);
        QPointF getScreenPos( Heightmap::Position pos, double* dist, bool use_heightmap_value = true );
        QPointF getWidgetPos( Heightmap::Position pos, double* dist, bool use_heightmap_value = true );
        Heightmap::Position getHeightmapPos( QPointF widget_coordinates, bool useRenderViewContext = true );
        Heightmap::Position getPlanePos( QPointF widget_coordinates, bool* success = 0, bool useRenderViewContext = true );
        QPointF widget_coordinates( QPointF window_coordinates );
        QPointF window_coordinates( QPointF widget_coordinates );
        float getHeightmapValue( Heightmap::Position pos, Heightmap::Reference* ref = 0, float* find_local_max = 0, bool fetch_interpolation = false, bool* is_valid_value = 0 );

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

        void setPosition( Heightmap::Position pos );

        float last_ysize;
        floatAni orthoview;
        //QTransform projectionTransform;
        //QTransform modelviewTransform;
        //QTransform viewTransform;

        // TODO need to be able to update a QWidget, signal?
        // is this data/function model or view?

        RenderModel* model;

        QGLWidget *glwidget;

        // graphicsview belongs in rendercontroller but this simplifies access from other classes
        GraphicsView* graphicsview;

        Support::ToolSelector* toolSelector();

        Support::ToolSelector* tool_selector;

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
        void userinput_update( bool request_high_fps = true );

    private slots:
        void clearCaches();
        void finishedWorkSectionSlot();

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
        void populateTodoList();

        /**
          */
        void postPaint();

        /**
          */
        void paintingForeground();

        /**
          */
        void finishedWorkSection();

    private:
        /// Similiar to QGLWidget::initializeGL()
        void initializeGL();

        /// Similiar to QGLWidget::resizeGL()
        void resizeGL( int width, int height );

        /// Similiar to QGLWidget::paintGL()
        void paintGL();

        void queueRepaint();

        void drawCollection(int, Signal::FinalSource*, float yscale);

        void setStates();
        void setLights();
        void defaultStates();
        void setupCamera();
        void computeChannelColors();

        boost::scoped_ptr<TaskTimer> _work_timer;
        boost::scoped_ptr<TaskTimer> _render_timer;
        boost::scoped_ptr<GlFrameBuffer> _renderview_fbo;

        bool _inited;
        unsigned _last_width;
        unsigned _last_height;
        unsigned _last_x;
        unsigned _last_y;
        int _try_gc;
        QTimer* _update_timer;

        float _last_length;
        double modelview_matrix[16], projection_matrix[16];
        int viewport_matrix[4];

        std::vector<float4> channel_colors;
    };
} // namespace Tools

#endif // RENDERVIEW_H
