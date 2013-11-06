#ifndef RENDERVIEW_H
#define RENDERVIEW_H

#pragma once

#include "rendermodel.h"
#include "support/toolselector.h"
#include "commentview.h"
#include "sawe/toolmainloop.h"
#include "tools/commands/viewstate.h"

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
    class GraphicsView;

    class SaweDll RenderView: public QGraphicsScene
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
        // TODO use a pointer to a smart pointer or something that has a semantic meaning instead of a magical value of ref.
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
        Tools::Commands::ViewState viewstate;

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
        float last_length();

		template<class Archive> void serialize_items(Archive& ar, const unsigned int version) {
			QList<QGraphicsItem *> itms = items();
			/*foreach( QGraphicsItem * itm, itms ) {
				if (dynamic_cast<
			}
			this->*/
		}

        const std::vector<tvector<4> >& channelColors() const { return channel_colors; }

        void emitTransformChanged();
        void emitAxisChanged();

    public slots:
        void setLastUpdateSize( Signal::UnsignedIntervalType length );
        void userinput_update( bool request_high_fps = true, bool post_update = true, bool cheat_also_high=true );

    signals:
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
         * @brief populateTodoList. Use 'Qt::DirectConnection'
         */
        void populateTodoList();

        /**
         * @brief postPaint. Use 'Qt::DirectConnection'
         */
        void postPaint();

        /**
         * @brief paintingForeground. Use 'Qt::DirectConnection'
         */
        void paintingForeground();

        /**
         * @brief finishedWorkSection. Use 'Qt::AutoConnection'
         */
        void finishedWorkSection();


        /**
         * @brief postUpdate. Use 'Qt::DirectConnection'
         */
        void postUpdate();


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
        void restartUpdateTimer();

    private:
        /// Similiar to QGLWidget::initializeGL()
        void initializeGL();

        /// Similiar to QGLWidget::resizeGL()
        void resizeGL( int width, int height, int ratio );

        /// Similiar to QGLWidget::paintGL()
        void paintGL();

        void drawCollection(int channel, float yscale);

        void setStates();
        void setLights();
        void defaultStates();
        void setupCamera();
        void setRotationForAxes(bool);
        void computeChannelColors();

        boost::scoped_ptr<TaskTimer> _render_timer;
        boost::scoped_ptr<GlFrameBuffer> _renderview_fbo;

        bool _inited;
        unsigned _last_width;
        unsigned _last_height;
        unsigned _last_x;
        unsigned _last_y;
        int _try_gc;
        QPointer<QTimer> _update_timer;

        /**
          Adjusting sleep between frames based on fps.
          */
        Timer _last_frame;
        float _target_fps;

        Signal::UnsignedIntervalType _last_update_size;

        double modelview_matrix[16], projection_matrix[16];
        int viewport_matrix[4];

        std::vector<tvector<4> > channel_colors;
    };
} // namespace Tools

#endif // RENDERVIEW_H
