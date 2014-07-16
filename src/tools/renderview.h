#ifndef RENDERVIEW_H
#define RENDERVIEW_H

#pragma once

#include "rendermodel.h"
#include "support/toolselector.h"
#include "commentview.h"
#include "sawe/toolmainloop.h"
#include "tools/commands/viewstate.h"
#include "sawe/sawedll.h"

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

        virtual bool event( QEvent * e );
        virtual bool eventFilter(QObject* o, QEvent* e);
        virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
        virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
        virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);

        // Owned by commandInvoker
        QPointer<Tools::Commands::ViewState> viewstate;

        RenderModel* model;

        QGLWidget *glwidget;

        // graphicsview belongs in rendercontroller but this simplifies access from other classes
        GraphicsView* graphicsview;
        Support::ToolSelector* tool_selector;

        unsigned last_width() { return _last_width; }
        unsigned last_height() { return _last_height; }

        glProjection gl_projection;
        unsigned _last_width;  // This is viewport, gl_projection is also viewport
        unsigned _last_height;
        unsigned _last_x;
        unsigned _last_y;

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
        void redraw();

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
        int draw_more = 0;

        /// Similiar to QGLWidget::initializeGL()
        void initializeGL();

        /// Similiar to QGLWidget::resizeGL()
        void resizeGL( int width, int height, int ratio );

        /// Similiar to QGLWidget::paintGL()
        void paintGL();

        void drawCollection(int channel, float yscale);

        void setStates();
        void defaultStates();
        void setupCamera();
        void setRotationForAxes(bool);
        void computeChannelColors();

        boost::scoped_ptr<TaskTimer> _render_timer;
        boost::scoped_ptr<GlFrameBuffer> _renderview_fbo;

        bool _inited;

        int _try_gc;
        QPointer<QTimer> _update_timer;

        /**
          Adjusting sleep between frames based on fps.
          */
        Timer _last_frame;
        float _target_fps;

        Signal::UnsignedIntervalType _last_update_size;

        std::vector<tvector<4> > channel_colors;
    };
} // namespace Tools

#endif // RENDERVIEW_H
