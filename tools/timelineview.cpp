#include "timelineview.h"

// Sonic AWE tools
#include "toolfactory.h"
#include "ui/mainwindow.h"
#include "rendercontroller.h"

// Sonic AWE lib
#include "heightmap/renderer.h"

// gpumisc
#include <CudaException.h>
#include <GlException.h>
#include <glPushContext.h>
#include <cuda_vector_types_op.h>

// boost
#include <boost/assert.hpp>

// qt
#include <QMouseEvent>
#include <QDockWidget>
#include <QTimer>

#undef max

//#define TIME_PAINTGL
#define TIME_PAINTGL if(0)

using namespace Signal;

namespace Tools {

TimelineView::
        TimelineView( Sawe::Project* p, RenderView* render_view)
:   QGLWidget( 0, render_view->glwidget, Qt::WindowFlags(0) ),
    _xscale( 1 ),
    _xoffs( 0 ),
    _barHeight( 0.1f ),
    _length( 0 ),
    _width( 0 ),
    _height( 0 ),
    _project( p ),
    _render_view( render_view ),
    _except_count( 0 )
{
    BOOST_ASSERT( _render_view );

    if (!context() || !context()->isSharing())
    {
        throw std::invalid_argument("Failed to open a second OpenGL window. Couldn't find a valid rendering context to share.");
    }
}


TimelineView::
        ~TimelineView()
{
    TaskTimer("~TimelineView").suppressTiming();
}


void TimelineView::
        userinput_update()
{
    _project->worker.requested_fps(30);
    // this will leave room for others to paint as well, calling 'update' wouldn't
    QTimer::singleShot(0, this, SLOT(update()));
}


void TimelineView::
        getLengthNow()
{
    _length = std::max( 1.f, _project->worker.source()->length());
}


void TimelineView::
        initializeGL()
{
    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glClearDepth(1.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    {   // Antialiasing
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        glDisable(GL_POLYGON_SMOOTH);
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
        glEnable(GL_BLEND);
    }

    glShadeModel(GL_SMOOTH);
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
}


void TimelineView::
        resizeGL( int width, int height )
{
    height = height?height:1;

    glViewport( 0, 0, _width = width, _height = height );

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    _barHeight = 20.f/height;
    //glOrtho(0,1,-_barHeight,1, -10,10);
    glOrtho(0,1,0,1, -10,10);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


void TimelineView::
        paintGL()
{
    TIME_PAINTGL TaskTimer tt("TimelineView::paintGL");

    _except_count = 0;
    try {
        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glPushMatrixContext mc(GL_MODELVIEW);

        { // Render
            // Set up camera position
            float look_ahead = 0; // 1
            float h = look_ahead - 0.5f*_length/_xscale;
            if (_xscale<1) _xscale = 1;
            if (_xoffs<h) _xoffs = h;
            if (_xoffs>_length+h) _xoffs = _length+h;

            setupCamera( false );
            glViewport( 0, _height*_barHeight, _width, _height*(1-_barHeight) );

            {
                glPushMatrixContext mc(GL_MODELVIEW);

                _render_view->drawCollections();

                // TODO what should be rendered in the timelineview?
                // Not arbitrary tools but
                // _project->tools().selection_view.drawSelection();
                _render_view->model->renderer->drawFrustum();
            }
        }

        {
            // Draw little bar for entire signal at the bottom of the timeline
            //glPushMatrixContext mc(GL_MODELVIEW);

            setupCamera( true );
            glViewport( 0, 0, (GLint)_width, (GLint)_height*_barHeight );

            _render_view->drawCollections();

            glViewport( 0, 0, (GLint)_width, (GLint)_height );
            setupCamera( true );
            glScalef(1,1,_barHeight);
            //glTranslatef(0,0,-1);

            glColor4f( 0.75, 0.75,0.75, .5);
            glLineWidth(2);
            glBegin(GL_LINES);
                glVertex3f(0,1,1);
                glVertex3f(_length,1,1);
            glEnd();

            float x1 = _xoffs;
            float x4 = _xoffs+_length/_xscale;
            float x2 = x1*.9 + x4*.1;
            float x3 = x1*.1 + x4*.9;
            glBegin( GL_TRIANGLE_STRIP );
                glColor4f(0,0,0,.5);
                glVertex3f(x1,1,0);
                glVertex3f(x1,1,1);
                glColor4f(0,0,0,.25);
                glVertex3f(x2,1,0);
                glVertex3f(x2,1,1);
                glVertex3f(x3,1,0);
                glVertex3f(x3,1,1);
                glColor4f(0,0,0,.5);
                glVertex3f(x4,1,0);
                glVertex3f(x4,1,1);
            glEnd();

            _render_view->model->renderer->drawFrustum(0.75);
        }

        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();

        _except_count = 0;
    } catch (const CudaException &x) {
        if (1<++_except_count) throw;

        TaskTimer("TimelineView::paintGL SWALLOWED CUDAEXCEPTION\n%s", x.what()).suppressTiming();;
        _render_view->clearCaches();
    } catch (const GlException &x) {
        if (1<++_except_count) throw;

        TaskTimer("TimelineView::paintGL SWALLOWED GLEXCEPTION\n%s", x.what()).suppressTiming();
        _render_view->clearCaches();
    }
}


void TimelineView::
        setupCamera( bool staticTimeLine )
{
    // Make sure that the camera focus point is within the timeline
    {
        float t = _render_view->model->renderer->camera[0];
        float new_t = -1;

        switch(0) // Both 1 and 2 might feel annoying, don't do them :)
        {
        case 1: // Clamp the timeline, prevent moving to much.
                // This might be both annoying and confusing
            if (t < _xoffs) _xoffs = t;
            if (t > _xoffs + _length/_xscale ) _xoffs = t - _length/_xscale;
            break;

        case 2: // Clamp the render view
                // This might be just annoying
            if (t < _xoffs) new_t = _xoffs;
            if (t > _xoffs + _length/_xscale ) new_t = _xoffs + _length/_xscale;

            if (0<=new_t)
            {
                float f = _render_view->model->renderer->camera[2];
                _render_view->setPosition( new_t, f);
            }
            break;
        }
    }

    glLoadIdentity();

    glRotatef( 90, 1, 0, 0 );
    glRotatef( 180, 0, 1, 0 );

    glScalef(-1/_length, 1, 1);

    if (!staticTimeLine) {
        glScalef(_xscale, 1, 1);
        glTranslatef(-_xoffs, 0, 0);
    }
}


} // namespace Tools
