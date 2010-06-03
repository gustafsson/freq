#include "sawe-timelinewidget.h"
#include <boost/assert.hpp>
#include <GlException.h>
#include <CudaException.h>
#include <glPushContext.h>
#include <QMouseEvent>

using namespace Signal;

namespace Sawe {

TimelineWidget::
        TimelineWidget(  pSink displayWidget )
:   QGLWidget( 0, dynamic_cast<QGLWidget*>(displayWidget.get()), Qt::WindowFlags(0) ),
    _displayWidget(displayWidget)
{
    BOOST_ASSERT( dynamic_cast<DisplayWidget*>(displayWidget.get()) );
}

TimelineWidget::
        ~TimelineWidget()
{
    TaskTimer tt("~TimelineWidget");
}

void TimelineWidget::
        put(Signal::pBuffer , Signal::pSource )
{
    update();
}

void TimelineWidget::
        add_expected_samples( const Signal::SamplesIntervalDescriptor& )
{
    update();
}

void TimelineWidget::
        initializeGL()
{
    glShadeModel(GL_SMOOTH);

    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glClearDepth(1.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    {   // Antialiasing
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        glEnable(GL_POLYGON_SMOOTH);
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
        glEnable(GL_BLEND);
    }

    GLfloat LightAmbient[]= { 0.5f, 0.5f, 0.5f, 1.0f };
    GLfloat LightDiffuse[]= { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat LightPosition[]= { 0.0f, 0.0f, 2.0f, 1.0f };
    glShadeModel(GL_SMOOTH);
    glLightfv(GL_LIGHT1, GL_AMBIENT, LightAmbient);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, LightDiffuse);
    glLightfv(GL_LIGHT1, GL_POSITION,LightPosition);
    glEnable(GL_LIGHT1);
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
}

void TimelineWidget::
        resizeGL( int width, int height )
{
    height = height?height:1;

    glViewport( 0, 0, (GLint)width, (GLint)height );

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0,1,0,1, -10,10);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void TimelineWidget::
        paintGL()
{
    TaskTimer tt("TimelineWidget::paintGL");
    static int exceptCount = 0;
    try {
        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glPushMatrixContext mc;

        // Set up camera position
        float length = std::max( 1.f, getDisplayWidget()->worker()->source()->length());

        {
            glLoadIdentity();

            glRotatef( 90, 1, 0, 0 );
            glRotatef( 180, 0, 1, 0 );

            glScalef(-1/length, 1, 1);
        }
        glColor4f( 0, 1,0, 1);
        glBegin(GL_LINE_LOOP);
        glVertex3f(0,0,0);
        glVertex3f(length,0,0);
        glVertex3f(length,0,1);
        glEnd();

        { // Render
            getDisplayWidget()->renderer()->draw();
            getDisplayWidget()->drawSelection();
            getDisplayWidget()->renderer()->drawFrustum();
        }

        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();

        exceptCount = 0;
    } catch (const CudaException &x) {
        TaskTimer("TimelineWidget::paintGL SWALLOWED CUDAEXCEPTION %s", x.what()).suppressTiming();;
        if (1<++exceptCount) throw;
    } catch (const GlException &x) {
        TaskTimer("TimelineWidget::paintGL SWALLOWED GLEXCEPTION %s", x.what()).suppressTiming();
        if (1<++exceptCount) throw;
    }
}


void TimelineWidget::
        mousePressEvent ( QMouseEvent * e )
{
    makeCurrent();

    int x = e->x(), y = this->height() - e->y();

    moveButton.press( e->x(), this->height() - e->y() );

    GLvector current;
    if( moveButton.spacePos(x, y, current[0], current[1]) )
        getDisplayWidget()->setPosition( current[0], current[1] );
}

void TimelineWidget::
        mouseReleaseEvent ( QMouseEvent * )
{
    moveButton.release();
}


void TimelineWidget::
        mouseMoveEvent ( QMouseEvent * e )
{
    mousePressEvent(e);
}

DisplayWidget * TimelineWidget::
        getDisplayWidget()
{
    DisplayWidget* w = dynamic_cast<DisplayWidget*>(_displayWidget.get());
    BOOST_ASSERT( w );
    return w;
}

} // namespace Sawe
