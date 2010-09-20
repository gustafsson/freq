#include <CudaException.h>
#include "timelineview.h"
#include <boost/assert.hpp>
#include <GlException.h>
#include <glPushContext.h>
#include <QMouseEvent>
#include "ui/displaywidget.h"
#include "ui/displaywidget.h"
#include <QDockWidget>
#include "toolfactory.h"

#undef max

//#define TIME_PAINTGL
#define TIME_PAINTGL if(0)

using namespace Signal;

namespace Tools {

TimelineView::
        TimelineView( Sawe::Project* p, QGLWidget* displaywidget )
:   QGLWidget( 0, displaywidget, Qt::WindowFlags(0) ),
    _xscale( 1 ),
    _xoffs( 0 ),
    _barHeight( 0.1f ),
    _movingTimeline( 0 ),
    _project( p )
{
    BOOST_ASSERT( displaywidget );

    if (!context() || !context()->isSharing())
    {
        throw std::invalid_argument("Failed to open a second OpenGL window. Couldn't find a valid rendering context to share.");
    }

    QMainWindow *MainWindow = dynamic_cast<QMainWindow*>(displaywidget->parentWidget());
    QDockWidget* dock = new QDockWidget(MainWindow);
    dock->setObjectName(QString::fromUtf8("dockWidgetTimeline"));
    dock->setMinimumSize(QSize(42, 79));
    dock->setMaximumSize(QSize(524287, 524287));
    dock->setContextMenuPolicy(Qt::NoContextMenu);
    dock->setFeatures(QDockWidget::DockWidgetFeatureMask);
    dock->setWidget(this);
    dock->show();
    MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(8), dock);

    Ui::DisplayWidget* d = dynamic_cast<Ui::DisplayWidget*>(displaywidget);
    connect(d, SIGNAL(renderingParametersChanged()), this, SLOT(update()));
}


TimelineView::
        ~TimelineView()
{
    TaskTimer tt("~TimelineView");
}


/*void TimelineView::
        put(Signal::pBuffer , Signal::pOperation )
{
    update();
}


void TimelineView::
        add_expected_samples( const Signal::Intervals& )
{
    update();
}*/


void TimelineView::
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


void TimelineView::
        resizeGL( int width, int height )
{
    height = height?height:1;

    glViewport( 0, 0, (GLint)width, (GLint)height );

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    _barHeight = 20.f/height;
    glOrtho(0,1,-_barHeight,1, -10,10);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


void TimelineView::
        paintGL()
{
    TIME_PAINTGL TaskTimer tt("TimelineView::paintGL");

    static int exceptCount = 0;
    try {
        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glPushMatrixContext mc;

        { // Render
            // Set up camera position
            float length = std::max( 1.f, _project->worker.source()->length());
            float h = 1 - 0.5f*length/_xscale;
            if (_xscale<1) _xscale = 1;
            if (_xoffs<h) _xoffs = h;
            if (_xoffs>length+h) _xoffs = length+h;

            setupCamera( false );

            // Draw debug triangle
            /*glColor4f( 0, 1,0, 1);
            glBegin(GL_LINE_LOOP);
            glVertex3f(0,0,0);
            glVertex3f(length,0,0);
            glVertex3f(length,0,1);
            glEnd();*/

            {
                glPushMatrixContext a;

                _project->tools().render_view.renderer->draw( 0.f );
                _project->tools().selection_view.drawSelection();
                _project->tools().render_view.renderer->drawFrustum();
            }
        }

        {
            glPushMatrixContext mc;

            setupCamera( true );

            glScalef(1,1,_barHeight);
            glTranslatef(0,0,-1);
            _project->tools().render_view.renderer->draw( 0.f );

            float length = std::max( 1.f, _project->worker.source()->length());
            glColor4f( 0.75, 0.75,0.75, .5);
            glLineWidth(2);
            glBegin(GL_LINES);
                glVertex3f(0,1,1);
                glVertex3f(length,1,1);
            glEnd();

            float x1 = _xoffs;
            float x4 = _xoffs+length/_xscale;
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

            _project->tools().render_view.renderer->drawFrustum(0.75);
        }

        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();

        exceptCount = 0;
    } catch (const CudaException &x) {
        if (1<++exceptCount) throw;
        else TaskTimer("TimelineView::paintGL SWALLOWED CUDAEXCEPTION\n%s", x.what()).suppressTiming();;
    } catch (const GlException &x) {
        if (1<++exceptCount) throw;
        else TaskTimer("TimelineView::paintGL SWALLOWED GLEXCEPTION\n%s", x.what()).suppressTiming();
    }
}


void TimelineView::
        setupCamera( bool staticTimeLine )
{
    float length = std::max( 1.f, _project->worker.source()->length());

    if (0 == "Make sure that the camera focus point is within the timeline")
    {
        float t = _project->tools().render_view.renderer->camera[0];
        if (t < _xoffs) _xoffs = t;
        if (t > _xoffs + length/_xscale ) _xoffs = t - length/_xscale;
    }

    glLoadIdentity();

    glRotatef( 90, 1, 0, 0 );
    glRotatef( 180, 0, 1, 0 );

    glScalef(-1/length, 1, 1);

    if (!staticTimeLine) {
        glScalef(_xscale, 1, 1);
        glTranslatef(-_xoffs, 0, 0);
    }
}


void TimelineView::
        wheelEvent ( QWheelEvent *e )
{
    makeCurrent();
    setupCamera();

    int x = e->x(), y = height() - e->y();
    float ps = 0.0005f;

    GLvector current;
    moveButton.spacePos(x, y, current[0], current[1]);

    float f = 1.f - ps * e->delta();
    _xscale *= f;

    setupCamera();

    GLvector newPos;
    moveButton.spacePos(x, y, newPos[0], newPos[1]);

    //_xoffs -= current[0]/prevscale*_xscale-newPos[0];
    //_xoffs = current[0] - _xscale*(x/(float)width());
    _xoffs -= newPos[0]-current[0];

    setupCamera();

    GLvector newPos2;
    moveButton.spacePos(x, y, newPos2[0], newPos2[1]);

    // float tg = _oldoffs + x * prevscale;
    // float tg2 = _newoffs + x/(float)width() * _xscale;
    //_xoffs -= x/(float)width() * (prevscale-_xscale);

    if (0) printf("[%d, %d] -> [%g, %g, %g] -> (%g, %g)\n",
           x, y,
           current[0], newPos[0], newPos2[0],
           _xscale, _xoffs);

    update();
}


void TimelineView::
        mousePressEvent ( QMouseEvent * e )
{
    makeCurrent();
    setupCamera();

    int x = e->x(), y = height() - e->y();

    GLvector prev;
    moveButton.spacePos(prev[0], prev[1]);

    GLvector current;
    moveButton.spacePos(x, y, current[0], current[1]);

    if (e->buttons() & Qt::LeftButton)
    {
        if (0 == _movingTimeline)
        {
            if (current[1]>=0)  _movingTimeline = 1;
            else                _movingTimeline = 2;
        }

        switch ( _movingTimeline )
        {
        case 1:
            _project->tools().render_view.setPosition( current[0], current[1] );
            break;
        case 2:
            {
                setupCamera( true );
                moveButton.spacePos(x, y, current[0], current[1]);

                float length = std::max( 1.f, _project->worker.source()->length());
                _xoffs = current[0] - 0.5f*length/_xscale;
            }
            break;
        }
    }

    if (moveButton.isDown() && (e->buttons() & Qt::RightButton))
    {
        _xoffs -= current[0] - prev[0];
    }

    moveButton.press( x, y );
    update();
}


void TimelineView::
        mouseReleaseEvent ( QMouseEvent * e)
{
    if (0 == (e->buttons() & Qt::LeftButton)) {
        _movingTimeline = 0;
    }
    moveButton.release();
}


void TimelineView::
        mouseMoveEvent ( QMouseEvent * e )
{
    mousePressEvent(e);
}


} // namespace Tools
