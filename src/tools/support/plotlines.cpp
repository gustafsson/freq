#include "plotlines.h"
#include "paintline.h"
#include "heightmap/update/blockkernel.h" // amplitudevalueruntime

#include "tools/renderview.h"

#include "glPushContext.h"
#include "GlException.h"
#include "glstate.h"

#include <QColor>

#ifdef LEGACY_OPENGL

using namespace std;

namespace Tools {
namespace Support {

PlotLines::
        PlotLines(RenderView* render_view)
    :
    render_view_(render_view),
    rand_color_offs_( rand()/(float)RAND_MAX ),
    display_list_( 0 )
{
    rand_color_offs_ = 0;
    connect(render_view, SIGNAL(transformChanged()), SLOT(resetDisplayList()) );
    connect(render_view, SIGNAL(axisChanged()), SLOT(resetDisplayList()) );
    connect(render_view, SIGNAL(painting()), SLOT(draw()) );
}


void PlotLines::
        clear()
{
    lines_.clear();
}


void PlotLines::
        clear( const Signal::Intervals& I, float fs )
{
    bool newColors = false;

    for (Lines::iterator itr = lines_.begin(); itr != lines_.end(); )
    {
        for (Line::Data::iterator d = itr->second.data.begin(); d != itr->second.data.end(); )
        {
            if (I.testSample(d->first*fs))
                itr->second.data.erase( d++ );
            else
                d++;
        }

        if (itr->second.data.empty())
        {
            lines_.erase( itr++ );
            newColors = true;
        }
        else
            itr++;
    }

    if (newColors)
        recomputeColors();
}


void PlotLines::
        set( Time t, float hz, float a )
{
    line(0).data[t] = Value( hz, a );
    resetDisplayList();
}


void PlotLines::
        set( LineIdentifier id, Time t, float hz, float a )
{
    line(id).data[t] = Value( hz, a );
    resetDisplayList();
}


PlotLines::Line& PlotLines::
        line( LineIdentifier id )
{
    if (lines_.find(id) == lines_.end())
    {
        lines_[id];

        recomputeColors();
    }

    return lines_[id];
}


void PlotLines::
        draw()
{
    if (lines_.empty())
        return;

    glPushMatrixContext push_model( GL_MODELVIEW );

    glScalef(1, render_view_->model->render_settings.y_scale, 1);

    if (0 == display_list_)
    {
        display_list_ = glGenLists( 1 );
        glNewList( display_list_, GL_COMPILE_AND_EXECUTE );

        for (Lines::iterator itr = lines_.begin(); itr != lines_.end(); itr++ )
        {
            draw(itr->second);
        }

        glEndList();
    }
    else
    {
        glCallList( display_list_ );
    }
}


void PlotLines::
        resetDisplayList()
{
    if ( 0 != display_list_)
        glDeleteLists(display_list_, 1);
    display_list_ = 0;
}


void PlotLines::
        draw(Line& l)
{
    const Heightmap::FreqAxis& fa = render_view_->model->display_scale();
    Heightmap::AmplitudeValueRuntime height = render_view_->model->amplitude_axis();

    GlException_CHECK_ERROR();

    glPushAttribContext ac;
    GlState::glEnable (GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(false);

    glBegin(GL_TRIANGLE_STRIP);
    for (Line::Data::iterator d = l.data.begin(); d != l.data.end(); ++d )
    {
        float scale = fa.getFrequencyScalar( d->second.hz );
        glColor4f( l.R, l.G, l.B, 0.5*l.A);
        glVertex3f( d->first, 0, scale );
        glColor4f( l.R, l.G, l.B, l.A );
        glVertex3f( d->first, height(d->second.a), scale );
    }
    glEnd();

    glLineWidth(1.6f);
    glBegin(GL_LINE_STRIP);
    for (Line::Data::iterator d = l.data.begin(); d != l.data.end(); ++d )
    {
        float scale = fa.getFrequencyScalar( d->second.hz );
        glColor4f( l.R, l.G, l.B, l.A*d->second.a);
        glVertex3f( d->first, height(d->second.a), scale );
    }
    glEnd();
    glLineWidth(0.5f);

    glPointSize( 6.f );
    glBegin(GL_POINTS);
    for (Line::Data::iterator d = l.data.begin(); d != l.data.end(); ++d )
    {
        float scale = fa.getFrequencyScalar( d->second.hz );
        glVertex3f( d->first, d->second.a, scale );
        glColor4f( l.R, l.G, l.B, min(1.f, l.A*d->second.a*5));
        glVertex3f( d->first, height(d->second.a), scale );
    }
    glEnd();
    glPointSize( 1.f );
    glDepthMask(true);
    GlState::glDisable (GL_BLEND);

    GlException_CHECK_ERROR();
}


void PlotLines::
        recomputeColors()
{
    unsigned N = lines_.size();

    // Set colors
    float i = 0, d;
    for (Lines::iterator itr = lines_.begin(); itr != lines_.end(); itr++)
    {
        QColor c = QColor::fromHsvF( modf(rand_color_offs_ + i++/(float)N, &d), 1, 0.3, 0.5 );
        itr->second.R = c.redF();
        itr->second.G = c.greenF();
        itr->second.B = c.blueF();
        itr->second.A = c.alphaF();
    }

    resetDisplayList();
}

} // namespace Support
} // namespace Tools

#endif // LEGACY_OPENGL
