#include "plotlines.h"
#include "paintline.h"

#include "tools/rendermodel.h"

#include <glPushContext.h>
#include <GlException.h>

#include <QColor>

namespace Tools {
namespace Support {

PlotLines::
        PlotLines(RenderModel* render_model)
    :
    render_model_(render_model)
{
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
}


void PlotLines::
        set( LineIdentifier id, Time t, float hz, float a )
{
    line(id).data[t] = Value( hz, a );
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
    for (Lines::iterator itr = lines_.begin(); itr != lines_.end(); itr++ )
    {
        draw(itr->second);
    }
}


void PlotLines::
        draw(Line& l)
{
    GlException_CHECK_ERROR();

    glPushAttribContext ac;
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(false);
    glColor4f( l.R, l.G, l.B, l.A);

    glBegin(GL_TRIANGLE_STRIP);
    for (Line::Data::iterator d = l.data.begin(); d != l.data.end(); ++d )
    {
        float scale = render_model_->display_scale().getFrequencyScalar( d->second.hz );
        glVertex3f( d->first, 0, scale );
        glVertex3f( d->first, d->second.a, scale );
    }
    glEnd();

    glLineWidth(1.6f);
    glBegin(GL_LINE_STRIP);
    for (Line::Data::iterator d = l.data.begin(); d != l.data.end(); ++d )
    {
        float scale = render_model_->display_scale().getFrequencyScalar( d->second.hz );
        glVertex3f( d->first, d->second.a, scale );
    }
    glEnd();
    glLineWidth(0.5f);
    glDepthMask(true);

    GlException_CHECK_ERROR();
}


void PlotLines::
        recomputeColors()
{
    float N = lines_.size();

    // Set colors
    float i = 0;
    for (Lines::iterator itr = lines_.begin(); itr != lines_.end(); itr++)
    {
        QColor c = QColor::fromHsvF( i++/N, 1, 1 );
        itr->second.R = c.redF();
        itr->second.G = c.greenF();
        itr->second.B = c.blueF();
        itr->second.A = c.alphaF();
    }
}

} // namespace Support
} // namespace Tools
