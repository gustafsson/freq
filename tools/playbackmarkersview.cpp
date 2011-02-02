#include "playbackmarkersview.h"
#include "glPushContext.h"
#include <boost/foreach.hpp>

namespace Tools {

PlaybackMarkersView::PlaybackMarkersView(PlaybackMarkersModel* model)
    :
    enabled(false),
    model_(model),
    is_adding_marker_(false),
    highlighted_marker_(-1)
{
}


void PlaybackMarkersView::
        setHighlightMarker( float t, bool is_adding_marker )
{
    highlighted_marker_ = t;
    is_adding_marker_ = is_adding_marker;
}


void PlaybackMarkersView::
        draw()
{
    drawMarkers();
}


void PlaybackMarkersView::
        drawMarkers()
{
    float y = 1;

    glPushAttribContext ac;
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(false);
    glColor4f( 0, 0, 0, enabled ? .2f : 0.15f);

    glBegin(GL_QUADS);

    BOOST_FOREACH( const float& m, model_->markers() )
    {
        glVertex3f( m, 0, 0 );
        glVertex3f( m, y, 0 );
        glVertex3f( m, y, 1 );
        glVertex3f( m, 0, 1 );

        if (m == *model_->currentMarker())
        {
            glVertex3f( m, 0, 0 );
            glVertex3f( m, y, 0 );
            glVertex3f( m, y, 1 );
            glVertex3f( m, 0, 1 );
        }

        if ( !is_adding_marker_ && m == highlighted_marker_)
        {
            glVertex3f( m, 0, 0 );
            glVertex3f( m, y, 0 );
            glVertex3f( m, y, 1 );
            glVertex3f( m, 0, 1 );
        }
    }

    if ( is_adding_marker_ && 0 <= highlighted_marker_ )
        // draw adding marker
    {
        glVertex3f( highlighted_marker_, 0, 0 );
        glVertex3f( highlighted_marker_, y, 0 );
        glVertex3f( highlighted_marker_, y, 1 );
        glVertex3f( highlighted_marker_, 0, 1 );
    }

    glEnd();

    glLineWidth(1.6f);
    glPolygonOffset(1.f, 1.f);
    glBegin(GL_LINES);

    BOOST_FOREACH( const float& m, model_->markers() )
    {
        glVertex3f( m, y, 0 );
        glVertex3f( m, y, 1 );

        if (m == *model_->currentMarker())
        {
            glVertex3f( m, y, 0 );
            glVertex3f( m, y, 1 );
        }

        if ( !is_adding_marker_ && m == highlighted_marker_)
        {
            glVertex3f( m, y, 0 );
            glVertex3f( m, y, 1 );
        }
    }

    if ( is_adding_marker_ && 0 <= highlighted_marker_ )
        // draw adding marker
    {
        glVertex3f( highlighted_marker_, y, 0 );
        glVertex3f( highlighted_marker_, y, 1 );
    }

    glEnd();
    glLineWidth(0.5f);
    glDepthMask(true);
}


} // namespace Tools
