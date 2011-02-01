#include "playbackmarkersview.h"

namespace Tools {

PlaybackMarkersView::PlaybackMarkersView(PlaybackMarkersModel* model)
    :
    model_(model)
{
}


void PlaybackMarkersView::
        draw()
{
    drawMarkers();
    drawAddingMarker();
}


void PlaybackMarkersView::
        drawAddingMarker()
{
    if (adding_marker_t_<0)
        return;
}


void PlaybackMarkersView::
        drawMarkers()
{
    float y = 1;

    glPushAttribContext ac;
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(false);
    glColor4f( 0, 0, 0, enabled ? .5 : 0.2);

    glBegin(GL_QUADS);

    BOOST_FOREACH( const float& m, model_->markers() )
    {
            glVertex3f( m, 0, 0 );
            glVertex3f( m, y, 0 );
            glVertex3f( m, y, 1 );
            glVertex3f( m, 0, 1 );
    }

    glEnd();

    glLineWidth(1.6f);
    glPolygonOffset(1.f, 1.f);
    glBegin(GL_LINES);

    BOOST_FOREACH( const float& m, model_->markers() )
    {
        glVertex3f( m, y, 0 );
        glVertex3f( m, y, 1 );
    }

    glEnd();
    glLineWidth(0.5f);
    glDepthMask(true);
}


} // namespace Tools
