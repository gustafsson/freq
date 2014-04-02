#include "playbackmarkersview.h"

#include "sawe/project.h"
#include "support/toolglbrush.h"

#include <boost/foreach.hpp>

namespace Tools {

PlaybackMarkersView::PlaybackMarkersView(PlaybackMarkersModel* model, Sawe::Project* project)
    :
    enabled(false),
    project_(project),
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

    Support::ToolGlBrush tgb(enabled);

    glBegin(GL_QUADS);

    BOOST_FOREACH( const float& m, model_->markers() )
    {
        glVertex3f( m, 0, 0 );
        glVertex3f( m, y, 0 );
        glVertex3f( m, y, 1 );
        glVertex3f( m, 0, 1 );

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

    PlaybackMarkersModel::Markers::iterator itr = model_->currentMarker();
    float a = *itr, b;
    itr++;
    if ( itr != model_->markers().end() )
        b = *itr;
    else {
//        b = project_->worker.length();
        Signal::OperationDesc::Extent x = project_->processing_chain ().read ()->extent(project_->default_target ());
        b = x.interval.get_value_or (Signal::Interval()).count();
    }

    glVertex3f( a, 0, -.1 );
    glVertex3f( a, y, -.1 );
    glVertex3f( a, y, 1.1 );
    glVertex3f( a, 0, 1.1 );

    glVertex3f( a, 0, -.1 );
    glVertex3f( a, y, -.1 );
    glVertex3f( b, y, -.1 );
    glVertex3f( b, 0, -.1 );

    glVertex3f( a, y, 1.1 );
    glVertex3f( a, 0, 1.1 );
    glVertex3f( b, 0, 1.1 );
    glVertex3f( b, y, 1.1 );

    glVertex3f( b, 0, -.1 );
    glVertex3f( b, y, -.1 );
    glVertex3f( b, y, 1.1 );
    glVertex3f( b, 0, 1.1 );

    glEnd();

    glLineWidth(1.6f);
    glBegin(GL_LINES);

    BOOST_FOREACH( const float& m, model_->markers() )
    {
        glVertex3f( m, y, 0 );
        glVertex3f( m, y, 1 );

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

    glVertex3f( a, y, -.1 );
    glVertex3f( b, y, -.1 );

    glVertex3f( a, y, 1.1 );
    glVertex3f( b, y, 1.1 );

    glVertex3f( a, y, -.1 );
    glVertex3f( a, y, 1.1 );

    glVertex3f( b, y, -.1 );
    glVertex3f( b, y, 1.1 );

    glEnd();
    glLineWidth(0.5f);
}


} // namespace Tools
