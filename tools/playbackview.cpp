#include "playbackview.h"

#include "playbackmodel.h"
#include "renderview.h"
#include "selectionmodel.h"
#include "adapters/playback.h"
#include "filters/ellipse.h"

#include <glPushContext.h>

// Qt
#include <QTimer>

namespace Tools
{


PlaybackView::
        PlaybackView(PlaybackModel* model, RenderView* render_view)
            :
            model(model),
            follow_play_marker( false ),
            _render_view( render_view ),
            _playbackMarker(-1)
{

}


void PlaybackView::
        update()
{
    emit update_view();
}


void PlaybackView::
        draw()
{
    drawPlaybackMarker();
}


void PlaybackView::
        locatePlaybackMarker()
{
    _playbackMarker = -1;

    // No selection
    if (0 == model->selection)
        return;

    // No playback instance
    if (!model->playback()) {
        return;
    }

    // Playback has stopped
    if (model->playback()->isStopped()) {
        return;
    }

    _playbackMarker = model->playback()->time();

    if (follow_play_marker)
    {
        Tools::RenderView& r = *_render_view;
        r.model->_qx = _playbackMarker;
    }

    update();
}



void PlaybackView::
        drawPlaybackMarker()
{
    if (0>_playbackMarker)
        return;

    if (drawPlaybackMarkerInEllipse())
        return;

    //glEnable(GL_BLEND);
    glDepthMask(false);
    //glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f( 0, 0, 0, .5);

    float
        t = _playbackMarker,
        y = 1,
        z1 = 0,
        z2 = 1;


    glBegin(GL_QUADS);
        glVertex3f( t, 0, z1 );
        glVertex3f( t, 0, z2 );
        glVertex3f( t, y, z2 );
        glVertex3f( t, y, z1 );
    glEnd();

    //glDisable(GL_BLEND);
    glDepthMask(true);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glPolygonOffset(1.f, 1.f);
    glBegin(GL_QUADS);
        glVertex3f( t, 0, z1 );
        glVertex3f( t, 0, z2 );
        glVertex3f( t, y, z2 );
        glVertex3f( t, y, z1 );
    glEnd();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}


bool PlaybackView::
        drawPlaybackMarkerInEllipse()
{
    Filters::Ellipse* e = dynamic_cast<Filters::Ellipse*>(
            model->selection->current_filter_.get() );
    if (!e)
        return false;

    glPushAttribContext ac;
    //glEnable(GL_BLEND);
    glDisable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(false);
    glColor4f( 0, 0, 0, .5);

    Tfr::FreqAxis const& fa =
            _render_view->model->collection->display_scale();
    float
            s1 = fa.getFrequencyScalar( e->_f1 ),
            s2 = fa.getFrequencyScalar( e->_f2 );

    float
        t = _playbackMarker,
        x = e->_t1,
        y = 1,
        z = s1,
        _rx = e->_t2 - e->_t1,
        _rz = s2 - s1,
        z1 = z-sqrtf(1 - (x-t)*(x-t)/_rx/_rx)*_rz,
        z2 = z+sqrtf(1 - (x-t)*(x-t)/_rx/_rx)*_rz;


    glBegin(GL_QUADS);
        glVertex3f( t, 0, z1 );
        glVertex3f( t, 0, z2 );
        glVertex3f( t, y, z2 );
        glVertex3f( t, y, z1 );
    glEnd();

    glDepthMask(true);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glPolygonOffset(1.f, 1.f);
    glBegin(GL_TRIANGLE_STRIP);
        glVertex3f( t, 0, z1 );
        glVertex3f( t, 0, z2 );
        glVertex3f( t, y, z2 );
        glVertex3f( t, y, z1 );
    glEnd();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    return true;
}


} // namespace Tools
