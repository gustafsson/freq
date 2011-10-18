#include "playbackview.h"

#include "playbackmodel.h"
#include "renderview.h"
#include "selectionmodel.h"
#include "adapters/playback.h"
#include "filters/ellipse.h"
#include "tfr/cwt.h"

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
            just_started( false ),
            _render_view( render_view ),
            _playbackMarker(-1),
            _qtimer( new QTimer )
{
    connect(_qtimer, SIGNAL(timeout()), SLOT(emit_update_view()));
    _qtimer->setSingleShot( true );
}


PlaybackView::
        ~PlaybackView()
{
    delete _qtimer;
}


void PlaybackView::
        update()
{
    float fps = 30;
    if (!_qtimer->isActive())
        _qtimer->start( 1000/fps );
}


void PlaybackView::
        emit_update_view()
{
    emit update_view(false);
    Tfr::Cwt::Singleton().wavelet_time_support( Tfr::Cwt::Singleton().wavelet_default_time_support() );
}


void PlaybackView::
        draw()
{
    locatePlaybackMarker();
    drawPlaybackMarker();
}


void PlaybackView::
        locatePlaybackMarker()
{
    float prev_pos = _playbackMarker;
    _playbackMarker = -1;

    // No selection
    if (0 == model->selection)
        return;

    // No playback instance
    if (!model->playback()) {
        return;
    }

    // Playback has stopped/or hasn't started
    bool is_stopped = model->playback()->isStopped();
    is_stopped &= model->playback()->invalid_samples().empty();
    if (!is_stopped)
        just_started = false;
    is_stopped &= !just_started;
    if (is_stopped && 0>=prev_pos) {
        return;
    }

    bool is_paused = model->playback()->isPaused();
    if (!is_paused) {
        update();
    }

    // Playback has recently stopped
    if (is_stopped) {
        emit playback_stopped();
        return;
    }

    _playbackMarker = model->playback()->time();

    if (follow_play_marker)
    {
        Tools::RenderView& r = *_render_view;
        if (r.model->_qx != _playbackMarker)
        {
            r.model->_qx = _playbackMarker;

            r.userinput_update(true, false);
        }
    }
}



void PlaybackView::
        drawPlaybackMarker()
{
    if (0>_playbackMarker)
        return;

    glPushAttribContext ac;

    glColor4f( 0, 0, 0, .5);

    if (drawPlaybackMarkerInEllipse())
        return;


    glDepthMask(false);

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
            model->selection->current_selection().get() );
    if (!e || model->playbackTarget->post_sink()->filter() != model->selection->current_selection())
        return false;

    glDepthMask(false);

    Tfr::FreqAxis const& fa =
            _render_view->model->display_scale();
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


    glBegin(GL_TRIANGLE_STRIP);
        glVertex3f( t, 0, z1 );
        glVertex3f( t, 0, z2 );
        glVertex3f( t, y, z1 );
        glVertex3f( t, y, z2 );
    glEnd();

    glDepthMask(true);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glBegin(GL_QUADS);
        glVertex3f( t, 0, z1 );
        glVertex3f( t, 0, z2 );
        glVertex3f( t, y, z2 );
        glVertex3f( t, y, z1 );
    glEnd();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    return true;
}


} // namespace Tools
