#include "playbackview.h"

#include "playbackmodel.h"
#include "renderview.h"
#include "selectionmodel.h"
#include "adapters/playback.h"
#include "filters/ellipse.h"
#include "tfr/cwt.h"

#include "glPushContext.h"

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
    emit update_view();
    auto td = _render_view->model->transform_descs ().write ();
    Tfr::Cwt& cwt = td->getParam<Tfr::Cwt>();
    cwt.wavelet_time_support( cwt.wavelet_default_time_support() );
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

    Adapters::Playback* playback = dynamic_cast<Adapters::Playback*>(model->playback().get ());

    // Playback has stopped/or hasn't started
    bool is_stopped = playback->isStopped();
    is_stopped &= playback->invalid_samples().empty();
    if (!is_stopped)
        just_started = false;
    is_stopped &= !just_started;
    if (is_stopped && 0>=prev_pos) {
        return;
    }

    bool is_paused = playback->isPaused();
    if (!is_paused) {
        update();
    }

    is_stopped |= playback->hasReachedEnd ();
    // Playback has recently stopped
    if (is_stopped) {
        emit playback_stopped();
        return;
    }

    _playbackMarker = playback->time();
    if (_playbackMarker<=0)
        _playbackMarker = -1;

    if (follow_play_marker && 0<_playbackMarker)
    {
        Tools::RenderView& r = *_render_view;
        if (r.model->camera.q[0] != _playbackMarker)
        {
            r.model->camera.q[0] = _playbackMarker;

            r.redraw();
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
    auto current_selection = model->selection->current_selection();
    if (!current_selection)
        return false;

    auto w = current_selection.write ();
    Filters::Ellipse* e = dynamic_cast<Filters::Ellipse*>( w.get() );
//Use Signal::Processing namespace
//    if (!e || model->playbackTarget->post_sink()->filter() != model->selection->current_selection())
    if (!e)
        return false;

    glDepthMask(false);

    Heightmap::FreqAxis const& fa =
            _render_view->model->display_scale();
    float
            s1 = fa.getFrequencyScalar( e->_centre_f ),
            s2 = fa.getFrequencyScalar( e->_centre_plus_radius_f );

    float
        t = _playbackMarker,
        x = e->_centre_t,
        y = 1,
        z = s1,
        _rx = e->_centre_plus_radius_t - e->_centre_t,
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
