#include "playbackview.h"

#include "playbackmodel.h"
#include "renderview.h"
#include "selectionmodel.h"
#include "adapters/playback.h"

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
        r._qx = _playbackMarker;
    }

    update();
}


void PlaybackView::
        drawPlaybackMarker()
{
    if (0>_playbackMarker)
        return;

    //glEnable(GL_BLEND);
    glDepthMask(false);
    //glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f( 0, 0, 0, .5);

    MyVector* selection = model->selection->selection;

    float
        t = _playbackMarker,
        x = selection[0].x,
        y = 1,
        z = selection[0].z,
        _rx = selection[1].x-selection[0].x,
        _rz = selection[1].z-selection[0].z,
        z1 = z-sqrtf(1 - (x-t)*(x-t)/_rx/_rx)*_rz,
        z2 = z+sqrtf(1 - (x-t)*(x-t)/_rx/_rx)*_rz;


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

} // namespace Tools
