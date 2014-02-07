#ifndef PLAYBACKVIEW_H
#define PLAYBACKVIEW_H

#include <QObject>

class QTimer;

namespace Tools
{
    class PlaybackModel;
    class RenderView;

    class PlaybackView: public QObject
    {
        Q_OBJECT
    public:
        PlaybackView(PlaybackModel* model, RenderView *render_view);
        ~PlaybackView();

        PlaybackModel* model;

        bool follow_play_marker;
        bool just_started;

        void update();

    signals:
        void update_view();
        void playback_stopped();

    public slots:
        void emit_update_view();
        void draw();
        void locatePlaybackMarker();

    private:
        RenderView *_render_view;
        float _playbackMarker;
        QTimer *_qtimer;

        void drawPlaybackMarker();
        bool drawPlaybackMarkerInEllipse();
    };
} // namespace Tools
#endif // PLAYBACKVIEW_H
