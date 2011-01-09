#ifndef PLAYBACKVIEW_H
#define PLAYBACKVIEW_H

#include <QObject>

namespace Tools
{
    class PlaybackModel;
    class RenderView;

    class PlaybackView: public QObject
    {
        Q_OBJECT
    public:
        PlaybackView(PlaybackModel* model, RenderView *render_view);

        PlaybackModel* model;

        bool follow_play_marker;

    signals:
        void update_view();

    public slots:
        void update();
        void draw();
        void locatePlaybackMarker();

    private:
        RenderView *_render_view;
        float _playbackMarker;

        void drawPlaybackMarker();
        bool drawPlaybackMarkerInEllipse();
    };
} // namespace Tools
#endif // PLAYBACKVIEW_H
