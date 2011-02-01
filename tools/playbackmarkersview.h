#ifndef PLAYBACKMARKERSVIEW_H
#define PLAYBACKMARKERSVIEW_H

#include "playbackmarkersmodel.h"

#include <QObject>

namespace Tools {

class PlaybackMarkersView: public QObject
{
    Q_OBJECT
public:
    PlaybackMarkersView(PlaybackMarkersModel* model);

    PlaybackMarkersModel* model() { return model_; }

    void setAddingMarker( float t ) { adding_marker_t_; }
    void setHighlightMarker( Markers::iterator highlighted_marker ) { highlighted_marker_ = highlighted_marker; }

public slots:
    /// Connected in EllipseController
    virtual void draw();

private:
    void drawAddingMarker();
    void drawMarkers();

    PlaybackMarkersModel* model_;
    float adding_marker_t_;
    Markers::iterator highlighted_marker_;
};

} // namespace Tools

#endif // PLAYBACKMARKERSVIEW_H
