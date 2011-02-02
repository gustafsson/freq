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

    bool enabled;
    void setHighlightMarker( float t, bool is_adding_marker );

public slots:
    /// Connected in EllipseController
    virtual void draw();

private:
    void drawMarkers();

    PlaybackMarkersModel* model_;
    bool is_adding_marker_;
    float highlighted_marker_;
};

} // namespace Tools

#endif // PLAYBACKMARKERSVIEW_H
