#ifndef SHOWPROCESSING_H
#define SHOWPROCESSING_H

#include <QQuickItem>
#include "squircle.h"
#include "selectionrenderer.h"

/**
 * @brief The ShowProcessing class uses SelectionRenderer to show what is
 * currently being processed.
 */
class ShowProcessing : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(Squircle* heightmap READ heightmap WRITE setHeightmap NOTIFY heightmapChanged)
public:
    explicit ShowProcessing(QQuickItem *parent = 0);

    Squircle* heightmap() const { return heightmap_; }
    void setHeightmap(Squircle*s);

signals:
    void heightmapChanged();

public slots:

private slots:
    void onRendererChanged(SquircleRenderer* renderer);
    void onWindowChanged(QQuickWindow *win);
    void sync();

private:
    Squircle* heightmap_ = 0;
    SelectionRenderer* selection_renderer_1 = 0;
    SelectionRenderer* selection_renderer_2 = 0;
};

#endif // SHOWPROCESSING_H
