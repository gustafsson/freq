#include "showprocessing.h"
#include "signal/processing/targetneeds.h"

#include <QQuickWindow>

ShowProcessing::ShowProcessing(QQuickItem *parent) :
    QQuickItem(parent)
{
    connect (this, SIGNAL(windowChanged(QQuickWindow*)), SLOT(onWindowChanged(QQuickWindow*)));
}


void ShowProcessing::
        setHeightmap(Squircle*s)
{
    if (heightmap_) disconnect(heightmap_, SIGNAL(rendererChanged(SquircleRenderer*)), this, SLOT(onRendererChanged(SquircleRenderer*)));

    heightmap_ = s;
    emit heightmapChanged ();
    connect(s, SIGNAL(rendererChanged(SquircleRenderer*)), SLOT(onRendererChanged(SquircleRenderer*)), Qt::DirectConnection);
}


void ShowProcessing::
        onRendererChanged(SquircleRenderer* renderer)
{
#ifdef _DEBUG
    // SquircleRenderer owns selection_renderer_
    selection_renderer_1 = new SelectionRenderer(renderer);
    selection_renderer_1->setRgba (0.0, 0.0, 0.0, 0.1);

    selection_renderer_2 = new SelectionRenderer(renderer);
    selection_renderer_2->setRgba (0.0, 0.0, 0.0, 0.05);
#endif
}


void ShowProcessing::
        onWindowChanged(QQuickWindow *win)
{
    if (win)
        connect(win, SIGNAL(beforeSynchronizing()), SLOT(sync()), Qt::DirectConnection);
}


void ShowProcessing::
        sync()
{
    if (!selection_renderer_1)
        return;
    auto target = heightmap_->renderModel ()->target_marker ();
    if (!target)
        return;

    auto needs = target->target_needs();
    Signal::Intervals out_of_date = needs->out_of_date();
    Signal::Intervals not_started = needs->not_started();

    Signal::Intervals currently_processing = out_of_date - not_started;

    selection_renderer_1->setSelection (not_started);
    selection_renderer_2->setSelection (currently_processing);
}
