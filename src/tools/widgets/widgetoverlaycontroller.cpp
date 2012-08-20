#include "widgetoverlaycontroller.h"

// gpumisc
#include "TaskTimer.h"
#include "demangle.h"

#include <QLayout>

// widgets
#include "rescalewidget.h"
#include "panwidget.h"
#include "rotatewidget.h"

namespace Tools {
namespace Widgets {

WidgetOverlayController::
        WidgetOverlayController(RenderView* view)
    : OverlayWidget(view),
      view_(view)
{
    setupLayout();
}


WidgetOverlayController::
        ~WidgetOverlayController()
{
}


void WidgetOverlayController::
        setupLayout()
{
    setCursor(Qt::CrossCursor);

//    setupLayoutRightAndBottom();
    setupLayoutCenter();
}


void WidgetOverlayController::
        setupLayoutCenter()
{    
    QHBoxLayout* h = new QHBoxLayout();
    h->addStretch();
    h->addWidget(new PanWidget(view_));
    h->addWidget(new RescaleWidget(view_));
    h->addWidget(new RotateWidget(view_));
    h->addStretch();

    QVBoxLayout* v = new QVBoxLayout(this);
    v->addStretch(3);
    v->addLayout(h);
    v->addStretch(1);
}


void WidgetOverlayController::
        setupLayoutRightAndBottom()
{
    QVBoxLayout* v = new QVBoxLayout();
    v->addSpacerItem(new QSpacerItem(50,50,QSizePolicy::Maximum,QSizePolicy::Maximum));
    v->addStretch();
    v->addWidget(new PanWidget(view_));
    v->addWidget(new RescaleWidget(view_));
    v->addWidget(new RotateWidget(view_));
    v->addStretch();

    QHBoxLayout* h = new QHBoxLayout();
    h->addSpacerItem(new QSpacerItem(50,50,QSizePolicy::Maximum,QSizePolicy::Maximum));
    h->addStretch();
    h->addWidget(new PanWidget(view_));
    h->addWidget(new RescaleWidget(view_));
    h->addWidget(new RotateWidget(view_));
    h->addStretch();

    QGridLayout* g = new QGridLayout(this);
    g->addLayout(v,0,1);
    g->addLayout(h,1,0);
}


void WidgetOverlayController::
        updatePosition()
{
    QRect r = sceneRect();
    move(r.topLeft());
    resize(r.size());
}


} // namespace Widgets
} // namespace Tools
