#include "renderview.h"
#include "sawe/project.h"

#include <QVBoxLayout>

namespace Tools
{

RenderView::
        RenderView(RenderModel* model)
            :
            _qx(0), _qy(0), _qz(.5f), // _qz(3.6f/5),
            model(model),
            displayWidget(0)
{
    setLayout( new QHBoxLayout() );

    renderer.reset( new Heightmap::Renderer( model->collection.get() ));
}


RenderView::
        ~RenderView()
{
    emit destroyingRenderView();
}


void RenderView::
        setPosition( float time, float f )
{
    _qx = time;
    _qz = f;

    // todo find length by other means
    float l = model->project->worker.source()->length();

    if (_qx<0) _qx=0;
    if (_qz<0) _qz=0;
    if (_qz>1) _qz=1;
    if (_qx>l) _qx=l;

    // todo isn't requested fps is a renderview property?
    model->project->worker.requested_fps(30);

    update();
}

} // namespace Tools
