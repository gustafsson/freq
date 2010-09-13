#include "renderview.h"

namespace Tools
{

RenderView::
        RenderView(RenderModel* model)
            :
            model(model),
            _qx(0), _qy(0), _qz(.5f) // _qz(3.6f/5),
{}

void RenderView::
        setPosition( float time, float f )
{
    _qx = time;
    _qz = f;

    float l = model->_project->worker->source()->length();

    if (_qx<0) _qx=0;
    if (_qz<0) _qz=0;
    if (_qz>1) _qz=1;
    if (_qx>l) _qx=l;

    model->_project->worker->requested_fps(30);
//    update();
}

} // namespace Tools
