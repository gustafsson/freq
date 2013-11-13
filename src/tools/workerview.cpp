#include "workerview.h"
#include "sawe/project.h"

#include "support/paintline.h"

namespace Tools {

WorkerView::
        WorkerView(Sawe::Project* project)
            :
    project_(project)
{
}


Sawe::Project* WorkerView::
        project()
{
    return project_;
}


void WorkerView::
        draw()
{
/*
//Use Signal::Processing namespace
    Signal::Intervals I = project_->worker.todo_list();
    float FS = project_->head->head_source()->sample_rate();

    std::vector<Heightmap::Position> pts(2);

    for (Signal::Intervals::const_iterator itr = I.begin(); itr != I.end(); ++itr)
    {
        pts[0].time = itr->first/FS;
        pts[1].time = itr->last/FS;
        pts[0].scale = 0.97;
        pts[1].scale = 0.97;
        Support::PaintLine::drawSlice( 2, &pts[0], 1,0,0 );
    }

    pts[0].time = project_->worker.latest_request.first/FS;
    pts[1].time = project_->worker.latest_request.last/FS;
    pts[0].scale = 0.95;
    pts[1].scale = 0.95;
    Support::PaintLine::drawSlice( 2, &pts[0], 0,0,1 );

    pts[0].time = project_->worker.latest_result.first/FS;
    pts[1].time = project_->worker.latest_result.last/FS;
    pts[0].scale = 0.93;
    pts[1].scale = 0.93;
    Support::PaintLine::drawSlice( 2, &pts[0], 0,1,0 );
*/
}

} // namespace Tools
