#include "toolmodel.h"

namespace Tools
{

    void ToolModel::
            removeFromRepo()
    {
        if (repo_)
            repo_->removeToolModel( this );
        repo_ = 0;
    }


    ToolModelP ToolRepo::
            addModel(ToolModel* model)
    {
        BOOST_ASSERT( model );

        ToolModelP modelp(model);
        tool_models_.insert( modelp );
        model->repo_ = this;

        foreach( ToolControllerP const& p, tool_controllers_)
            p->createView( modelp, this, project_ );

        return modelp;
    }

} // namespace Tools
