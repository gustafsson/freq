#ifndef TOOLMODEL_H
#define TOOLMODEL_H

#include <boost/shared_ptr.hpp>
#include <QWidget>

#include <boost/serialization/serialization.hpp>

namespace Sawe {
    class Project;
}

namespace Tools
{
    class RenderView;


    class ToolModel
    {
    public:
        virtual ~ToolModel() {}

		friend class boost::serialization::access;
        template<class Archive> void serialize(Archive& /*ar*/, const unsigned int /*version*/)
		{
			/* nothing to see here, move on */
		}
    };
    typedef boost::shared_ptr<ToolModel> ToolModelP;


	class ToolController: public QWidget
    {
    public:
        virtual ~ToolController() {}

        virtual void createView( ToolModel* model, Sawe::Project* p, RenderView* r ) = 0;
    };


} // namespace Tools

#endif // TOOLMODEL_H
