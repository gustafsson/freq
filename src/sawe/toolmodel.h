#ifndef TOOLMODEL_H
#define TOOLMODEL_H

#include "TaskTimer.h"
#include "exceptionassert.h"

#include <boost/shared_ptr.hpp>
#include <QWidget>
#include <QPointer>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/set.hpp>

namespace Sawe {
    class Project;
}

namespace Tools
{
    class ToolRepo;

    class ToolMainLoop;
    class RenderView;

    class ToolModel: public QObject
    {
        Q_OBJECT
    public:
        virtual ~ToolModel() {}

        void removeFromRepo();
        void emitModelChanged();

    signals:
        void modelChanged(Tools::ToolModel*);

    private:
        friend class ToolRepo;
        ToolRepo* repo_;

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

        virtual void createView( ToolModelP model, class ToolRepo* repo, Sawe::Project* p ) = 0;
    };
    typedef QPointer<ToolController> ToolControllerP;


    class ToolRepo
    {
    public:
        ToolRepo( Sawe::Project* project ) : project_(project) {}
        virtual ~ToolRepo() {}

        virtual RenderView* render_view() { EXCEPTION_ASSERT(false); return 0; } // can't make pure virtual because of serialization lib
        //virtual ToolMainLoop* mainloop() { EXCEPTION_ASSERT(false); return 0; } // can't make pure virtual because of serialization lib

        template<typename T>
        ToolModelP findToolModel(T* t)
        {
            foreach( const ToolModelP& model, tool_models_)
            {
                T* tst = dynamic_cast<T*>(model.get());
                if (tst == t)
                    return model;
            }
            return ToolModelP();
        }


        template<typename T>
        T* findToolModelType()
        {
            foreach( const ToolModelP& model, tool_models_)
            {
                T* tst = dynamic_cast<T*>(model.get());
                if (tst)
                    return tst;
            }
            return 0;
        }

        template<typename T>
        size_t removeToolModel(T* t)
        {
            ToolModelP p = findToolModel( t );
            return tool_models_.erase( p );
        }


        ToolModelP addModel(ToolModel* model);

    protected:
        std::set< ToolModelP > tool_models_;
        std::vector< ToolControllerP > tool_controllers_;
        Sawe::Project* project_;

        friend class boost::serialization::access;
        ToolRepo() { EXCEPTION_ASSERT( false ); }
        template<class Archive> void serialize(Archive& ar, const unsigned int /*version*/)
        {
            TaskInfo ti("ToolRepo::serialize");
            ar
                    & BOOST_SERIALIZATION_NVP(tool_models_);

            if (typename Archive::is_loading())
            {
                foreach( ToolModelP const& model, tool_models_)
                    model->repo_ = this;

                foreach( ToolControllerP const& p, tool_controllers_)
                    foreach( ToolModelP const& model, tool_models_)
                        p->createView( model, this, project_ );
            }
        }
    };

} // namespace Tools

#endif // TOOLMODEL_H
