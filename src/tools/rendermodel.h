#ifndef RENDERMODEL_H
#define RENDERMODEL_H

#include "tfr/freqaxis.h"
#include "signal/target.h"
#include "heightmap/amplitudeaxis.h"
#include "heightmap/renderer.h"
#include "heightmap/tfrmapping.h"
#include "sawe/toolmodel.h"
#include "tfr/transform.h"
#include "support/transformdescs.h"
#include "signal/processing/chain.h"
#include "signal/processing/targetmarker.h"
#include "support/renderoperation.h"

// gpumisc
#include <TAni.h>

// boost
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/version.hpp>

namespace Sawe {
    class Project;
}

namespace Heightmap
{
    class Collection;
}

namespace Tfr { class Filter; }

namespace Tools
{
    /**
     * @brief The RenderModel class
     *
     * TODO call set_extent when it's changed
     */
    class RenderModel: public ToolModel
    {
    public:
        RenderModel(Sawe::Project* p);
        ~RenderModel();

        void init(Signal::Processing::Chain::Ptr chain, Support::RenderOperationDesc::RenderTarget::Ptr rt);
        void resetSettings();

        Heightmap::TfrMap::Collections collections();

        void block_size(Heightmap::BlockSize);

        Tfr::FreqAxis display_scale();
        void display_scale(Tfr::FreqAxis x);

        Heightmap::AmplitudeAxis amplitude_axis();
        void amplitude_axis(Heightmap::AmplitudeAxis);

        Heightmap::TfrMapping tfr_mapping();
        Heightmap::TfrMap::Ptr tfr_map();
        Support::TransformDescs::Ptr transform_descs();


        Tfr::Filter* block_filter();

        Tfr::TransformDesc::Ptr transform_desc();
        void set_transform_desc(Tfr::TransformDesc::Ptr t);

        void recompute_extent();

        Signal::OperationDesc::Ptr renderOperationDesc();

        Signal::Processing::TargetMarker::Ptr target_marker();
        void set_filter(Signal::OperationDesc::Ptr o);
        Signal::OperationDesc::Ptr get_filter();

        //Signal::pTarget renderSignalTarget;
        boost::shared_ptr<Heightmap::Renderer> renderer;

        Sawe::Project* project() { return _project; }

        // TODO remove position and use renderer->camera instead
        float _qx, _qy, _qz; // camera focus point, i.e (10, 0, 0.5)
        float _px, _py, _pz, // camera position relative center, i.e (0, 0, -6)
            _rx, _ry, _rz; // rotation around center
        float effective_ry(); // take orthoview into account
        float xscale;
        float zscale;
        floatAni orthoview;

    private:
        friend class RenderView; // todo remove
        friend class RenderController; // todo remove
        friend class TimelineController; // todo remove
        Sawe::Project* _project; // project should probably be a member of RenderController instead
        Support::TransformDescs::Ptr transform_descs_;
        Heightmap::TfrMap::Ptr tfr_map_;
        Signal::OperationDesc::Ptr render_operation_desc_;
        Signal::Processing::TargetMarker::Ptr target_marker_;
        Signal::Processing::Chain::Ptr chain_;

        friend class boost::serialization::access;
        RenderModel() { EXCEPTION_ASSERT( false ); } // required for serialization to compile, is never called
        template<class Archive> void serialize(Archive& ar, const unsigned int version) {
            TaskInfo ti("RenderModel::serialize");
            ar
                    & BOOST_SERIALIZATION_NVP(_qx)
                    & BOOST_SERIALIZATION_NVP(_qy)
                    & BOOST_SERIALIZATION_NVP(_qz)
                    & BOOST_SERIALIZATION_NVP(_px)
                    & BOOST_SERIALIZATION_NVP(_py)
                    & BOOST_SERIALIZATION_NVP(_pz)
                    & BOOST_SERIALIZATION_NVP(_rx)
                    & BOOST_SERIALIZATION_NVP(_ry)
                    & BOOST_SERIALIZATION_NVP(_rz)
                    & BOOST_SERIALIZATION_NVP(xscale)
                    & BOOST_SERIALIZATION_NVP(zscale)
                    & boost::serialization::make_nvp("color_mode", renderer->color_mode)
                    & boost::serialization::make_nvp("y_scale", renderer->y_scale);

            if (typename Archive::is_loading())
                orthoview.reset( _rx >= 90 );

            if (version <= 0)
                ar & boost::serialization::make_nvp("draw_height_lines", renderer->draw_contour_plot);
            else
                ar & boost::serialization::make_nvp("draw_contour_plot", renderer->draw_contour_plot);

            ar
                    & boost::serialization::make_nvp("draw_piano", renderer->draw_piano)
                    & boost::serialization::make_nvp("draw_hz", renderer->draw_hz)
                    & boost::serialization::make_nvp("left_handed_axes", renderer->left_handed_axes);

            if (version >= 2)
            {
                float redundancy = renderer->redundancy();
                ar & BOOST_SERIALIZATION_NVP(redundancy);
                renderer->redundancy(redundancy);
            }
        }


        void setTestCamera();
    };
} // namespace Tools

BOOST_CLASS_VERSION(Tools::RenderModel, 2)

#endif // RENDERMODEL_H
