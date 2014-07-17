#ifndef RENDERMODEL_H
#define RENDERMODEL_H

#include "tfr/freqaxis.h"
#include "heightmap/amplitudeaxis.h"
#include "heightmap/render/renderblock.h"
#include "heightmap/render/rendersettings.h"
#include "heightmap/tfrmapping.h"
#include "heightmap/tfrmappings/stftblockfilter.h"
#include "heightmap/update/updatequeue.h"
#include "sawe/toolmodel.h"
#include "tfr/transform.h"
#include "support/transformdescs.h"
#include "signal/processing/chain.h"
#include "signal/processing/targetmarker.h"
#include "support/renderoperation.h"

// gpumisc
#include "TAni.h"

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

        void init(Signal::Processing::Chain::ptr chain, Support::RenderOperationDesc::RenderTarget::ptr rt);
        void resetCameraSettings();
        void resetBlockCaches();

        Heightmap::TfrMapping::Collections collections();

        void block_layout(Heightmap::BlockLayout);

        Heightmap::FreqAxis display_scale();
        void display_scale(Heightmap::FreqAxis x);

        Heightmap::AmplitudeAxis amplitude_axis();
        void amplitude_axis(Heightmap::AmplitudeAxis);

        Heightmap::TfrMapping::ptr tfr_mapping();
        Support::TransformDescs::ptr transform_descs();


        Tfr::TransformDesc::ptr transform_desc();
        void set_transform_desc(Tfr::TransformDesc::ptr t);

        void recompute_extent();
        void set_extent(Signal::OperationDesc::Extent extent);

        Signal::OperationDesc::ptr renderOperationDesc();

        Signal::Processing::TargetMarker::ptr target_marker();
        void set_filter(Signal::OperationDesc::ptr o);
        Signal::OperationDesc::ptr get_filter();

        Heightmap::TfrMappings::StftBlockFilterParams::ptr get_stft_block_filter_params();

        Heightmap::Update::UpdateQueue::ptr block_update_queue;

        Heightmap::Render::RenderSettings render_settings;
        Heightmap::Render::RenderBlock::ptr render_block;

        Sawe::Project* project() { return _project; }

        // TODO remove position and use render_settings.camera instead
        void setPosition( Heightmap::Position pos );
        Heightmap::Position position() const;

        float _qx, _qy, _qz; // camera focus point, i.e (10, 0, 0.5)
        float _px, _py, _pz, // camera position relative focus point, i.e (0, 0, -6)
            _rx, _ry, _rz; // rotation around focus point
        float effective_ry(); // take orthoview into account
        float xscale;
        float zscale;
        floatAni orthoview;

    private:
        friend class RenderView; // todo remove
        friend class RenderController; // todo remove
        friend class TimelineController; // todo remove
        Sawe::Project* _project; // project should probably be a member of RenderController instead
        Support::TransformDescs::ptr transform_descs_;
        Heightmap::TfrMapping::ptr tfr_map_;
        Signal::OperationDesc::ptr render_operation_desc_;
        Signal::Processing::TargetMarker::ptr target_marker_;
        Signal::Processing::Chain::ptr chain_;
        Heightmap::TfrMappings::StftBlockFilterParams::ptr stft_block_filter_params_;

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
                    & boost::serialization::make_nvp("color_mode", render_settings.color_mode)
                    & boost::serialization::make_nvp("y_scale", render_settings.y_scale);

            if (typename Archive::is_loading())
                orthoview.reset( _rx >= 90 );

            if (version <= 0)
                ar & boost::serialization::make_nvp("draw_height_lines", render_settings.draw_contour_plot);
            else
                ar & boost::serialization::make_nvp("draw_contour_plot", render_settings.draw_contour_plot);

            ar
                    & boost::serialization::make_nvp("draw_piano", render_settings.draw_piano)
                    & boost::serialization::make_nvp("draw_hz", render_settings.draw_hz)
                    & boost::serialization::make_nvp("left_handed_axes", render_settings.left_handed_axes);

            if (version >= 2)
            {
                float redundancy = render_settings.redundancy;
                ar & BOOST_SERIALIZATION_NVP(redundancy);
                render_settings.redundancy = redundancy;
            }
        }


        void setTestCamera();
    };
} // namespace Tools

BOOST_CLASS_VERSION(Tools::RenderModel, 2)

#endif // RENDERMODEL_H
