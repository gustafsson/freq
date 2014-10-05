#ifndef RENDERMODEL_H
#define RENDERMODEL_H

#include "heightmap/amplitudeaxis.h"
#include "heightmap/render/renderblock.h"
#include "heightmap/render/rendersettings.h"
#include "heightmap/tfrmapping.h"
#include "heightmap/tfrmappings/stftblockfilter.h"
#include "heightmap/update/updatequeue.h"
#include "signal/processing/chain.h"
#include "signal/processing/targetmarker.h"
#include "tfr/transform.h"
#include "tfr/freqaxis.h"

#include "support/transformdescs.h"
#include "support/rendercamera.h"
#include "support/renderoperation.h"

// gpumisc
#include "TAni.h"
#include "tasktimer.h"

// boost
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/version.hpp>

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
    class RenderModel
    {
    public:
        RenderModel();
        ~RenderModel();

        void init(Signal::Processing::Chain::ptr chain, Support::RenderOperationDesc::RenderTarget::ptr rt, Signal::Processing::TargetMarker::ptr target_marker = Signal::Processing::TargetMarker::ptr());
        void resetCameraSettings();
        void resetBlockCaches();

        Heightmap::TfrMapping::Collections collections() const;
        Signal::Processing::Chain::ptr chain();

        void block_layout(Heightmap::BlockLayout);

        Heightmap::FreqAxis display_scale();
        void display_scale(Heightmap::FreqAxis x);

        Heightmap::AmplitudeAxis amplitude_axis();
        void amplitude_axis(Heightmap::AmplitudeAxis);

        Heightmap::TfrMapping::ptr tfr_mapping() const;
        Support::TransformDescs::ptr transform_descs() const;

        Tfr::TransformDesc::ptr transform_desc() const;
        void set_transform_desc(Tfr::TransformDesc::ptr t);

        Signal::OperationDesc::Extent recompute_extent();
        void set_extent(Signal::OperationDesc::Extent extent);

        Signal::OperationDesc::ptr renderOperationDesc();

        Signal::Processing::TargetMarker::ptr target_marker();
        void set_filter(Signal::OperationDesc::ptr o);
        Signal::OperationDesc::ptr get_filter();

        Heightmap::TfrMappings::StftBlockFilterParams::ptr get_stft_block_filter_params();

        Heightmap::Update::UpdateQueue::ptr block_update_queue;

        Heightmap::Render::RenderSettings render_settings;
        Heightmap::Render::RenderBlock::ptr render_block;
        shared_state<Tools::Support::RenderCamera> camera;
        shared_state<glProjection> gl_projection;

        void setPosition( Heightmap::Position pos );
        Heightmap::Position position() const;

    private:
        friend class RenderView; // todo remove
        friend class RenderController; // todo remove
        friend class TimelineController; // todo remove
        Support::TransformDescs::ptr transform_descs_;
        Heightmap::TfrMapping::ptr tfr_map_;
        Signal::OperationDesc::ptr render_operation_desc_;
        Signal::Processing::TargetMarker::ptr target_marker_;
        Signal::Processing::Chain::ptr chain_;
        Heightmap::TfrMappings::StftBlockFilterParams::ptr stft_block_filter_params_;

        friend class boost::serialization::access;
        template<class Archive> void serialize(Archive& ar, const unsigned int version) {
            TaskInfo ti("RenderModel::serialize");
            auto camera = this->camera.write ();
            float _qx = camera->q[0],
                  _qy = camera->q[1],
                  _qz = camera->q[2];
            float _px  = camera->p[0],
                  _py = camera->p[1],
                  _pz = camera->p[2],
                _rx = camera->r[0],
                _ry = camera->r[1],
                _rz = camera->r[2];
            float xscale = camera->xscale;
            float zscale = camera->zscale;

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
                camera->orthoview.reset( _rx >= 90 );

            camera->q[0] = _qx;
            camera->q[1] = _qy;
            camera->q[2] = _qz;
            camera->p[0] = _px;
            camera->p[1] = _py;
            camera->p[2] = _pz;
            camera->r[0] = _rx;
            camera->r[1] = _ry;
            camera->r[2] = _rz;
            camera->xscale = xscale;
            camera->zscale = zscale;

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
