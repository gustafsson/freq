#ifndef TOOLTIPMODEL_H
#define TOOLTIPMODEL_H

// Tools
#include "commentview.h"

// Sonic AWE
#include "heightmap/reference.h"
#include "sawe/project.h"

// Qt
#include <QPointer>

// boost serialization
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/version.hpp>
#include <boost/mpl/bool.hpp>

namespace Tools {
class CommentController;

class TooltipModel: public ToolModel
{
public:
    TooltipModel();
    void setPtrs(RenderView *render_view, CommentController* comments);

    const Heightmap::Position& comment_pos();

    void showToolTip( Heightmap::Position p );
    unsigned guessHarmonicNumber( const Heightmap::Position& pos );
    float computeMarkerMeasure(const Heightmap::Position& pos, unsigned i, Heightmap::Reference* ref=0);

    Heightmap::Position pos;
    float frequency;
    float max_so_far;
    float compliance;
    unsigned markers;
    unsigned markers_auto;
    QPointer<CommentView> comment;
    enum AutoMarkersState
    {
        ManualMarkers,
        AutoMarkerWorking,
        AutoMarkerFinished
    } automarking;
    std::string automarkingStr();

    void toneName(std::string& primaryName, std::string& secondaryName, float& accuracy);
    std::string toneName();

private:
    CommentController* comments_;
    RenderView *render_view_;
    unsigned fetched_heightmap_values;
    ToolModelP comment_model;

private:
    //template <bool T> bool eval_boolean(boost::mpl::bool_<T> const&) { return T; }
    //template <typename T> bool eval_boolean2(T const&) { return true; }

    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive& ar, const unsigned int /*version*/)
    {
        bool is_saving = typename Archive::is_saving();
        TaskInfo("%s is_saving=%d comment_model=%p", __FUNCTION__, is_saving, comment_model.get());
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ToolModel);
        ar
                & BOOST_SERIALIZATION_NVP(pos.scale)
                & BOOST_SERIALIZATION_NVP(pos.time)
                & BOOST_SERIALIZATION_NVP(frequency)
                & BOOST_SERIALIZATION_NVP(max_so_far)
                & BOOST_SERIALIZATION_NVP(compliance)
                & BOOST_SERIALIZATION_NVP(markers)
                & BOOST_SERIALIZATION_NVP(markers_auto)
                & BOOST_SERIALIZATION_NVP(automarking)
                & BOOST_SERIALIZATION_NVP(comment_model);
    }
};

} // namespace Tools

#endif // TOOLTIPMODEL_H
