#ifndef COMMENTMODEL_H
#define COMMENTMODEL_H

#include <vector>
#include <string>
#include "heightmap/position.h"
#include "sawe/toolmodel.h"
#include <vector_types.h>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/version.hpp>
#include "tvector.h"

namespace Tools
{

const float UpdateModelPositionFromScreen = -2e30f;
const float UpdateScreenPositionFromWorld = -1e30f;

class CommentModel: public ToolModel
{
public:
    CommentModel();

    Heightmap::Position pos;
    std::string html;
    float scroll_scale;
    bool thumbnail;
    tvector<2, unsigned> window_size;
    bool freezed_position;
    tvector<2, float> screen_pos;

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive& ar, const unsigned int /*version*/)
    {
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ToolModel);
        ar
                & BOOST_SERIALIZATION_NVP(pos.scale)
                & BOOST_SERIALIZATION_NVP(pos.time)
                & BOOST_SERIALIZATION_NVP(html)
                & BOOST_SERIALIZATION_NVP(scroll_scale)
                & BOOST_SERIALIZATION_NVP(thumbnail)
                & BOOST_SERIALIZATION_NVP(window_size[0])
                & BOOST_SERIALIZATION_NVP(window_size[1])
                & BOOST_SERIALIZATION_NVP(freezed_position);
    }
};

} // namespace Tools

#endif // COMMENTMODEL_H
