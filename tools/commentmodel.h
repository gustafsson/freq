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

namespace Tools
{

class CommentModel: public ToolModel
{
public:
    CommentModel();

    Heightmap::Position pos;
    std::string html;
    float scroll_scale;
    bool thumbnail;
    uint2 window_size;
    bool freezed_position;
    bool move_on_hover;
    float2 screen_pos;

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
                & BOOST_SERIALIZATION_NVP(window_size.x)
                & BOOST_SERIALIZATION_NVP(window_size.y)
                & BOOST_SERIALIZATION_NVP(freezed_position);
    }
};

} // namespace Tools

#endif // COMMENTMODEL_H
