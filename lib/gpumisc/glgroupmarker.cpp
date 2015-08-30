#include "glgroupmarker.h"
#include "gl.h"

GlGroupMarker::GlGroupMarker(const char*title)
{
#if GL_EXT_debug_label
    glPushGroupMarkerEXT(0, title);
#endif
}


GlGroupMarker::~GlGroupMarker()
{
#if GL_EXT_debug_label
    glPopGroupMarkerEXT();
#endif
}
