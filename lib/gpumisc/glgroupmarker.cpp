#include "glgroupmarker.h"
#include "gl.h"

GlGroupMarker::GlGroupMarker(const char*title)
{
#if GL_EXT_debug_label
#ifdef __APPLE__
    glPushGroupMarkerEXT(0, title);
#endif
#endif
}


GlGroupMarker::~GlGroupMarker()
{
#if GL_EXT_debug_label
#ifdef __APPLE__
    glPopGroupMarkerEXT();
#endif
#endif
}
