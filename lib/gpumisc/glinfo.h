#ifndef GLINFO_H
#define GLINFO_H

#include <string>

/**
 * @brief The glinfo class should provide a human readable text string of a Qt managed open gl render context.
 */
class glinfo
{
public:
    static std::string pretty_format(const class QGLFormat&);
    static std::string pretty_format(const class QGLWidget&);
    static std::string driver_info();

    static void test();
};

#endif // GLINFO_H
