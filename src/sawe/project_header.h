#ifndef PROJECT_HEADER_H
#define PROJECT_HEADER_H

#if defined __cplusplus && !defined __OBJC__

#ifdef _MSC_VER
#include <stdlib.h> //  error C2381: 'exit' : redefinition; __declspec(noreturn) differs
#endif

// OpenGL
#include "gl.h" // from gpumisc
#ifndef __APPLE__
#   include <GL/glut.h>
#else
#   include <GLUT/glut.h>
#endif

// Sonic AWE
#include "heightmap/collection.h"
#include "heightmap/render/renderer.h"
#include "sawe/project.h"
#include "tfr/chunkfilter.h"
#include "tools/rendercontroller.h"
#include "ui/mainwindow.h"


// gpumisc
#include "tasktimer.h"
#include "ThreadChecker.h"
#include "deprecated.h"
#include "exceptionassert.h"
#include "expectexception.h"
#include "msc_stdc.h"
#include "shared_state.h"
#ifdef USE_CUDA
#include "cuda_vector_types_op.h"
#endif


// std
#include <iostream>
#include <list>
#include <math.h>
#include <ostream>
#include <set>
#include <stdarg.h>
#include <string>
#include <vector>


// Qt
#include <QtCore> // QWaitCondition etc
#include <QtGui> // QMouseEvent etc
#include <QtWidgets> // QApplication etc
#include <QtOpenGL> // QGLWidget


// boost
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/exception/all.hpp>
#include <boost/format.hpp>
#include <boost/graph/directed_graph.hpp>
#include <boost/noncopyable.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>
#include <boost/utility.hpp>
#include <boost/weak_ptr.hpp>

#endif // defined __cplusplus && !defined __OBJC__
#endif // PROJECT_HEADER_H
