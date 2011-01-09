#pragma once

// OpenGL
#include <gl.h> // from gpumisc
#ifndef __APPLE__
#   include <GL/glut.h>
#else
#   include <GLUT/glut.h>
#endif

// Sonic AWE
#include "heightmap/collection.h"
#include "heightmap/renderer.h"
#include "sawe/project.h"
#include "tfr/cwtfilter.h"
#include "tfr/stftfilter.h"
#include "tools/rendercontroller.h"
#include "ui/mainwindow.h"

// gpumisc
#include <TaskTimer.h>
#include <cuda_vector_types_op.h>

// Qt
#include <QDockWidget>
#include <QWheelEvent>
#include <QHBoxLayout>

// boost
#include <boost/archive/binary_iarchive.hpp> 
#include <boost/archive/binary_oarchive.hpp> 
#include <boost/serialization/base_object.hpp> 

