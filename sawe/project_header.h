#pragma once

#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>
#endif

#ifndef __APPLE__
 #include "GL/glew.h"
 #include <GL/glut.h>
#else
  #include "OpenGL/glu.h"
  #include <GLUT/glut.h>
#endif

#include "heightmap/collection.h"
#include "heightmap/renderer.h"
#include "sawe/project.h"
#include "tfr/cwtfilter.h"
#include "tfr/stftfilter.h"
#include "tools/rendercontroller.h"
#include <TaskTimer.h>

#include "ui/mainwindow.h"
#include <cuda_vector_types_op.h>
#include <QDockWidget>
#include <QWheelEvent>
#include <QHBoxLayout>
