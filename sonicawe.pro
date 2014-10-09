TEMPLATE = subdirs
CONFIG += ordered

SUBDIRS = \
    lib/justmisc \
    lib/backtrace \
    lib/gpumisc \
    lib/signal \
    lib/tfr \
    lib/filters \
    lib/heightmap \
    lib/tfrheightmap \
    lib/heightmapview \
    src

cache()
