#ifndef TFRFILTER_H
#define TFRFILTER_H

#include "sawe/selection.h"
#include "signal/operation.h"
#include "tfr/cwtfilter.h"

#include <list>
#include <boost/shared_ptr.hpp>


namespace Tfr {

class CwtRenderFilter: public CwtFilter
{
public:
    virtual void operator()( Chunk& ) { /* TODO render in heightmap! */ }
};

class SelectionFilter: public CwtFilter
{
public:
    SelectionFilter( Selection s );

    virtual void operator()( Chunk& );
    virtual Signal::Intervals ZeroedSamples( unsigned FS ) const;
    virtual Signal::Intervals NeededSamples( unsigned FS ) const;

    Selection s;

private:
    // TODO Why not copyable?
    SelectionFilter& operator=(const SelectionFilter& );
    SelectionFilter(const SelectionFilter& );
};


class EllipsFilter: public CwtFilter
{
public:
    EllipsFilter(float t1, float f1, float t2, float f2, bool save_inside=false);

    virtual void operator()( Chunk& );
    virtual Signal::Intervals ZeroedSamples( unsigned FS ) const;
    virtual Signal::Intervals NeededSamples( unsigned FS ) const;

    float _t1, _f1, _t2, _f2;
    bool _save_inside;
};


class SquareFilter: public CwtFilter
{
public:
    SquareFilter(float t1, float f1, float t2, float f2, bool save_inside=false);

    virtual void operator()( Chunk& );
    virtual Signal::Intervals ZeroedSamples( unsigned FS ) const;
    virtual Signal::Intervals NeededSamples( unsigned FS ) const;

    float _t1, _f1, _t2, _f2;
    bool _save_inside;
};


class MoveFilter: public CwtFilter
{
public:
    MoveFilter(float df);

    virtual void operator()( Chunk& );
    virtual Signal::Intervals ZeroedSamples( unsigned FS ) const;
    virtual Signal::Intervals NeededSamples( unsigned FS ) const;

    float _df;
};


class ReassignFilter: public CwtFilter
{
public:
    virtual void operator()( Chunk& );
};


class TonalizeFilter: public CwtFilter
{
public:
    virtual void operator()( Chunk& );
};
