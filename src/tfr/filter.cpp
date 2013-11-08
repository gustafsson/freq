#include "filter.h"
#include "signal/buffersource.h"
#include "tfr/chunk.h"
#include "tfr/transform.h"

#include "demangle.h"

#include <boost/format.hpp>

#include <QMutexLocker>

//#define TIME_Filter
#define TIME_Filter if(0)

//#define TIME_FilterReturn
#define TIME_FilterReturn if(0)


using namespace Signal;
using namespace boost;


namespace Tfr {

//////////// Filter
Filter::
        Filter( pOperation source )
            :
            DeprecatedOperation( source )
{}


Filter::
        Filter(Filter& f)
    :
      DeprecatedOperation(f)
{
    transform(f.transform ());
}


Signal::pBuffer Filter::
        read(  const Signal::Interval& I )
{
    TIME_Filter TaskTimer tt("%s Filter::read( %s )", vartype(*this).c_str(),
                             I.toString().c_str());

    QMutexLocker l(&_transform_mutex);
    pTransform t = _transform;
    l.unlock ();

    Signal::Interval required = requiredInterval(I, t);

    // If no samples would be non-zero, return zeros
    if (!(required - zeroed_samples_recursive()))
        return zeros(required);

    pBuffer b = source()->readFixedLength( required );
    if (this != affecting_source(required))
        return b;

    return process(b);
}


Signal::Interval Filter::
        requiredInterval( const Signal::Interval& I )
{
    return requiredInterval(I, transform());
}


Signal::pBuffer Filter::
        process(Signal::pBuffer b)
{
    pTransform t = _transform;
#ifdef _DEBUG
    const Interval I = b->getInterval ();
#endif

    pBuffer r;
    for (unsigned c=0; c<b->number_of_channels (); ++c)
    {
        ChunkAndInverse ci;
        ci.channel = c;
        ci.t = t;
        ci.inverse = b->getChannel (c);

        ci.chunk = (*t)( ci.inverse );

        #ifdef _DEBUG
            Interval cii = ci.chunk->getInterval().spanned ( ci.chunk->getCoveredInterval () );

            EXCEPTION_ASSERTX( cii & I, boost::format("cii = %s, I = %s") % cii % I);
        #endif

        bool applied_filter = (*this)( ci );
        if (applied_filter)
            ci.inverse = t->inverse (ci.chunk);

        if (!r)
            r.reset ( new Buffer(ci.inverse->getInterval (), ci.inverse->sample_rate (), b->number_of_channels ()));

        #ifdef _DEBUG
            Interval invinterval = ci.inverse->getInterval ();
            Interval i(I.first, I.first+1);
            if (!( i & invinterval ) && !applied_filter)
            {
                Signal::Interval required2 __attribute__ ((unused)) = requiredInterval(I, t);
                Interval cgi2 __attribute__ ((unused)) = ci.chunk->getInterval ();
                ci.inverse = b->getChannel (c);
                ci.chunk = (*t)( ci.inverse );
                ci.inverse = t->inverse (ci.chunk);
                EXCEPTION_ASSERT( i & invinterval );
            }
        #endif

        *r->getChannel (c) |= *ci.inverse;
    }

    return r;
}


DeprecatedOperation* Filter::
        affecting_source( const Interval& I )
{
    return DeprecatedOperation::affecting_source( I );
}


unsigned Filter::
        prev_good_size( unsigned current_valid_samples_per_chunk )
{
    return transform()->transformDesc()->prev_good_size( current_valid_samples_per_chunk, sample_rate() );
}


unsigned Filter::
        next_good_size( unsigned current_valid_samples_per_chunk )
{
    return transform()->transformDesc()->next_good_size( current_valid_samples_per_chunk, sample_rate() );
}


bool Filter::
        applyFilter( ChunkAndInverse& chunk )
{
    return (*this)( *chunk.chunk );
}


bool Filter::
        operator()( ChunkAndInverse& chunk )
{
    return applyFilter( chunk );
}


Tfr::pTransform Filter::
        transform()
{
    QMutexLocker l(&_transform_mutex);
    return _transform;
}


void Filter::
        transform( Tfr::pTransform t )
{
    QMutexLocker l(&_transform_mutex);

    if (_transform)
    {
        EXCEPTION_ASSERTX(typeid(*_transform) == typeid(*t), str(format("'transform' must be an instance of %s, was %s") % vartype(*_transform) % vartype(*t)));
    }

    if (_transform == t )
        return;

    _transform = t;

    l.unlock ();

    invalidate_samples( getInterval() );
}


TransformKernel::
        TransformKernel(Tfr::pTransform transform, pChunkFilter chunk_filter)
    :
      transform_(transform),
      chunk_filter_(chunk_filter)
{}


Signal::pBuffer TransformKernel::
        process(Signal::pBuffer b)
{
    pTransform t = transform_;

    pBuffer r;
    for (unsigned c=0; c<b->number_of_channels (); ++c)
      {
        ChunkAndInverse ci;
        ci.channel = c;
        ci.t = t;
        ci.inverse = b->getChannel (c);

        ci.chunk = (*t)( ci.inverse );

        bool compute_inverse = (*chunk_filter_)( ci );
        if (compute_inverse)
          {
            ci.inverse = t->inverse (ci.chunk);

            if (!r)
                r.reset ( new Buffer(ci.inverse->getInterval (), ci.inverse->sample_rate (), b->number_of_channels ()));

            *r->getChannel (c) |= *ci.inverse;
          }
        else
          {
            r.reset ( new Buffer(ci.chunk->getCoveredInterval (), b->sample_rate (), b->number_of_channels ()));
          }

      }

    return r;
}


Tfr::pTransform TransformKernel::
        transform()
{
    return transform_;
}


pChunkFilter TransformKernel::
        chunk_filter()
{
    return chunk_filter_;
}

FilterDesc::
        FilterDesc(Tfr::pTransformDesc d, FilterKernelDesc::Ptr f)
    :
      transform_desc_(d),
      chunk_filter_(f)
{

}


OperationDesc::Ptr FilterDesc::
        copy() const
{
    return OperationDesc::Ptr (new FilterDesc (this->transform_desc_, chunk_filter_));
}


Signal::Operation::Ptr FilterDesc::
        createOperation(Signal::ComputingEngine*engine) const
{
    Tfr::pTransform t = transform_desc_->createTransform ();
    pChunkFilter f = read1(chunk_filter_)->createChunkFilter (engine);
    return Signal::Operation::Ptr (new TransformKernel( t, f ));
}


Signal::Interval FilterDesc::
        requiredInterval(const Signal::Interval& I, Signal::Interval* expectedOutput) const
{
    return transform_desc_->requiredInterval (I, expectedOutput);
}


Signal::Interval FilterDesc::
        affectedInterval(const Signal::Interval& I) const
{
    return transform_desc_->affectedInterval (I);
}


QString FilterDesc::
        toString() const
{
    return (vartype(*chunk_filter_) + " on " + transform_desc_->toString ()).c_str();
}


bool FilterDesc::
        operator==(const Signal::OperationDesc&d) const
{
    if (const FilterDesc* f = dynamic_cast<const FilterDesc*>(&d))
    {
        const TransformDesc& a = *transform_desc_;
        const TransformDesc& b = *f->transformDesc ();
        return a == b;
       // return *f->transformDesc () == *transform_desc_;
    }
    return false;
}


Tfr::pTransformDesc FilterDesc::
        transformDesc() const
{
    return transform_desc_;
}

} // namespace Tfr
