#include "filter.h"
#include "tfr/chunk.h"
#include "tfr/transform.h"

using namespace Signal;
using namespace boost;


namespace Tfr {

TransformKernel::
        TransformKernel(Tfr::pTransform transform, pChunkFilter chunk_filter)
    :
      transform_(transform),
      chunk_filter_(chunk_filter)
{}


Signal::pBuffer TransformKernel::
        process(Signal::pBuffer b)
{
    chunk_filter_->set_number_of_channels(b->number_of_channels ());

    pBuffer r;
    for (unsigned c=0; c<b->number_of_channels (); ++c)
      {
        ChunkAndInverse ci;
        ci.channel = c;
        ci.t = transform_;
        ci.input = b->getChannel (c);
        ci.chunk = (*ci.t)( ci.input );

        (*chunk_filter_)( ci );

        bool compute_inverse = 0==dynamic_cast<ChunkFilter::NoInverseTag*>(chunk_filter_.get ());
        if (compute_inverse)
          {
            if (!ci.inverse)
                ci.inverse = ci.t->inverse (ci.chunk);

            if (!r)
                r.reset ( new Buffer(ci.inverse->getInterval (), ci.inverse->sample_rate (), b->number_of_channels ()));

            *r->getChannel (c) |= *ci.inverse;
          }
        else
          {
            // If chunk_filter_ has the NoInverseTag it shouldn't compute the inverse
            EXCEPTION_ASSERTX( 0==ci.inverse.get (), vartype(*chunk_filter_) );

            r.reset ( new Buffer(ci.chunk->getCoveredInterval (), b->sample_rate (), b->number_of_channels ()));
          }

      }

    return r;
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

    if (!f)
        return Signal::Operation::Ptr();

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
