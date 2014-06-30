#include "transformoperation.h"

#include "demangle.h"

#include "tfr/chunk.h"
#include "tfr/chunkfilter.h"
#include "tfr/transform.h"

using namespace Signal;
using namespace boost;


namespace Tfr {

class TransformOperationOperation: public Operation
{
public:
    TransformOperationOperation(pTransform t, pChunkFilter chunk_filter, bool no_inverse_tag);

    // Operation
    pBuffer process(pBuffer b);

private:
    pTransform transform_;
    pChunkFilter chunk_filter_;
    bool no_inverse_tag_;
};


TransformOperationOperation::
        TransformOperationOperation(Tfr::pTransform transform, pChunkFilter chunk_filter, bool no_inverse_tag)
    :
      transform_(transform),
      chunk_filter_(chunk_filter),
      no_inverse_tag_(no_inverse_tag)
{
}


Signal::pBuffer TransformOperationOperation::
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

        if (!no_inverse_tag_)
          {
            if (!ci.inverse)
                ci.inverse = ci.t->inverse (ci.chunk);

            if (!r)
                r.reset ( new Signal::Buffer(ci.inverse->getInterval (), ci.inverse->sample_rate (), b->number_of_channels ()));

            *r->getChannel (c) |= *ci.inverse;
          }
        else
          {
            // If chunk_filter_ has the NoInverseTag it shouldn't compute the inverse
            EXCEPTION_ASSERTX( !ci.inverse, vartype(*chunk_filter_) );

            r.reset ( new Signal::Buffer(ci.chunk->getCoveredInterval (), b->sample_rate (), b->number_of_channels ()));
          }

      }

    return r;
}


TransformOperationDesc::
        TransformOperationDesc(ChunkFilterDesc::ptr f)
    :
      chunk_filter_(f),
      transformDesc_(f.read ()->transformDesc()->copy())
{
}


OperationDesc::ptr TransformOperationDesc::
        copy() const
{
    //ChunkFilterDesc::Ptr chunk_filter = chunk_filter_.read ()->copy();
    return OperationDesc::ptr (new TransformOperationDesc (chunk_filter_));
}


Signal::Operation::ptr TransformOperationDesc::
        createOperation(Signal::ComputingEngine*engine) const
{
    Tfr::pTransform t = transformDesc_->createTransform ();
    pChunkFilter f;
    {
        auto c = chunk_filter_.write ();
        c->transformDesc (transformDesc_);
        f = c->createChunkFilter (engine);
    }
    bool no_inverse_tag = 0!=dynamic_cast<volatile ChunkFilter::NoInverseTag*>(f.get ());

    if (!f)
        return Signal::Operation::ptr();

    return Signal::Operation::ptr (new TransformOperationOperation( t, f, no_inverse_tag ));
}


Signal::Interval TransformOperationDesc::
        requiredInterval(const Signal::Interval& I, Signal::Interval* expectedOutput) const
{
    Signal::Interval J = transformDesc_->requiredInterval (I, expectedOutput);
//    TaskInfo ti(boost::format("In %s") % this->toString ().toStdString ());
//    TaskInfo(boost::format("requiredInterval (%s, %s) -> %s") % I % (expectedOutput?*expectedOutput:Signal::Interval()) % J);
    return J;
}


Signal::Interval TransformOperationDesc::
        affectedInterval(const Signal::Interval& I) const
{
    return transformDesc_->affectedInterval (I);
}


TransformOperationDesc::Extent TransformOperationDesc::
        extent() const
{
    return chunk_filter_.read ()->extent();
}


QString TransformOperationDesc::
        toString() const
{
    return (vartype(*chunk_filter_.get ()) + " on " + transformDesc_->toString ()).c_str();
}


bool TransformOperationDesc::
        operator==(const Signal::OperationDesc&d) const
{
    if (const TransformOperationDesc* f = dynamic_cast<const TransformOperationDesc*>(&d))
    {
        const TransformDesc& a = *transformDesc_;
        const TransformDesc& b = *f->transformDesc_;
        return a == b;
       // return *f->transformDesc () == *transform_desc_;
    }
    return false;
}


Tfr::pTransformDesc TransformOperationDesc::
        transformDesc() const
{
    return transformDesc_->copy();
}


void TransformOperationDesc::
        transformDesc(boost::shared_ptr<TransformDesc> td)
{
    transformDesc_ = td->copy ();
    chunk_filter_.write ()->transformDesc (td->copy ());
}


shared_state<ChunkFilterDesc>::read_ptr TransformOperationDesc::
        chunk_filter() const
{
    return chunk_filter_.read ();
}


shared_state<ChunkFilterDesc>::write_ptr TransformOperationDesc::
        chunk_filter()
{
    return chunk_filter_.write ();
}

} // namespace Tfr

#include "dummytransform.h"
#include "test/randombuffer.h"

namespace Tfr {

class DummyChunkFilter: public ChunkFilter, public ChunkFilter::NoInverseTag
{
public:
    DummyChunkFilter(int* i):i(i) {}

    void operator()( ChunkAndInverse& c ) {
        (*i)++;
    }

private:
    int* i;
};

class DummyChunkFilterDesc: public ChunkFilterDesc
{
public:
    DummyChunkFilterDesc(int* i):i(i) {}

    ChunkFilter::ptr createChunkFilter(Signal::ComputingEngine* engine) const {
        if (0 == engine)
            return ChunkFilter::ptr(new DummyChunkFilter(i));
        return ChunkFilter::ptr();
    }

private:
    int* i;
};

void TransformOperationDesc::
        test()
{
    // It should wrap all generic functionality in Signal::Operation and Tfr::Transform
    // so that ChunkFilters can explicilty do only the filtering.
    {
        int i = 0;
        ChunkFilterDesc::ptr cfd(new DummyChunkFilterDesc(&i));
        cfd.write ()->transformDesc(pTransformDesc(new Tfr::DummyTransformDesc));
        pTransformDesc t = cfd.read ()->transformDesc();
        TransformOperationDesc tod(cfd);

        EXCEPTION_ASSERT_EQUALS(
                    tod.affectedInterval (Signal::Interval(5,7)),
                    t->affectedInterval (Signal::Interval(5,7)));
        EXCEPTION_ASSERT_EQUALS(
                    tod.requiredInterval (Signal::Interval(5,7), 0),
                    t->requiredInterval (Signal::Interval(5,7),0));

        Signal::Operation::ptr o = tod.createOperation (0);
        Signal::pBuffer b = o->process (Test::RandomBuffer::smallBuffer ());
        EXCEPTION_ASSERT_EQUALS(i, (int)b->number_of_channels ());
    }
}

} // namespace Tfr
