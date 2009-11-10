#ifndef WAVELETTRANSFORM_H
#define WAVELETTRANSFORM_H

#include "transformdata.h"
#include "waveform.h"
#include <boost/shared_ptr.hpp>

class WavelettTransform
{
public:
    WavelettTransform( const char* filename );

    boost::shared_ptr<TransformData> getWavelettTransform();
    boost::shared_ptr<Waveform> getOriginalWaveform();
    boost::shared_ptr<Waveform> getInverseWaveform();

    boost::shared_ptr<TransformData> computeCompleteWavelettTransform();
    //boost::shared_ptr<TransformData> computeWavelettTransform( float startt, float endt, float lowf, float highf );
    boost::shared_ptr<Waveform> computeInverseWaveform();

private:
    boost::shared_ptr<TransformData> _transform;
    boost::shared_ptr<Waveform> _originalWaveform;
    boost::shared_ptr<Waveform> _inverseWaveform;
};

#endif // WAVELETTRANSFORM_H
