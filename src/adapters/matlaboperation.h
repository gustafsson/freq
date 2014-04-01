#ifndef ADAPTERS_MATLABOPERATION_H
#define ADAPTERS_MATLABOPERATION_H

#include "sawe/openfileerror.h"
#include "matlabfunction.h"
#include "signal/operation.h"
#include "signal/computingengine.h"

// boost
#include <boost/scoped_ptr.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/split_member.hpp>

namespace Tools {
    namespace Support {
        class PlotLines;
    }
}

namespace Adapters {

class MatlabOperation
{
public:
    MatlabOperation( MatlabFunctionSettings* settings );
    ~MatlabOperation();

    // Does only support mono, use first channel
    //virtual unsigned num_channels() { return std::min(1u, Signal::DeprecatedOperation::num_channels()); }

    std::string name();
    Signal::pBuffer process( Signal::pBuffer src );
    void invalidate_samples(const Signal::Intervals& I);

    void restart();
    void settings(MatlabFunctionSettings*);
    MatlabFunctionSettings* settings() { return _settings; }

    /// Will call invalidate_samples if new data is available
    bool dataAvailable();

    Signal::Interval intervalToCompute( const Signal::Interval& I );
    bool isWaiting();
    std::string functionName();

    boost::scoped_ptr<Tools::Support::PlotLines> plotlines;

protected:
    boost::scoped_ptr<MatlabFunction> _matlab;
    MatlabFunctionSettings* _settings;
    Signal::pBuffer ready_data;
    Signal::pBuffer sent_data;
    Signal::Intervals _invalid_returns;
    Signal::Intervals _invalid_samples;
    Signal::Intervals invalid_returns() { return _invalid_returns; }
    Signal::Intervals invalid_samples() { return _invalid_samples; }

private:
    friend class boost::serialization::access;
    MatlabOperation();
    template<class Archive> void save(Archive& ar, const unsigned int /*version*/) const {
        using boost::serialization::make_nvp;

        DefaultMatlabFunctionSettings settings;
        settings.scriptname_ =  _settings->scriptname();
        settings.redundant_ = _settings->overlap();
        settings.computeInOrder_ = _settings->computeInOrder();
        settings.chunksize_ = _settings->chunksize();
        settings.arguments_ = _settings->arguments();
        settings.argument_description_ = _settings->argument_description();
        ar & BOOST_SERIALIZATION_NVP(settings.scriptname_);
        ar & BOOST_SERIALIZATION_NVP(settings.chunksize_);
        ar & BOOST_SERIALIZATION_NVP(settings.computeInOrder_);
        ar & BOOST_SERIALIZATION_NVP(settings.redundant_);
        ar & BOOST_SERIALIZATION_NVP(settings.arguments_);
        ar & BOOST_SERIALIZATION_NVP(settings.argument_description_);
    }

#if defined(TARGET_reader)
    template<class Archive> void load(Archive& /*ar*/, const unsigned int /*version*/) {
        throw Sawe::OpenFileError("Sonic AWE Reader does not support Matlab/Octave interoperability");
    }
#else
    template<class Archive> void load(Archive& ar, const unsigned int version) {
        using boost::serialization::make_nvp;

        DefaultMatlabFunctionSettings* settingsp = new DefaultMatlabFunctionSettings();
        DefaultMatlabFunctionSettings& settings = *settingsp;
        ar & BOOST_SERIALIZATION_NVP(settings.scriptname_);
        ar & BOOST_SERIALIZATION_NVP(settings.chunksize_);
        ar & BOOST_SERIALIZATION_NVP(settings.computeInOrder_);
        ar & BOOST_SERIALIZATION_NVP(settings.redundant_);
        if (0<version)
        {
            ar & BOOST_SERIALIZATION_NVP(settings.arguments_);
            ar & BOOST_SERIALIZATION_NVP(settings.argument_description_);
        }
        settings.operation = this;

        this->settings(settingsp);
        invalidate_samples(Signal::Intervals());
    }
#endif
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};


class MatlabOperationWrapper: public Signal::Operation
{
public:
    MatlabOperationWrapper( MatlabFunctionSettings* settings );

    // Signal::Operation
    Signal::pBuffer process(Signal::pBuffer b);

private:
    boost::shared_ptr<MatlabOperation> matlab_operation_;
};

class MatlabOperationDesc: public Signal::OperationDesc
{
public:
    class MatlabComputingEngine: public Signal::ComputingEngine {};

    MatlabOperationDesc( MatlabFunctionSettings* settings );

    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    Signal::Interval affectedInterval( const Signal::Interval& I ) const;
    Signal::OperationDesc::ptr copy() const;
    Signal::Operation::ptr createOperation(Signal::ComputingEngine* engine) const;

private:
    MatlabFunctionSettings* settings;

public:
    static void test();
};

} // namespace Adapters

BOOST_CLASS_VERSION(Adapters::MatlabOperation, 1)

#endif // ADAPTERS_MATLABOPERATION_H
