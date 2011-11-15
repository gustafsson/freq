#ifndef ADAPTERS_MATLABOPERATION_H
#define ADAPTERS_MATLABOPERATION_H

#include "signal/operationcache.h"
#include "matlabfunction.h"

// boost
#include <boost/scoped_ptr.hpp>
#include <boost/serialization/split_member.hpp>

namespace Tools {
    namespace Support {
        class PlotLines;
    }
}

namespace Adapters {


class MatlabOperation: public Signal::OperationCache
{
public:
    MatlabOperation( Signal::pOperation source, MatlabFunctionSettings* settings );
    ~MatlabOperation();

    // Does only support mono, use first channel
    //virtual unsigned num_channels() { return std::min(1u, Signal::OperationCache::num_channels()); }

    virtual std::string name();
    virtual Signal::pBuffer readRaw( const Signal::Interval& I );
    virtual void invalidate_samples(const Signal::Intervals& I);

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
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation);
        ar & BOOST_SERIALIZATION_NVP(settings.scriptname_);
        ar & BOOST_SERIALIZATION_NVP(settings.chunksize_);
        ar & BOOST_SERIALIZATION_NVP(settings.computeInOrder_);
        ar & BOOST_SERIALIZATION_NVP(settings.redundant_);
        ar & BOOST_SERIALIZATION_NVP(settings.arguments_);
        ar & BOOST_SERIALIZATION_NVP(settings.argument_description_);
    }
    template<class Archive> void load(Archive& ar, const unsigned int version) {
        using boost::serialization::make_nvp;

        DefaultMatlabFunctionSettings* settingsp = new DefaultMatlabFunctionSettings();
        DefaultMatlabFunctionSettings& settings = *settingsp;
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation);
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
        invalidate_cached_samples(Signal::Intervals());
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

} // namespace Adapters

BOOST_CLASS_VERSION(Adapters::MatlabOperation, 1)

#endif // ADAPTERS_MATLABOPERATION_H
