#include "datastorage.h"
#include "cpumemorystorage.h"
#include "tasktimer.h"
#include <boost/scoped_ptr.hpp>
#include "throwInvalidArgument.h"
#include <stdio.h>
//#include <Volume Rendering API/VoxelData.h>

/**
Statistics performs different statistical methods on a given data 
set. The input parameter is a GpuCpuData* which allows
some of the methods to be executed on the Gpu.
*/
template<typename data_t>
class Statistics {
public:
    Statistics(typename DataStorage<data_t>::Ptr data, bool isSegmented = false, bool silent = false)
	:	data(data)
	{
		if(0==data)
			return;
                recompute(isSegmented, silent);
	}

	data_t getVal(data_t, bool isSegmented);

	void recompute(bool isSegmented=false, bool silent = false) {
                boost::scoped_ptr<TaskTimer> tt;
        if (!silent)
            tt.reset( new TaskTimer("Stats %u#%s",
                data->numberOfElements(), typeid(data_t).name()) );

		long double
			varianceSum = 0,
			diff,
			meanSum = 0;

		size_t
            n = data->numberOfElements();

		data_t
            *p = data->getCpuMemory(),
			maxVal, 
			minVal;
		
		min = max = p;
		maxVal = minVal = getVal(p[0], isSegmented);

		for (size_t i=0; i<n; i++) {
			meanSum += getVal(p[i], isSegmented);
		}

		mean = meanSum/=n;

		for (size_t i=0; i<n; i++) {
			data_t v = getVal(p[i], isSegmented);

			(diff = v) -= meanSum;
			varianceSum += diff*diff;

			if (minVal > v)
				min = &p[i],
				minVal = v;

			if (maxVal < v)
				max = &p[i],
				maxVal = v;
		}

		variance = varianceSum/n;
		std = sqrt(variance);

		if (0!=tt)
            printf(", max %.3g, min %.3g, mean %.3g, std dev %.3g", (double)maxVal,(double)minVal,
				(double)mean, (double)std);
	}

	data_t* getMax() { return max; }
	data_t* getMin() { return min; }
	data_t getMean() { return mean; }
	long double getVariance() { return variance; }
	data_t getStd() { return std; }

	template<typename data_t2>
	bool operator==(const Statistics<data_t2>& b) const {
		if (0==data || 0==b.data)
			return false;
		return
			*max == *b.max &&
			*min == *b.min &&
			mean == b.mean &&
			variance == b.variance &&
			std == b.std;
	}
protected:
    typedef typename DataStorage<data_t>::Ptr DataType;
    DataType data;

private:
	data_t* max;
	data_t* min;
	data_t mean;
	long double variance;
	data_t std;

	// could be useful for multithreading floats
	data_t calcMean(const size_t start, const size_t end) {
		double
			n = end-start,
			m1 = n/2,
			m2 = (n+1)/2;

		if ( 0 == n ) {
			throw std::logic_error(__LOCATION__);
		}
		if ( 1 == n ) {
			return data.getCpuMemory()[start];
		}

		data_t
			first = mean(start, start + m1),
			second = mean(start + m2, end);

		if (m1 == m2) {
			return (first/2.0 + second/2.0);
		} else {
			data_t middle = data.getCpuMemory()[start + m1];
			return (first*m1 + second*m2 + middle)/n;
		}
	}
};

/*
Application specific:

template<>
inline unsigned short Statistics<unsigned short>::getVal(unsigned short val, bool isSegmented) {
	unsigned short
		nhb = NO_HIGHBITS,
		mrb = MAX_RAWBITS,
		mhb = MAX_HIGHBITS,
		rb = RAWBITS(val),
		hb = HIGHBITS(val),
		uhb = USE_HIGHER_BITS(val);

	if (isSegmented)
		val = USE_HIGHER_BITS(val);

	return val;
}

template<>
inline short Statistics<short>::getVal(short val, bool isSegmented) {
	if (isSegmented)
		val = USE_HIGHER_BITS(val);

	return val;
}
*/

template<typename data_t>
inline data_t Statistics<data_t>::getVal(data_t val, bool) {
	return val;
}
