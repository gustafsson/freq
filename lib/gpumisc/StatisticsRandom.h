// dead code
// StatisticsRandom wasn't updated when class Statistics was updated to use DataStorage.
#if 0

#include "GpuCpuData.h"
#include "tasktimer.h"
#include <time.h>
#include <string.h>

/**
StatisticsRandom performs various statistical methods on a given 
data set. The input parameter is a GpuCpuData<data_t>* which allows
some of the methods to be executed on the Gpu.
<p>
The methods of StatisticsRandom only takes into account a small 
random subset of the original data and gives estimates values of
min, max, standard deviation (std) and variance (var).
*/
template<typename data_t>
class StatisticsRandom {
public:
        StatisticsRandom(GpuCpuData<data_t>* data, size_t numberOfEntries = 2<<10)
	:	data(data)
	{
		recompute(numberOfEntries);
	}

	void recompute(size_t numberOfEntries) {
/*		TaskTimer tt("Statistics %s #%u of %u",
			typeid(data_t).name(),
			numberOfEntries,
			data->getNumberOfElements1D());
*/
		long double
			varianceSum = 0,
			diff,
			meanSum = 0;

		size_t
			n = data->getNumberOfElements1D();

		data_t
			*p = data->getCpuMemory();
		
		min = max = p;

		srand( clock() );
		for (size_t i=0; i<numberOfEntries; i++) {
			size_t index = (double)rand()/(RAND_MAX+1)*n;
			meanSum += p[index];
		}

		mean = meanSum/=numberOfEntries;

		for (size_t i=0; i<numberOfEntries; i++) {
			size_t index = (double)rand()/(RAND_MAX+1)*n;

			data_t& v = p[index];
			(diff = v) -= meanSum;
			varianceSum += diff*diff;

			if (*min > v)
				min = &v;

			if (*max < v)
				max = &v;
		}

		variance = varianceSum/numberOfEntries;
		std = sqrt(variance);

//		tt.info("max %g, min %g, mean %g, standard deviation %g", (double)*max,(double)*min,
//			(double)mean, (double)std);
	}

	data_t* getMax() { return max; }
	data_t* getMin() { return min; }
	data_t getMean() { return mean; }
	long double getVariance() { return variance; }
	data_t getStd() { return std; }

protected:
	GpuCpuData<data_t>* data;

private:
	data_t* max;
	data_t* min;
	data_t mean;
	long double variance;
	data_t std;
};

class AnytypeStats {
public:
	template<typename data_t>
	AnytypeStats( GpuCpuVoidData* data, data_t* type, size_t numberOfEntries = 1000 )
	{
		if (data)
			recompute( data, type, numberOfEntries );
		else {
			voidData = 0;
			diffSum = 0;
		}
	}

	bool operator<(const AnytypeStats& b) const {
		return diffSum < b.diffSum;
	}

	template<typename data_t>
	void recompute(GpuCpuVoidData* voidData, data_t* type, size_t numberOfEntries = 1000 ) {
		GpuCpuData<data_t> data( 
			voidData->getCpuMemoryVoid(),
			GpuCpuData<data_t>::getNumberOfElements(voidData->getSizeInBytes()),
			GpuCpuVoidData::CpuMemory,
			true);

		StatisticsRandom<data_t> stat(&data, numberOfEntries);
		max = *stat.getMax();
		min = *stat.getMin();
		mean = stat.getMean();
		variance = stat.getVariance();
		std = stat.getStd();
		this->voidData = voidData;

		undecoratedName = typeid(data_t).name();
#ifdef _WIN32
                decoratedName = typeid(data_t).raw_name();
#endif // _WIN32
                sizeOfType = sizeof(data_t);

		diffSum = computeDiffSum( &data, 1 );
	}

	template<typename data_t>
        long double recomputeDiffSum(GpuCpuVoidData* voidData, data_t* , size_t stride=1) {
		GpuCpuData<data_t> data( 
			voidData->getCpuMemoryVoid(),
			GpuCpuData<data_t>::getNumberOfElements(voidData->getSizeInBytes()),
			GpuCpuVoidData::CpuMemory,
			true);

		return diffSum = computeDiffSum( &data, stride );
	}

	template<typename data_t>
	bool isType() {
            #ifdef _WIN32
                return 0==strcmp(typeid(data_t).raw_name(), decoratedName);
            #else
                return 0==strcmp(typeid(data_t).name(), undecoratedName);
            #endif
	}
	long double recomputeDiffSum(size_t stride) {
		if (isType<signed char>()) return recomputeDiffSum( voidData, (signed char*)0, stride);
		if (isType<unsigned char>()) return recomputeDiffSum( voidData, (unsigned char*)0, stride);
		if (isType<signed short>()) return recomputeDiffSum( voidData, (signed short*)0, stride);
		if (isType<unsigned short>()) return recomputeDiffSum( voidData, (unsigned short*)0, stride);
		if (isType<signed long>()) return recomputeDiffSum( voidData, (signed long*)0, stride);
		if (isType<unsigned long>()) return recomputeDiffSum( voidData, (unsigned long*)0, stride);
		if (isType<signed long long>()) return recomputeDiffSum( voidData, (signed long long*)0, stride);
		if (isType<unsigned long long>()) return recomputeDiffSum( voidData, (unsigned long long*)0, stride);
		if (isType<float>()) return recomputeDiffSum( voidData, (float*)0, stride);
		if (isType<double>()) return recomputeDiffSum( voidData, (double*)0, stride);
		if (isType<long double>()) return recomputeDiffSum( voidData, (long double*)0, stride);
		return 0;
	}

	const char* getName() { return undecoratedName; }
#ifdef _WIN32
	const char* getRawName() { return decoratedName; }
#endif
	size_t getSizeOf() { return sizeOfType; }

	long double getMax() { return max; }
	long double getMin() { return min; }
	long double getMean() { return mean; }
	long double getVariance() { return variance; }
	long double getStd() { return std; }
	long double getDiffSum() { return diffSum; }

protected:
	GpuCpuVoidData* voidData;
        const char* undecoratedName;
        #ifdef _WIN32
                const char* decoratedName;
        #endif // _WIN32
	size_t sizeOfType;

	long double max;
	long double min;
	long double mean;
	long double variance;
	long double std;

	long double diffSum;

	template<typename data_t>
	long double computeDiffSum( GpuCpuData<data_t>* data, size_t stride=1 ) {
		size_t
			n = data->getNumberOfElements1D(),
			stripeLength = 100,
			numberOfStripes = 10000;

		data_t
			*p = data->getCpuMemory(),
			*prev = 0;

		long double diffSum = 0;

		for (unsigned c=0; c<numberOfStripes; c++) {
                        double baseIndexD = rand() / (((double)RAND_MAX) +1) * ((double)n - stripeLength*stride);
                        if (baseIndexD<0) return 100000000.;
			size_t baseIndex = baseIndexD;
			prev = &p[baseIndex];

			for (unsigned i=1; i<stripeLength; i++) {
				data_t &v = prev[stride];

				long double d = (long double)v-*prev;
				diffSum += abs(d);

				prev = &v;
			}
		}

//		diffSum /= (getMax()-getMin());
//		diffSum /= (getMax()-getMin());
		diffSum /= getStd();

		if (getMax()<0.00001)
			return LONG_MAX;
		return diffSum;
	}
};
#endif
