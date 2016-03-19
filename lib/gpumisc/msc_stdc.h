#pragma once

#ifdef _MSC_VER
#ifndef __CUDACC__

template<typename T>
T inline log2(const T& v) {
	return log(v)/log(2.f);
}

float inline log2f(const float& v) { return log2(v); }

template<typename T>
T inline exp2(const T& v) {
	return exp(v*log(2.f));
}

float inline exp2f(const float& v) { return exp2(v); }

double inline pow(int X, int Y) {
	return pow((double)X, (double)Y);
}

template<typename T>
T inline fmax(const T& a, const T& b) {
	return a>b?a:b;
}

template<typename T>
T inline fmin(const T& a, const T& b) {
	return a<b?a:b;
}

#define isnan _isnan

#endif // !__CUDACC__
#endif // _WIN32

