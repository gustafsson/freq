/**
 class for smooth transitions
 TAni is a transparent float

 johan.b.gustafsson@gmail.com
 */
#pragma once

#include <float.h>
#include <cmath>

template<typename floater=float>
class TAni {
private:
	floater b,v,a,minv,maxv;
	float totTime;
	float t;
public:
	TAni(floater _v=0, float speed = 1.0, floater minv = -FLT_MAX, floater maxv = FLT_MAX)
        :	minv(minv),
                maxv(maxv),
                totTime(speed),
                t(0)
	{b=v=a=_v;}

		  operator floater() {return v;}
	operator const floater() const {return v;}
		  floater& operator&() {return b;}
	const floater& operator&() const {return b;}

	floater reset(floater f) {
		return b=v=a=f;
	}

	TAni& operator=(floater f)
	{
		if( f<minv )
			f = minv;
		else if( f>maxv )
			f = maxv;

		if(b==f)
			return *this;
		floater 
			vprim = 4*(b-a)*(t>totTime/2?totTime-t:t),
//			vprimMax = 2*( b-a ); // assume that this is the peak of the new movement
//			v2prim = ~ vprim, but the equation still unknown 4*( 2*(f-a2) )*(t2>totTime/2?1-t2:t2),
			v2primMax = 4*(f-a)*totTime; // assume that the given value is the peak of the new movement
		if(a==v)	// start from rest
			t=0; 
		else if( (vprim > 0) != (v2primMax > 0)) // change dir
		{	
			// start over from rest, will jump vprim
			t = 0;
			a = v;
		}
        else if( std::abs(vprim) >= std::abs(v2primMax) ) // it's currently going too fast for this goal (b)
		{
			a = 2*v-f;
			// jump vprim to the new peak
			// a = f-2*(f - b);//	== -f + 2*b// adjust source
			t = .5f*totTime;	// max speed, decrease from here
		}
		else // it's not going too fast, accelerate
		{
			// at tim t, value v, goal f(=b): solve equation for a
			float 
				T = totTime,
				s = T-t;
			if (t>.5*totTime) {
				// not safe as t -> totTime
				if (s < 0.0001) {
					a = 2*v-f;
				} else {
					a = (v-f)*(T*T)/(2*s*s)+f;
				}
			} else {
				// safe for all t,v,f, T*T>0 const
				a = (v - 2*f*(t*t)/(T*T))/((T*T)-2*(t*t));
			}
		}
		/*(v-b)*(totTime*totTime)*(totTime-t)*(totTime-t)+2*b = a
			(v - (b - 2*b*(totTime-t)*(totTime-t)/(totTime*totTime)))/(2*(totTime-t)*(totTime-t)/(totTime*totTime))=  a

		v = b - 2*(b-a)*s*s/(T*T)
		v = b - 2*b*s*s/(T*T) + 2*a*s*s/(T*T)
		v - b + 2*b*s*s/(T*T) = 2*a*s*s/(T*T)
		(v - b + 2*b*s*s/(T*T))/(2*s*s/(T*T)) = a
		((v - b)*(T*T)/(2*s*s) + b) = a*/
		b=f;
		return *this;
	}

	bool TimeStep(float time=0.01f)
	{
		t+=time;
		if(t >= totTime) {v=a=b; t = totTime; }
		else if(t > .5*totTime)	v = b - 2*(b-a)*(totTime-t)*(totTime-t)/(totTime*totTime);
		else							v = a + 2*(b-a)*(t*t)/(totTime*totTime);
		return v!=b;
	}
};

typedef TAni<float> floatAni;
