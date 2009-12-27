#ifndef TIMESCALEPLANE_H
#define TIMESCALEPLANE_H

typedef TransformData TimeScaleData;

class TimeScalePlane 
{
public:
private:
	// Original waveform
	Waveform _waveform;

	// Slots of transforms, as many as there are space for in the GPU ram
	list<TimeScaleData> _slots;

	// Tree structure for the signal... map-style, download (compute) when requested for.
	// With the difference that x-zoom is not coupled to y-zoom
	// x-zoom == t-zoom, time space
	// y-zoom == f-zoom, scale(~frequency) space
	// t-zoom level n has "2^n" == 1<<n steps between each sample, t-zoom=0 maximum zoom, arbitrary zoom-out possible.
	// could also measure in S.I. units, t-zoom in seconds. f-zoom hz=1/seconds. where the zoom-level names amont per data-element
	// pros: equivalent for all signals, as the axes are assumed to always be [s] and [1/s]
	// cons: the highest zoom level does not correspond to actual data samples (well it doesn't anyway...)
	// there is always an translation to SI units whichever zoom-type that is choosen
	// f-zoom level n has "2^n" steps between each scale... 
	
	int fzoom, tzoom; // size of one sample is: (delta t, delta f)= (2^tzoom, 2^fzoom) [s, 1/s]
			  // this does (of course) not correspond to 
	
}

#endif // TIMESCALEPLANE_H
/*
git daemon --verbose --base-path=/home/johan/git
sudo -H -u git gitosis-init < ~/id_rsa.pub
stat /home/git/repositories/gitosis-admin.git/hooks/post-update
sudo chmod 755 /home/git/repositories/gitosis-admin.git/hooks/post-update
git clone git://jovia.net/sonicawe.git
sudo -u git git-daemon --base-path=/home/johan

git daemon --verbose --export-all /home/johan/dev/sonic-gpu/sonicawe
                               --interpolated-path=/pub/%IP/%D
                               /pub/192.168.1.200/software
                               /pub/10.10.220.23/software
*/
