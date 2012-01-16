Welcome to the Sonic AWE test suite. 

This file describes how to setup the suite in order to execute the tests in Sonic AWE.

In Windows:
	Make sure you include the sonicawe-base folder containing all necessary dll files in you PATH variable.

**************************************************************************************
Tests requiring the music-1.ogg music file to run : 
	-sonicawe-end2end\deleteselection
	-sonicawe-end2end\openaudio

Just copy the .ogg file to the test's main directory.
**************************************************************************************
Tests which need a "gold" comparison file:
	-sonicawe-end2end\deleteselection
	-sonicawe-end2end\openaudio
	-sonicawe-end2end\opengui
	-sonicawe-end2end\testcommon
	
Run the tests once to produce the original image to test against. This file will be named test-cpu/cuda-result.png. Rename this file in each concerned test folder to test-cpu/cuda-gold.png. Run the test again.
**************************************************************************************

Additional test information.

When running a test using an image diff based evaluation, do note that the current version of Sonic AWE could produce a difference in the visualization which can be ignored. This 
only concerns the very first few chuncks of the visualization though. The reason for this is because the first frequency sample is taken at time 0 and the result of the transform 
could vary.  

Concerned tests : 
	-sonicawe-end2end\deleteselection
	-sonicawe-end2end\openaudio
	



