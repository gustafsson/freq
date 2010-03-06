// GLSL fragment shader
varying vec3 eyeSpacePos;
varying vec3 worldSpaceNormal;
varying vec3 eyeSpaceNormal;
uniform vec4 deepColor;    // = vec4(0.0, 0.0, 0.1, 1.0);
uniform vec4 shallowColor; // = vec4(0.1, 0.4, 0.3, 1.0);
uniform vec4 skyColor;     // = vec4(0.5, 0.5, 0.5, 1.0);
uniform vec3 lightDir;     // = vec3(0.0, 1.0, 0.0);
varying float intensity;

vec4 setWavelengthColor( float wavelengthScalar ) {
    vec4 spectrum[7];
        /* white background */
    spectrum[0] = vec4( 1, 1, 1, 0 ),
    spectrum[1] = vec4( 0, 0, 1, 0 ),
    spectrum[2] = vec4( 0, 1, 1, 0 ),
    spectrum[3] = vec4( 0, 1, 0, 0 ),
    spectrum[4] = vec4( 1, 1, 0, 0 ),
    spectrum[5] = vec4( 1, 0, 1, 0 ),
    spectrum[6] = vec4( 1, 0, 0, 0 );
        /* black background
        { 0, 0, 0 },
        { 1, 0, 1 },
        { 0, 0, 1 },
        { 0, 1, 1 },
        { 0, 1, 0 },
        { 1, 1, 0 },
        { 1, 0, 0 }}; */

    int count = 6;//sizeof(spectrum)/sizeof(spectrum[0])-1;
    float f = float(count)*wavelengthScalar;
    int i = int(min(f, float(count)));
    int j = int(min(f+1.0, float(count)));
    float t = f-float(i);

    vec4 rgb = mix(spectrum[i], spectrum[j], t);
    return rgb;
}

void main()
{
/*
    vec3 eyeVector              = normalize(eyeSpacePos);
    vec3 eyeSpaceNormalVector   = normalize(eyeSpaceNormal);
    vec3 worldSpaceNormalVector = normalize(worldSpaceNormal);

    float facing    = max(0.0, dot(eyeSpaceNormalVector, -eyeVector));
    float fresnel   = pow(1.0 - facing, 5.0); // Fresnel approximation
    float diffuse   = max(0.0, dot(worldSpaceNormalVector, lightDir));
    
    vec4 waterColor = mix(shallowColor, deepColor, facing);
    */
//    gl_FragColor = gl_Color;
//    gl_FragColor = vec4(fresnel);
//    gl_FragColor = vec4(diffuse);
//    gl_FragColor = waterColor;
//    gl_FragColor = waterColor*diffuse;
//    gl_FragColor = waterColor*diffuse + skyColor*fresnel;
//    gl_FragColor = pow(1-intensity,5);
//    gl_FragColor = setWavelengthColor( intensity );
    gl_FragColor = setWavelengthColor( 1.0-pow(1.0-clamp(intensity, 0.0, 1.0),5.0) );

}
