#version 410 core

// Interpolated values from the vertex shaders
in vec2 UV;
in vec3 Position_worldspace;
in vec3 Position_modelspace;
in vec3 Normal_modelspace;
in vec3 Normal_cameraspace;
in vec3 EyeDirection_cameraspace;
in vec3 LightDirection_cameraspace;
in vec3 class_id_color;

// Ouput data
out vec3 color;

// Values that stay constant for the whole mesh.
uniform sampler2D myTextureSampler;
uniform mat4 MV;
uniform vec3 LightPosition_worldspace;
uniform int UseTexture; // is 1 if texture is used 0 if white color should be used


void main(){

	// Material properties
	if(UseTexture == 0){
        color = ((Normal_modelspace * 0.5) + vec3(0.5, 0.5, 0.5));
    }else if(UseTexture == 1){
        // Normal of the computed fragment, in camera space
        vec3 n = normalize( Normal_modelspace);
        // Direction of the light (from the fragment to the light)
        vec3 l = normalize(vec3(0,-1,1));
        // Cosine of the angle between the normal and the light direction,
        // clamped above 0
        //  - light is at the vertical of the triangle -> 1
        //  - light is perpendicular to the triangle -> 0
        //  - light is behind the triangle -> 0
        float cosTheta = clamp( dot( n,l ), 0,1 );

        vec3 material = texture2D( myTextureSampler, UV ).rgb;
        color =material * 0.75 + material * 0.35 * cosTheta;
    }else{
        vec3 n = normalize( Normal_modelspace);
        color = class_id_color * 0.75 + n * 0.1;
    }
}

