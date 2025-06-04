#version 460

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inTangent;
layout(location = 3) in vec3 inBinormal;
layout(location = 4) in mat3 inNormalMatrix;

layout(location = 0) out vec4 fragColor;

layout(set = 2, binding = 0) uniform MaterialParams {
    vec3 diffuse;
    float shininess;
    vec3 specular;
    float indexOfRefraction;
    vec3 ambient;
    float opacity;
    vec3 emission;
    uint hasTexture;
    uint isNeuralTexture;
} materialParams;

layout(set = 2, binding = 1) uniform sampler2D diffuseTexture;
layout(set = 2, binding = 2) uniform sampler2D specularTexture;
layout(set = 2, binding = 3) uniform sampler2D ambientTexture;
layout(set = 2, binding = 4) uniform sampler2D emissionTexture;
layout(set = 2, binding = 5) uniform sampler2D normalTexture;

layout(set = 2, binding = 6) uniform usampler2D neuralVQGridL0G0;
layout(set = 2, binding = 7) uniform usampler2D neuralVQGridL0G1;
// L1
layout(set = 2, binding = 8) uniform usampler2D neuralVQGridL1G0;
layout(set = 2, binding = 9) uniform usampler2D neuralVQGridL1G1;
// L2
layout(set = 2, binding = 10) uniform usampler2D neuralVQGridL2G0;
layout(set = 2, binding = 11) uniform usampler2D neuralVQGridL2G1;
// L3
layout(set = 2, binding = 12) uniform usampler2D neuralVQGridL3G0;
layout(set = 2, binding = 13) uniform usampler2D neuralVQGridL3G1;

layout(set = 3, binding = 0) uniform sampler2D environmentMap;
layout(set = 3, binding = 1) readonly buffer PaletteBuffer { float palette_values[]; } paletteBuffer; // Palette

layout(push_constant) uniform PushConstant {
    uint renderMode;
    uint discardOnZeroAlpha;
    vec3 lineColor;
} pushConstant;

const uint DIFFUSE_TEXTURE_BIT  = 1 << 0;
const uint SPECULAR_TEXTURE_BIT = 1 << 1;
const uint AMBIENT_TEXTURE_BIT = 1 << 2;
const uint EMISSION_TEXTURE_BIT = 1 << 3;
const uint NORMAL_TEXTURE_BIT = 1 << 4;

void main() {
    if (materialParams.isNeuralTexture == 1u) {
        uint vq_index = texture(neuralVQGridL0G0, inTexCoord).r;
        float palette_val = 0.5; // Default gray
        if (paletteBuffer.palette_values.length() > 0) {
            // This logic is too simple for a real VQ system but good for a first step.
            // A real VQ system uses the VQ index to look up a *vector* from a codebook,
            // then that vector might be further processed or directly used.
            // If the palette is small (e.g. 4 values) and vq_index is large, we need proper mapping.
            // The provided 'palette' has 4 values. 'vq_codebook_patches_packed_uint8' has shape [256, 4].
            // This means each of the 256 codebook entries is a 4-tuple of indices into the palette.
            // This part needs to be fleshed out based on how the VQ codebook and palette are structured.
            // For now, extremely simplified:
            palette_val = paletteBuffer.palette_values[vq_index % paletteBuffer.palette_values.length()];
        }
        
        fragColor = vec4(vec3(palette_val), 1.0);
    }else {
      // Diffuse (default)
      if ((materialParams.hasTexture & DIFFUSE_TEXTURE_BIT) != 0) {
          fragColor = texture(diffuseTexture, inTexCoord);
          if (pushConstant.discardOnZeroAlpha == 1 && fragColor.a == 0.0) {
              discard;
          }
      } else {
          fragColor = vec4(materialParams.diffuse, 1.0);
      }
      
      // Specular
      if (pushConstant.renderMode == 1) {
          if ((materialParams.hasTexture & SPECULAR_TEXTURE_BIT) != 0) {
              fragColor = texture(specularTexture, inTexCoord);
          } else {
              fragColor = vec4(materialParams.specular, 1.0);
          }
      }

      // Ambient
      if (pushConstant.renderMode == 2) {
          if ((materialParams.hasTexture & AMBIENT_TEXTURE_BIT) != 0) {
              fragColor = texture(ambientTexture, inTexCoord);
          } else {
              fragColor = vec4(materialParams.ambient, 1.0);
          }
      }

      // Emission
      if (pushConstant.renderMode == 3) {
          if ((materialParams.hasTexture & EMISSION_TEXTURE_BIT) != 0) {
              fragColor = texture(emissionTexture, inTexCoord);
          } else {
              fragColor = vec4(materialParams.emission, 1.0);
          }
      }

      // Shininess
      if (pushConstant.renderMode == 4) {
          fragColor = vec4(vec3(1.0 / log(materialParams.shininess)), 1.0);
      }

      // Index of refraction
      if (pushConstant.renderMode == 5) {
          fragColor = vec4(vec3(materialParams.indexOfRefraction), 1.0);
      }

      // Opacity
      if (pushConstant.renderMode == 6) {
          fragColor = vec4(vec3(materialParams.opacity), 1.0);
      }

      // Normal
      if (pushConstant.renderMode == 7) {
          if ((materialParams.hasTexture & NORMAL_TEXTURE_BIT) != 0) {
              mat3 TBN = mat3(normalize(inTangent), normalize(inBinormal), normalize(inNormal));
              vec3 normal = texture(normalTexture, inTexCoord).rgb * 2.0 - 1.0;
              fragColor.rgb = normalize(inNormalMatrix * TBN * normal);
          } else {
              fragColor = vec4(inNormal, 1.0);
          }
          fragColor.rgb = fragColor.rgb * 0.5 + 0.5;
      }

      // Texture coordinates
      if (pushConstant.renderMode == 8) {
          fragColor = vec4(inTexCoord, 0.0, 1.0);
      }

      // NTC placeholder
      if (pushConstant.renderMode == 9) {
          fragColor = vec4(0.5, 0.5, 0.5, 1.0);
      }

      // Line render
      if (pushConstant.renderMode == 10) {
          fragColor = vec4(pushConstant.lineColor, 1.0);
      }
  }
}

