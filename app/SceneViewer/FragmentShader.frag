#version 460
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_shader_16bit_storage : require

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inTangent;
layout(location = 3) in vec3 inBinormal;
layout(location = 4) in mat3 inNormalMatrix;

layout(location = 0) out vec4 fragColor;

// --- Constants ---
#define MAX_NEURAL_FEATURE_GRID_LEVELS 4
#define MAX_MLP_LAYERS 3
#define RESOLUTION 1024

const uint DIFFUSE_TEXTURE_BIT  = 1 << 0;
const uint SPECULAR_TEXTURE_BIT = 1 << 1;
const uint AMBIENT_TEXTURE_BIT = 1 << 2;
const uint EMISSION_TEXTURE_BIT = 1 << 3;
const uint NORMAL_TEXTURE_BIT = 1 << 4;

// --- Push Constants ---
layout(push_constant) uniform PushConstant {
    uint renderMode;
    uint discardOnZeroAlpha;
    vec3 lineColor;
    float lod;
} pushConstant;


// =========================================================================
// Set 2: Per-Material Bindings
// =========================================================================

layout(set = 2, binding = 0) uniform MaterialParamsUBO {
    // Standard params
    vec3 diffuse;
    float shininess;
    vec3 specular;
    float indexOfRefraction;
    vec3 ambient;
    float opacity;
    vec3 emission;
    uint hasTexture;
    uint isNeuralTexture;
    float pad0;
    float pad1;
    float pad2;
    uvec4 channelCounts[MAX_NEURAL_FEATURE_GRID_LEVELS];
    uvec4 featureGridShapes[MAX_NEURAL_FEATURE_GRID_LEVELS][2];
    uvec4 mlpLayerShapes[MAX_MLP_LAYERS];
} materialParams;

// Standard Textures (bindings 1-5)
layout(set = 2, binding = 1) uniform sampler2D diffuseTexture;
layout(set = 2, binding = 2) uniform sampler2D specularTexture;
layout(set = 2, binding = 3) uniform sampler2D ambientTexture;
layout(set = 2, binding = 4) uniform sampler2D emissionTexture;
layout(set = 2, binding = 5) uniform sampler2D normalTexture;

// --- Neural VQ Grids (bindings 6-13) ---
// These are now texture handles for use with `texelFetch`
layout(set = 2, binding = 6) uniform usampler2DArray neuralVQGridL0G0;
layout(set = 2, binding = 7) uniform usampler2DArray neuralVQGridL0G1;
layout(set = 2, binding = 8) uniform usampler2DArray neuralVQGridL1G0;
layout(set = 2, binding = 9) uniform usampler2DArray neuralVQGridL1G1;
layout(set = 2, binding = 10) uniform usampler2DArray neuralVQGridL2G0;
layout(set = 2, binding = 11) uniform usampler2DArray neuralVQGridL2G1;
layout(set = 2, binding = 12) uniform usampler2DArray neuralVQGridL3G0;
layout(set = 2, binding = 13) uniform usampler2DArray neuralVQGridL3G1;

// --- Neural Channel Selection SSBOs (bindings 14-21) ---
layout(set = 2, binding = 14) readonly buffer Grid0L0_Channels { uint16_t indices[]; } grid0l0_channels;
layout(set = 2, binding = 15) readonly buffer Grid1L0_Channels { uint16_t indices[]; } grid1l0_channels;
layout(set = 2, binding = 16) readonly buffer Grid0L1_Channels { uint16_t indices[]; } grid0l1_channels;
layout(set = 2, binding = 17) readonly buffer Grid1L1_Channels { uint16_t indices[]; } grid1l1_channels;
layout(set = 2, binding = 18) readonly buffer Grid0L2_Channels { uint16_t indices[]; } grid0l2_channels;
layout(set = 2, binding = 19) readonly buffer Grid1L2_Channels { uint16_t indices[]; } grid1l2_channels;
layout(set = 2, binding = 20) readonly buffer Grid0L3_Channels { uint16_t indices[]; } grid0l3_channels;
layout(set = 2, binding = 21) readonly buffer Grid1L3_Channels { uint16_t indices[]; } grid1l3_channels;

// --- Neural MLP Layer SSBOs (bindings 22+) ---
// Using arrays of SSBOs would be ideal but is not core GLSL. We declare them individually.
layout(set = 2, binding = 22) readonly buffer MlpL0_W_Buffer { float16_t data[]; } mlpL0_W;
layout(set = 2, binding = 23) readonly buffer MlpL0_B_Buffer { float16_t data[]; } mlpL0_B;
layout(set = 2, binding = 24) readonly buffer MlpL1_W_Buffer { float16_t data[]; } mlpL1_W;
layout(set = 2, binding = 25) readonly buffer MlpL1_B_Buffer { float16_t data[]; } mlpL1_B;
layout(set = 2, binding = 26) readonly buffer MlpL2_W_Buffer { float16_t data[]; } mlpL2_W;
layout(set = 2, binding = 27) readonly buffer MlpL2_B_Buffer { float16_t data[]; } mlpL2_B;


// =========================================================================
// Set 3: Global Bindings
// =========================================================================
layout(set = 3, binding = 0) uniform sampler2D environmentMap;
layout(set = 3, binding = 1) readonly buffer PaletteBuffer  { float data[]; } paletteBuffer;
layout(set = 3, binding = 2) readonly buffer VQCodebookBuffer { uint data[]; } vqCodebookBuffer;
layout(set = 3, binding = 3) readonly buffer PositionalEncodingBuffer { float data[]; } posEncodingBuffer;

// =========================================================================
// NEURAL TEXTURING HELPER FUNCTIONS
// =========================================================================

const int POS_ENCODING_TABLE_SIZE = 8;
const int POS_ENCODING_FEATURES_PER_DIM = 5; // 3 octaves * 2 offsets - 1 skipped

uint unpack_2bit_value(uint byte, uint index_in_byte) {
    return (byte >> (index_in_byte * 2)) & 0x03u;
}

float get_feature_from_codebook(uint vq_index, uint feature_index) {
    const uint bytes_per_patch = 4; // 16 features * 2 bits/feature = 32 bits = 4 bytes
    uint byte_offset_in_patch = feature_index / 4;
    uint total_byte_offset = (vq_index * bytes_per_patch) + byte_offset_in_patch;

    uint dword_index = total_byte_offset / 4;
    uint byte_index_in_dword = total_byte_offset % 4;
    uint packed_dword = vqCodebookBuffer.data[dword_index];
    uint packed_byte = (packed_dword >> (byte_index_in_dword * 8)) & 0xFF;

    uint index_in_byte = feature_index % 4;
    uint palette_index = (packed_byte >> (index_in_byte * 2)) & 0x03u;
    return paletteBuffer.data[palette_index];
}

void append_positional_encoding_static(inout float features[128], int base_feature_index, vec2 coords) {
    ivec2 int_coords = ivec2(coords);
    int ix = int_coords.x % POS_ENCODING_TABLE_SIZE;
    int iy = int_coords.y % POS_ENCODING_TABLE_SIZE;

    // Calculate the starting offset into the buffer for our x and y table entries
    int base_lookup_idx_x = ix * POS_ENCODING_FEATURES_PER_DIM;
    int base_lookup_idx_y = iy * POS_ENCODING_FEATURES_PER_DIM;

    // Manually unroll all 11 feature appends.
    // The compiler can optimize these static-indexed writes much better.
    // Octave 0, Offset 0.0 (skipped for x)
    features[base_feature_index + 0]  = posEncodingBuffer.data[base_lookup_idx_x + 0]; // Corresponds to octave 0, offset 0.0
    features[base_feature_index + 1]  = posEncodingBuffer.data[base_lookup_idx_y + 0];

    // Octave 1
    features[base_feature_index + 2]  = posEncodingBuffer.data[base_lookup_idx_x + 1]; // Corresponds to octave 1, offset 0.5
    features[base_feature_index + 3]  = posEncodingBuffer.data[base_lookup_idx_y + 1];
    features[base_feature_index + 4]  = posEncodingBuffer.data[base_lookup_idx_x + 2]; // Corresponds to octave 1, offset 0.0
    features[base_feature_index + 5]  = posEncodingBuffer.data[base_lookup_idx_y + 2];

    // Octave 2
    features[base_feature_index + 6]  = posEncodingBuffer.data[base_lookup_idx_x + 3]; // Corresponds to octave 2, offset 0.5
    features[base_feature_index + 7]  = posEncodingBuffer.data[base_lookup_idx_y + 3];
    features[base_feature_index + 8]  = posEncodingBuffer.data[base_lookup_idx_x + 4]; // Corresponds to octave 2, offset 0.0
    features[base_feature_index + 9]  = posEncodingBuffer.data[base_lookup_idx_y + 4];

    // Final zero
    features[base_feature_index + 10] = 0.0;
}

float interpolate_vq_patch(uint vq_idx, uint p_idx00, uint p_idx10, uint p_idx01, uint p_idx11, vec2 frac_coords_patch) {
    // Fetch the four neighboring feature values from the VQ codebook
    float feat_p00 = get_feature_from_codebook(vq_idx, p_idx00); // Top-left sub-feature
    float feat_p10 = get_feature_from_codebook(vq_idx, p_idx10); // Top-right sub-feature
    float feat_p01 = get_feature_from_codebook(vq_idx, p_idx01); // Bottom-left sub-feature
    float feat_p11 = get_feature_from_codebook(vq_idx, p_idx11); // Bottom-right sub-feature

    // First, interpolate along the X-axis
    float interp_x1 = mix(feat_p00, feat_p10, frac_coords_patch.x);
    float interp_x2 = mix(feat_p01, feat_p11, frac_coords_patch.x);

    // Then, interpolate the results along the Y-axis
    return mix(interp_x1, interp_x2, frac_coords_patch.y);
}

vec4 evaluate_neural_texture(vec2 uv, float lod) {
    int level_idx = 0;
    if (lod < 4.0) level_idx = 0;
    else if (lod < 6.0) level_idx = 1;
    else if (lod < 8.0) level_idx = 2;
    else level_idx = 3;

    const int MAX_TOTAL_FEATURES = 128;
    float features[MAX_TOTAL_FEATURES];
    int feature_count = 0;
    
    vec2 frac_coords;

    // --- 2. Grid 0 Feature Gathering (4 scaled features per channel) ---
    uint num_selections_g0 = materialParams.channelCounts[level_idx].x;
    if (num_selections_g0 > 0) {
        uvec3 grid_dims = materialParams.featureGridShapes[level_idx][0].xyz;
        vec2 rotated_uv = -vec2(uv.y, uv.x);
        vec2 grid_coords = rotated_uv * vec2(grid_dims.z, grid_dims.y);
        ivec2 base_coords_int = ivec2(floor(grid_coords));
        frac_coords = fract(grid_coords);

        float w11 = frac_coords.x * frac_coords.y;
        float w10 = frac_coords.x * (1.0 - frac_coords.y);
        float w01 = (1.0 - frac_coords.x) * frac_coords.y;
        float w00 = (1.0 - frac_coords.x) * (1.0 - frac_coords.y);
        
        ivec2 fetch_coord_int = ivec2(int(grid_coords.x), int(grid_coords.y)) % ivec2(grid_dims.z, grid_dims.y);

        vec2 patch_coords = frac_coords * 4.0;
        ivec2 patch_base_int = ivec2(floor(patch_coords));
        uint idx00 = uint(clamp(patch_base_int.y,     0, 3)) * 4 + uint(clamp(patch_base_int.x,     0, 3));
        uint idx01 = uint(clamp(patch_base_int.y,     0, 3)) * 4 + uint(clamp(patch_base_int.x + 1, 0, 3));
        uint idx10 = uint(clamp(patch_base_int.y + 1, 0, 3)) * 4 + uint(clamp(patch_base_int.x,     0, 3));
        uint idx11 = uint(clamp(patch_base_int.y + 1, 0, 3)) * 4 + uint(clamp(patch_base_int.x + 1, 0, 3));

        ivec2 n00 = (base_coords_int + ivec2(0,0)) % ivec2(grid_dims.z, grid_dims.y);
        ivec2 n10 = (base_coords_int + ivec2(1,0)) % ivec2(grid_dims.z, grid_dims.y);
        ivec2 n01 = (base_coords_int + ivec2(0,1)) % ivec2(grid_dims.z, grid_dims.y);
        ivec2 n11 = (base_coords_int + ivec2(1,1)) % ivec2(grid_dims.z, grid_dims.y);

        uint vq_idx_00; 
        uint vq_idx_10; 
        uint vq_idx_01; 
        uint vq_idx_11; 

        for (int i = 0; i < num_selections_g0; ++i) {
            uint channel_to_sample;
            switch(level_idx) {
                case 0:
                    channel_to_sample = uint(grid0l0_channels.indices[i]);
                    vq_idx_00 = uint(texelFetch(neuralVQGridL0G0, ivec3(n00, channel_to_sample), 0).r);
                    vq_idx_10 = uint(texelFetch(neuralVQGridL0G0, ivec3(n10, channel_to_sample), 0).r);
                    vq_idx_01 = uint(texelFetch(neuralVQGridL0G0, ivec3(n01, channel_to_sample), 0).r);
                    vq_idx_11 = uint(texelFetch(neuralVQGridL0G0, ivec3(n11, channel_to_sample), 0).r);
                    break;
                case 1:
                    channel_to_sample = uint(grid0l1_channels.indices[i]);
                    vq_idx_00 = uint(texelFetch(neuralVQGridL1G0, ivec3(n00, channel_to_sample), 0).r);
                    vq_idx_10 = uint(texelFetch(neuralVQGridL1G0, ivec3(n10, channel_to_sample), 0).r);
                    vq_idx_01 = uint(texelFetch(neuralVQGridL1G0, ivec3(n01, channel_to_sample), 0).r);
                    vq_idx_11 = uint(texelFetch(neuralVQGridL1G0, ivec3(n11, channel_to_sample), 0).r);
                    break;
                case 2:
                    channel_to_sample = uint(grid0l2_channels.indices[i]);
                    vq_idx_00 = uint(texelFetch(neuralVQGridL2G0, ivec3(n00, channel_to_sample), 0).r);
                    vq_idx_10 = uint(texelFetch(neuralVQGridL2G0, ivec3(n10, channel_to_sample), 0).r);
                    vq_idx_01 = uint(texelFetch(neuralVQGridL2G0, ivec3(n01, channel_to_sample), 0).r);
                    vq_idx_11 = uint(texelFetch(neuralVQGridL2G0, ivec3(n11, channel_to_sample), 0).r);
                    break;
                case 3:
                    channel_to_sample = uint(grid0l3_channels.indices[i]);
                    vq_idx_00 = uint(texelFetch(neuralVQGridL3G0, ivec3(n00, channel_to_sample), 0).r);
                    vq_idx_10 = uint(texelFetch(neuralVQGridL3G0, ivec3(n10, channel_to_sample), 0).r);
                    vq_idx_01 = uint(texelFetch(neuralVQGridL3G0, ivec3(n01, channel_to_sample), 0).r);
                    vq_idx_11 = uint(texelFetch(neuralVQGridL3G0, ivec3(n11, channel_to_sample), 0).r);
                    break;
            }

            float f00 = get_feature_from_codebook(vq_idx_00, idx00);
            float f01 = get_feature_from_codebook(vq_idx_01, idx01);
            float f10 = get_feature_from_codebook(vq_idx_10, idx10);
            float f11 = get_feature_from_codebook(vq_idx_11, idx11);

            features[feature_count++] = f00 * w00;
            features[feature_count++] = f01 * w01;
            features[feature_count++] = f10 * w10;
            features[feature_count++] = f11 * w11;
        }
    }

    uint num_selections_g1 = materialParams.channelCounts[level_idx].y;
    if (num_selections_g1 > 0) {
        uvec3 grid_dims = materialParams.featureGridShapes[level_idx][1].xyz;
        vec2 rotated_uv = -vec2(uv.y, uv.x);
        vec2 grid_coords = rotated_uv * vec2(grid_dims.z, grid_dims.y);
        ivec2 base_coords_int = ivec2(floor(grid_coords));
        vec2 frac_coords = fract(grid_coords);

        float w11 = frac_coords.x * frac_coords.y;
        float w10 = frac_coords.x * (1.0 - frac_coords.y);
        float w01 = (1.0 - frac_coords.x) * frac_coords.y;
        float w00 = (1.0 - frac_coords.x) * (1.0 - frac_coords.y);

        ivec2 n00 = (base_coords_int + ivec2(0,0)) % ivec2(grid_dims.z, grid_dims.y);
        ivec2 n10 = (base_coords_int + ivec2(1,0)) % ivec2(grid_dims.z, grid_dims.y);
        ivec2 n01 = (base_coords_int + ivec2(0,1)) % ivec2(grid_dims.z, grid_dims.y);
        ivec2 n11 = (base_coords_int + ivec2(1,1)) % ivec2(grid_dims.z, grid_dims.y);

        vec2 patch_coords = frac_coords * 4.0;
        ivec2 patch_base_int = ivec2(floor(patch_coords));
        vec2 patch_frac = fract(patch_coords);

        uint p_idx00 = uint(clamp(patch_base_int.y,     0, 3)) * 4 + uint(clamp(patch_base_int.x,     0, 3));
        uint p_idx10 = uint(clamp(patch_base_int.y,     0, 3)) * 4 + uint(clamp(patch_base_int.x + 1, 0, 3));
        uint p_idx01 = uint(clamp(patch_base_int.y + 1, 0, 3)) * 4 + uint(clamp(patch_base_int.x,     0, 3));
        uint p_idx11 = uint(clamp(patch_base_int.y + 1, 0, 3)) * 4 + uint(clamp(patch_base_int.x + 1, 0, 3));

        uint vq_idx_00, vq_idx_10, vq_idx_01, vq_idx_11;
        for (int i = 0; i < num_selections_g1; ++i) {
            uint channel_to_sample;
            switch(level_idx) {
                case 0:
                    channel_to_sample = uint(grid1l0_channels.indices[i]);
                    vq_idx_00 = uint(texelFetch(neuralVQGridL0G1, ivec3(n00, channel_to_sample), 0).r);
                    vq_idx_10 = uint(texelFetch(neuralVQGridL0G1, ivec3(n10, channel_to_sample), 0).r);
                    vq_idx_01 = uint(texelFetch(neuralVQGridL0G1, ivec3(n01, channel_to_sample), 0).r);
                    vq_idx_11 = uint(texelFetch(neuralVQGridL0G1, ivec3(n11, channel_to_sample), 0).r);
                    break;
                case 1:
                    channel_to_sample = uint(grid1l1_channels.indices[i]);
                    vq_idx_00 = uint(texelFetch(neuralVQGridL1G1, ivec3(n00, channel_to_sample), 0).r);
                    vq_idx_10 = uint(texelFetch(neuralVQGridL1G1, ivec3(n10, channel_to_sample), 0).r);
                    vq_idx_01 = uint(texelFetch(neuralVQGridL1G1, ivec3(n01, channel_to_sample), 0).r);
                    vq_idx_11 = uint(texelFetch(neuralVQGridL1G1, ivec3(n11, channel_to_sample), 0).r);
                    break;
                case 2:
                    channel_to_sample = uint(grid1l2_channels.indices[i]);
                    vq_idx_00 = uint(texelFetch(neuralVQGridL2G1, ivec3(n00, channel_to_sample), 0).r);
                    vq_idx_10 = uint(texelFetch(neuralVQGridL2G1, ivec3(n10, channel_to_sample), 0).r);
                    vq_idx_01 = uint(texelFetch(neuralVQGridL2G1, ivec3(n01, channel_to_sample), 0).r);
                    vq_idx_11 = uint(texelFetch(neuralVQGridL2G1, ivec3(n11, channel_to_sample), 0).r);
                    break;
                case 3:
                    channel_to_sample = uint(grid1l3_channels.indices[i]);
                    vq_idx_00 = uint(texelFetch(neuralVQGridL3G1, ivec3(n00, channel_to_sample), 0).r);
                    vq_idx_10 = uint(texelFetch(neuralVQGridL3G1, ivec3(n10, channel_to_sample), 0).r);
                    vq_idx_01 = uint(texelFetch(neuralVQGridL3G1, ivec3(n01, channel_to_sample), 0).r);
                    vq_idx_11 = uint(texelFetch(neuralVQGridL3G1, ivec3(n11, channel_to_sample), 0).r);
                    break;
            }

            float v00 = interpolate_vq_patch(vq_idx_00, p_idx00, p_idx10, p_idx01, p_idx11, patch_frac);
            float v10 = interpolate_vq_patch(vq_idx_10, p_idx00, p_idx10, p_idx01, p_idx11, patch_frac);
            float v01 = interpolate_vq_patch(vq_idx_01, p_idx00, p_idx10, p_idx01, p_idx11, patch_frac);
            float v11 = interpolate_vq_patch(vq_idx_11, p_idx00, p_idx10, p_idx01, p_idx11, patch_frac);

            float interp_x1 = mix(v00, v10, frac_coords.x);
            float interp_x2 = mix(v01, v11, frac_coords.x);
            float final_feature = mix(interp_x1, interp_x2, frac_coords.y);

            features[feature_count++] = final_feature;
        }
    }
    //TODO insert padding with 0s between last feature and the positional encoding for different LODS

    // --- 4. Append Positional Encoding & LOD ---
    if(num_selections_g0 > 0 || num_selections_g1 > 0){
        vec2 absolute_coords = uv * RESOLUTION;
        append_positional_encoding_static(features, feature_count, absolute_coords);
        feature_count += 11;
    }
    features[feature_count++] = lod;

    // --- 5. MLP ---
    const float LEAKY_RELU_SLOPE = 0.01;

    // ---- Layer 0 ----
    uint layer0_out_channels = materialParams.mlpLayerShapes[0].y;
    uint layer0_in_channels = materialParams.mlpLayerShapes[0].x;
    
    // Activations should be full 32-bit floats for precision
    float layer0_activations[32]; // Keep this as float

    for(int out_ch = 0; out_ch < layer0_out_channels; ++out_ch) {
        // Load bias (F16) and convert to float
        float accumulator = float(mlpL0_B.data[out_ch]); 
        
        for(int in_ch = 0; in_ch < min(uint(feature_count), layer0_in_channels); ++in_ch) {
            // Load weight (F16), convert to float for the multiplication
            float weight = float(mlpL0_W.data[out_ch * layer0_in_channels + in_ch]);
            accumulator += features[in_ch] * weight;
        }
        layer0_activations[out_ch] = max(accumulator, accumulator * LEAKY_RELU_SLOPE);
    }

    // ---- Layer 1 ----
    uint layer1_out_channels = materialParams.mlpLayerShapes[1].y;
    uint layer1_in_channels = materialParams.mlpLayerShapes[1].x;
    float layer1_activations[32]; // Keep as float

    for(int out_ch = 0; out_ch < layer1_out_channels; ++out_ch) {
        float accumulator = float(mlpL1_B.data[out_ch]);
        
        // Input to this layer comes from the previous layer's activations
        for(int in_ch = 0; in_ch < min(layer0_out_channels, layer1_in_channels); ++in_ch) {
            float weight = float(mlpL1_W.data[out_ch * layer1_in_channels + in_ch]);
            accumulator += layer0_activations[in_ch] * weight;
        }
        layer1_activations[out_ch] = max(accumulator, accumulator * LEAKY_RELU_SLOPE);
    }
    
    // ---- Layer 2 (Final Layer) ----
    uint final_out_channels = materialParams.mlpLayerShapes[2].y;
    uint final_in_channels = materialParams.mlpLayerShapes[2].x;
    vec4 final_color = vec4(0.0, 0.0, 0.0, 1.0);

    for(int out_ch = 0; out_ch < final_out_channels; ++out_ch) {
        float accumulator = float(mlpL2_B.data[out_ch]);
        
        for(int in_ch = 0; in_ch < min(layer1_out_channels, final_in_channels); ++in_ch) {
            float weight = float(mlpL2_W.data[out_ch * final_in_channels + in_ch]);
            accumulator += layer1_activations[in_ch] * weight;
        }
        
        if(out_ch < 4) {
          final_color[out_ch] = accumulator;
        }
    }

    // Return the calculated color
    return final_color; 
}

void main() {
    if (materialParams.isNeuralTexture == 1u) {
        fragColor = evaluate_neural_texture(inTexCoord, pushConstant.lod);
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

