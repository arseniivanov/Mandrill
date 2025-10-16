#version 460
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_NV_cooperative_vector : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#extension GL_EXT_shader_realtime_clock : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_ARB_gpu_shader_int64 : require

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
#define MAX_MATERIAL_CHANNELS 16 

const uint DIFFUSE_TEXTURE_BIT  = 1 << 0;
const uint SPECULAR_TEXTURE_BIT = 1 << 1;
const uint AMBIENT_TEXTURE_BIT = 1 << 2;
const uint EMISSION_TEXTURE_BIT = 1 << 3;
const uint NORMAL_TEXTURE_BIT = 1 << 4;

#define USE_COOP_VEC

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
    vec4 denormMean[MAX_MATERIAL_CHANNELS / 4];
    vec4 denormStd[MAX_MATERIAL_CHANNELS / 4];
    uint denormChannelCount;
    float denorm_pad0;
    float denorm_pad1;
    float denorm_pad2;
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

layout(set = 3, binding = 4) buffer TimingBuffer {
    uint64_t timestamps[4]; // 0: start, 1: after grid0, 2: after grid1, 3: end
} timingBuffer;

// =========================================================================
// NEURAL TEXTURING HELPER FUNCTIONS
// =========================================================================

const int POS_ENCODING_TABLE_SIZE = 8;
const int POS_ENCODING_FEATURES_PER_DIM = 6; // 3 octaves * 2 offsets

float get_feature_from_codebook(uint vq_index, uint feature_index) {
    // 1. Get the 32-bit packed data for the entire patch.
    uint packed_dword = vqCodebookBuffer.data[vq_index];

    // 2. Calculate the direct bit offset.
    // Each feature is 2 bits. feature_index=0 is at bit 0, feature_index=13 is at bit 26.
    uint bit_offset = feature_index * 2;

    // 3. Shift the entire dword right by the offset and mask the lowest 2 bits.
    uint palette_index = (packed_dword >> bit_offset) & 0x03u; // 0x03u is binary 11

    // 4. Look up the value in the palette.
    return paletteBuffer.data[palette_index];
}

int mod(int x, int m) {
    return (x % m + m) % m;
}

void append_positional_encoding_static(inout float features[128], int base_feature_index, vec2 coords) {
    ivec2 int_coords = ivec2(coords);
    int ix = mod(int_coords.x, POS_ENCODING_TABLE_SIZE);
    int iy = mod(int_coords.y, POS_ENCODING_TABLE_SIZE);

    // Calculate the starting offset into the buffer for our x and y table entries
    int base_lookup_idx_x = ix * POS_ENCODING_FEATURES_PER_DIM;
    int base_lookup_idx_y = iy * POS_ENCODING_FEATURES_PER_DIM;

    // Manually unroll all 12 feature appends.
    features[base_feature_index + 0] = posEncodingBuffer.data[base_lookup_idx_x + 0];
    features[base_feature_index + 1] = posEncodingBuffer.data[base_lookup_idx_x + 1];
    features[base_feature_index + 2] = posEncodingBuffer.data[base_lookup_idx_x + 2];
    features[base_feature_index + 3] = posEncodingBuffer.data[base_lookup_idx_x + 3];
    features[base_feature_index + 4] = posEncodingBuffer.data[base_lookup_idx_x + 4];
    features[base_feature_index + 5] = posEncodingBuffer.data[base_lookup_idx_x + 5];

    features[base_feature_index + 6] = posEncodingBuffer.data[base_lookup_idx_y + 0];
    features[base_feature_index + 7] = posEncodingBuffer.data[base_lookup_idx_y + 1];
    features[base_feature_index + 8] = posEncodingBuffer.data[base_lookup_idx_y + 2];
    features[base_feature_index + 9] = posEncodingBuffer.data[base_lookup_idx_y + 3];
    features[base_feature_index + 10] = posEncodingBuffer.data[base_lookup_idx_y + 4];
    features[base_feature_index + 11] = posEncodingBuffer.data[base_lookup_idx_y + 5];
}

vec4 evaluate_neural_texture(vec2 uv, float lod) {
    bool should_profile = (ivec2(gl_FragCoord.xy) == ivec2(RESOLUTION/2, RESOLUTION/2));

    // CHECKPOINT 0: Start of the function
    if (should_profile) {
        timingBuffer.timestamps[0] = clockRealtimeEXT();
    }

    int level_idx = 0;
    if (lod < 4.0) level_idx = 0;
    else if (lod < 6.0) level_idx = 1;
    else if (lod < 8.0) level_idx = 2;
    else level_idx = 3;

    const int MAX_TOTAL_FEATURES = 128;
    float features[MAX_TOTAL_FEATURES];
    int feature_count = 0;
    
    vec2 python_uv = vec2(1.0 - uv.y, uv.x);
    //vec2 python_uv = vec2(uv.x, 1.0 - uv.y);
    vec2 absolute_coords = python_uv * RESOLUTION;

    // --- 2. Grid 0 Feature Gathering (4 scaled features per channel) ---
    uint num_selections_g0 = materialParams.channelCounts[level_idx].x;
    if (num_selections_g0 > 0) {
        uvec3 grid_dims = materialParams.featureGridShapes[level_idx][0].xyz; //feature grid shape (channels, width, height)

        // We simulate an unfolded feature space (64x64 -> 256x256 for LOD 0)
        uvec2 conceptual_map_size = uvec2(grid_dims.z * 4u, grid_dims.y * 4u);

        vec2 python_space_float_coord = python_uv * (vec2(conceptual_map_size));
        
        ivec2 unwrapped_p00_coord = ivec2(floor(python_space_float_coord)); //Coordinate in uncompressed (0, grid_dims) space
        vec2 frac = fract(python_space_float_coord);

        //Bilinear weights for the coordinates in the uncompressed feature space
        float w11 = frac.x * frac.y;
        float w10 = frac.x * (1.0 - frac.y);
        float w01 = (1.0 - frac.x) * frac.y;
        float w00 = (1.0 - frac.x) * (1.0 - frac.y);
        
        //Wrapped coordinates in uncompressed (256,256)-space
        ivec2 unwrapped_p10_coord = unwrapped_p00_coord + ivec2(1, 0);
        ivec2 unwrapped_p01_coord = unwrapped_p00_coord + ivec2(0, 1);
        ivec2 unwrapped_p11_coord = unwrapped_p00_coord + ivec2(1, 1);

        uvec2 p00_coord = uvec2(unwrapped_p00_coord) % conceptual_map_size;
        uvec2 p10_coord = uvec2(unwrapped_p10_coord) % conceptual_map_size;
        uvec2 p01_coord = uvec2(unwrapped_p01_coord) % conceptual_map_size;
        uvec2 p11_coord = uvec2(unwrapped_p11_coord) % conceptual_map_size;

        uint idx00 = (p00_coord.y % 4u) * 4u + (p00_coord.x % 4u);
        uint idx10 = (p10_coord.y % 4u) * 4u + (p10_coord.x % 4u);
        uint idx01 = (p01_coord.y % 4u) * 4u + (p01_coord.x % 4u);
        uint idx11 = (p11_coord.y % 4u) * 4u + (p11_coord.x % 4u);

        uvec2 n00 = uvec2(p00_coord.x / 4u, p00_coord.y / 4u);
        uvec2 n10 = uvec2(p10_coord.x / 4u, p10_coord.y / 4u);
        uvec2 n01 = uvec2(p01_coord.x / 4u, p01_coord.y / 4u);
        uvec2 n11 = uvec2(p11_coord.x / 4u, p11_coord.y / 4u);

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
            float f10 = get_feature_from_codebook(vq_idx_10, idx10);
            float f01 = get_feature_from_codebook(vq_idx_01, idx01);
            float f11 = get_feature_from_codebook(vq_idx_11, idx11);

            features[feature_count++] = f00 * w00;
            features[feature_count++] = f01 * w01;
            features[feature_count++] = f10 * w10;
            features[feature_count++] = f11 * w11;
            //features[feature_count++] = 0.0f;
            //features[feature_count++] = 0.0f;
            //features[feature_count++] = 0.0f;
            //features[feature_count++] = 0.0f;
        }
    }

    if (should_profile) {
        timingBuffer.timestamps[1] = clockRealtimeEXT();
    }

    uint num_selections_g1 = materialParams.channelCounts[level_idx].y;
    if (num_selections_g1 > 0) {
        uvec3 grid_dims = materialParams.featureGridShapes[level_idx][1].xyz;
        uvec2 conceptual_map_size = uvec2(grid_dims.z * 4u, grid_dims.y * 4u);

        // 1. Convert to Python's top-left coordinate system. This is the key first step.
        vec2 python_space_float_coord = python_uv * vec2(conceptual_map_size);
        
        // 2. Get the base coordinate and fractional part for interpolation.
        ivec2 unwrapped_p00_coord = ivec2(floor(python_space_float_coord));
        vec2 frac = fract(python_space_float_coord);

        // 4. Find all four neighbors in the unwrapped space BEFORE applying the modulus.
        ivec2 unwrapped_p10_coord = unwrapped_p00_coord + ivec2(1, 0);
        ivec2 unwrapped_p01_coord = unwrapped_p00_coord + ivec2(0, 1);
        ivec2 unwrapped_p11_coord = unwrapped_p00_coord + ivec2(1, 1);

        // 5. Apply wrapping to each coordinate individually.
        uvec2 p00_coord = uvec2(unwrapped_p00_coord) % conceptual_map_size;
        uvec2 p10_coord = uvec2(unwrapped_p10_coord) % conceptual_map_size;
        uvec2 p01_coord = uvec2(unwrapped_p01_coord) % conceptual_map_size;
        uvec2 p11_coord = uvec2(unwrapped_p11_coord) % conceptual_map_size;

        uint idx00 = (p00_coord.y % 4u) * 4u + (p00_coord.x % 4u);
        uint idx10 = (p10_coord.y % 4u) * 4u + (p10_coord.x % 4u);
        uint idx01 = (p01_coord.y % 4u) * 4u + (p01_coord.x % 4u);
        uint idx11 = (p11_coord.y % 4u) * 4u + (p11_coord.x % 4u);

        uvec2 n00 = uvec2(p00_coord.x / 4u, p00_coord.y / 4u);
        uvec2 n10 = uvec2(p10_coord.x / 4u, p10_coord.y / 4u);
        uvec2 n01 = uvec2(p01_coord.x / 4u, p01_coord.y / 4u);
        uvec2 n11 = uvec2(p11_coord.x / 4u, p11_coord.y / 4u);

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

            float f00 = get_feature_from_codebook(vq_idx_00, idx00);
            float f10 = get_feature_from_codebook(vq_idx_10, idx10);
            float f01 = get_feature_from_codebook(vq_idx_01, idx01);
            float f11 = get_feature_from_codebook(vq_idx_11, idx11);

            float interp_x1 = mix(f00, f10, frac.x);
            float interp_x2 = mix(f01, f11, frac.x);
            float final_feature = mix(interp_x1, interp_x2, frac.y);

            features[feature_count++] = final_feature;
        }
    }
    //TODO insert padding with 0s between last feature and the LOD for different LODS
    //TODO normalize LODS

    //Hardcoded normalizied LOD0 
    features[feature_count++] = -1.0f;

    append_positional_encoding_static(features, feature_count, absolute_coords);
    feature_count += 12;


    if (should_profile) {
        timingBuffer.timestamps[2] = clockRealtimeEXT();
    }

    // --- 5. MLP ---
    const float LEAKY_RELU_SLOPE = 0.01;

#ifdef USE_COOP_VEC
#define COOP_VECTOR_TYPE gl_ComponentTypeFloat16NV
    const uint layer0_out_channels = 32;
    const uint layer0_in_channels  = 32;
    const uint layer1_out_channels = 32;
    const uint layer1_in_channels  = 32;

    // --------------------
    // Input as coop-vector
    // --------------------
    coopvecNV<float16_t, 32> input_features_vec;
    for (int i = 0; i < 32; ++i) {
        input_features_vec[i] = float16_t(features[i]);
    }

    // --------------------
    // Layer 0: MatMul + Bias + LeakyReLU
    // --------------------
    coopvecNV<float16_t, 32> layer0_activations_vec;

    coopVecMatMulAddNV(
        layer0_activations_vec,         // out: activations of layer 0
        input_features_vec,             // in: feature vector
        gl_ComponentTypeFloat16NV,
        mlpL0_W.data, 0,                // layer 0 weight matrix
        gl_ComponentTypeFloat16NV,
        mlpL0_B.data, 0,                // layer 0 bias vector
        gl_ComponentTypeFloat16NV,
        layer0_out_channels,            // M
        layer0_in_channels,             // K
        gl_CooperativeVectorMatrixLayoutColumnMajorNV, // matches your memory layout
        false,                          // no transpose (flip if you see wrong results)
        0u
    );

    // Apply Leaky ReLU activation in-place
    layer0_activations_vec =
        max(layer0_activations_vec,
            layer0_activations_vec * float16_t(LEAKY_RELU_SLOPE));

    // --------------------
    // Layer 1: MatMul + Bias + LeakyReLU
    // --------------------
    coopvecNV<float16_t, 32> layer1_activations_vec;

    coopVecMatMulAddNV(
        layer1_activations_vec,
        layer0_activations_vec,
        gl_ComponentTypeFloat16NV,
        mlpL1_W.data, 0,
        gl_ComponentTypeFloat16NV,
        mlpL1_B.data, 0,
        gl_ComponentTypeFloat16NV,
        layer1_out_channels,
        layer1_in_channels,
        gl_CooperativeVectorMatrixLayoutColumnMajorNV,
        false,
        0u
    );

    // Apply Leaky ReLU
    layer1_activations_vec =
        max(layer1_activations_vec,
            layer1_activations_vec * float16_t(LEAKY_RELU_SLOPE));

    // --------------------
    // Convert layer 1 output back to float for final layer
    // --------------------
    float layer1_activations[32];
    for (int i = 0; i < 32; ++i) {
        layer1_activations[i] = float(layer1_activations_vec[i]);
    }

#else
    // --------------------
    // CPU-style fallback (layer 0 + 1 as scalar loops)
    // --------------------
    uint layer0_out_channels = materialParams.mlpLayerShapes[0].y;
    uint layer0_in_channels  = materialParams.mlpLayerShapes[0].x;

    float layer0_activations[32];

    for (int out_ch = 0; out_ch < layer0_out_channels; ++out_ch) {
        float accumulator = float(mlpL0_B.data[out_ch]);
        for (int in_ch = 0; in_ch < min(uint(feature_count), layer0_in_channels); ++in_ch) {
            float weight = float(mlpL0_W.data[out_ch * layer0_in_channels + in_ch]);
            accumulator += features[in_ch] * weight;
        }
        layer0_activations[out_ch] =
            max(accumulator, accumulator * LEAKY_RELU_SLOPE);
    }

    uint layer1_out_channels = materialParams.mlpLayerShapes[1].y;
    uint layer1_in_channels  = materialParams.mlpLayerShapes[1].x;

    float layer1_activations[32];

    for (int out_ch = 0; out_ch < layer1_out_channels; ++out_ch) {
        float accumulator = float(mlpL1_B.data[out_ch]);
        for (int in_ch = 0; in_ch < min(layer0_out_channels, layer1_in_channels); ++in_ch) {
            float weight = float(mlpL1_W.data[out_ch * layer1_in_channels + in_ch]);
            accumulator += layer0_activations[in_ch] * weight;
        }
        layer1_activations[out_ch] =
            max(accumulator, accumulator * LEAKY_RELU_SLOPE);
    }
#endif

    // ---- Layer 2 (Final Layer) ----
    uint final_out_channels = materialParams.mlpLayerShapes[2].y;
    uint final_in_channels  = materialParams.mlpLayerShapes[2].x;
    vec4 final_color = vec4(0.0, 0.0, 0.0, 1.0);

    for (int out_ch = 0; out_ch < final_out_channels; ++out_ch) {
        float accumulator = float(mlpL2_B.data[out_ch]);
        for (int in_ch = 0; in_ch < min(layer1_out_channels, final_in_channels); ++in_ch) {
            float weight = float(mlpL2_W.data[out_ch * final_in_channels + in_ch]);
            accumulator += layer1_activations[in_ch] * weight;
        }
        if (out_ch < materialParams.denormChannelCount) {
            float mean = materialParams.denormMean[out_ch / 4][out_ch % 4];
            float std  = materialParams.denormStd[out_ch / 4][out_ch % 4];
            accumulator = accumulator * std + mean;
        }
        if (out_ch < 4) {
            final_color[out_ch] = accumulator;
        }
    }

    if (distance(vec2(0.0,0.0), python_uv) < 0.02){
        final_color = vec4(1.0,0.0,1.0,1.0);
        return final_color;
      }
    // Return the calculated color

    if (should_profile) {
        timingBuffer.timestamps[3] = clockRealtimeEXT();
    }

    return final_color; 
}

void main() {
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

      // NTC 
      if (pushConstant.renderMode == 9) {
          vec2 uv = inTexCoord;
          fragColor = evaluate_neural_texture(uv, pushConstant.lod);

      // Line render
      if (pushConstant.renderMode == 10) {
          fragColor = vec4(pushConstant.lineColor, 1.0);
      }
  }
}
