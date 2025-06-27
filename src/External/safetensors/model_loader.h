// model_loader.hpp
#ifndef MODEL_LOADER_HPP
#define MODEL_LOADER_HPP

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <vector>

// Include the provided safetensors.hh directly, as its types are used in our
// structs. Ensure this file is in your include path or the same directory.
#include "safetensors.hh"

// --- Data Structures to Hold Loaded Model ---

struct Palette {
    std::vector<float> values; // Should be size 4
};

struct VQCodebook {
    std::vector<uint8_t> packed_data;
    std::vector<size_t> shape; // Shape of the packed_data (e.g., [K, packed_bytes_per_patch])
                               // Note: safetensors.hh uses size_t for shape dimensions
};

struct FeatureGridData {
    std::string name;
    std::vector<uint8_t> data_uint8;
    std::vector<size_t> shape;

    int level_idx = -1;
    int grid_type = -1; // 0 for G0, 1 for G1
    int channel_idx = -1;
    std::string base_feature_key;

    FeatureGridData(const std::string& n = "");
};

struct ChannelSelections {
    std::vector<uint16_t> grid0_selected_channels;
    std::vector<uint16_t> grid1_selected_channels;
};

struct MLPLayer {
    std::string material_id;
    int layer_idx = -1;

    std::vector<uint8_t> weights_raw_bytes;
    std::vector<size_t> weights_shape;
    safetensors::dtype weights_dtype;

    std::vector<uint8_t> bias_raw_bytes;
    std::vector<size_t> bias_shape;
    safetensors::dtype bias_dtype;

    MLPLayer();

    template <typename T> std::vector<T> get_weights_typed() const
    {
        if (weights_raw_bytes.empty() || sizeof(T) == 0 || weights_raw_bytes.size() % sizeof(T) != 0) {
            if (!weights_raw_bytes.empty())
                std::cerr << "Warning: MLP Layer '" << material_id << "' L" << layer_idx
                          << " - Invalid weights data or type mismatch for typed access." << std::endl;
            return {};
        }
        std::vector<T> typed_weights(weights_raw_bytes.size() / sizeof(T));
        std::memcpy(typed_weights.data(), weights_raw_bytes.data(), weights_raw_bytes.size());
        return typed_weights;
    }

    template <typename T> std::vector<T> get_bias_typed() const
    {
        if (bias_raw_bytes.empty() || sizeof(T) == 0 || bias_raw_bytes.size() % sizeof(T) != 0) {
            if (!bias_raw_bytes.empty())
                std::cerr << "Warning: MLP Layer '" << material_id << "' L" << layer_idx
                          << " - Invalid bias data or type mismatch for typed access." << std::endl;
            return {};
        }
        std::vector<T> typed_bias(bias_raw_bytes.size() / sizeof(T));
        std::memcpy(typed_bias.data(), bias_raw_bytes.data(), bias_raw_bytes.size());
        return typed_bias;
    }
};

struct MaterialSpecificData {
    std::string id;
    std::map<int, ChannelSelections> level_channel_selections;
    std::vector<MLPLayer> mlp_layers;
    std::vector<float> denorm_mean;
    std::vector<float> denorm_std;
};

struct SafetensorsModelData {
    Palette palette;
    VQCodebook vq_codebook;
    std::map<std::string, FeatureGridData> named_feature_grids;
    std::map<std::string, MaterialSpecificData> materials;

    bool uses_vq = true;
    bool uses_combined_features = false;
    std::string combined_feature_key_name;
    int max_level_idx = -1;

    SafetensorsModelData();
};

bool load_model_from_safetensors(const std::string& filename, SafetensorsModelData& model_data);
void print_model_data_summary(const SafetensorsModelData& data);

#endif // MODEL_LOADER_HPP
