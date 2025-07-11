#define SAFETENSORS_CPP_IMPLEMENTATION
#include "model_loader.h"
#include <cmath>
#include <cstdint>
#include <limits>


// Define SAFETENSORS_CPP_IMPLEMENTATION in exactly one .cc file
// This is now handled by safetensors.hh when SAFETENSORS_CPP_IMPLEMENTATION is
// defined before including it. No, it should be here if model_loader.cpp is the
// one .cc file defining it. If main.cpp defines it, then remove from here.
// Let's assume main.cpp defines it for now. #define
// SAFETENSORS_CPP_IMPLEMENTATION // Will be in main.cpp

// --- Constructor Implementations ---
FeatureGridData::FeatureGridData(const std::string& n) : name(n)
{
}

MLPLayer::MLPLayer() : weights_dtype(safetensors::dtype::kFLOAT32), bias_dtype(safetensors::dtype::kFLOAT32)
{ // Default to something, will
  // be overwritten
  // kFLOAT32 is just a placeholder; it gets set from the tensor.
  // Using kBOOL or similar as "NONE" could also work if needed, but the
  // safetensors.hh doesn't seem to have an explicit NONE/UNDEFINED dtype. Check
  // if bias_raw_bytes is empty to see if bias exists.
}
SafetensorsModelData::SafetensorsModelData() : uses_vq(false), uses_combined_features(false), max_level_idx(-1)
{
}

// Helper to convert a 16-bit float (represented as uint16_t) to a 32-bit float for printing
float fp16_to_fp32(uint16_t half)
{
    uint32_t sign = (half >> 15) & 0x0001;
    uint32_t exponent = (half >> 10) & 0x001F;
    uint32_t mantissa = half & 0x03FF;

    if (exponent == 0) {
        if (mantissa == 0) { // Plus or minus zero
            return sign ? -0.0f : 0.0f;
        } else { // Denormalized number
            while (!(mantissa & 0x0400)) {
                mantissa <<= 1;
                exponent--;
            }
            exponent++;
            mantissa &= ~0x0400;
            float val = std::ldexp(static_cast<float>(mantissa), -10);
            return sign ? -val : val;
        }
    } else if (exponent == 31) {
        if (mantissa == 0) { // Inf
            return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        } else { // NaN
            return std::numeric_limits<float>::quiet_NaN();
        }
    }

    exponent = exponent + (127 - 15);
    mantissa = mantissa << 13;

    uint32_t full_float_bits = (sign << 31) | (exponent << 23) | mantissa;
    float result;
    std::memcpy(&result, &full_float_bits, sizeof(float));
    return result;
}

// --- Helper functions for parsing tensor names (same as before) ---
bool parse_denorm_tensor_name(const std::string& name, std::string& material_id, bool& is_mean)
{
    static const std::regex denorm_regex("denorm_([a-zA-Z0-9_\\.\\-]+)_(mean|std)");
    std::smatch match;
    if (std::regex_match(name, match, denorm_regex)) {
        material_id = match[1].str();
        is_mean = (match[2].str() == "mean");
        return true;
    }
    return false;
}

bool parse_mlp_tensor_name(const std::string& name, std::string& material_id, int& layer_idx, bool& is_weights)
{
    static const std::regex mlp_regex("mlp_([a-zA-Z0-9_\\.\\-]+)_layer_(\\d+)_(weights|bias)");
    std::smatch match;
    if (std::regex_match(name, match, mlp_regex)) {
        material_id = match[1].str();
        layer_idx = std::stoi(match[2].str());
        is_weights = (match[3].str() == "weights");
        return true;
    }
    return false;
}

bool parse_channel_selection_name(const std::string& name, std::string& material_id, int& level_idx,
                                  int& grid_type_0or1)
{
    static const std::regex sel_regex("([a-zA-Z0-9_\\.\\-]+)_level_(\\d+)_grid(0|1)_channel_selection_uint16");
    std::smatch match;
    if (std::regex_match(name, match, sel_regex)) {
        material_id = match[1].str();
        level_idx = std::stoi(match[2].str());
        grid_type_0or1 = std::stoi(match[3].str());
        return true;
    }
    return false;
}

void parse_feature_grid_name_details(FeatureGridData& fgd)
{
    static const std::regex vq_grid_regex("([a-zA-Z0-9_\\.\\-]+)_level_(\\d+)_grid_(\\d+)_vq_indices_uint8");
    static const std::regex raw_packed_grid_regex("packed_feature_grid_([a-zA-Z0-9_\\.\\-]+)_L(\\d+)_G(\\d+)_ch(\\d+)_"
                                                  "uint8");
    std::smatch match;

    if (std::regex_match(fgd.name, match, vq_grid_regex)) {
        fgd.base_feature_key = match[1].str();
        fgd.level_idx = std::stoi(match[2].str());
        fgd.grid_type = std::stoi(match[3].str());
    } else if (std::regex_match(fgd.name, match, raw_packed_grid_regex)) {
        fgd.base_feature_key = match[1].str();
        fgd.level_idx = std::stoi(match[2].str());
        fgd.grid_type = std::stoi(match[3].str());
        fgd.channel_idx = std::stoi(match[4].str());
    }
}

// Helper to copy data from safetensors_t storage
void copy_tensor_data(std::vector<uint8_t>& dest, const safetensors::safetensors_t& st_data,
                      const safetensors::tensor_t& tensor_info)
{
    size_t start_offset = tensor_info.data_offsets[0];
    size_t end_offset = tensor_info.data_offsets[1];
    size_t byte_length = end_offset - start_offset;

    if (byte_length == 0) {
        dest.clear();
        return;
    }

    const uint8_t* data_source_ptr = nullptr;
    size_t source_total_size = 0;

    if (st_data.mmaped) {
        data_source_ptr = st_data.databuffer_addr;
        source_total_size = st_data.databuffer_size;
    } else {
        data_source_ptr = st_data.storage.data();
        source_total_size = st_data.storage.size();
    }

    if (!data_source_ptr) {
        std::cerr << "Error: Data source pointer is null." << std::endl;
        dest.clear();
        return;
    }

    if (end_offset > source_total_size) {
        std::cerr << "Error: Tensor data_offsets [" << start_offset << ", " << end_offset
                  << "] exceed data buffer size " << source_total_size << std::endl;
        dest.clear();
        return;
    }

    dest.resize(byte_length);
    std::memcpy(dest.data(), data_source_ptr + start_offset, byte_length);
}

// --- Main Loading Function ---
bool load_model_from_safetensors(const std::string& filename, SafetensorsModelData& model_data)
{
    safetensors::safetensors_t st_data; // Changed from SafeTensors to safetensors_t
    std::string warn, err;

    // Using load_from_file which populates st_data.storage
    bool ret = safetensors::load_from_file(filename, &st_data, &warn, &err);

    if (!warn.empty()) {
        std::cout << "SAFETENSORS WARN: " << warn << "\n";
    }
    if (!ret) {
        std::cerr << "Failed to load safetensors file: " << filename << "\n";
        if (!err.empty())
            std::cerr << "  ERR: " << err << "\n";
        return false;
    }

    // Example of how you might check metadata if Python saved it in
    // st_data.metadata (Using ordered_dict<std::string> st_data.metadata;)
    // std::string uses_vq_str;
    // if (st_data.metadata.at("uses_vq_codebook", &uses_vq_str)) { // Check for
    // key existence
    //    model_data.uses_vq = (uses_vq_str == "True");
    // }

    // Iterate through tensors using the ordered_dict interface
    const auto& tensor_keys = st_data.tensors.keys();
    for (const std::string& name : tensor_keys) {
        safetensors::tensor_t tensor_info;
        if (!st_data.tensors.at(name, &tensor_info)) {
            std::cerr << "Error: Could not retrieve tensor info for key: " << name << std::endl;
            continue;
        }

        static const std::regex vq_grid_regex("([a-zA-Z0-9_\\.\\-]+)_level_(\\d+)_grid_(\\d+)_vq_indices_uint8");
        static const std::regex raw_packed_grid_regex(
            "packed_feature_grid_([a-zA-Z0-9_\\.\\-]+)_L(\\d+)_G(\\d+)_uint32");
        static const std::regex raw_packed_grid_dims_regex(
            "packed_feature_grid_([a-zA-Z0-9_\\.\\-]+)_L(\\d+)_G(\\d+)_uint32_dims");

        std::smatch match;

        if (name == "vq_codebook_palette_float32") {
            if (tensor_info.dtype != safetensors::dtype::kFLOAT32) {
                std::cerr << "Error: Palette tensor '" << name << "' has unexpected dtype. Expected F32." << std::endl;
                continue;
            }
            std::vector<uint8_t> raw_palette_data;
            copy_tensor_data(raw_palette_data, st_data, tensor_info);
            if (raw_palette_data.size() % sizeof(float) != 0) {
                std::cerr << "Error: Palette data size not multiple of float size." << std::endl;
                continue;
            }
            size_t num_elements = raw_palette_data.size() / sizeof(float);
            model_data.palette.values.resize(num_elements);
            std::memcpy(model_data.palette.values.data(), raw_palette_data.data(), raw_palette_data.size());
        } else if (name == "vq_codebook_patches_packed_uint32") {
            if (tensor_info.dtype != safetensors::dtype::kUINT32) {
                std::cerr << "Error: VQ Codebook tensor '" << name << "' has unexpected dtype. Expected U32."
                          << std::endl;
                model_data.uses_vq = true; // Set flag when codebook is found
                continue;
            }
            copy_tensor_data(model_data.vq_codebook.packed_data, st_data, tensor_info);
            model_data.vq_codebook.shape = tensor_info.shape;
            model_data.uses_vq = true;
        } else if (std::regex_match(name, match, raw_packed_grid_regex)) {
            model_data.uses_vq = false; // This is a non-VQ model

            // Construct a consistent name to use as a map key
            std::string grid_map_key = "packed_L" + match[2].str() + "_G" + match[3].str();

            FeatureGridData& fgd = model_data.named_feature_grids[grid_map_key];
            if (fgd.name.empty()) {
                fgd.name = grid_map_key;
                fgd.base_feature_key = match[1].str();
                fgd.level_idx = std::stoi(match[2].str());
                fgd.grid_type = std::stoi(match[3].str());
            }

            copy_tensor_data(fgd.raw_data_bytes, st_data, tensor_info);
            fgd.shape = tensor_info.shape; // Shape of the packed uint32 data
        }
        // --- Packed Raw Dimensions Handling ---
        else if (std::regex_match(name, match, raw_packed_grid_dims_regex)) {
            model_data.uses_vq = false; // This is a non-VQ model

            std::string grid_map_key = "packed_L" + match[2].str() + "_G" + match[3].str();
            FeatureGridData& fgd = model_data.named_feature_grids[grid_map_key];
            if (fgd.name.empty()) {
                fgd.name = grid_map_key;
                fgd.base_feature_key = match[1].str();
                fgd.level_idx = std::stoi(match[2].str());
                fgd.grid_type = std::stoi(match[3].str());
            }

            std::vector<uint8_t> dims_bytes;
            copy_tensor_data(dims_bytes, st_data, tensor_info);
            if (dims_bytes.size() == 3 * sizeof(int32_t)) {
                const int32_t* dims_ptr = reinterpret_cast<const int32_t*>(dims_bytes.data());
                fgd.original_dims.assign(dims_ptr, dims_ptr + 3);
            } else {
                std::cerr << "Error: Dimensions tensor '" << name << "' has incorrect size." << std::endl;
            }
        } else if (name.rfind("mlp_", 0) == 0) { // Starts with "mlp_"
            std::string material_id_mlp;
            int layer_idx_mlp;
            bool is_weights_mlp;
            if (parse_mlp_tensor_name(name, material_id_mlp, layer_idx_mlp, is_weights_mlp)) {
                MaterialSpecificData& mat_data = model_data.materials[material_id_mlp];
                if (mat_data.id.empty())
                    mat_data.id = material_id_mlp;

                MLPLayer* layer_ptr = nullptr;
                auto it = std::find_if(mat_data.mlp_layers.begin(), mat_data.mlp_layers.end(),
                                       [layer_idx_mlp](const MLPLayer& l) { return l.layer_idx == layer_idx_mlp; });
                if (it == mat_data.mlp_layers.end()) {
                    mat_data.mlp_layers.emplace_back();
                    layer_ptr = &mat_data.mlp_layers.back();
                    layer_ptr->material_id = material_id_mlp;
                    layer_ptr->layer_idx = layer_idx_mlp;
                } else {
                    layer_ptr = &(*it);
                }

                if (is_weights_mlp) {
                    copy_tensor_data(layer_ptr->weights_raw_bytes, st_data, tensor_info);
                    layer_ptr->weights_shape = tensor_info.shape;
                    layer_ptr->weights_dtype = tensor_info.dtype;
                } else { // Bias
                    copy_tensor_data(layer_ptr->bias_raw_bytes, st_data, tensor_info);
                    layer_ptr->bias_shape = tensor_info.shape;
                    layer_ptr->bias_dtype = tensor_info.dtype;
                }
            } else {
                std::cout << "Warning: Could not parse MLP tensor name: " << name << std::endl;
            }
        } else if (name.find("_channel_selection_uint16") != std::string::npos) {
            std::string material_id_sel;
            int level_idx_sel;
            int grid_type_sel;
            if (parse_channel_selection_name(name, material_id_sel, level_idx_sel, grid_type_sel)) {
                if (tensor_info.dtype != safetensors::dtype::kUINT16) {
                    std::cerr << "Error: Channel selection tensor '" << name << "' has unexpected dtype. Expected U16."
                              << std::endl;
                    continue;
                }
                MaterialSpecificData& mat_data = model_data.materials[material_id_sel];
                if (mat_data.id.empty())
                    mat_data.id = material_id_sel;

                ChannelSelections& sels = mat_data.level_channel_selections[level_idx_sel];

                std::vector<uint8_t> raw_selection_data;
                copy_tensor_data(raw_selection_data, st_data, tensor_info);
                if (raw_selection_data.size() % sizeof(uint16_t) != 0) {
                    std::cerr << "Error: Channel selection data size not multiple of "
                                 "uint16_t size for "
                              << name << std::endl;
                    continue;
                }
                size_t num_elements = raw_selection_data.size() / sizeof(uint16_t);

                std::vector<uint16_t>* target_vec =
                    (grid_type_sel == 0) ? &sels.grid0_selected_channels : &sels.grid1_selected_channels;
                target_vec->resize(num_elements);
                std::memcpy(target_vec->data(), raw_selection_data.data(), raw_selection_data.size());

                model_data.max_level_idx = std::max(model_data.max_level_idx, level_idx_sel);
            } else {
                std::cout << "Warning: Could not parse channel selection tensor name: " << name << std::endl;
            }
        } else if (name.find("_vq_indices_uint8") != std::string::npos || name.rfind("packed_feature_grid_", 0) == 0) {
            if (tensor_info.dtype != safetensors::dtype::kUINT8) {
                std::cerr << "Error: Feature grid tensor '" << name << "' has unexpected dtype. Expected U8."
                          << std::endl;
                continue;
            }
            FeatureGridData fgd(name);
            copy_tensor_data(fgd.raw_data_bytes, st_data, tensor_info);
            fgd.shape = tensor_info.shape;
            parse_feature_grid_name_details(fgd); // Parse after getting name
            model_data.named_feature_grids[name] = fgd;

            if (name.find("_vq_indices_uint8") != std::string::npos) { // This is a VQ Index Grid
                model_data.uses_vq = true;                             // Definitely using VQ if we find these

                if (!fgd.base_feature_key.empty()) {
                    if (model_data.combined_feature_key_name.empty()) {
                        // This is the first VQ index grid encountered, assume its base key is *the* shared key
                        model_data.combined_feature_key_name = fgd.base_feature_key;
                        model_data.uses_combined_features =
                            true; // If VQ grids are found, and they have a base_key, assume combined for now
                        std::cout << "Info: Set combined_feature_key_name to '" << model_data.combined_feature_key_name
                                  << "' based on VQ grid: " << name << std::endl;
                    } else if (model_data.combined_feature_key_name != fgd.base_feature_key) {
                        // We found another VQ index grid with a *different* base key.
                        // This is unexpected if all VQ grids are truly shared under one key.
                        std::cerr << "Warning: Multiple base_feature_keys ('" << model_data.combined_feature_key_name
                                  << "' and '" << fgd.base_feature_key << "') found for VQ index grids. "
                                  << "Sticking with the first one. This could indicate a model configuration issue."
                                  << std::endl;
                        // Keep uses_combined_features = true based on the first key.
                    }
                    // If combined_feature_key_name matches fgd.base_feature_key, no action needed.
                } else {
                    std::cerr
                        << "Warning: VQ index grid '" << name
                        << "' has an empty base_feature_key after parsing. Cannot determine combined status properly."
                        << std::endl;
                }
            } else if (name.rfind("packed_feature_grid_", 0) == 0) { // This is a Raw Packed Grid
                // Logic for raw packed grids (if you use them) would go here.
                // For now, it doesn't automatically set combined_feature_key_name unless specifically designed.
                // If raw packed grids are also always combined under a specific key, add similar logic.
                // If they can be per-material, then uses_combined_features might remain false.
                // Current VQ-focused logic doesn't alter combined_feature_key_name for raw packed.
            }

            if (fgd.level_idx != -1) {
                model_data.max_level_idx = std::max(model_data.max_level_idx, fgd.level_idx);
            }
        } else if (name.rfind("denorm_", 0) == 0) { // Starts with "denorm_"
            std::string material_id_denorm;
            bool is_mean_denorm;
            if (parse_denorm_tensor_name(name, material_id_denorm, is_mean_denorm)) {
                if (tensor_info.dtype != safetensors::dtype::kFLOAT32) {
                    std::cerr << "Error: Denorm tensor '" << name << "' has unexpected dtype. Expected F32."
                              << std::endl;
                    continue;
                }
                MaterialSpecificData& mat_data = model_data.materials[material_id_denorm];
                if (mat_data.id.empty())
                    mat_data.id = material_id_denorm;

                std::vector<float>* target_vec = is_mean_denorm ? &mat_data.denorm_mean : &mat_data.denorm_std;

                std::vector<uint8_t> raw_data;
                copy_tensor_data(raw_data, st_data, tensor_info);
                if (raw_data.size() % sizeof(float) != 0) {
                    std::cerr << "Error: Denorm data for '" << name << "' is not a multiple of float size."
                              << std::endl;
                    continue;
                }
                target_vec->resize(raw_data.size() / sizeof(float));
                std::memcpy(target_vec->data(), raw_data.data(), raw_data.size());
            }
        }

        else { // End of the 'else if' for feature grids
            std::cout << "Info: Unhandled tensor: " << name << std::endl;
        }
        // --- End of Corrected Logic for FeatureGrids ---
    }

    // Final check for uses_combined_features is now simpler:
    // If combined_feature_key_name was set (which happens if VQ grids were found and had a base key),
    // then uses_combined_features should be true. Otherwise, it remains its default (false).
    // The default is false, it only becomes true if a combined_feature_key_name is successfully set from a VQ grid.
    // If uses_vq is true but combined_feature_key_name is STILL empty after the loop,
    // it implies VQ grids were found but had no parsable base_feature_key, which is an error state.
    if (model_data.uses_vq && model_data.combined_feature_key_name.empty() && !model_data.named_feature_grids.empty()) {
        std::cerr << "Error: VQ is active and VQ grids were loaded, but no combined_feature_key_name could be "
                     "established. "
                  << "This indicates an issue with parsing VQ grid names or an unexpected model structure."
                  << std::endl;
        model_data.uses_combined_features = false; // Cannot assume combined without a key
    }
    // Note: uses_combined_features is already true if combined_feature_key_name was set.
    // If it's not set, it remains false.

    for (auto& mat_pair : model_data.materials) {
        std::sort(mat_pair.second.mlp_layers.begin(), mat_pair.second.mlp_layers.end(),
                  [](const MLPLayer& a, const MLPLayer& b) { return a.layer_idx < b.layer_idx; });
    }

    std::cout << "\n--- CPU Bias Value Verification ---" << std::endl;
    for (const auto& mat_pair : model_data.materials) {
        const MaterialSpecificData& mat_data = mat_pair.second;
        if (mat_data.mlp_layers.empty())
            continue;

        const MLPLayer& last_layer =
            mat_data.mlp_layers.back(); // Get the last layer (e.g., Conv2d with 3 output channels)

        std::cout << "Material: '" << mat_data.id << "', Last MLP Layer (idx " << last_layer.layer_idx << ") Bias: ";

        if (last_layer.bias_raw_bytes.empty()) {
            std::cout << "(no bias tensor found)" << std::endl;
            continue;
        }

        if (last_layer.bias_dtype == safetensors::dtype::kFLOAT32) {
            size_t num_floats = last_layer.bias_raw_bytes.size() / sizeof(float);
            const float* bias_values = reinterpret_cast<const float*>(last_layer.bias_raw_bytes.data());
            std::cout << "[F32] ";
            for (size_t i = 0; i < num_floats; ++i) {
                std::cout << std::fixed << bias_values[i] << " ";
            }
        } else if (last_layer.bias_dtype == safetensors::dtype::kFLOAT16) {
            size_t num_halfs = last_layer.bias_raw_bytes.size() / sizeof(uint16_t);
            const uint16_t* bias_raw_half_values = reinterpret_cast<const uint16_t*>(last_layer.bias_raw_bytes.data());
            std::cout << "[F16] ";
            for (size_t i = 0; i < num_halfs; ++i) {
                float val = fp16_to_fp32(bias_raw_half_values[i]);
                std::cout << std::fixed << val << " ";
            }
        } else {
            std::cout << "(unsupported dtype for printing)";
        }
        std::cout << std::endl;
    }
    std::cout << "--- End CPU Bias Verification ---\n" << std::endl;

    print_model_data_summary(model_data);

    return true;
} // End of load_model_from_safetensors

void print_model_data_summary(const SafetensorsModelData& data)
{
    std::cout << "\n--- Loaded Safetensors Model Summary ---" << std::endl;
    std::cout << "Palette values (" << data.palette.values.size() << "): ";
    for (float v : data.palette.values)
        std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "Uses VQ: " << (data.uses_vq ? "Yes" : "No") << std::endl;
    if (data.uses_vq && !data.vq_codebook.packed_data.empty()) {
        std::cout << "  VQ Codebook shape: [";
        for (size_t i = 0; i < data.vq_codebook.shape.size(); ++i)
            std::cout << data.vq_codebook.shape[i] << (i == data.vq_codebook.shape.size() - 1 ? "" : ", ");
        std::cout << "], Data size: " << data.vq_codebook.packed_data.size() << " bytes" << std::endl;
    }

    std::cout << "Uses Combined Features: " << (data.uses_combined_features ? "Yes" : "No") << std::endl;
    if (data.uses_combined_features) {
        std::cout << "  Combined Feature Key Name: " << data.combined_feature_key_name << std::endl;
    }
    std::cout << "Max Feature Level Index Found: " << data.max_level_idx << std::endl;

    std::cout << "\nFeature Grids (" << data.named_feature_grids.size() << " total):" << std::endl;
    for (const auto& pair : data.named_feature_grids) {
        const FeatureGridData& fgd = pair.second;
        std::cout << "  - Name: " << fgd.name << ", Shape: [";
        for (size_t i = 0; i < fgd.shape.size(); ++i)
            std::cout << fgd.shape[i] << (i == fgd.shape.size() - 1 ? "" : ", ");
        std::cout << "], Data size: " << fgd.raw_data_bytes.size() << " bytes" << std::endl;
        std::cout << "    Parsed: base_key='" << fgd.base_feature_key << "', L=" << fgd.level_idx
                  << ", G=" << fgd.grid_type << ", ch=" << fgd.channel_idx << std::endl;
    }

    std::cout << "\nMaterials (" << data.materials.size() << " total):" << std::endl;
    for (const auto& mat_pair : data.materials) {
        const MaterialSpecificData& mat_data = mat_pair.second;
        std::cout << "  Material ID: " << mat_data.id << std::endl;

        std::cout << "    Channel Selections (by level):" << std::endl;
        for (const auto& sel_pair : mat_data.level_channel_selections) {
            std::cout << "      Level " << sel_pair.first << ":" << std::endl;
            std::cout << "        Grid0 (" << sel_pair.second.grid0_selected_channels.size() << " indices): ";
            if (sel_pair.second.grid0_selected_channels.empty())
                std::cout << "(none)";
            else
                for (size_t i = 0; i < std::min(size_t(5), sel_pair.second.grid0_selected_channels.size()); ++i)
                    std::cout << sel_pair.second.grid0_selected_channels[i] << " ";
            if (sel_pair.second.grid0_selected_channels.size() > 5)
                std::cout << "...";
            std::cout << std::endl;

            std::cout << "        Grid1 (" << sel_pair.second.grid1_selected_channels.size() << " indices): ";
            if (sel_pair.second.grid1_selected_channels.empty())
                std::cout << "(none)";
            else
                for (size_t i = 0; i < std::min(size_t(5), sel_pair.second.grid1_selected_channels.size()); ++i)
                    std::cout << sel_pair.second.grid1_selected_channels[i] << " ";
            if (sel_pair.second.grid1_selected_channels.size() > 5)
                std::cout << "...";
            std::cout << std::endl;
        }

        std::cout << "    MLP Layers (" << mat_data.mlp_layers.size() << " total):" << std::endl;
        for (const auto& layer : mat_data.mlp_layers) {
            std::cout << "      Layer " << layer.layer_idx << ": Weights shape [";
            for (size_t i = 0; i < layer.weights_shape.size(); ++i)
                std::cout << layer.weights_shape[i] << (i == layer.weights_shape.size() - 1 ? "" : ", ");
            std::cout << "], Dtype: " << safetensors::get_dtype_str(layer.weights_dtype); // Use helper
            if (!layer.bias_raw_bytes.empty()) {
                std::cout << ", Bias shape [";
                for (size_t i = 0; i < layer.bias_shape.size(); ++i)
                    std::cout << layer.bias_shape[i] << (i == layer.bias_shape.size() - 1 ? "" : ", ");
                std::cout << "], Dtype: " << safetensors::get_dtype_str(layer.bias_dtype);
            }
            std::cout << std::endl;
        }
        std::cout << "    Denorm Mean (" << mat_data.denorm_mean.size() << " values): ";
        for (float v : mat_data.denorm_mean)
            std::cout << v << " ";
        std::cout << std::endl;

        std::cout << "    Denorm Std (" << mat_data.denorm_std.size() << " values): ";
        for (float v : mat_data.denorm_std)
            std::cout << v << " ";
        std::cout << std::endl;
    }
    std::cout << "--- End of Summary ---" << std::endl;
}
