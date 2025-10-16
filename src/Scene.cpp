#include "Scene.h"

#include "Extension.h"
#include "Helpers.h"
#include "Log.h"
#include "Pipeline.h"

#include "glm/fwd.hpp"
#include "tiny_obj_loader.h"
#include <complex>
#include <cstddef>
#include <cstdint>

using namespace Mandrill;

enum MaterialTextureBit {
    DIFFUSE_TEXTURE_BIT = 1 << 0,
    SPECULAR_TEXTURE_BIT = 1 << 1,
    AMBIENT_TEXTURE_BIT = 1 << 2,
    EMISSION_TEXTURE_BIT = 1 << 3,
    NORMAL_TEXTURE_BIT = 1 << 4,
};

Node::Node()
{
    mTransform = glm::identity<glm::mat4>();
    mVisible = true;
    mpTransformDevice = nullptr;
}

Node::~Node()
{
}

// REPLACE the existing Node::render function with this:
void Node::render(VkCommandBuffer cmd, const ptr<Camera> pCamera, const ptr<const Scene> pScene) const
{
    if (!mVisible) {
        return;
    }

    std::memcpy(mpTransformDevice + pScene->mpSwapchain->getInFlightIndex(), &mTransform, sizeof(glm::mat4));

    VkPipelineLayout lastBoundLayout = VK_NULL_HANDLE;

    for (auto meshIndex : mMeshIndices) {
        const Mesh& mesh = pScene->mMeshes[meshIndex];
        const Material& material = pScene->mMaterials[mesh.materialIndex];

        // 1. Select the correct pipeline for this mesh's material
        ptr<const Pipeline> pCurrentPipeline = mpPipeline; // Use mpPipeline as the default/standard pipeline
        if (material.isNeuralTexture && pScene->hasNeuralModel()) {
            pCurrentPipeline = pScene->mNeuralModelData.uses_vq ? mpVqPipeline : mpNonVqPipeline;
        }

        if (!pCurrentPipeline) {
            Log::Warning("Node::render - Skipping mesh {} because no suitable pipeline is set.", meshIndex);
            continue;
        }

        VkPipelineLayout currentLayout = pCurrentPipeline->getLayout();

        // 2. If pipeline (and layout) has changed, bind it and the global descriptor sets
        if (currentLayout != lastBoundLayout) {
            pCurrentPipeline->bind(cmd);

            VkDeviceSize alignment =
                pScene->mpDevice->getProperties().physicalDevice.limits.minUniformBufferOffsetAlignment;
            uint32_t cameraDescriptorOffset = static_cast<uint32_t>(
                Helpers::alignTo(sizeof(CameraMatrices), alignment) * pScene->mpSwapchain->getInFlightIndex());
            pCamera->getDescriptor()->bind(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, currentLayout, 0,
                                           cameraDescriptorOffset);

            uint32_t nodeDescriptorOffset = static_cast<uint32_t>(Helpers::alignTo(sizeof(glm::mat4), alignment) *
                                                                  pScene->mpSwapchain->getInFlightIndex());
            pDescriptor->bind(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, currentLayout, 1, nodeDescriptorOffset);

            if (pScene->mpEnvironmentMapDescriptor) {
                pScene->mpEnvironmentMapDescriptor->bind(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, currentLayout, 3);
            }
            lastBoundLayout = currentLayout;
        }

        // 3. Bind the material-specific descriptor set
        material.pDescriptor->bind(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, currentLayout, 2);

        // 4. Bind vertex/index buffers and draw
        std::array<VkBuffer, 1> vertexBuffers = {pScene->mpVertexBuffer->getBuffer()};
        std::array<VkDeviceSize, 1> offsets = {mesh.deviceVerticesOffset};
        vkCmdBindVertexBuffers(cmd, 0, count(vertexBuffers), vertexBuffers.data(), offsets.data());
        vkCmdBindIndexBuffer(cmd, pScene->mpIndexBuffer->getBuffer(), mesh.deviceIndicesOffset, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, count(mesh.indices), 1, 0, 0, 0);
    }
}

Scene::Scene(ptr<Device> pDevice, ptr<Swapchain> pSwapchain, bool supportRayTracing)
    : mpDevice(pDevice), mpSwapchain(pSwapchain), mSupportRayTracing(supportRayTracing), mVertexCount(0),
      mIndexCount(0), mHasNeuralModel(false), m_pLastSetSamplerInScene(nullptr)
{
    const uint8_t data[] = {0xff, 0x00, 0xff, 0xff, 0x88, 0x00, 0xff, 0xff,
                            0x88, 0x00, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff};
    mpMissingTexture =
        make_ptr<Texture>(pDevice, Texture::Type::Texture2D, VK_FORMAT_R8G8B8A8_UNORM, data, 2, 2, 1, 4, false);
}

Scene::~Scene()
{
}

void Scene::render(VkCommandBuffer cmd, const ptr<Camera> pCamera) const
{
    for (auto& node : mNodes) {
        node.render(cmd, pCamera, shared_from_this());
    }
}

ptr<Node> Scene::addNode()
{
    Node node = {};

    mNodes.push_back(node);

    return ptr<Node>(&*(mNodes.end() - 1), [](Node*) {});
}


void Scene::loadNeuralModel(const std::filesystem::path& modelPath)
{
    if (modelPath.empty()) {
        Log::Warning("Neural model path is empty. Skipping loading.");
        mHasNeuralModel = false;
        return;
    }

    Log::Info("Loading neural model from {}", modelPath.string());
    if (::load_model_from_safetensors(modelPath.string(), mNeuralModelData)) { // Use global namespace for your loader
        mHasNeuralModel = true;
        mNeuralModelPath = modelPath; // Store for potential reloads
        Log::Info("Successfully loaded neural model: {}", modelPath.string());
        // print_model_data_summary(mNeuralModelData); // Optional: for debugging

        // This assumes mMaterials is already populated by addMeshFromFile
        for (auto& mat : mMaterials) {
            // Ensure mat.name is populated correctly from tinyobj::material_t::name
            if (mNeuralModelData.materials.count(mat.neuralLinkName)) {
                mat.isNeuralTexture = true;
                mat.pCpuNeuralMaterialData = &mNeuralModelData.materials.at(mat.neuralLinkName);
                mat.params.isNeuralTexture = 1; // For UBO
                Log::Info("Linked OBJ material '{}' to neural material ID '{}'", mat.name,
                          mat.pCpuNeuralMaterialData->id);
            } else {
                mat.params.isNeuralTexture = 0; // Default for non-neural / non-linked materials
            }
        }

    } else {
        Log::Error("Failed to load neural model: {}", modelPath.string());
        mHasNeuralModel = false;
    }
}

uint32_t Scene::addMesh(const std::vector<Vertex> vertices, const std::vector<uint32_t> indices, uint32_t materialIndex)
{
    Mesh mesh = {
        .vertices = vertices,
        .indices = indices,
        .materialIndex = materialIndex,
    };

    mMeshes.push_back(mesh);

    return count(mMeshes) - 1;
}

template <typename T, typename... Rest> inline void hashCombine(std::size_t& seed, T const& v, Rest&&... rest)
{
    std::hash<T> hasher;
    seed ^= (hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    int i[] = {0, (hashCombine(seed, std::forward<Rest>(rest)), 0)...};
    (void)(i);
}

namespace std
{
    template <> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const
        {
            size_t h = 0;
            hashCombine(h, vertex.position, vertex.normal, vertex.texcoord, vertex.tangent, vertex.binormal);
            return h;
        }
    };
} // namespace std


std::string toLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

// The deriveBaseNameFromTexture function - this is the core new logic
// It takes a texture filename (just the name, not the full path) and returns the derived base.
std::string deriveBaseNameFromTextureFilename(const std::string& textureFilename)
{
    if (textureFilename.empty()) {
        return ""; // Or handle as an error/warning
    }

    std::filesystem::path p(textureFilename);
    std::string stem_original = p.stem().string(); // e.g., "sponza_bricks_a_diff" from "sponza_bricks_a_diff.png"
    std::string stem_lower = toLower(stem_original);

    std::vector<std::string> keywords_to_remove_sorted = {
        "displacement", "displace_inv", "displace", "combined_orm", "normal",    "nor_gl",   "roughness",
        "rough",        "occlusion",    "ambient",  "metallic",     "metalness", "specular", "spec",
        "emissive",     "diffuse",      "albedo",   "color",        "diff",      "mask",     "disp",
        "bump",         "emi",          "emt",      "mra",          "ao",
    };
    // Sort by length descending
    std::sort(keywords_to_remove_sorted.rbegin(), keywords_to_remove_sorted.rend(),
              [](const std::string& a, const std::string& b) { return a.length() < b.length(); });

    std::string current_stem_to_process = stem_original;
    std::string current_stem_lower_to_process = stem_lower;

    for (const auto& keyword : keywords_to_remove_sorted) {
        std::string keyword_l = toLower(keyword); // keyword is already lower from list
        size_t pos = current_stem_lower_to_process.rfind(keyword_l);

        if (pos != std::string::npos) {
            // Check if it's a "whole word" suffix or delimited part
            bool is_actual_suffix = (pos + keyword_l.length() == current_stem_lower_to_process.length());
            bool preceded_by_delimiter = (pos > 0 && (current_stem_lower_to_process[pos - 1] == '_' ||
                                                      current_stem_lower_to_process[pos - 1] == '.' ||
                                                      current_stem_lower_to_process[pos - 1] == '-'));

            if (is_actual_suffix || preceded_by_delimiter) {
                std::string new_candidate_stem;
                if (pos > 0 && preceded_by_delimiter) { // "base_keyword" -> "base"
                    new_candidate_stem = current_stem_to_process.substr(0, pos - 1);
                } else if (pos > 0 && is_actual_suffix) { // "basekeyword" -> "base"
                    new_candidate_stem = current_stem_to_process.substr(0, pos);
                } else if (pos == 0 && is_actual_suffix &&
                           keyword_l.length() < current_stem_lower_to_process.length()) { // "keyword_base" -> "base"
                    new_candidate_stem = current_stem_to_process.substr(keyword_l.length());
                    // trim leading delimiter if any
                    if (!new_candidate_stem.empty() &&
                        (new_candidate_stem.front() == '_' || new_candidate_stem.front() == '.' ||
                         new_candidate_stem.front() == '-')) {
                        new_candidate_stem = new_candidate_stem.substr(1);
                    }
                } else if (pos == 0 && is_actual_suffix &&
                           keyword_l.length() ==
                               current_stem_lower_to_process.length()) { // "keyword" -> "" (then handled)
                    new_candidate_stem = "";
                } else {
                    continue; // Not a clean strip, try next keyword
                }


                // Trim trailing delimiters from the new candidate
                while (!new_candidate_stem.empty() &&
                       (new_candidate_stem.back() == '_' || new_candidate_stem.back() == '.' ||
                        new_candidate_stem.back() == '-')) {
                    new_candidate_stem.pop_back();
                }

                if (!new_candidate_stem.empty()) {
                    // If something was stripped and result is not empty, update and continue trying to strip more from
                    // the result current_stem_to_process = new_candidate_stem; current_stem_lower_to_process =
                    // toLower(current_stem_to_process); Log::Debug("Stripped '{}', intermediate base name: '{}'",
                    // keyword, current_stem_to_process); continue; // Found and stripped one, try stripping more from
                    // this result (optional)

                    // OR: Assume one primary keyword suffix is enough (simpler)
                    // Fix for mtl.file windows dash being a part of the stem
                    auto pos = new_candidate_stem.find_last_of('\\');
                    if (pos != std::string::npos) {
                        auto real_stem = new_candidate_stem.substr(0, pos);
                        new_candidate_stem = new_candidate_stem.substr(pos + 1);
                    }
                    Log::Debug("Derived base name for texture file '{}' by stripping '{}': result '{}'",
                               textureFilename, keyword, new_candidate_stem);

                    return new_candidate_stem;

                } else if (stem_original == keyword_l) { // Original stem was just the keyword
                    Log::Info("Texture file '{}' stem is just the keyword '{}'. Using original stem '{}' as base name.",
                              textureFilename, keyword, stem_original);
                    return stem_original; // Safetensor key is likely "diffuse" itself if filename was "diffuse.png"
                }
            }
        }
    }

    // If no keywords were stripped, or stripping resulted in empty and original wasn't just a keyword
    if (current_stem_to_process == stem_original) {
        auto pos = current_stem_to_process.find_last_of('\\');
        if (pos != std::string::npos) {
            auto real_stem = current_stem_to_process.substr(0, pos);
            current_stem_to_process = current_stem_to_process.substr(pos + 1);
        }

        Log::Info("No keywords stripped from texture file '{}' (stem: '{}'). Using original stem as base name.",
                  textureFilename, current_stem_to_process);
    }

    return current_stem_to_process; // Return whatever is left (could be original stem)
}

std::vector<uint32_t> Scene::addMeshFromFile(const std::filesystem::path& path,
                                             const std::filesystem::path& materialPath)
{
    std::vector<uint32_t> newMeshIndices;

    Log::Info("Loading {}", path.string());

    tinyobj::ObjReaderConfig readerConfig;
    readerConfig.mtl_search_path = materialPath.string();
    readerConfig.triangulate = true;

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(path.string(), readerConfig)) {
        if (!reader.Error().empty()) {
            Log::Error("TinyObjReader: {}", reader.Error());
        }
        Log::Error("Failed to load {}", path.string());
        return newMeshIndices;
    }

    if (!reader.Warning().empty()) {
        Log::Warning("TinyObjReader: {}", reader.Warning());
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    uint32_t baseMaterialIndex = count(mMaterials);
    std::vector<uint32_t> tinyobjIndexToSceneIndexMap;
    tinyobjIndexToSceneIndexMap.reserve(materials.size());

    for (const auto& material : materials) {
        Material mat;
        mat.params.isNeuralTexture = 0;
        mat.name = material.name;

        // --- Derive neuralLinkName (logic is unchanged) ---
        std::string name_for_linking = material.name;
        if (!material.diffuse_texname.empty()) {
            std::string derived = deriveBaseNameFromTextureFilename(material.diffuse_texname);
            if (!derived.empty()) {
                name_for_linking = derived;
            }
        } else if (!material.specular_texname.empty()) {
            std::string derived = deriveBaseNameFromTextureFilename(material.specular_texname);
            if (!derived.empty()) {
                name_for_linking = derived;
            }
        }
        mat.neuralLinkName = name_for_linking;
        Log::Info("Material (original MTL: '{}') processed for linking as: '{}'", mat.name, mat.neuralLinkName);

        // --- Standard material properties (logic is unchanged) ---
        mat.params.diffuse = {material.diffuse[0], material.diffuse[1], material.diffuse[2]};
        mat.params.specular = {material.specular[0], material.specular[1], material.specular[2]};
        mat.params.ambient = {material.ambient[0], material.ambient[1], material.ambient[2]};
        mat.params.emission = {material.emission[0], material.emission[1], material.emission[2]};
        mat.params.shininess = material.shininess;
        mat.params.indexOfRefraction = material.ior;
        mat.params.opacity = material.dissolve;

        auto setTexture = [this, path, materialPath](std::string textureName, std::string& textureKey) {
            if (!textureName.empty()) {
                std::replace(textureName.begin(), textureName.end(), '\\', '/');
                auto fullPath = std::filesystem::canonical(path.parent_path() / textureName);
                textureKey = fullPath.string();
                addTexture(textureKey);
                return true;
            }
            // For setTexture, we no longer need to pass the whole map or the missing texture.
            // The logic in createDescriptors handles missing textures correctly.
            return false;
        };

        mat.params.hasTexture = 0;
        if (setTexture(material.diffuse_texname, mat.diffuseTexturePath)) {
            mat.params.hasTexture |= DIFFUSE_TEXTURE_BIT;
        }
        if (setTexture(material.specular_texname, mat.specularTexturePath)) {
            mat.params.hasTexture |= SPECULAR_TEXTURE_BIT;
        }
        if (setTexture(material.ambient_texname, mat.ambientTexturePath)) {
            mat.params.hasTexture |= AMBIENT_TEXTURE_BIT;
        }
        if (setTexture(material.emissive_texname, mat.emissionTexturePath)) {
            mat.params.hasTexture |= EMISSION_TEXTURE_BIT;
        }
        if (setTexture(material.normal_texname, mat.normalTexturePath)) {
            mat.params.hasTexture |= NORMAL_TEXTURE_BIT;
        }

        // <<< ADDED: Store the new material and map its original index to its new global index
        uint32_t newSceneIndex = count(mMaterials);
        mMaterials.push_back(mat);
        tinyobjIndexToSceneIndexMap.push_back(newSceneIndex);
    }

    for (auto& shape : shapes) {
        // One mesh per material in shape
        std::set<int> matIDs;
        for (int matID : shape.mesh.material_ids) {
            matIDs.insert(matID);
        }

        std::vector<Mesh> shapeMesh(matIDs.size());

        // Loop over faces
        size_t indexOffset = 0;
        std::vector<uint32_t> indices(matIDs.size(), 0);
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shape.mesh.num_face_vertices[f]);
            int tinyobjMaterialIndex = shape.mesh.material_ids[f]; // This is the original index from tinyobj

            // <<< MODIFIED: Handle case with no material
            if (tinyobjMaterialIndex < 0 || tinyobjMaterialIndex >= tinyobjIndexToSceneIndexMap.size()) {
                // This face has no material or an invalid one. Skip it or assign a default.
                // For now, we just skip the vertices of this face.
                indexOffset += fv;
                continue;
            }

            // Loop over vertices in the face
            for (size_t v = 0; v < fv; v++) {
                Vertex vert = {};

                tinyobj::index_t idx = shape.mesh.indices[indexOffset + v];
                vert.position.x = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                vert.position.y = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                vert.position.z = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                if (idx.normal_index >= 0) {
                    vert.normal.x = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    vert.normal.y = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    vert.normal.z = attrib.normals[3 * size_t(idx.normal_index) + 2];
                }

                if (idx.texcoord_index >= 0) {
                    vert.texcoord.x = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    vert.texcoord.y = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                }

                auto meshIndex = std::distance(matIDs.find(tinyobjMaterialIndex), matIDs.end()) - 1;
                shapeMesh[meshIndex].vertices.push_back(vert);
                shapeMesh[meshIndex].indices.push_back(indices[meshIndex]);
                // <<< MODIFIED: Use the map to get the correct, definitive scene material index
                shapeMesh[meshIndex].materialIndex = tinyobjIndexToSceneIndexMap.at(tinyobjMaterialIndex);
                indices[meshIndex] += 1;
            }

            indexOffset += fv;
        }

        for (auto& mesh : shapeMesh) {
            // <<< MODIFIED: Ensure we don't add empty meshes
            if (!mesh.vertices.empty()) {
                mMeshes.push_back(mesh);
                newMeshIndices.push_back(count(mMeshes) - 1);
            }
        }
    }

    // Calculate tangent space for each face (triangle)
    for (uint32_t i = 0; i < count(newMeshIndices); i++) {
        Mesh& mesh = mMeshes[newMeshIndices.at(i)];
        for (uint32_t j = 0; j < count(mesh.indices); j += 3) {
            Vertex& v0 = mesh.vertices[j + 0];
            Vertex& v1 = mesh.vertices[j + 1];
            Vertex& v2 = mesh.vertices[j + 2];

            glm::vec3 e1 = v1.position - v0.position;
            glm::vec3 e2 = v2.position - v0.position;

            glm::vec2 duv1 = v1.texcoord - v0.texcoord;
            glm::vec2 duv2 = v2.texcoord - v0.texcoord;

            float f = 1.0f / (duv1.x * duv2.y - duv2.x * duv1.y);

            glm::vec3 t =
                glm::normalize(glm::vec3(f * (duv2.y * e1.x - duv1.y * e2.x), f * (duv2.y * e1.y - duv1.y * e2.y),
                                         f * (duv2.y * e1.z - duv1.y * e2.z)));
            glm::vec3 b =
                glm::normalize(glm::vec3(f * (-duv2.x * e1.x + duv1.x * e2.x), f * (-duv2.x * e1.y + duv1.x * e2.y),
                                         f * (-duv2.x * e1.z + duv1.x * e2.z)));

            mesh.vertices[j + 0].tangent = t;
            mesh.vertices[j + 1].tangent = t;
            mesh.vertices[j + 2].tangent = t;

            mesh.vertices[j + 0].binormal = b;
            mesh.vertices[j + 1].binormal = b;
            mesh.vertices[j + 2].binormal = b;
        }
    }

    // Remove duplicates
    for (uint32_t i = 0; i < count(newMeshIndices); i++) {
        Mesh& mesh = mMeshes[newMeshIndices.at(i)];
        std::unordered_map<Vertex, uint32_t> uniqueVertices;
        std::vector<Vertex> newVertices;
        std::vector<uint32_t> newIndices;
        uint32_t index = 0;
        for (uint32_t j = 0; j < count(mesh.indices); j++) {
            Vertex v = mesh.vertices[j];

            if (uniqueVertices.count(v) == 0) {
                uniqueVertices[v] = index;
                newVertices.push_back(v);
                index += 1;
            }

            newIndices.push_back(uniqueVertices[v]);
        }

        mesh.vertices = newVertices;
        mesh.indices = newIndices;
    }

    // Add to statistics
    for (auto index : newMeshIndices) {
        mVertexCount += count(mMeshes[index].vertices);
        mIndexCount += count(mMeshes[index].indices);
    }

    return newMeshIndices;
}

void executeSingleTimeCommands(Mandrill::ptr<Mandrill::Device> device, std::function<void(VkCommandBuffer)> functor)
{
    if (!device) {
        Mandrill::Log::Error("executeSingleTimeCommands: Device pointer is null.");
        return;
    }
    VkDevice vkDevice = device->getDevice(); // Corrected: Renamed from mDevice to vkDevice for clarity
    VkCommandPool commandPool = device->getCommandPool();
    VkQueue queue = device->getQueue();

    if (vkDevice == VK_NULL_HANDLE || commandPool == VK_NULL_HANDLE || queue == VK_NULL_HANDLE) {
        Mandrill::Log::Error("executeSingleTimeCommands: Vulkan device, command pool, or queue is null.");
        return;
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    if (vkAllocateCommandBuffers(vkDevice, &allocInfo, &commandBuffer) != VK_SUCCESS) {
        Mandrill::Log::Error("Failed to allocate single-time command buffer!");
        return;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        Mandrill::Log::Error("Failed to begin single-time command buffer!");
        vkFreeCommandBuffers(vkDevice, commandPool, 1, &commandBuffer);
        return;
    }

    functor(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        Mandrill::Log::Error("Failed to end single-time command buffer!");
        vkFreeCommandBuffers(vkDevice, commandPool, 1, &commandBuffer);
        return;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VkResult submitResult = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (submitResult != VK_SUCCESS) {
        Mandrill::Log::Error("Failed to submit single-time command buffer! Error: {}", static_cast<int>(submitResult));
        vkFreeCommandBuffers(vkDevice, commandPool, 1, &commandBuffer);
        return;
    }

    VkResult waitResult = vkQueueWaitIdle(queue);
    if (waitResult != VK_SUCCESS) {
        Mandrill::Log::Error("Failed to wait for queue idle! Error: {}", static_cast<int>(waitResult));
    }

    vkFreeCommandBuffers(vkDevice, commandPool, 1, &commandBuffer);
}

void createPositionalEncodingData(std::vector<float>& outData)
{
    const int TABLE_SIZE = 8;       // The data repeats over an 8-unit grid
    const int FEATURES_PER_DIM = 6; // 3 octaves * 2 offsets - 1 skipped pair

    outData.resize(TABLE_SIZE * FEATURES_PER_DIM);

    auto tri = [](float x, float offset) {
        float val = x - offset;
        // This is the C++ implementation of Python's `%` operator for floats
        float mod_val = val - 2.0f * std::floor(val / 2.0f);
        return (2.0f * std::abs(mod_val - 1.0f) - 1.0f);
    };

    for (int pos = 0; pos < TABLE_SIZE; ++pos) {
        int feature_idx = 0;
        for (int octave = 0; octave < 3; ++octave) {
            float div = static_cast<float>(1 << octave);
            for (int i = 0; i < 2; ++i) {
                float offset = (i == 0) ? 0.5f : 0.0f;
                if (octave == 0 && i == 0) {
                    continue; // Skip the first pair as in the shader
                }

                float value = tri(static_cast<float>(pos) / div, offset);
                outData[pos * FEATURES_PER_DIM + feature_idx] = value;
                feature_idx++;
            }
        }
        outData[pos * FEATURES_PER_DIM + feature_idx] = 0.0f;
    }
}

void Scene::compile()
{
    if (mpMissingTexture->getSampler() == VK_NULL_HANDLE) {
        Log::Error("Scene: Sampler must be set before calling compile()");
    }

    if (mHasNeuralModel) {
        Log::Info("Compiling scene with Neural Model data...");

        std::vector<float> posEncodingData;
        createPositionalEncodingData(posEncodingData);
        VkDeviceSize posEncBufferSize = sizeof(float) * posEncodingData.size();
        if (posEncBufferSize > 0) {
            mpPositionalEncodingBuffer = make_ptr<Buffer>(
                mpDevice, posEncBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            ptr<Buffer> pStagingBuffer =
                make_ptr<Buffer>(mpDevice, posEncBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            pStagingBuffer->copyFromHost(posEncodingData.data(), posEncBufferSize, 0);
            executeSingleTimeCommands(mpDevice, [&](VkCommandBuffer cmd) {
                VkBufferCopy copyRegion{};
                copyRegion.size = posEncBufferSize;
                vkCmdCopyBuffer(cmd, pStagingBuffer->getBuffer(), mpPositionalEncodingBuffer->getBuffer(), 1,
                                &copyRegion);
            });
            Log::Info("Uploaded positional encoding buffer to GPU: {} bytes", posEncBufferSize);
        }
        // Create and zero-out the timing buffer
        const int numTimestamps = 4;
        VkDeviceSize timingBufferSize = sizeof(uint64_t) * numTimestamps;
        mpTimingBuffer = make_ptr<Buffer>(mpDevice, timingBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        uint64_t* pTimingData = static_cast<uint64_t*>(mpTimingBuffer->getHostMap());
        memset(pTimingData, 0, timingBufferSize);
        Log::Info("Created GPU timing buffer: {} bytes", timingBufferSize);

        // Palette Buffer
        if (!mNeuralModelData.palette.values.empty()) {
            VkDeviceSize paletteBufferSize = sizeof(float) * mNeuralModelData.palette.values.size();
            if (paletteBufferSize > 0) {
                mpPaletteBuffer =
                    make_ptr<Buffer>(mpDevice, paletteBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                mpPaletteBuffer->copyFromHost(mNeuralModelData.palette.values.data(), paletteBufferSize, 0);
                Log::Info("Uploaded palette buffer to GPU: {} floats", mNeuralModelData.palette.values.size());
            }
        } else {
            Log::Warning("Neural model data: Palette values are empty.");
        }

        // VQ Codebook Buffer
        if (mNeuralModelData.uses_vq) {
            if (!mNeuralModelData.vq_codebook.packed_data.empty()) {
                VkDeviceSize codebookBufferSize = sizeof(uint8_t) * mNeuralModelData.vq_codebook.packed_data.size();
                if (codebookBufferSize > 0) {
                    mpVQCodebookBuffer =
                        make_ptr<Buffer>(mpDevice, codebookBufferSize,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                    ptr<Buffer> pStagingBuffer =
                        make_ptr<Buffer>(mpDevice, codebookBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                    pStagingBuffer->copyFromHost(mNeuralModelData.vq_codebook.packed_data.data(), codebookBufferSize,
                                                 0);
                    executeSingleTimeCommands(mpDevice, [&](VkCommandBuffer cmd) {
                        VkBufferCopy copyRegion{};
                        copyRegion.size = codebookBufferSize;
                        vkCmdCopyBuffer(cmd, pStagingBuffer->getBuffer(), mpVQCodebookBuffer->getBuffer(), 1,
                                        &copyRegion); // Corrected to getBuffer()
                    });
                    Log::Info("Uploaded VQ Codebook buffer to GPU: {} bytes", codebookBufferSize);
                }
            } else {
                Log::Warning("Neural model data: VQ enabled, but VQ Codebook packed_data is empty.");
            }
        } else {
            Log::Info("Compiling neural data for Packed Raw Feature model...");
            for (const auto& fg_pair : mNeuralModelData.named_feature_grids) {
                const FeatureGridData& fgd = fg_pair.second;

                // We only care about grids with original_dims, which indicates they are our packed raw grids.
                if (fgd.original_dims.empty())
                    continue;

                std::pair<int, int> gridKey = {fgd.level_idx, fgd.grid_type};
                if (fgd.level_idx != -1 && fgd.grid_type != -1 &&
                    mSharedPackedRawGridBuffers.find(gridKey) == mSharedPackedRawGridBuffers.end()) {

                    if (!fgd.raw_data_bytes.empty()) {
                        VkDeviceSize bufferSize = fgd.raw_data_bytes.size();

                        // Create a GPU buffer for this packed grid
                        ptr<Buffer> pGridBuffer = make_ptr<Buffer>(
                            mpDevice, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

                        // Stage and copy the data
                        ptr<Buffer> pStagingBuffer = make_ptr<Buffer>(
                            mpDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                        pStagingBuffer->copyFromHost(fgd.raw_data_bytes.data(), bufferSize, 0);
                        executeSingleTimeCommands(mpDevice, [&](VkCommandBuffer cmd) {
                            VkBufferCopy copyRegion{};
                            copyRegion.size = bufferSize;
                            vkCmdCopyBuffer(cmd, pStagingBuffer->getBuffer(), pGridBuffer->getBuffer(), 1, &copyRegion);
                        });

                        mSharedPackedRawGridBuffers[gridKey] = pGridBuffer;
                        Log::Info("Created SHARED Packed Raw Grid Buffer L{}G{}, {} bytes.", fgd.level_idx,
                                  fgd.grid_type, bufferSize);

                    } else {
                        Log::Warning("Packed Raw Grid L{}G{} has empty data. Skipping.", fgd.level_idx, fgd.grid_type);
                    }
                }
            }
        }
        // SHARED VQ Index Grid Textures
        if (mNeuralModelData.uses_vq && mNeuralModelData.uses_combined_features) {
            const std::string& sharedGridsBaseKey = mNeuralModelData.combined_feature_key_name;
            Log::Info("Attempting to create shared VQ grids. Uses VQ: {}, Uses Combined: {}, SharedKey: '{}'",
                      mNeuralModelData.uses_vq, mNeuralModelData.uses_combined_features,
                      sharedGridsBaseKey); // ADD THIS
            if (sharedGridsBaseKey.empty()) {
                Log::Info("Neural model data: Combined VQ enabled, but combined_feature_key_name is empty.");
            } else {
                Log::Info("Creating shared VQ Index Textures using base key: '{}'", sharedGridsBaseKey);
                for (const auto& fg_pair : mNeuralModelData.named_feature_grids) {
                    const FeatureGridData& fgd = fg_pair.second;
                    if (fgd.base_feature_key == sharedGridsBaseKey &&
                        fgd.name.find("_vq_indices_uint8") != std::string::npos) { // Ensure it's a VQ index grid
                        std::pair<int, int> gridKey = {fgd.level_idx, fgd.grid_type};
                        if (fgd.level_idx != -1 && fgd.grid_type != -1 &&
                            mSharedVQIndexTextures.find(gridKey) == mSharedVQIndexTextures.end()) {
                            if (fgd.shape.size() == 3) {
                                uint32_t channels = static_cast<uint32_t>(fgd.shape[0]);
                                uint32_t height = static_cast<uint32_t>(fgd.shape[1]);
                                uint32_t width = static_cast<uint32_t>(fgd.shape[2]);

                                // Create a single TALL texture by stacking channels vertically
                                uint32_t total_height = height * channels;
                                size_t expected_size = static_cast<size_t>(width) * total_height;

                                if (width > 0 && height > 0 && channels > 0 && !fgd.raw_data_bytes.empty() &&
                                    fgd.raw_data_bytes.size() == expected_size) {

                                    // Create the flattened 2D texture
                                    ptr<Texture> pGridTexture =
                                        make_ptr<Texture>(mpDevice, Texture::Type::Texture2DArray, VK_FORMAT_R8_UINT,
                                                          fgd.raw_data_bytes.data(), width, height, channels, 1, false);

                                    if (m_pLastSetSamplerInScene) {
                                        pGridTexture->setSampler(m_pLastSetSamplerInScene);
                                    } else {
                                        Log::Error("Critical: m_pLastSetSamplerInScene is null for VQ index textures.");
                                    }

                                    // Store the texture and the ORIGINAL shape for the shader
                                    mSharedVQIndexTextures[gridKey] = pGridTexture;
                                    mSharedVQIndexOriginalShapes[gridKey] = glm::uvec4(channels, height, width, 0.0);

                                    Log::Info("Created SHARED VQ Index Texture (key '{}'), L{}G{}, Original Shape: "
                                              "{}x{}x{}, Flattened to: {}x{}",
                                              sharedGridsBaseKey, fgd.level_idx, fgd.grid_type, channels, height, width,
                                              width, total_height);
                                } else {
                                    Log::Warning("SHARED VQ Index Grid L{}G{} (key '{}') invalid dims/data. Name: {}. "
                                                 "Skipping. Got shape [{}, {}, {}], expected size {}, got {}",
                                                 fgd.level_idx, fgd.grid_type, sharedGridsBaseKey, fgd.name, channels,
                                                 height, width, expected_size, fgd.raw_data_bytes.size());
                                }
                            } else {
                                Log::Warning("SHARED VQ Index Grid L{}G{} (key '{}') does not have 3 dimensions. Shape "
                                             "size: {}. Skipping.",
                                             fgd.level_idx, fgd.grid_type, sharedGridsBaseKey, fgd.shape.size());
                            }
                        }
                    }
                }
            }
        }

        for (Material& mat : mMaterials) {
            if (mat.isNeuralTexture) {
                // For each level and grid, find the corresponding shared shape and assign it to the material's UBO
                // struct
                for (int l = 0; l < MAX_NEURAL_FEATURE_GRID_LEVELS; ++l) {
                    for (int g = 0; g < 2; ++g) {
                        auto key = std::make_pair(l, g);
                        if (mNeuralModelData.uses_vq) {
                            // VQ Path: Use the shape of the VQ index texture
                            if (mSharedVQIndexOriginalShapes.count(key)) {
                                mat.params.featureGridShapes[l][g] = mSharedVQIndexOriginalShapes.at(key);
                            } else {
                                mat.params.featureGridShapes[l][g] = glm::uvec4(0);
                            }
                        } else {
                            // Non-VQ Path: Use the original_dims from the loaded packed grid data
                            std::string grid_map_key = "packed_L" + std::to_string(l) + "_G" + std::to_string(g);
                            if (mNeuralModelData.named_feature_grids.count(grid_map_key)) {
                                const auto& fgd = mNeuralModelData.named_feature_grids.at(grid_map_key);
                                if (fgd.original_dims.size() == 3) {
                                    mat.params.featureGridShapes[l][g] = glm::uvec4(fgd.original_dims[0], // C
                                                                                    fgd.original_dims[1], // H
                                                                                    fgd.original_dims[2], // W
                                                                                    0                     // Unused
                                    );
                                } else {
                                    mat.params.featureGridShapes[l][g] = glm::uvec4(0);
                                }
                            } else {
                                mat.params.featureGridShapes[l][g] = glm::uvec4(0);
                            }
                        }
                    }
                }

                Log::Info("Processing neural data for material: {}", mat.neuralLinkName);
                const MaterialSpecificData& specific_mat_data = *mat.pCpuNeuralMaterialData;

                // --- Channel Selection Data ---
                for (int l = 0; l < MAX_NEURAL_FEATURE_GRID_LEVELS; ++l) {
                    auto it = specific_mat_data.level_channel_selections.find(l);
                    if (it != specific_mat_data.level_channel_selections.end()) {
                        const ChannelSelections& selections = it->second;

                        // Grid 0
                        mat.params.channelCounts[l].x = uint32_t(selections.grid0_selected_channels.size());
                        if (!selections.grid0_selected_channels.empty()) {
                            VkDeviceSize bufferSize = sizeof(uint16_t) * selections.grid0_selected_channels.size();
                            ptr<Buffer> pGpuBuffer =
                                make_ptr<Buffer>(mpDevice, bufferSize,
                                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                            ptr<Buffer> pStaging = make_ptr<Buffer>(
                                mpDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                            pStaging->copyFromHost(selections.grid0_selected_channels.data(), bufferSize);
                            executeSingleTimeCommands(mpDevice, [&](VkCommandBuffer cmd) {
                                VkBufferCopy cr{};
                                cr.size = bufferSize;
                                vkCmdCopyBuffer(cmd, pStaging->getBuffer(), pGpuBuffer->getBuffer(), 1, &cr);
                            });
                            mat.neuralChannelSelectionBuffers[{l, 0}] = pGpuBuffer;
                            Log::Debug("Uploaded channel selection for '{}' L{}G0: {} indices", mat.neuralLinkName, l,
                                       selections.grid0_selected_channels.size());
                        }

                        // Grid 1
                        mat.params.channelCounts[l].y = uint32_t(selections.grid1_selected_channels.size());
                        if (!selections.grid1_selected_channels.empty()) {
                            VkDeviceSize bufferSize = sizeof(uint16_t) * selections.grid1_selected_channels.size();
                            ptr<Buffer> pGpuBuffer =
                                make_ptr<Buffer>(mpDevice, bufferSize,
                                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                            ptr<Buffer> pStaging = make_ptr<Buffer>(
                                mpDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                            pStaging->copyFromHost(selections.grid1_selected_channels.data(), bufferSize);
                            executeSingleTimeCommands(mpDevice, [&](VkCommandBuffer cmd) {
                                VkBufferCopy cr{};
                                cr.size = bufferSize;
                                vkCmdCopyBuffer(cmd, pStaging->getBuffer(), pGpuBuffer->getBuffer(), 1, &cr);
                            });
                            mat.neuralChannelSelectionBuffers[{l, 1}] = pGpuBuffer;
                            Log::Debug("Uploaded channel selection for '{}' L{}G1: {} indices", mat.neuralLinkName, l,
                                       selections.grid1_selected_channels.size());
                        }
                        mat.params.channelCounts[l].z = 0.0;
                        mat.params.channelCounts[l].w = 0.0;
                    } else {
                        // No selections for this level
                        mat.params.channelCounts[l] = glm::uvec4(0);
                    }
                }

                // --- MLP Layer Data ---
                if (specific_mat_data.mlp_layers.size() > MAX_MLP_LAYERS) {
                    Log::Warning("Material '{}' has {} MLP layers, but layout only supports {}. Truncating.",
                                 mat.neuralLinkName, specific_mat_data.mlp_layers.size(), MAX_MLP_LAYERS);
                }
                for (const MLPLayer& sm_layer : specific_mat_data.mlp_layers) {
                    if (mat.mlpWeightBuffers.size() >= MAX_MLP_LAYERS)
                        break; // Adhere to layout limit

                    if (!sm_layer.weights_shape.empty()) {
                        // Assuming weights_shape is [out_channels, in_channels, 1, 1]
                        mat.params.mlpLayerShapes[sm_layer.layer_idx].x = sm_layer.weights_shape[1];
                        mat.params.mlpLayerShapes[sm_layer.layer_idx].y = sm_layer.weights_shape[0];
                    }

                    // Weights
                    if (!sm_layer.weights_raw_bytes.empty()) {
                        VkDeviceSize bufferSize = sm_layer.weights_raw_bytes.size();
                        ptr<Buffer> pGpuBuffer = make_ptr<Buffer>(
                            mpDevice, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                        ptr<Buffer> pStaging = make_ptr<Buffer>(mpDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                        pStaging->copyFromHost(sm_layer.weights_raw_bytes.data(), bufferSize);
                        executeSingleTimeCommands(mpDevice, [&](VkCommandBuffer cmd) {
                            VkBufferCopy cr{};
                            cr.size = bufferSize;
                            vkCmdCopyBuffer(cmd, pStaging->getBuffer(), pGpuBuffer->getBuffer(), 1, &cr);
                        });
                        mat.mlpWeightBuffers[sm_layer.layer_idx] = pGpuBuffer;
                        Log::Info("Uploaded MLP L{} weights for '{}', {} bytes", sm_layer.layer_idx, mat.neuralLinkName,
                                  bufferSize);
                    } else {
                        mat.mlpWeightBuffers[sm_layer.layer_idx] = nullptr;
                    }

                    // Bias
                    if (!sm_layer.bias_raw_bytes.empty()) {
                        VkDeviceSize bufferSize = sm_layer.bias_raw_bytes.size();
                        ptr<Buffer> pGpuBuffer = make_ptr<Buffer>(
                            mpDevice, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                        ptr<Buffer> pStaging = make_ptr<Buffer>(mpDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                        pStaging->copyFromHost(sm_layer.bias_raw_bytes.data(), bufferSize);
                        executeSingleTimeCommands(mpDevice, [&](VkCommandBuffer cmd) {
                            VkBufferCopy cr{};
                            cr.size = bufferSize;
                            vkCmdCopyBuffer(cmd, pStaging->getBuffer(), pGpuBuffer->getBuffer(), 1, &cr);
                        });
                        mat.mlpBiasBuffers[sm_layer.layer_idx] = pGpuBuffer;
                        Log::Info("Uploaded MLP L{} bias for '{}', {} bytes", sm_layer.layer_idx, mat.neuralLinkName,
                                  bufferSize);
                    } else {
                        mat.mlpBiasBuffers[sm_layer.layer_idx] = nullptr;
                    }
                }
            }
        }
        Log::Info("Finished processing Neural Model data.");
    } else {
        Log::Info("No Neural Model data to process for scene compilation.");
    }

    mVertexCount = 0;
    mIndexCount = 0;
    for (const auto& mesh : mMeshes) {
        mVertexCount += count(mesh.vertices);
        mIndexCount += count(mesh.indices);
    }

    Log::Info("Compiling scene with {} meshes, {} vertices and {} indices", count(mMeshes), mVertexCount, mIndexCount);

    // Define usage flags based on whether RT is supported
    VkBufferUsageFlags vertexBufferUsage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags indexBufferUsage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (mSupportRayTracing) {
        vertexBufferUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        indexBufferUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    }

    // Allocate Host-Visible buffers for the rasterization path
    if (mVertexCount > 0) {
        mpVertexBuffer = make_ptr<Buffer>(mpDevice, sizeof(Vertex) * mVertexCount, vertexBufferUsage,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    if (mIndexCount > 0) {
        mpIndexBuffer = make_ptr<Buffer>(mpDevice, sizeof(uint32_t) * mIndexCount, indexBufferUsage,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    // Allocate Transform and Material UBOs
    uint32_t copies = mpSwapchain->getFramesInFlightCount();
    VkDeviceSize alignment = mpDevice->getProperties().physicalDevice.limits.minUniformBufferOffsetAlignment;

    VkDeviceSize transformsSize = Helpers::alignTo(sizeof(glm::mat4), alignment) * mNodes.size() * copies;
    mpTransforms = make_ptr<Buffer>(mpDevice, transformsSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkDeviceSize materialParamsSize = Helpers::alignTo(sizeof(MaterialParams), alignment) * mMaterials.size();
    mpMaterialParams = make_ptr<Buffer>(mpDevice, materialParamsSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Associate nodes with the transform buffer (FIXES DANGLING POINTER)
    glm::mat4* transforms_ptr = static_cast<glm::mat4*>(mpTransforms->getHostMap());
    for (uint32_t i = 0; i < count(mNodes); i++) {
        mNodes[i].mpTransformDevice = transforms_ptr + i * copies;
    }


    VkDeviceSize alignedSize = Helpers::alignTo(sizeof(MaterialParams), alignment);

    // Get a raw byte pointer to the start of the mapped buffer

    uint8_t* base_ptr = static_cast<uint8_t*>(mpMaterialParams->getHostMap());

    // Loop and copy to the CORRECT, ALIGNED locations
    for (uint32_t i = 0; i < mMaterials.size(); i++) {
        // Calculate the correct byte offset for the current material
        VkDeviceSize currentOffset = i * alignedSize;
        Material& mat = mMaterials[i];

        // Start with the existing parameters from the material
        MaterialParams params_to_copy = mat.params;

        // If it's a neural material, augment it with the loaded denorm data
        if (mat.isNeuralTexture && mat.pCpuNeuralMaterialData) {
            const auto& cpu_data = *mat.pCpuNeuralMaterialData;
            uint32_t count = static_cast<uint32_t>(cpu_data.denorm_mean.size());

            params_to_copy.denormChannelCount = count;

            if (count > MAX_MATERIAL_CHANNELS) {
                Log::Warning("Material '{}' has {} denorm channels, but layout only supports {}. Clamping.",
                             mat.neuralLinkName, count, MAX_MATERIAL_CHANNELS);
                count = MAX_MATERIAL_CHANNELS;
            }

            // Zero out the arrays first to prevent garbage data in padding
            memset(params_to_copy.denormMean, 0, sizeof(params_to_copy.denormMean));
            memset(params_to_copy.denormStd, 0, sizeof(params_to_copy.denormStd));

            // Copy the valid data
            if (count > 0) {
                memcpy(params_to_copy.denormMean, cpu_data.denorm_mean.data(), sizeof(float) * count);
                memcpy(params_to_copy.denormStd, cpu_data.denorm_std.data(), sizeof(float) * count);
            }
        } else {
            params_to_copy.denormChannelCount = 0;
        }

        // Get the destination pointer by adding the byte offset to the base pointer
        MaterialParams* dest_ptr = reinterpret_cast<MaterialParams*>(base_ptr + currentOffset);

        // Now copy the fully populated struct to the correct, calculated destination
        *dest_ptr = params_to_copy;
    }

    // Create dummy buffer for unused descriptor slots
    if (!mpDummyStorageBuffer) {
        const uint32_t dummyData = 0;
        mpDummyStorageBuffer = make_ptr<Buffer>(mpDevice, sizeof(dummyData),
                                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        ptr<Buffer> pStagingBuffer =
            make_ptr<Buffer>(mpDevice, sizeof(dummyData), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        pStagingBuffer->copyFromHost(&dummyData, sizeof(dummyData));

        executeSingleTimeCommands(mpDevice, [&](VkCommandBuffer cmd) {
            VkBufferCopy copyRegion{};
            copyRegion.size = sizeof(dummyData);
            vkCmdCopyBuffer(cmd, pStagingBuffer->getBuffer(), mpDummyStorageBuffer->getBuffer(), 1, &copyRegion);
        });
        Log::Info("Created dummy storage buffer for unused descriptor slots.");
    }

    createDescriptors();
}

void Scene::syncToDevice()
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    VkDeviceSize verticesOffset = 0;
    VkDeviceSize indicesOffset = 0;

    for (auto& node : mNodes) {
        for (auto meshIndex : node.mMeshIndices) {
            auto& mesh = mMeshes[meshIndex];

            size_t vertSize = mesh.vertices.size() * sizeof(Vertex);
            size_t indxSize = mesh.indices.size() * sizeof(uint32_t);

            vertices.insert(vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
            indices.insert(indices.end(), mesh.indices.begin(), mesh.indices.end());

            mesh.deviceVerticesOffset = verticesOffset;
            mesh.deviceIndicesOffset = indicesOffset;

            verticesOffset += vertSize;
            indicesOffset += indxSize;
        }
    }

    mpVertexBuffer->copyFromHost(vertices.data(), verticesOffset, 0);
    mpIndexBuffer->copyFromHost(indices.data(), indicesOffset, 0);
}

void Scene::updateAccelerationStructure(VkBuildAccelerationStructureFlagsKHR flags)
{
    if (!mSupportRayTracing) {
        Log::Error("Cannot build acceleration structure for a scene that was not initiated to support ray tracing. Set "
                   "supportRayTracing to TRUE during scene creation.");
        return;
    }

    if (!mpAccelerationStructure) {
        mpAccelerationStructure = make_ptr<AccelerationStructure>(mpDevice, shared_from_this(), flags);

        // After acceleration structure is built, we can create the descriptors
        createDescriptors();

        // The acceleration structure will be built during the creation so we can return here
        return;
    }

    mpAccelerationStructure->update(flags);
}

void Scene::bindRayTracingDescriptors(VkCommandBuffer cmd, ptr<Camera> pCamera, VkPipelineLayout layout)
{
    VkDeviceSize alignment = mpDevice->getProperties().physicalDevice.limits.minUniformBufferOffsetAlignment;
    uint32_t cameraDescriptorOffset =
        static_cast<uint32_t>(Helpers::alignTo(sizeof(CameraMatrices), alignment) * mpSwapchain->getInFlightIndex());
    pCamera->getDescriptor()->bind(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, layout, 0, cameraDescriptorOffset);

    if (mpEnvironmentMapDescriptor) {
        mpEnvironmentMapDescriptor->bind(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, layout, 3);
    }

    mpRayTracingDescriptor->bind(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, layout, 4);
}

void Scene::setSampler(const ptr<Sampler> pSampler)
{
    m_pLastSetSamplerInScene = pSampler;
    mpMissingTexture->setSampler(pSampler);

    for (auto& texture_pair : mTextures) { // std::unordered_map<std::string, ptr<Texture>>
        texture_pair.second->setSampler(pSampler);
    }

    // Also apply sampler to neural VQ grid textures
    if (mHasNeuralModel) {
        for (auto& mat : mMaterials) {
            if (mat.isNeuralTexture) {
                for (auto& vq_grid_pair : mat.neuralVQGrids) {
                    if (vq_grid_pair.second) {
                        vq_grid_pair.second->setSampler(pSampler);
                    }
                }
            }
        }
    }
}

ptr<Layout> Scene::getLayout()
{
    return getLayout(false, false);
}

ptr<Layout> Scene::getLayout(bool forNeural, bool forVqModel)

{
    std::vector<LayoutDesc> desc;

    // 0.0: Camera matrices
    // 1.0: Model matrix
    // 2.0: Material parameters
    // 2.1: Material diffuse texture
    // 2.2: Material specular texture
    // 2.3: Material ambient texture
    // 2.4: Material emission texture
    // 2.5: Material normal texture
    // 3.0: Environment map texture
    desc.emplace_back(0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                      VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    desc.emplace_back(1, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_ALL_GRAPHICS);
    desc.emplace_back(2, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS);
    desc.emplace_back(2, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_ALL_GRAPHICS);
    desc.emplace_back(2, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_ALL_GRAPHICS);
    desc.emplace_back(2, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_ALL_GRAPHICS);
    desc.emplace_back(2, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_ALL_GRAPHICS);
    desc.emplace_back(2, 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_ALL_GRAPHICS);

    if (!forNeural) {
        // This is a standard pipeline, no neural bindings.
        desc.emplace_back(3, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT);
        if (mSupportRayTracing) {
            // 4.0: Acceleration structure
            // 4.1: Scene vertex buffer
            // 4.2: Scene index buffer
            // 4.3: Scene material buffer
            // 4.4: Scene texture array
            // 4.5: Instance data buffer
            // 5.0: Output storage image
            desc.emplace_back(4, 0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
            desc.emplace_back(4, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
            desc.emplace_back(4, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
            desc.emplace_back(4, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
            desc.emplace_back(4, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                              count(mTextures));
            desc.emplace_back(4, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
            desc.emplace_back(5, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_ALL);
        }
        return make_ptr<Layout>(mpDevice, desc);
    }

    uint32_t currentBinding = 6;
    // 2.6 - 2.13: Neural VQ Grids (usampler2D)
    for (int i = 0; i < MAX_NEURAL_FEATURE_GRID_LEVELS * 2; ++i) {
        if (forVqModel) {
            // VQ models expect textures (samplers)
            desc.emplace_back(2, currentBinding++, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              VK_SHADER_STAGE_FRAGMENT_BIT);
        } else {
            // Non-VQ models expect raw data buffers (SSBOs)
            desc.emplace_back(2, currentBinding++, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT);
        }
    }

    // 2.14 - 2.21: Neural Channel Selection Lists (SSBOs)
    for (int i = 0; i < MAX_NEURAL_FEATURE_GRID_LEVELS * 2; ++i) {
        desc.emplace_back(2, currentBinding++, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    // 2.22 onwards: MLP Layer Buffers (SSBOs)
    for (int i = 0; i < MAX_MLP_LAYERS; ++i) {
        desc.emplace_back(2, currentBinding++, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                          VK_SHADER_STAGE_FRAGMENT_BIT); // Weights
        desc.emplace_back(2, currentBinding++, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                          VK_SHADER_STAGE_FRAGMENT_BIT); // Biases
    }

    // --- Set 3: Global Bindings ---
    // 3.0: Environment map texture
    desc.emplace_back(3, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                      VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT);
    // 3.1: Neural Palette Buffer (SSBO)
    desc.emplace_back(3, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT);
    // 3.2: Neural VQ Codebook Buffer (SSBO)
    desc.emplace_back(3, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT);
    // 3.3: Neural Positional Encoding Buffer (SSBO) // <<< ADD THIS LINE
    desc.emplace_back(3, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT);
    // 3.4 Timing buffer
    desc.emplace_back(3, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT);

    return make_ptr<Layout>(mpDevice, desc);
}

void Scene::addTexture(std::string texturePath)
{
    if (texturePath.empty()) {
        return;
    }

    if (mTextures.contains(texturePath)) {
        return;
    }

    auto pTexture = make_ptr<Texture>(mpDevice, Texture::Type::Texture2D, VK_FORMAT_R8G8B8A8_UNORM, texturePath, true);
    mTextures.insert(std::make_pair(texturePath, pTexture));
}

void Scene::createDescriptors()
{

    mpStandardLayout = getLayout(false, false);
    if (mHasNeuralModel) {
        mpVqNeuralLayout = getLayout(true, true);
        mpNonVqNeuralLayout = getLayout(true, false);
    }

    // Part B: Create Descriptors for Nodes (Set 1)
    // All layouts share the same structure for Set 1, so we can use the standard one.
    VkDescriptorSetLayout nodeLayout = mpStandardLayout->getDescriptorSetLayouts()[1];
    VkDeviceSize offset = 0;
    for (auto& node : mNodes) {
        std::vector<DescriptorDesc> desc;
        desc.emplace_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, mpTransforms);
        desc.back().range = sizeof(glm::mat4);
        desc.back().offset = offset;
        offset += Helpers::alignTo(sizeof(glm::mat4),
                                   mpDevice->getProperties().physicalDevice.limits.minUniformBufferOffsetAlignment) *
                  mpSwapchain->getFramesInFlightCount();
        node.pDescriptor = std::make_unique<Descriptor>(mpDevice, desc, nodeLayout);
    }

    // Part C: Create Descriptors for Materials (Set 2)
    VkDeviceSize alignment = mpDevice->getProperties().physicalDevice.limits.minUniformBufferOffsetAlignment;
    VkDeviceSize alignedSize = Helpers::alignTo(sizeof(MaterialParams), alignment);
    for (uint32_t i = 0; i < mMaterials.size(); ++i) {
        auto& mat = mMaterials[i];
        std::vector<DescriptorDesc> desc;

        // Determine which layout to use for this specific material
        ptr<Layout> pMaterialLayout;
        if (mat.isNeuralTexture && mHasNeuralModel) {
            pMaterialLayout = mNeuralModelData.uses_vq ? mpVqNeuralLayout : mpNonVqNeuralLayout;
        } else {
            pMaterialLayout = mpStandardLayout;
        }

        // --- Bindings 0-5: Common Data (UBO, Standard Textures) ---
        desc.emplace_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, mpMaterialParams);
        desc.back().offset = i * alignedSize;
        desc.back().range = sizeof(MaterialParams);
        desc.emplace_back(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          !mat.diffuseTexturePath.empty() ? mTextures.at(mat.diffuseTexturePath) : mpMissingTexture);
        desc.emplace_back(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          !mat.specularTexturePath.empty() ? mTextures.at(mat.specularTexturePath) : mpMissingTexture);
        desc.emplace_back(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          !mat.ambientTexturePath.empty() ? mTextures.at(mat.ambientTexturePath) : mpMissingTexture);
        desc.emplace_back(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          !mat.emissionTexturePath.empty() ? mTextures.at(mat.emissionTexturePath) : mpMissingTexture);
        desc.emplace_back(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          !mat.normalTexturePath.empty() ? mTextures.at(mat.normalTexturePath) : mpMissingTexture);

        // --- Bindings 6+: Neural Data (if applicable) ---
        if (mat.isNeuralTexture && mHasNeuralModel) {
            // Bindings 6-13: Neural grid features (Either Textures for VQ or Buffers for Raw)
            for (int level = 0; level < MAX_NEURAL_FEATURE_GRID_LEVELS; ++level) {
                for (int grid_type = 0; grid_type < 2; ++grid_type) {
                    auto key = std::make_pair(level, grid_type);
                    if (mNeuralModelData.uses_vq) {
                        if (mSharedVQIndexTextures.count(key) && mSharedVQIndexTextures.at(key))
                            desc.emplace_back(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                              mSharedVQIndexTextures.at(key));
                        else
                            desc.emplace_back(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, mpMissingTexture);
                    } else {
                        if (mSharedPackedRawGridBuffers.count(key) && mSharedPackedRawGridBuffers.at(key))
                            desc.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mSharedPackedRawGridBuffers.at(key));
                        else
                            desc.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mpDummyStorageBuffer);
                    }
                }
            }
            // Bindings 14-21: Channel Selection SSBOs
            for (int level = 0; level < MAX_NEURAL_FEATURE_GRID_LEVELS; ++level) {
                for (int grid_type = 0; grid_type < 2; ++grid_type) {
                    auto key = std::make_pair(level, grid_type);
                    if (mat.neuralChannelSelectionBuffers.count(key) && mat.neuralChannelSelectionBuffers.at(key))
                        desc.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mat.neuralChannelSelectionBuffers.at(key));
                    else
                        desc.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mpDummyStorageBuffer);
                }
            }
            // Bindings 22+: MLP Layer Buffers
            for (int layer_idx = 0; layer_idx < MAX_MLP_LAYERS; ++layer_idx) {
                if (mat.mlpWeightBuffers.count(layer_idx) && mat.mlpWeightBuffers.at(layer_idx))
                    desc.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mat.mlpWeightBuffers.at(layer_idx));
                else
                    desc.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mpDummyStorageBuffer);
                if (mat.mlpBiasBuffers.count(layer_idx) && mat.mlpBiasBuffers.at(layer_idx))
                    desc.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mat.mlpBiasBuffers.at(layer_idx));
                else
                    desc.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mpDummyStorageBuffer);
            }
        }

        auto materialDescSetLayout = pMaterialLayout->getDescriptorSetLayouts()[2];
        mat.pDescriptor = std::make_unique<Descriptor>(mpDevice, desc, materialDescSetLayout);
    }

    // Part D: Create Global Descriptors (Set 3)
    ptr<Layout> pGlobalLayout = mHasNeuralModel ? mpVqNeuralLayout : mpStandardLayout;
    std::vector<DescriptorDesc> globalDesc;
    globalDesc.emplace_back(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            mpEnvironmentMap ? mpEnvironmentMap : mpMissingTexture);
    if (mHasNeuralModel) {
        globalDesc.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                mpPaletteBuffer ? mpPaletteBuffer : mpDummyStorageBuffer);
        globalDesc.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (mpVQCodebookBuffer && mNeuralModelData.uses_vq)
                                                                       ? mpVQCodebookBuffer
                                                                       : mpDummyStorageBuffer);
        globalDesc.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                mpPositionalEncodingBuffer ? mpPositionalEncodingBuffer : mpDummyStorageBuffer);
        globalDesc.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                mpTimingBuffer ? mpTimingBuffer : mpDummyStorageBuffer);
    }

    auto layoutSet3 = pGlobalLayout->getDescriptorSetLayouts()[3];
    mpEnvironmentMapDescriptor = std::make_unique<Descriptor>(mpDevice, globalDesc, layoutSet3);
    Log::Info("Created descriptors successfully.");
    // Add extra descriptors for ray tracing (set 4)
    if (mSupportRayTracing) {
        std::vector<DescriptorDesc> desc;

        // Get a list of the textures
        std::vector<ptr<Texture>> textures;
        std::transform(mTextures.begin(), mTextures.end(), std::back_inserter(textures),
                       [](const auto& entry) { return entry.second; });
        ptr<std::vector<ptr<Texture>>> pTextures = make_ptr<std::vector<ptr<Texture>>>(textures);

        desc.emplace_back(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, mpAccelerationStructure);
        desc.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mpVertexBuffer);
        desc.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mpIndexBuffer);
        desc.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mpMaterialBuffer);
        desc.emplace_back(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, pTextures, 0, 0, count(textures));
        desc.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mpInstanceDataBuffer);

        auto layout = mpStandardLayout->getDescriptorSetLayouts()[4];
        mpRayTracingDescriptor = std::make_unique<Descriptor>(mpDevice, desc, layout);
    }
}

Scene::ShaderTimingResults Scene::getTimingResults()
{
    ShaderTimingResults results{};
    if (!mpTimingBuffer || !mHasNeuralModel) {
        return results;
    }

    const int numTimestamps = 4;
    const size_t bufferSize = sizeof(uint64_t) * numTimestamps;

    // Get the pointer to the mapped GPU memory
    const void* pGpuData = mpTimingBuffer->getHostMap();
    if (!pGpuData) {
        Log::Warning("Timing buffer is not host-mapped. Cannot read results.");
        return results;
    }

    // Copy the data from the GPU-mapped memory into a local CPU array
    uint64_t timestamps[numTimestamps];
    std::memcpy(timestamps, pGpuData, bufferSize);

    // If the start or end timestamp is zero, the shader hasn't written fresh data yet.
    if (timestamps[0] == 0 || timestamps[3] == 0) {
        return results;
    }

    results.grid0_ns = timestamps[1] - timestamps[0];
    results.grid1_ns = timestamps[2] - timestamps[1];
    results.mlp_ns = timestamps[3] - timestamps[2];
    results.total_ns = timestamps[3] - timestamps[0];

    // Reset the buffer on the CPU side so we know when the next frame's data is ready.
    // We can write back to the mapped pointer directly.
    void* pWritableGpuData = mpTimingBuffer->getHostMap();
    memset(pWritableGpuData, 0, bufferSize);

    return results;
}
