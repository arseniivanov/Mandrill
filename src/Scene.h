#pragma once

#include "Common.h"

#include "AccelerationStructure.h"
#include "Camera.h"
#include "Descriptor.h"
#include "Device.h"
#include "Layout.h"
#include "Sampler.h"
#include "Swapchain.h"
#include "Texture.h"

#include "model_loader.h"
#define MAX_NEURAL_FEATURE_GRID_LEVELS 4
#define MAX_MLP_LAYERS 3


struct pair_hash {
    template <class T1, class T2> std::size_t operator()(const std::pair<T1, T2>& p) const
    {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

namespace Mandrill
{
    struct Vertex {
        alignas(16) glm::vec3 position;
        alignas(16) glm::vec3 normal;
        alignas(16) glm::vec2 texcoord;
        alignas(16) glm::vec3 tangent;
        alignas(16) glm::vec3 binormal;

        bool operator==(const Vertex& other) const
        {
            return position == other.position && normal == other.normal && texcoord == other.texcoord &&
                   tangent == other.tangent && binormal == other.binormal;
        }
    };

    struct Mesh {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        uint32_t materialIndex{};

        // Device offset are set when uploading to device
        VkDeviceSize deviceVerticesOffset{};
        VkDeviceSize deviceIndicesOffset{};
    };

    struct alignas(16) MaterialParams {
        glm::vec3 diffuse;
        float shininess;
        glm::vec3 specular;
        float indexOfRefraction;
        glm::vec3 ambient;
        float opacity;
        glm::vec3 emission;
        uint32_t hasTexture;
        uint32_t isNeuralTexture;
        uint32_t channelCounts[MAX_NEURAL_FEATURE_GRID_LEVELS][2];
        glm::uvec3 featureGridShapes[MAX_NEURAL_FEATURE_GRID_LEVELS][2];
    };

    struct alignas(16) MaterialDevice {
        MaterialParams params;
        uint32_t diffuseTextureIndex;
        uint32_t specularTextureIndex;
        uint32_t ambientTextureIndex;
        uint32_t emissionTextureIndex;
        uint32_t normalTextureIndex;
    };

    struct Material {
        MaterialParams params{};
        MaterialParams* paramsDevice{}; // This is set during Scene::compile()
        VkDeviceSize paramsOffset{};    // This is set during Scene::compile()

        std::string diffuseTexturePath;
        std::string specularTexturePath;
        std::string ambientTexturePath;
        std::string emissionTexturePath;
        std::string normalTexturePath;

        ptr<Descriptor> pDescriptor;

        // Added field NTC below
        std::string name;
        bool isNeuralTexture = false;
        const MaterialSpecificData* pCpuNeuralMaterialData = nullptr;
        // Key: {level_idx, grid_type (0 or 1)}
        std::map<std::pair<int, int>, ptr<Buffer>> neuralChannelSelectionBuffers;

        // MLP layer buffers
        std::vector<ptr<Buffer>> mlpWeightBuffers;
        std::vector<ptr<Buffer>> mlpBiasBuffers;

        std::map<std::pair<int, int>, ptr<Texture>> neuralVQGrids;
        std::map<std::pair<int, int>, glm::ivec2> neuralVQGridShapes;
    };

    struct InstanceData {
        uint32_t verticesOffset; // Offset into global vertex buffer
        uint32_t indicesOffset;  // Offset into global index buffer
    };

    class Scene; // Forward declare scene so Node can befriend it
    class Pipeline;

    class Node
    {
    public:
        /// <summary>
        /// Create a new scene node.
        /// </summary>
        /// <returns></returns>
        MANDRILL_API Node();

        /// <summary>
        /// Destructor for scene node.
        /// </summary>
        /// <returns></returns>
        MANDRILL_API ~Node();

        /// <summary>
        /// Render a node in the scene.
        /// </summary>
        /// <param name="cmd">Command buffer to use for rendering</param>
        /// <param name="pCamera">Camera that defines which camera matrices to use</param>
        /// <param name="pScene">Scene which the node belongs to</param>
        /// <returns></returns>
        MANDRILL_API void render(VkCommandBuffer cmd, const ptr<Camera> pCamera, const ptr<const Scene> pScene) const;

        /// <summary>
        /// Add a mesh to the node.
        /// </summary>
        /// <param name="meshIndex">Mesh index that was received during mesh creation</param>
        /// <returns></returns>
        MANDRILL_API void addMesh(uint32_t meshIndex)
        {
            mMeshIndices.push_back(meshIndex);
        }

        /// <summary>
        /// Set pipeline to use when rendering node
        /// </summary>
        /// <param name="pPipeline">Pipeline to use</param>
        /// <returns></returns>
        MANDRILL_API void setPipeline(ptr<Pipeline> pPipeline)
        {
            mpPipeline = pPipeline;
        }

        /// <summary>
        /// Get the TRS transform of the node.
        /// </summary>
        /// <returns>4x4 matrix containing transform</returns>
        MANDRILL_API glm::mat4 getTransform() const
        {
            return mTransform;
        }

        /// <summary>
        /// Set the TRS transform of the node.
        /// </summary>
        /// <param name="transform">Transform to use</param>
        /// <returns></returns>
        MANDRILL_API void setTransform(glm::mat4 transform)
        {
            mTransform = transform;
        }

        /// <summary>
        /// Set weather the node should be rendered or not.
        /// </summary>
        /// <param name="visible">True to render the node, otherwise false</param>
        /// <returns></returns>
        MANDRILL_API void setVisible(bool visible)
        {
            mVisible = visible;
        }

        /// <summary>
        /// Get the visibility of the node.
        /// </summary>
        /// <returns></returns>
        MANDRILL_API bool getVisible() const
        {
            return mVisible;
        }

        /// <summary>
        /// Get the mesh indices
        /// </summary>
        /// <returns>Vector of mesh indices</returns>
        MANDRILL_API std::vector<uint32_t>& getMeshIndices()
        {
            return mMeshIndices;
        }

    private:
        friend Scene;

        ptr<Pipeline> mpPipeline;

        std::vector<uint32_t> mMeshIndices;

        glm::mat4 mTransform;
        glm::mat4* mpTransformDevice;
        ptr<Descriptor> pDescriptor;
        std::vector<VkDescriptorSet> mDescriptors;

        bool mVisible;

        std::vector<Node*> mChildren;
    };

    class Scene : public std::enable_shared_from_this<Scene>
    {
    public:
        MANDRILL_NON_COPYABLE(Scene)

        /// <summary>
        /// Create a new scene.
        ///
        /// The scene will use descriptor sets as follows:
        ///     Camera matrix (struct CameraMatrices): Set = 0, Binding = 0, Type =
        ///     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC Model matrix (mat4): Set = 1, Binding = 0, Type =
        ///     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC Material params (struct MaterialParams): Set = 2, Binding = 0,
        ///     Type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER Material diffuse texture: Set = 2, Binding = 1, Type =
        ///     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE Material specular texture: Set = 2, Binding = 2, Type =
        ///     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER Material ambient texture: Set = 2, Binding = 3, Type =
        ///     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER Material emission texture: Set = 2, Binding = 4, Type =
        ///     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER Material normal texture: Set = 2, Binding = 5, Type =
        ///     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER Acceleration structure: Set = 3, Binding = 0, Type =
        ///     VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR Storage output image: Set = 3, Binding = 1, Type =
        ///     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER Scene vertex buffer: Set = 3, Binding = 2, Type =
        ///     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER Scene index buffer: Set = 3, Binding = 3, Type =
        ///     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER Scene material buffer: Set = 3, Binding = 4, Type =
        ///     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER Scene texture array: Set = 3, Binding = 5, Type =
        ///     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER Output storage image: Set = 4, Binding = 0, Type =
        ///     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
        ///
        ///     NB: Set 3 and 4 is only available when using ray tracing
        ///
        /// </summary>
        /// <param name="pDevice">Device to use</param>
        /// <param name="pSwapchain">Swapchain is used to determine how many descriptors will be used</param>
        /// <param name="supportRayTracing">Prepare scene for handling acceleration sturctures</param>
        /// <returns></returns>
        MANDRILL_API Scene(ptr<Device> pDevice, ptr<Swapchain> pSwapchain, bool supportRayTracing = false);

        /// <summary>
        /// Destructor for scene.
        /// </summary>
        /// <returns></returns>
        MANDRILL_API ~Scene();

        /// <summary>
        /// Render all the nodes in the scene.
        /// </summary>
        /// <param name="cmd">Command buffer to use for rendering</param>
        /// <param name="pCamera">Camera that defines which camera matrices to use</param>
        /// <returns></returns>
        MANDRILL_API void render(VkCommandBuffer cmd, const ptr<Camera> pCamera) const;

        /// <summary>
        /// Add a node to the scene.
        /// </summary>
        /// <returns>A ptr to the created node</returns>
        MANDRILL_API ptr<Node> addNode();

        /// <summary>
        /// Add a material to the scene.
        /// </summary>
        /// <param name="material">Material struct defining the new material</param>
        /// <returns>Material index that can be used to create new meshes</returns>
        MANDRILL_API uint32_t addMaterial(Material material);

        /// <summary>
        /// Add a mesh to the scene.
        /// </summary>
        /// <param name="vertices">List of vertices that make up the mesh</param>
        /// <param name="indices">List of indices that describes how the vertices are connected</param>
        /// <param name="materialIndex">Which material should be used for the mesh</param>
        /// <returns>Mesh index that can be added to a node in the scene</returns>
        MANDRILL_API uint32_t addMesh(const std::vector<Vertex> vertices, const std::vector<uint32_t> indices,
                                      uint32_t materialIndex);

        /// <summary>
        /// Add several meshes to a scene by reading them from an OBJ-file.
        /// </summary>
        /// <param name="path">Path to the OBJ-file</param>
        /// <param name="materialPath">Path to where the material files are stored (leave to default if the materials
        /// are in the same directory as the OBJ-file)</param> <returns>List of mesh indices that can be added to a node
        /// in the scene</returns>
        MANDRILL_API std::vector<uint32_t> addMeshFromFile(const std::filesystem::path& path,
                                                           const std::filesystem::path& materialPath = "");

        /// <summary>
        /// Calculate sizes of buffers and allocate resources. Call this after all nodes have been added.
        /// </summary>
        /// <returns></returns>
        MANDRILL_API void compile();

        /// <summary>
        /// Synchronize buffers to device.
        ///
        /// Upload all buffers from host to device. This function should be called after updates have been done to
        /// the scene.
        ///
        /// Node transforms and material parameters are kept host coherent and can be changed without requiring a new
        /// sync.
        ///
        /// </summary>
        /// <returns></returns>
        MANDRILL_API void syncToDevice();

        /// <summary>
        /// Build and update acceleration structure of the scene. This is needed for ray tracing the scene. First time
        /// called, it will build a new acceleration structure. Subsequent calls will update the TLAS to account for
        /// updates in the instance transforms.
        /// </summary>
        /// <param name="flags">Flags used for building acceleration structure</param>
        /// <returns></returns>
        MANDRILL_API void updateAccelerationStructure(
            VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

        /// <summary>
        /// Bind descriptors for ray tracing.
        ///
        /// This will bind the descriptor sets 0 (camera matrices) and 3 (acceleration structure and scene buffers).
        /// Descriptor set 4 should contain the storage image to render the ray-traced image to, and has to be bound
        /// elsewhere.
        ///
        /// </summary>
        /// <returns></returns>
        MANDRILL_API void bindRayTracingDescriptors(VkCommandBuffer cmd, ptr<Camera> pCamera, VkPipelineLayout layout);

        /// <summary>
        /// Set sampler to use for rendering materials.
        /// </summary>
        /// <param name="pSampler">Sampler to be used for all materials</param>
        /// <returns></returns>
        MANDRILL_API void setSampler(const ptr<Sampler> pSampler);

        /// <summary>
        /// Get the layout used by the scene.
        ///
        /// The layout is as follows:
        ///
        /// Set.Binding: Descriptor type [usage]
        ///
        /// 0.0: VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC  [Camera matrices]
        /// 1.0: VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC  [Model matrix]
        /// 2.0: VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER          [Material parameters]
        /// 2.1: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER  [Material diffuse texture]
        /// 2.2: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER  [Material specular texture]
        /// 2.3: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER  [Material ambient texture]
        /// 2.4: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER  [Material emission texture]
        /// 2.5: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER  [Material normal texture]
        ///
        /// If ray-tracing support is requested, the following is also available:
        ///
        /// 3.0: VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR      [Acceleration structure]
        /// 3.1: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER                  [Scene vertex buffer]
        /// 3.2: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER                  [Scene index buffer]
        /// 3.3: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER                  [Scene material buffer]
        /// 3.4: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER (array)  [Scene texture array]
        /// 3.5: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER                  [Instance data buffer]
        /// 4.0: VK_DESCRIPTOR_TYPE_STORAGE_IMAGE                   [Output storage image]
        ///
        /// </summary>
        /// <returns>Pointer to layout</returns>
        MANDRILL_API ptr<Layout> getLayout();

        MANDRILL_API void loadNeuralModel(const std::filesystem::path& modelPath);
        /// <summary>
        /// Get the list of all nodes in the scene.
        /// </summary>
        /// <returns>Vector of nodes</returns>
        MANDRILL_API std::vector<Node>& getNodes()
        {
            return mNodes;
        }

        /// <summary>
        /// Get the number of vertices in the scene.
        /// </summary>
        /// <returns>Number of vertices</returns>
        MANDRILL_API uint32_t getVertexCount() const
        {
            return mVertexCount;
        }

        /// <summary>
        /// Get the number of indices in the scene.
        /// </summary>
        /// <returns>Number of indices</returns>
        MANDRILL_API uint32_t getIndexCount() const
        {
            return mIndexCount;
        }

        /// <summary>
        /// Get the number of meshes in the scene.
        /// </summary>
        /// <returns>Number of meshes</returns>
        MANDRILL_API uint32_t getMeshCount()
        {
            return count(mMeshes);
        }

        /// <summary>
        /// Get the number of materials in the scene.
        /// </summary>
        /// <returns>Number of materials</returns>
        MANDRILL_API uint32_t getMaterialCount() const
        {
            return count(mMaterials);
        }

        /// <summary>
        /// Get the number of textures in the scene.
        /// </summary>
        /// <returns>Number of textures</returns>
        MANDRILL_API uint32_t getTextureCount() const
        {
            return count(mTextures);
        }

        /// <summary>
        /// Get the number of vertices in a mesh.
        /// </summary>
        /// <param name="meshIndex">Index of mesh to get the vertices from</param>
        /// <returns>Number of vertices</returns>
        MANDRILL_API uint32_t getMeshVertexCount(uint32_t meshIndex) const
        {
            return count(mMeshes[meshIndex].vertices);
        }

        /// <summary>
        /// Get the number of indices in a mesh.
        /// </summary>
        /// <param name="meshIndex">Index of mesh to get the vertices from</param>
        /// <returns>Number of indices</returns>
        MANDRILL_API uint32_t getMeshIndexCount(uint32_t meshIndex) const
        {
            return count(mMeshes[meshIndex].indices);
        }

        /// <summary>
        /// Get the address of a mesh's vertices buffer.
        /// </summary>
        /// <param name="meshIndex">Index of mesh to look up</param>
        /// <returns>Device address</returns>
        MANDRILL_API VkDeviceAddress getMeshVertexAddress(uint32_t meshIndex) const
        {
            return mpVertexBuffer->getDeviceAddress() + mMeshes[meshIndex].deviceVerticesOffset;
        }

        /// <summary>
        /// Get the address of a mesh's indices buffer.
        /// </summary>
        /// <param name="meshIndex">Index of the mesh to look up</param>
        /// <returns>Device address</returns>
        MANDRILL_API VkDeviceAddress getMeshIndexAddress(uint32_t meshIndex) const
        {
            return mpIndexBuffer->getDeviceAddress() + mMeshes[meshIndex].deviceIndicesOffset;
        }

        /// <summary>
        /// Get the matieral index of a mesh.
        /// </summary>
        /// <param name="meshIndex">Index of the mesh to look up</param>
        /// <returns></returns>
        MANDRILL_API uint32_t getMeshMaterialIndex(uint32_t meshIndex) const
        {
            return mMeshes[meshIndex].materialIndex;
        }

        /// <summary>
        /// Get the acceleration structure of the scene.
        /// </summary>
        /// <returns>Pointer to acceleration structure or nullptr</returns>
        MANDRILL_API ptr<AccelerationStructure> getAccelerationStructure() const
        {
            return mpAccelerationStructure;
        }

        /// <summary>
        /// Set an environment map for the scene.
        /// </summary>
        /// <param name="pTexture">Texture to use as environment map</param>
        /// <returns></returns>
        MANDRILL_API void setEnvironmentMap(ptr<Texture> pTexture)
        {
            mpEnvironmentMap = pTexture;
        }

        MANDRILL_API bool hasNeuralModel()
        {
            return mHasNeuralModel;
        }

    private:
        friend Node;
        // NTC Additions
        SafetensorsModelData mNeuralModelData;
        bool mHasNeuralModel = false;
        std::filesystem::path mNeuralModelPath;
        ptr<Sampler> m_pLastSetSamplerInScene;

        ptr<Buffer> mpPaletteBuffer;
        ptr<Buffer> mpVQCodebookBuffer;                                     // For "vq_codebook_patches_packed_uint8"
        std::map<std::pair<int, int>, ptr<Texture>> mSharedVQIndexTextures; // Key: {level_idx, grid_type}
        std::map<std::pair<int, int>, glm::ivec2> mSharedVQIndexShapes;     // Key: {level_idx, grid_type}
        std::unordered_map<std::pair<int, int>, glm::uvec3, pair_hash> mSharedVQIndexOriginalShapes;
        ptr<Buffer> mpDummyStorageBuffer;

        // NTC additions end
        void addTexture(std::string texturePath);
        void createDescriptors();

        ptr<Device> mpDevice;
        ptr<Swapchain> mpSwapchain;

        std::vector<Mesh> mMeshes;
        std::vector<Node> mNodes;
        std::vector<Material> mMaterials;
        std::unordered_map<std::string, ptr<Texture>> mTextures;
        ptr<Texture> mpEnvironmentMap;
        ptr<Descriptor> mpEnvironmentMapDescriptor;

        ptr<Buffer> mpVertexBuffer;
        ptr<Buffer> mpIndexBuffer;
        ptr<Buffer> mpTransforms;
        ptr<Buffer> mpMaterialParams;

        ptr<Texture> mpMissingTexture;

        bool mSupportRayTracing;
        ptr<AccelerationStructure> mpAccelerationStructure;
        ptr<Descriptor> mpRayTracingDescriptor;
        ptr<Buffer> mpMaterialBuffer; // Almost same as mpMaterialParams but for ray tracing
        ptr<Buffer> mpInstanceDataBuffer;

        uint32_t mVertexCount;
        uint32_t mIndexCount;
    };
}; // namespace Mandrill
