#pragma once

#include "Common.h"

#include "Device.h"
#include "Layout.h"
#include "Pass.h"
#include "Scene.h"
#include "Shader.h"

namespace Mandrill
{
    static std::vector<VkVertexInputBindingDescription> defaultBindingDescriptions = {{{
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    }}};

    static std::vector<VkVertexInputAttributeDescription> defaultAttributeDescriptions = {{
        {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, position),
        },
        {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, normal),
        },
        {
            .location = 2,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = offsetof(Vertex, texcoord),
        },
        {
            .location = 3,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, tangent),
        },
        {
            .location = 4,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, binormal),
        },
    }};

    struct PipelineDesc {
        std::vector<VkVertexInputBindingDescription> bindingDescriptions;
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
        VkPolygonMode polygonMode;
        VkBool32 depthTest;
        VkBool32 blend;
        VkBool32 alphaToCoverage;

        MANDRILL_API
        PipelineDesc(
            std::vector<VkVertexInputBindingDescription> bindingDescriptions = defaultBindingDescriptions,
            std::vector<VkVertexInputAttributeDescription> attributeDescriptions = defaultAttributeDescriptions,
            VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL, VkBool32 depthTest = VK_TRUE, VkBool32 blend = VK_FALSE,
            VkBool32 alphaToCoverage = VK_FALSE)
            : bindingDescriptions(bindingDescriptions), attributeDescriptions(attributeDescriptions),
              polygonMode(polygonMode), depthTest(depthTest), blend(blend), alphaToCoverage(alphaToCoverage)
        {
        }
    };

    class Pipeline
    {
    public:
        MANDRILL_NON_COPYABLE(Pipeline)

        /// <summary>
        /// Create a new pipeline.
        /// </summary>
        /// <param name="pDevice">Device to use</param>
        /// <param name="pPass">Pass to use</param>
        /// <param name="pLayout">Layout to use</param>
        /// <param name="pShader">Shader to use</param>
        /// <param name="desc">Description of pipeline</param>
        MANDRILL_API Pipeline(ptr<Device> pDevice, ptr<Pass> pPass, ptr<Layout> pLayout, ptr<Shader> pShader,
                              const PipelineDesc& desc = PipelineDesc());

        /// <summary>
        /// Destructor for pipeline.
        /// </summary>
        MANDRILL_API ~Pipeline();

        /// <summary>
        /// Bind a pipeline for rendering and set its dynamic states.
        /// </summary>
        /// <param name="cmd">Command buffer to use</param>
        /// <returns></returns>
        MANDRILL_API void bind(VkCommandBuffer cmd) const;

        /// <summary>
        /// Recreate a pipeline. Call this if shader source code has changed and should be reloaded.
        /// </summary>
        /// <returns></returns>
        MANDRILL_API void recreate();

        /// <summary>
        /// Get the pipeline handle.
        /// </summary>
        /// <returns>Pipeline handle</returns>
        MANDRILL_API VkPipeline getPipeline() const
        {
            return mPipeline;
        }

        /// <summary>
        /// Get the pipeline layout handle.
        /// </summary>
        /// <returns>Pipeline layout handle</returns>
        MANDRILL_API VkPipelineLayout getLayout() const
        {
            return mPipelineLayout;
        }

        /// <summary>
        /// Set the cull mode.
        /// </summary>
        /// <param name="cullMode">New cull mode</param>
        /// <returns></returns>
        MANDRILL_API void setCullMode(VkCullModeFlagBits cullMode)
        {
            mCullMode = cullMode;
        }

        /// <summary>
        /// Set the front face.
        /// </summary>
        /// <param name="frontFace">New front face</param>
        /// <returns></returns>
        MANDRILL_API void setFrontFace(VkFrontFace frontFace)
        {
            mFrontFace = frontFace;
        }

        /// <summary>
        /// Set line width.
        /// </summary>
        /// <param name="lineWidth">New line width</param>
        /// <returns></returns>
        MANDRILL_API void setLineWidth(float lineWidth)
        {
            mLineWidth = lineWidth;
        }

    protected:
        virtual void createPipeline();
        virtual void destroyPipeline();

        ptr<Device> mpDevice;
        ptr<Pass> mpPass;

        ptr<Shader> mpShader;
        ptr<Layout> mpLayout;

        VkPipeline mPipeline;
        VkPipelineLayout mPipelineLayout;

    private:
        VkPolygonMode mPolygonMode;
        VkBool32 mDepthTest;
        VkBool32 mBlend;
        VkBool32 mAlphaToCoverage;

        std::vector<VkVertexInputBindingDescription> mBindingDescriptions;
        std::vector<VkVertexInputAttributeDescription> mAttributeDescriptions;
        VkCullModeFlagBits mCullMode = VK_CULL_MODE_NONE;
        VkFrontFace mFrontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        float mLineWidth = 1.0f;
    };
} // namespace Mandrill
