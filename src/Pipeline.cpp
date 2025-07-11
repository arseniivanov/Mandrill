#include "Pipeline.h"

#include "Error.h"
#include "Helpers.h"

using namespace Mandrill;

Pipeline::Pipeline(ptr<Device> pDevice, ptr<Pass> pPass, ptr<Layout> pLayout, ptr<Shader> pShader,
                   const PipelineDesc& desc)
    : mpDevice(pDevice), mpShader(pShader), mpLayout(pLayout), mpPass(pPass), mPipeline(nullptr),
      mBindingDescriptions(desc.bindingDescriptions), mAttributeDescriptions(desc.attributeDescriptions),
      mPolygonMode(desc.polygonMode), mDepthTest(desc.depthTest), mBlend(desc.blend),
      mAlphaToCoverage(desc.alphaToCoverage)
{
    const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts = mpLayout->getDescriptorSetLayouts();
    const std::vector<VkPushConstantRange>& pushConstantLayout = mpLayout->getPushConstantRanges();

    VkPipelineLayoutCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = count(descriptorSetLayouts),
        .pSetLayouts = descriptorSetLayouts.data(),
        .pushConstantRangeCount = count(pushConstantLayout),
        .pPushConstantRanges = pushConstantLayout.data(),
    };

    Check::Vk(vkCreatePipelineLayout(mpDevice->getDevice(), &ci, nullptr, &mPipelineLayout));

    if (pPass) {
        createPipeline();
    }
}

Pipeline::~Pipeline()
{
    destroyPipeline();
    vkDestroyPipelineLayout(mpDevice->getDevice(), mPipelineLayout, nullptr);
}

void Pipeline::bind(VkCommandBuffer cmd) const
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipeline);

    // Set dynamic states
    vkCmdSetFrontFace(cmd, mFrontFace);
    vkCmdSetCullMode(cmd, mCullMode);
    vkCmdSetLineWidth(cmd, mLineWidth);

    VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(mpPass->getExtent().width),
        .height = static_cast<float>(mpPass->getExtent().height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = {
        .offset = {0, 0},
        .extent = mpPass->getExtent(),
    };
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void Pipeline::recreate()
{
    destroyPipeline();
    mpShader->reload();
    createPipeline();
}

void Pipeline::createPipeline()
{
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = count(mBindingDescriptions),
        .pVertexBindingDescriptions = mBindingDescriptions.data(),
        .vertexAttributeDescriptionCount = count(mAttributeDescriptions),
        .pVertexAttributeDescriptions = mAttributeDescriptions.data(),
    };

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,   VK_DYNAMIC_STATE_LINE_WIDTH,
    };

    VkPipelineDynamicStateCreateInfo dynamicState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = count(dynamicStates),
        .pDynamicStates = dynamicStates.data(),
    };

    VkPipelineViewportStateCreateInfo viewportState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    VkPipelineRasterizationStateCreateInfo rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = mPolygonMode,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
    };

    VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = mpPass->getSampleCount(),
        .sampleShadingEnable = VK_FALSE,
        .alphaToCoverageEnable = mAlphaToCoverage,
    };

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {
        .blendEnable = mBlend,
        .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    VkPipelineDepthStencilStateCreateInfo depthStencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = mDepthTest,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .minDepthBounds = 0.0f,
        .maxDepthBounds = 1.0f,
    };

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachmentStates;
    for (uint32_t i = 0; i < count(mpPass->getColorAttachments()); i++) {
        colorBlendAttachmentStates.push_back(colorBlendAttachment);
    }

    VkPipelineColorBlendStateCreateInfo colorBlending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .attachmentCount = count(colorBlendAttachmentStates),
        .pAttachments = colorBlendAttachmentStates.data(),
    };

    auto stages = mpShader->getStages();
    VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo = mpPass->getPipelineRenderingCreateInfo();

    VkGraphicsPipelineCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &pipelineRenderingCreateInfo,
        .stageCount = count(stages),
        .pStages = stages.data(),
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = mPipelineLayout,
        .renderPass = nullptr,
    };
    Check::Vk(vkCreateGraphicsPipelines(mpDevice->getDevice(), VK_NULL_HANDLE, 1, &ci, nullptr, &mPipeline));
}

void Pipeline::destroyPipeline()
{
    vkDeviceWaitIdle(mpDevice->getDevice());
    vkDestroyPipeline(mpDevice->getDevice(), mPipeline, nullptr);
}
