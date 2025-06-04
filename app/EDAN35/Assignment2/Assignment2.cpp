#include "Mandrill.h"

#include <filesystem> // C++17
#include <string>

using namespace Mandrill;

class Assignment2 : public App
{
public:
    enum {
        GBUFFER_PASS = 0,
        RESOLVE_PASS = 1,
    };

    struct PushConstants {
        int renderMode;
    };

    static std::shared_ptr<Image> createColorAttachmentImage(std::shared_ptr<Device> pDevice, uint32_t width,
                                                             uint32_t height)
    {
        return std::make_shared<Image>(pDevice, width, height, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_B8G8R8A8_UNORM,
                                       VK_IMAGE_TILING_OPTIMAL,
                                       VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    void transitionAttachmentsForGBuffer(VkCommandBuffer cmd)
    {
        for (auto& attachment : mColorAttachments) {
            Helpers::imageBarrier(cmd, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, attachment->getImage());
        }
    }

    void transitionAttachmentsForResolve(VkCommandBuffer cmd)
    {
        for (auto& attachment : mColorAttachments) {
            Helpers::imageBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, attachment->getImage());
        }
    }

    void createAttachments()
    {
        uint32_t width = mpSwapchain->getExtent().width;
        uint32_t height = mpSwapchain->getExtent().height;
        VkFormat depthFormat = Helpers::findDepthFormat(mpDevice);

        mColorAttachments.clear();
        mColorAttachments.push_back(createColorAttachmentImage(mpDevice, width, height));
        mColorAttachments.push_back(createColorAttachmentImage(mpDevice, width, height));
        mColorAttachments.push_back(createColorAttachmentImage(mpDevice, width, height));
        mpDepthAttachment = std::make_shared<Image>(
            mpDevice, width, height, 1, 1, VK_SAMPLE_COUNT_1_BIT, depthFormat, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // Create image views and transition image layouts
        VkCommandBuffer cmd = Helpers::cmdBegin(mpDevice);

        for (auto& attachment : mColorAttachments) {
            attachment->createImageView(VK_IMAGE_ASPECT_COLOR_BIT);

            // Transition to correct layout for descriptor creation
            Helpers::imageBarrier(cmd, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  attachment->getImage());
        }

        // Transition depth attachment for depth use
        VkImageSubresourceRange depthSubresourceRange = {.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                                                         .baseMipLevel = 0,
                                                         .levelCount = 1,
                                                         .baseArrayLayer = 0,
                                                         .layerCount = 1};
        if (depthFormat == VK_FORMAT_D32_SFLOAT_S8_UINT || depthFormat == VK_FORMAT_D24_UNORM_S8_UINT) {
            depthSubresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        Helpers::imageBarrier(
            cmd, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, mpDepthAttachment->getImage(),
            &depthSubresourceRange);

        Helpers::cmdEnd(mpDevice, cmd);

        // Create descriptor for resolve pass input attachments
        std::vector<DescriptorDesc> descriptorDesc;
        for (auto& attachment : mColorAttachments) {
            descriptorDesc.emplace_back(VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, attachment);
        }
        mpColorAttachmentDescriptor =
            std::make_shared<Descriptor>(mpDevice, descriptorDesc, mpResolveLayout->getDescriptorSetLayouts()[0]);
    }

    Assignment2() : App("Assignment2: Deferred Shading and Shadow Maps", 1280, 720)
    {
        // Create a Vulkan instance and device
        mpDevice = std::make_shared<Device>(mpWindow);

        // Create a swapchain with 2 frames in flight
        mpSwapchain = std::make_shared<Swapchain>(mpDevice, 2);

        // Create scene
        mpScene = std::make_shared<Scene>(mpDevice, mpSwapchain);

        // Create layouts for rendering the scene in first pass and resolve in the second pass
        auto pGBufferLayout = mpScene->getLayout();

        // 3 color attachments: position, normal, albedo
        std::vector<LayoutDesc> layoutDesc;
        layoutDesc.emplace_back(0, 0, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VK_SHADER_STAGE_FRAGMENT_BIT);
        layoutDesc.emplace_back(0, 1, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VK_SHADER_STAGE_FRAGMENT_BIT);
        layoutDesc.emplace_back(0, 2, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VK_SHADER_STAGE_FRAGMENT_BIT);
        mpResolveLayout = std::make_shared<Layout>(mpDevice, layoutDesc);

        // Add push constant to layout so we can set render mode in resolve pipeline
        VkPushConstantRange pushConstantRange = {
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(PushConstants),
        };
        mpResolveLayout->addPushConstantRange(pushConstantRange);

        // Create two passes
        createAttachments();
        mpGBufferPass = std::make_shared<Pass>(mpDevice, mColorAttachments, mpDepthAttachment);
        mpResolvePass = std::make_shared<Pass>(mpDevice, mpSwapchain->getExtent(), mpSwapchain->getImageFormat());

        // Create two shaders (and pipelines) for G-buffer and resolve pass respectively
        std::vector<ShaderDesc> shaderDesc;
        shaderDesc.emplace_back("Assignment2/GBuffer.vert", "main", VK_SHADER_STAGE_VERTEX_BIT);
        shaderDesc.emplace_back("Assignment2/GBuffer.frag", "main", VK_SHADER_STAGE_FRAGMENT_BIT);
        auto pGBufferShader = std::make_shared<Shader>(mpDevice, shaderDesc);

        shaderDesc.clear();
        shaderDesc.emplace_back("Assignment2/Resolve.vert", "main", VK_SHADER_STAGE_VERTEX_BIT);
        shaderDesc.emplace_back("Assignment2/Resolve.frag", "main", VK_SHADER_STAGE_FRAGMENT_BIT);
        auto pResolveShader = std::make_shared<Shader>(mpDevice, shaderDesc);

        // Create pipelines
        PipelineDesc pipelineDesc;
        pipelineDesc.depthTest = VK_TRUE;
        mPipelines.emplace_back(
            std::make_shared<Pipeline>(mpDevice, mpGBufferPass, pGBufferLayout, pGBufferShader, pipelineDesc));

        pipelineDesc.depthTest = VK_FALSE;
        mPipelines.emplace_back(
            std::make_shared<Pipeline>(mpDevice, mpResolvePass, mpResolveLayout, pResolveShader, pipelineDesc));

        // Create a sampler that will be used to render materials
        mpSampler = std::make_shared<Sampler>(mpDevice);

        // Load scene
        auto meshIndices =
            mpScene->addMeshFromFile("/home/arseni/repositories/ntc-renderer/Mandrill/res/sponza/sponza.obj",
                                     "/home/arseni/repositories/ntc-renderer/Mandrill/res/sponza");
        std::shared_ptr<Node> pNode = mpScene->addNode();
        pNode->setPipeline(mPipelines[GBUFFER_PASS]); // Render scene with first pass pipeline
        for (auto meshIndex : meshIndices) {
            pNode->addMesh(meshIndex);
        }
        // Scale down the model
        pNode->setTransform(glm::scale(glm::vec3(0.01f)));

        mpScene->setSampler(mpSampler);
        mpScene->compile();
        mpScene->syncToDevice();

        // Activate back-face culling for G-buffer pass
        mPipelines[GBUFFER_PASS]->setCullMode(VK_CULL_MODE_BACK_BIT);

        // Setup camera
        mpCamera = std::make_shared<Camera>(mpDevice, mpWindow, mpSwapchain);
        mpCamera->setPosition(glm::vec3(5.0f, 0.0f, 0.0f));
        mpCamera->setTarget(glm::vec3(0.0f, 0.0f, 0.0f));
        mpCamera->setFov(60.0f);

        // Initialize GUI
        App::createGUI(mpDevice, mpResolvePass);
    }

    ~Assignment2()
    {
        App::destroyGUI(mpDevice);
    }

    void update(float delta)
    {
        if (!keyboardCapturedByGUI() && !mouseCapturedByGUI()) {
            mpCamera->update(delta, getCursorDelta());
        }
    }

    void render() override
    {
        // Check if camera matrix and attachments need to be updated
        if (mpSwapchain->recreated()) {
            mpCamera->updateAspectRatio();
            createAttachments();
            mpGBufferPass->update(mColorAttachments, mpDepthAttachment);
            mpResolvePass->update(mpSwapchain->getExtent());
        }

        // Acquire frame from swapchain and prepare rasterizer
        VkCommandBuffer cmd = mpSwapchain->acquireNextImage();

        // Transition color attachments to correct image layout
        transitionAttachmentsForGBuffer(cmd);

        // Begin G-Buffer pass
        mpGBufferPass->begin(cmd, glm::vec4(0.2f, 0.6f, 1.0f, 1.0f));

        // Render scene
        mpScene->render(cmd, mpCamera);

        // End the G-Buffer pass without any implicit image transitions
        vkCmdEndRendering(cmd);

        // Transition color attachments to correct image layout
        transitionAttachmentsForResolve(cmd);

        mpResolvePass->begin(cmd);

        // Bind the pipeline for the full-screen quad
        mPipelines[RESOLVE_PASS]->bind(cmd);

        // Bind descriptors for color attachments
        mpColorAttachmentDescriptor->bind(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipelines[RESOLVE_PASS]->getLayout(),
                                          0);

        // Push constants
        PushConstants pushConstants = {
            .renderMode = mRenderMode,
        };
        vkCmdPushConstants(cmd, mPipelines[RESOLVE_PASS]->getLayout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(PushConstants), &pushConstants);

        // Render full-screen quad to resolve final composition
        vkCmdDraw(cmd, 3, 1, 0, 0);

        // Draw GUI
        App::renderGUI(cmd);

        // Submit command buffer to rasterizer and present swapchain frame
        mpResolvePass->end(cmd);
        mpSwapchain->present(cmd, mpResolvePass->getOutput());
    }

    void appGUI(ImGuiContext* pContext)
    {
        ImGui::SetCurrentContext(pContext);

        // Render the base GUI, the menu bar with it's subwindows
        App::baseGUI(mpDevice, mpSwapchain, mPipelines);

        if (ImGui::Begin("Assignment 2")) {
            const char* renderModes[] = {
                "Resolved",
                "Position",
                "Normal",
                "Albedo",
            };
            ImGui::Combo("Render mode", &mRenderMode, renderModes, IM_ARRAYSIZE(renderModes));
        }

        ImGui::End();
    }

    void appKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        App::baseKeyCallback(window, key, scancode, action, mods, mpDevice, mpSwapchain, mPipelines);
    }

    void appCursorPosCallback(GLFWwindow* pWindow, double xPos, double yPos)
    {
        App::baseCursorPosCallback(pWindow, xPos, yPos);
    }

    void appMouseButtonCallback(GLFWwindow* pWindow, int button, int action, int mods)
    {
        App::baseMouseButtonCallback(pWindow, button, action, mods, mpCamera);
    }


private:
    std::shared_ptr<Device> mpDevice;
    std::shared_ptr<Swapchain> mpSwapchain;
    std::shared_ptr<Pass> mpGBufferPass;
    std::shared_ptr<Pass> mpResolvePass;
    std::vector<std::shared_ptr<Pipeline>> mPipelines;

    std::shared_ptr<Layout> mpResolveLayout;

    std::vector<std::shared_ptr<Image>> mColorAttachments;
    std::shared_ptr<Descriptor> mpColorAttachmentDescriptor;

    std::shared_ptr<Image> mpDepthAttachment;

    std::shared_ptr<Sampler> mpSampler;
    std::shared_ptr<Scene> mpScene;
    std::shared_ptr<Camera> mpCamera;

    int mRenderMode = 0;
};

int main()
{
    Assignment2 app = Assignment2();
    app.run();
    return 0;
}
