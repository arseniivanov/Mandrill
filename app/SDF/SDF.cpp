#include "Mandrill.h"

using namespace Mandrill;

class SDF : public App
{
public:
    struct PushConstant {
        glm::vec3 resolution;
        float time;
        glm::vec4 mouse;
    };

    SDF() : App("SDF", 1000, 563)
    {
        // Create a Vulkan instance and device
        std::vector<const char*> extensions = {
            VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        };
        mpDevice = std::make_shared<Device>(mpWindow, extensions);

        // Create a swapchain with 2 frames in flight
        mpSwapchain = std::make_shared<Swapchain>(mpDevice, 2);

        // Create a layout (push descriptor):
        std::vector<LayoutDesc> layoutDesc;
        auto pLayout =
            std::make_shared<Layout>(mpDevice, layoutDesc, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);
        VkPushConstantRange pushConstantRange = {
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(PushConstant),
        };
        pLayout->addPushConstantRange(pushConstantRange);

        // Create a pass with 1 color attachment, depth attachment and multisampling
        mpPass = std::make_shared<Pass>(mpDevice, mpSwapchain->getExtent(), mpSwapchain->getImageFormat(), 1, true,
                                        mpDevice->getSampleCount());

        // Prepare vertex binding and attribute descriptions with empty vectors (only fullscreen triangles are used)
        std::vector<VkVertexInputBindingDescription> emptyBindingDescription;
        std::vector<VkVertexInputAttributeDescription> emptyAttributeDescription;
        PipelineDesc pipelineDesc = PipelineDesc(emptyBindingDescription, emptyAttributeDescription);
        pipelineDesc.depthTest = VK_FALSE;
        pipelineDesc.blend = VK_FALSE;

        // Create a pipeline
        std::vector<ShaderDesc> shaderDesc;
        shaderDesc.emplace_back("SDF/Fullscreen.vert", "main", VK_SHADER_STAGE_VERTEX_BIT);
        shaderDesc.emplace_back("SDF/SDF.frag", "main", VK_SHADER_STAGE_FRAGMENT_BIT);
        std::shared_ptr<Shader> pShader = std::make_shared<Shader>(mpDevice, shaderDesc);
        mpPipeline = std::make_shared<Pipeline>(mpDevice, mpPass, pLayout, pShader, pipelineDesc);

        // Initialize GUI
        App::createGUI(mpDevice, mpPass);
    }

    ~SDF()
    {
        App::destroyGUI(mpDevice);
    }

    void update(float delta)
    {
    }

    void render() override
    {
        // Check if camera matrix and attachments need to be updated
        if (mpSwapchain->recreated()) {
            mpPass->update(mpSwapchain->getExtent());
        }

        // Acquire frame from swapchain and prepare rasterizer
        VkCommandBuffer cmd = mpSwapchain->acquireNextImage();
        mpPass->begin(cmd, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

        // Push constants
        float aspectRatio =
            static_cast<float>(mpSwapchain->getExtent().width) / static_cast<float>(mpSwapchain->getExtent().height);
        PushConstant pushConstant = {
            .resolution = glm::vec3(mpSwapchain->getExtent().width, mpSwapchain->getExtent().height, aspectRatio),
            .time = mTime,
            .mouse = glm::vec4(mMouseCursorX, mMouseCursorY, mMouseClickX, mMouseClickY),
        };
        vkCmdPushConstants(cmd, mpPipeline->getLayout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pushConstant,
                           &pushConstant);

        // Bind pipeline and draw fullscreen triangle
        mpPipeline->bind(cmd);
        vkCmdDraw(cmd, 3, 1, 0, 0);

        // Draw GUI
        App::renderGUI(cmd);

        // Submit command buffer to rasterizer and present swapchain frame
        mpPass->end(cmd);
        mpSwapchain->present(cmd, mpPass->getOutput());
    }

    void appGUI(ImGuiContext* pContext)
    {
        ImGui::SetCurrentContext(pContext);
        App::baseGUI(mpDevice, mpSwapchain, mpPipeline);
    }

    void appKeyCallback(GLFWwindow* pWindow, int key, int scancode, int action, int mods)
    {
        App::baseKeyCallback(pWindow, key, scancode, action, mods, mpDevice, mpSwapchain, mpPipeline);
    }

    void appCursorPosCallback(GLFWwindow* pWindow, double xPos, double yPos)
    {
        App::baseCursorPosCallback(pWindow, xPos, yPos);

        if (mMouseClick) {
            mMouseCursorX = static_cast<float>(xPos);
            mMouseCursorY = static_cast<float>(yPos);
        }
    }

    void appMouseButtonCallback(GLFWwindow* pWindow, int button, int action, int mods)
    {
        App::baseMouseButtonCallback(pWindow, button, action, mods, nullptr);

        if (action == GLFW_PRESS) {
            mMouseClick = true;
            mMouseClickX = mMouseCursorX;
            mMouseClickY = mMouseCursorY;
        } else if (action == GLFW_RELEASE) {
            mMouseClick = false;
        }
    }

private:
    std::shared_ptr<Device> mpDevice;
    std::shared_ptr<Swapchain> mpSwapchain;
    std::shared_ptr<Pass> mpPass;
    std::shared_ptr<Pipeline> mpPipeline;

    bool mMouseClick = false;
    float mMouseCursorX = 628.0f;
    float mMouseCursorY = 251.0f;
    float mMouseClickX = 0.0f;
    float mMouseClickY = 0.0f;
};

int main()
{
    SDF app = SDF();
    app.run();
    return 0;
}
