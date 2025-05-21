#include "App.h"

#include "Error.h"
#include "Helpers.h"
#include "Log.h"

#include "IconData.h"
#include "RobotoFontData.h"
#include "stb_image.h"


#if MANDRILL_WINDOWS
// Include windows.h first
#include <windows.h>
// Line to stop formatter
#include <shellapi.h>
#endif

using namespace Mandrill;

static void errorCallback(int errorCode, const char* pDescription)
{
    Log::error("GLFW error {}: {}", errorCode, pDescription);
}

App::App(const std::string& title, uint32_t width, uint32_t height) : mWidth(width), mHeight(height)
{
    Log::info("=== Mandrill {}.{}.{} ===", MANDRILL_VERSION_MAJOR, MANDRILL_VERSION_MINOR, MANDRILL_VERSION_PATCH);

    Log::info("Initializing GLFW");
    initGLFW(title, width, height);

    Log::info("Initializing ImGUI");
    initImGUI();

    mFullscreen = false;
}

App::~App()
{
    glfwDestroyWindow(mpWindow);
    glfwTerminate();
}

void App::run()
{
    Log::info("Running...");

    while (!glfwWindowShouldClose(mpWindow)) {
        mDelta = static_cast<float>(glfwGetTime());
        glfwSetTime(0.0);

        mDeltaSmooth = kSmoothingFactor * mDelta + (1 - kSmoothingFactor) * mDeltaSmooth;

        // Mouse cursor movement
        mCursorDeltaX = mCursorX - mCursorPrevX;
        mCursorDeltaY = mCursorY - mCursorPrevY;
        mCursorPrevX = mCursorX;
        mCursorPrevY = mCursorY;

        auto& io = ImGui::GetIO();
        mKeyboardCapturedByGUI = io.WantCaptureKeyboard;
        mMouseCapturedByGUI = io.WantCaptureMouse;

        update(mDelta);

        ImGuiContext* pContext = newFrameGUI();

        appGUI(pContext);

        render();

        glfwPollEvents();
    }

    Log::info("Exiting...");
}

void App::createGUI(ptr<Device> pDevice, ptr<Pass> pPass)
{
    // Create descriptor pool for ImGUI
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
    };

    VkDescriptorPoolCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 1,
        .poolSizeCount = count(poolSizes),
        .pPoolSizes = poolSizes.data(),
    };

    Check::Vk(vkCreateDescriptorPool(pDevice->getDevice(), &ci, nullptr, &mDescriptorPool));

    // Setup platform for ImGUI
    ImGui_ImplVulkan_InitInfo info = {
        .Instance = pDevice->getInstance(),
        .PhysicalDevice = pDevice->getPhysicalDevice(),
        .Device = pDevice->getDevice(),
        .QueueFamily = pDevice->getQueueFamily(),
        .Queue = pDevice->getQueue(),
        .DescriptorPool = mDescriptorPool,
        .MinImageCount = 2,
        .ImageCount = 2,
        .MSAASamples = pPass->getSampleCount(),
        .PipelineCache = VK_NULL_HANDLE,
        .UseDynamicRendering = true,
        .PipelineRenderingCreateInfo = pPass->getPipelineRenderingCreateInfo(),
        .Allocator = nullptr,
    };

    ImGui_ImplVulkan_Init(&info);

    // Create fonts texture
    ImGuiIO& io = ImGui::GetIO();
    mFont = io.Fonts->AddFontFromMemoryTTF((void*)res_Roboto_ttf, // Or whatever xxd named it
                                           res_Roboto_ttf_len,    // Or whatever xxd named it
                                           16.0f);

    ImGui_ImplVulkan_CreateFontsTexture();

    // Set style
    ImGuiStyle& style = ImGui::GetStyle();

    style.WindowPadding = ImVec2(5, 5);
    style.WindowRounding = 5.0f;
    style.FramePadding = ImVec2(5, 5);
    style.FrameRounding = 4.0f;
    style.FrameBorderSize = 0.0f;
    style.ItemSpacing = ImVec2(12, 8);
    style.ItemInnerSpacing = ImVec2(8, 6);
    style.IndentSpacing = 25.0f;
    style.ScrollbarSize = 15.0f;
    style.ScrollbarRounding = 9.0f;
    style.GrabMinSize = 5.0f;
    style.GrabRounding = 3.0f;

    mCreatedGUI = true;
}

void App::destroyGUI(ptr<Device> pDevice)
{
    vkDeviceWaitIdle(pDevice->getDevice());

    ImGui_ImplVulkan_DestroyFontsTexture();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    vkDestroyDescriptorPool(pDevice->getDevice(), mDescriptorPool, nullptr);

    mCreatedGUI = false;
}

ImGuiContext* App::newFrameGUI()
{
    if (!mCreatedGUI) {
        return nullptr;
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::StyleColorsLight();
    ImGui::PushFont(mFont);

    return ImGui::GetCurrentContext();
}

void App::baseGUI(ptr<Device> pDevice, ptr<Swapchain> pSwapchain, ptr<Pipeline> pPipeline)
{
    std::vector<ptr<Pipeline>> pipelines;
    pipelines.push_back(pPipeline);
    App::baseGUI(pDevice, pSwapchain, pipelines);
}

void App::baseGUI(ptr<Device> pDevice, ptr<Swapchain> pSwapchain, std::vector<ptr<Pipeline>> pPipelines)
{
    if (!mCreatedGUI) {
        return;
    }

    if (mShowMainMenu && ImGui::BeginMainMenuBar()) {

        if (ImGui::BeginMenu("File")) {

            if (ImGui::MenuItem("Exit", "ESC / Alt + F4", false)) {
                glfwSetWindowShouldClose(mpWindow, GLFW_TRUE);
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Edit")) {
            bool vsync = pDevice->getVsync();
            if (ImGui::MenuItem("Verical sync", "V", &vsync)) {
                pDevice->setVsync(vsync);
                pSwapchain->recreate();
            }

            if (ImGui::MenuItem("Reload shaders", "R")) {
                for (auto& pipeline : pPipelines) {
                    if (pipeline) {
                        pipeline->recreate();
                    }
                }
                pSwapchain->recreate();
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {

            if (ImGui::MenuItem("Hide menu", "F2", false)) {
                mShowMainMenu = !mShowMainMenu;
            }

            if (ImGui::MenuItem("Frame rate", "F3", false)) {
                mShowFrameRate = !mShowFrameRate;
            }

            if (ImGui::MenuItem("Toggle fullscreen", "F11", false)) {
                toggleFullscreen();
            }

            if (ImGui::MenuItem("Take screenshot", "F12", false)) {
                takeScreenshot(pDevice, pSwapchain);
            }

            if (ImGui::MenuItem("Reset to initial framesize", "", false)) {
                resetFramebufferSize();
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("Show controls", "F1", false)) {
                mShowHelp = !mShowHelp;
            }

            if (ImGui::MenuItem("About", "", false)) {
                mShowAbout = !mShowAbout;
            }

            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }

    if (mShowFrameRate) {
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs;
        ImGui::SetNextWindowPos(ImVec2(10.0f, 30.0f), ImGuiCond_Appearing);
        if (ImGui::Begin("Frame rate", &mShowFrameRate, flags)) {
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 255, 255));
            ImGui::Text("Frametime: %.2f ms", mDeltaSmooth * 1000.0f);
            ImGui::Text("FPS: %.2f", 1.0f / mDeltaSmooth);
            ImGui::PopStyleColor();
        }
        ImGui::End();
    }

    if (mShowHelp) {
        if (ImGui::Begin("Help", &mShowHelp)) {
            ImGui::Text("Camera movement:");
            ImGui::Text("\tW: Move forward");
            ImGui::Text("\tS: Move backward");
            ImGui::Text("\tA: Move left");
            ImGui::Text("\tD: Move right");
            ImGui::Text("\tE: Move up");
            ImGui::Text("\tQ: Move down");
            ImGui::Text("\tArrow keys: Pan");
            ImGui::Text("\tPeriod: Zoom in");
            ImGui::Text("\tComma: Zoom out");
            ImGui::Text("\tShift: Speed up movement");
            ImGui::Text("\tCtrl: Slow down movement");
            ImGui::Text("\tLeft mouse button: Click and drag to pan");
            ImGui::Text("\tRight mouse button: Capture/release mouse for panning");
        }
        ImGui::End();
    }

    if (mShowAbout) {
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                                 ImGuiWindowFlags_AlwaysAutoResize;
        int width, height;
        glfwGetWindowSize(mpWindow, &width, &height);
        ImGui::SetNextWindowPos(ImVec2(0.5f * width, 0.5f * height), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        if (ImGui::Begin("About", &mShowAbout, flags)) {
            ImGui::Text("%s v%d.%d.%d", MANDRILL_NAME, MANDRILL_VERSION_MAJOR, MANDRILL_VERSION_MINOR,
                        MANDRILL_VERSION_PATCH);
            ImGui::Text(
                "This is an education and research graphics framework based on Vulkan, written and used at Lund "
                "University.");
            ImGui::Text(
                "Latest source code is available from the git repository and is released under the MIT License.");

            if (ImGui::Button("Go to repo")) {
#if MANDRILL_WINDOWS
                ShellExecute(0, 0, "https://github.com/rikardolajos/Mandrill/tree/master", 0, 0, SW_SHOW);
#elif MANDRILL_LINUX
                if (std::system("xdg-open https://github.com/rikardolajos/Mandrill/tree/master")) {
                    Log::error("Unable to open browser");
                }
#endif
            }

            ImGui::SameLine();
            if (ImGui::Button("Close")) {
                mShowAbout = false;
            }
        }
        ImGui::End();
    }
}

void App::renderGUI(VkCommandBuffer cmd) const
{
    if (!mCreatedGUI) {
        return;
    }

    ImGui::PopFont();

    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

void App::baseKeyCallback(GLFWwindow* pWindow, int key, int scancode, int action, int mods, ptr<Device> pDevice,
                          ptr<Swapchain> pSwapchain, ptr<Pipeline> pPipeline)
{
    std::vector<ptr<Pipeline>> pipelines;
    pipelines.push_back(pPipeline);
    App::baseKeyCallback(pWindow, key, scancode, action, mods, pDevice, pSwapchain, pipelines);
}

void App::baseKeyCallback(GLFWwindow* pWindow, int key, int scancode, int action, int mods, ptr<Device> pDevice,
                          ptr<Swapchain> pSwapchain, std::vector<ptr<Pipeline>> pipelines)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(mpWindow, 1);
    }

    if (key == GLFW_KEY_F1 && action == GLFW_PRESS) {
        mShowHelp = !mShowHelp;
    }

    if (key == GLFW_KEY_F2 && action == GLFW_PRESS) {
        mShowMainMenu = !mShowMainMenu;
    }

    if (key == GLFW_KEY_F3 && action == GLFW_PRESS) {
        mShowFrameRate = !mShowFrameRate;
    }

    if (key == GLFW_KEY_F11 && action == GLFW_PRESS) {
        toggleFullscreen();
    }

    if (key == GLFW_KEY_F12 && action == GLFW_PRESS) {
        takeScreenshot(pDevice, pSwapchain);
    }

    if (key == GLFW_KEY_V && action == GLFW_PRESS) {
        bool vsync = pDevice->getVsync();
        pDevice->setVsync(!vsync);
        pSwapchain->recreate();
    }

    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        for (auto& pipeline : pipelines) {
            if (pipeline) {
                pipeline->recreate();
            }
        }

        pSwapchain->recreate();
    }
}

void App::baseCursorPosCallback(GLFWwindow* pWindow, double xPos, double yPos)
{
    mCursorX = static_cast<float>(xPos);
    mCursorY = static_cast<float>(yPos);
}

void App::baseMouseButtonCallback(GLFWwindow* pWindow, int button, int action, int mods, ptr<Camera> pCamera)
{
    if (button == GLFW_MOUSE_BUTTON_2 && action == GLFW_PRESS) {
        bool captured = pCamera->toggleMouseCapture();
        glfwSetInputMode(pWindow, GLFW_CURSOR, captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);

        ImGuiIO& io = ImGui::GetIO();
        io.SetAppAcceptingEvents(!captured);
        if (captured) {
            ImGui::SetWindowFocus();
        }
    }
}

void App::initGLFW(const std::string& title, uint32_t width, uint32_t height)
{
    if (glfwInit() == GLFW_FALSE) {
        Log::error("Failed to initialze GLFW");
        Check::GLFW();
    }

    glfwSetErrorCallback(errorCallback);

    // Save video mode of fullscreen mode
    mpMonitor = glfwGetPrimaryMonitor();
    if (!mpMonitor) {
        Log::error("Failed to find primary monitor");
        Check::GLFW();
    }
    const GLFWvidmode* mode = glfwGetVideoMode(mpMonitor);
    std::memcpy(&mFullscreenMode, mode, sizeof(GLFWvidmode));

    glfwWindowHint(GLFW_RED_BITS, mFullscreenMode.redBits);
    glfwWindowHint(GLFW_GREEN_BITS, mFullscreenMode.greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, mFullscreenMode.blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, mFullscreenMode.refreshRate);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    Check::GLFW();

    std::string fullTitle = std::format("Mandrill: {}", title);
    mpWindow = glfwCreateWindow(width, height, fullTitle.c_str(), nullptr, nullptr);
    if (!mpWindow) {
        Log::error("Failed to create window");
        Check::GLFW();
    }

    GLFWimage image = {};
    image.pixels = stbi_load_from_memory(res_icon_png, static_cast<int>(res_icon_png_len), &image.width, &image.height,
                                         nullptr, 4);
    if (image.pixels) {
        glfwSetWindowIcon(mpWindow, 1, &image);
        stbi_image_free(image.pixels);
    } else {
        Log::error("Failed to load icon.png");
    }

    if (!glfwVulkanSupported()) {
        Log::error("Failed to find Vulkan");
        Check::GLFW();
    }

    glfwSetWindowUserPointer(mpWindow, this);
    glfwSetKeyCallback(mpWindow, keyCallbackEntry);
    glfwSetCursorPosCallback(mpWindow, cursorPosCallbackEntry);
    glfwSetMouseButtonCallback(mpWindow, mouseButtonCallbackEntry);
    Check::GLFW();
}

void App::initImGUI()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls

    ImGui_ImplGlfw_InitForVulkan(mpWindow, true);
}

static void saveScreenshot(uint8_t* data, uint32_t width, uint32_t height, uint32_t channels)
{
    char timestamp[64];
    time_t t = std::time(nullptr);
    std::strftime(timestamp, 64, "%G-%m-%d_%H-%M-%S", std::localtime(&t));
    std::filesystem::path filename = std::format("{}.png", timestamp);
    stbi_write_png(filename.string().c_str(), width, height, channels, data, width * 4);
    auto fullpath = std::filesystem::current_path() / filename;

    Log::info("Screenshot saved to {}", fullpath.string());
}

void App::takeScreenshot(ptr<Device> pDevice, ptr<Swapchain> pSwapchain)
{
    Log::info("Taking screenshot and saving to disk...");

    const uint32_t channels = 4;

    pSwapchain->waitForInFlightImage();

    Helpers::transitionImageLayout(pDevice, pSwapchain->getImage(), pSwapchain->getImageFormat(),
                                   VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1);
    VkDeviceSize size = pSwapchain->getExtent().width * pSwapchain->getExtent().height * channels;
    Buffer buffer(pDevice, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Helpers::copyImageToBuffer(pDevice, pSwapchain->getImage(), buffer.getBuffer(), pSwapchain->getExtent().width,
                               pSwapchain->getExtent().height);
    Helpers::transitionImageLayout(pDevice, pSwapchain->getImage(), pSwapchain->getImageFormat(),
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, 1);
    uint8_t* data = static_cast<uint8_t*>(buffer.getHostMap());

    // BGR -> RGB
    for (uint32_t i = 0; i < size; i += channels) {
        std::swap(data[i], data[i + 2]);
    }

    // Run in a thread since saving takes forever
    auto thread =
        std::thread(saveScreenshot, data, pSwapchain->getExtent().width, pSwapchain->getExtent().height, channels);
    thread.detach();
}

void App::toggleFullscreen()
{
    if (mFullscreen) {
        // Set windowed
        int xpos = (mFullscreenMode.width - mWidth) / 2;
        int ypos = (mFullscreenMode.height - mHeight) / 2;
        glfwSetWindowMonitor(mpWindow, NULL, xpos, ypos, mWidth, mHeight, mFullscreenMode.refreshRate);
    } else {
        // Set fullscreen
        glfwSetWindowMonitor(mpWindow, mpMonitor, 0, 0, mFullscreenMode.width, mFullscreenMode.height,
                             mFullscreenMode.refreshRate);
    }
    Check::GLFW();
    mFullscreen = !mFullscreen;
}

void App::resetFramebufferSize()
{
    // Set windowed
    int xpos = (mFullscreenMode.width - mWidth) / 2;
    int ypos = (mFullscreenMode.height - mHeight) / 2;
    glfwSetWindowMonitor(mpWindow, NULL, xpos, ypos, mWidth, mHeight, mFullscreenMode.refreshRate);

    Check::GLFW();
    mFullscreen = false;
}

void App::keyCallbackEntry(GLFWwindow* pWindow, int key, int scancode, int action, int mods)
{
    App* pApp = static_cast<App*>(glfwGetWindowUserPointer(pWindow));
    pApp->appKeyCallback(pWindow, key, scancode, action, mods);
}

void App::cursorPosCallbackEntry(GLFWwindow* pWindow, double xPos, double yPos)
{
    App* pApp = static_cast<App*>(glfwGetWindowUserPointer(pWindow));
    pApp->appCursorPosCallback(pWindow, xPos, yPos);
}

void App::mouseButtonCallbackEntry(GLFWwindow* pWindow, int button, int action, int mods)
{
    App* pApp = static_cast<App*>(glfwGetWindowUserPointer(pWindow));
    pApp->appMouseButtonCallback(pWindow, button, action, mods);
}
