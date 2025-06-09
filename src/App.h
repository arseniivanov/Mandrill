#pragma once

#include "Common.h"

#include "Camera.h"
#include "Device.h"
#include "Pass.h"
#include "Pipeline.h"
#include "Swapchain.h"

namespace Mandrill
{

    class MANDRILL_API App
    {
    public:
        /// <summary>
        /// Create a new Mandrill app.
        ///
        /// This class should be inherited by any application using Mandrill, and all the pure virtual functions have to
        /// be overriden.
        ///
        /// </summary>
        /// <param name="title">Title to be shown in the window frame</param>
        /// <param name="width">Width of window</param>
        /// <param name="height">Height of window</param>
        App(const std::string& title, uint32_t width = 1280, uint32_t height = 720);

        /// <summary>
        /// Destructor for app.
        /// </summary>
        ~App();

        /// <summary>
        /// This is the main function of an application. A typical usecase would be to inherit <c>App</c> and then
        /// invoke the <c>run()</c> function. For a class <c>SampleApp</c> that inherits <c>App</c> this is how it could
        /// be used:
        ///
        /// <code>
        /// int main()
        /// {
        ///     SampleApp app = SampleApp();
        ///     app.run();
        ///     return 0;
        /// }
        /// </code>
        /// </summary>
        void run();

    protected:
        /// <summary>
        /// Update scene objects.
        ///
        /// This function is called once per <c>run()</c> loop and can be used to update scene objects like meshes and
        /// cameras that should animate.
        ///
        /// This function should be overridden.
        /// </summary>
        /// <param name="delta">Time passed since last frame</param>
        virtual void update(float delta) = 0;

        /// <summary>
        /// Render app.
        ///
        /// This function is called once per <c>run()</c> loop is used to populate the command buffer for rendering.
        /// Typical use case is to acquire a frame from the swapchain, populate the command buffer and then present the
        /// to the screen using a render pass.
        ///
        /// This is an example:
        ///
        /// <code>
        /// void render() override
        /// {
        ///     VkCommandBuffer cmd = mpSwapchain->acquireNextImage();
        ///     mpPass->frameBegin(cmd, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        ///
        ///     // Populate command buffer using cmd here...
        ///
        ///     mpPass->frameEnd(cmd);
        ///     mpSwapchain->present(cmd, mpPass->getOutput());
        /// }
        /// </code>
        ///
        /// This function should be overridden.
        /// </summary>
        virtual void render() = 0;

        /// <summary>
        /// Draw application specific GUI of a Mandrill application.
        ///
        /// Use this function to draw GUI that is specific to the application. This function gives access to the current
        /// ImGUI context being used by Mandrill. Make sure to set the context before interacting with ImGUI:
        ///
        /// <code>
        /// ImGui::SetCurrentContext(pContext);
        /// </code>
        ///
        /// This function should be overridden.
        /// </summary>
        /// <param name="pContext">Current ImGUI context</param>
        virtual void appGUI(ImGuiContext* pContext) = 0;

        // virtual void onMouseEvent() = 0;

        /// <summary>
        /// Create a GUI for the application.
        ///
        /// This initializes all the resources for ImGUI.
        /// </summary>
        /// <param name="pDevice">Device to use for resource allocations</param>
        /// <param name="pPass">Pass to use for rendering the GUI</param>
        void createGUI(ptr<Device> pDevice, ptr<Pass> pPass);

        /// <summary>
        /// Destroy the GUI for the application.
        ///
        /// This will clean-up all the allocated resources of ImGUI.
        /// </summary>
        /// <param name="pDevice">Device that was used for resources allocations</param>
        void destroyGUI(ptr<Device> pDevice);

        /// <summary>
        /// Draw the base GUI of a Mandrill application.
        ///
        /// This function will draw a menu bar and allow for basic features like showing framerate and taking
        /// screenshots. The menu also allows for controlling different aspects of the rendering context and
        /// therefore needs access to the device, swapchain, render pass and pipelines.
        /// </summary>
        /// <param name="pDevice">Currently active device</param>
        /// <param name="pSwapchain">Swapchain that should be recreated on changes</param>
        /// <param name="pPipline">Pipeline that should be recreated</param>
        void baseGUI(ptr<Device> pDevice, ptr<Swapchain> pSwapchain, ptr<Pipeline> pPipeline);

        /// <summary>
        /// Draw the base GUI of a Mandrill application.
        ///
        /// This function will draw a menu bar and allow for basic features like showing framerate and taking
        /// screenshots. The menu also allows for controlling different aspects of the rendering context and
        /// therefore needs access to the device, swapchain, render pass and pipelines.
        /// </summary>
        /// <param name="pDevice">Currently active device</param>
        /// <param name="pSwapchain">Swapchain that should be recreated on changes</param>
        /// <param name="pPipelines">Pipelines that should be recreated</param>
        void baseGUI(ptr<Device> pDevice, ptr<Swapchain> pSwapchain, std::vector<ptr<Pipeline>> pPipelines);

        /// <summary>
        /// Render the GUI by writing the state to a command buffer.
        ///
        /// Call this while populating command buffer.
        /// </summary>
        /// <param name="cmd">Command buffer to write to</param>
        void renderGUI(VkCommandBuffer cmd) const;

        /// <summary>
        /// App keyboard callback function. This will handle keyboard commands associated with the base application
        /// menus. Read more in the GLFW documentation: https://www.glfw.org/docs/3.3/input_guide.html#input_key
        /// </summary>
        /// <param name="pWindow">The window that received the event</param>
        /// <param name="key">The keyboard key that was pressed or released</param>
        /// <param name="scancode">The system-specific scancode of the key</param>
        /// <param name="action">`GLFW_PRESS`, `GLFW_RELEASE` or `GLFW_REPEAT`. Future releases may add more
        /// actions</param> <param name="mods">Bit field describing which modifier keys were held down</param>
        /// <param name="pDevice">Currently active device</param>
        /// <param name="pSwapchain">Swapchain that should be recreated on changes</param>
        /// <param name="pPipeline">Pipeline that should be recreated</param>
        void baseKeyCallback(GLFWwindow* pWindow, int key, int scancode, int action, int mods, ptr<Device> pDevice,
                             ptr<Swapchain> pSwapchain, ptr<Pipeline> pPipeline);

        /// <summary>
        /// App keyboard callback function. This will handle keyboard commands associated with the base application
        /// menus. Read more in the GLFW documentation: https://www.glfw.org/docs/3.3/input_guide.html#input_key
        /// </summary>
        /// <param name="pWindow">The window that received the event</param>
        /// <param name="key">The keyboard key that was pressed or released</param>
        /// <param name="scancode">The system-specific scancode of the key</param>
        /// <param name="action">`GLFW_PRESS`, `GLFW_RELEASE` or `GLFW_REPEAT`. Future releases may add more
        /// actions</param> <param name="mods">Bit field describing which modifier keys were held down</param>
        /// <param name="pDevice">Currently active device</param>
        /// <param name="pSwapchain">Swapchain that should be recreated on changes</param>
        /// <param name="pPipelines">Pipelines that should be recreated</param>
        void baseKeyCallback(GLFWwindow* pWindow, int key, int scancode, int action, int mods, ptr<Device> pDevice,
                             ptr<Swapchain> pSwapchain, std::vector<ptr<Pipeline>> pPipelines);

        /// <summary>
        /// Virtual function for app to override. Just invoke <code>baseKeyCallback()</code> to get standard
        /// keybindings. See <code>baseKeyCallback()</code> for more details.
        /// </summary>
        /// <param name="pWindow">The window that received the event</param>
        /// <param name="key">The keyboard key that was pressed or released</param>
        /// <param name="scancode">The system-specific scancode of the key</param>
        /// <param name="action">`GLFW_PRESS`, `GLFW_RELEASE` or `GLFW_REPEAT`. Future releases may add more
        /// actions</param> <param name="mods">Bit field describing which modifier keys were held down</param>
        virtual void appKeyCallback(GLFWwindow* pWindow, int key, int scancode, int action, int mods) = 0;

        /// <summary>
        /// App cursor position callback function. This will handle track mouse movements and is required for
        /// getCursorDelta() to work. Read more in the GLFW documentation:
        /// https://www.glfw.org/docs/3.3/input_guide.html#input_mouse
        /// </summary>
        /// <param name="pWindow">The window that received the event</param>
        /// <param name="xPos">The new cursor x-coordinate, relative to the left edge of the content area.</param>
        /// <param name="yPos">The new cursor y-coordinate, relative to the top edge of the content area.</param>
        void baseCursorPosCallback(GLFWwindow* pWindow, double xPos, double yPos);

        /// <summary>
        /// Virtual function for app to override. Just invoke <code>baseCursorPosCallback()</code> to get standard
        /// keybindings. See <code>baseCursorPosCallback()</code> for more details.
        /// </summary>
        /// <param name="pWindow">The window that received the event</param>
        /// <param name="xPos">The new cursor x-coordinate, relative to the left edge of the content area.</param>
        /// <param name="yPos">The new cursor y-coordinate, relative to the top edge of the content area.</param>
        virtual void appCursorPosCallback(GLFWwindow* pWindow, double xPos, double yPos) = 0;

        /// <summary>
        /// App mouse button callback function. This will handle camera capturing the mouse cursor. Read more in the
        /// GLFW docutmentation: https://www.glfw.org/docs/3.3/input_guide.html#input_mouse
        /// </summary>
        /// <param name="pWindow">The window that received the event</param>
        /// <param name="button">The mouse button that was pressed or released.</param>
        /// <param name="action">One of `GLFW_PRESS` or `GLFW_RELEASE`. Future releases may add more actions.</param>
        /// <param name="mods">Bit field describing which modifier keys were held down.</param>
        /// <param name="pCamera">Current active camera to handle mouse capture for.</param>
        void baseMouseButtonCallback(GLFWwindow* pWindow, int button, int action, int mods, ptr<Camera> pCamera);

        /// <summary>
        /// Virtual function for app to override. Just invoke <code>baseMouseButtonCallback()</code> to get standard
        /// keybindings. See <code>baseMouseButtonCallback()</code> for more details.
        /// </summary>
        /// <param name="pWindow">The window that received the event</param>
        /// <param name="button">The mouse button that was pressed or released.</param>
        /// <param name="action">One of `GLFW_PRESS` or `GLFW_RELEASE`. Future releases may add more actions.</param>
        /// <param name="mods">Bit field describing which modifier keys were held down.</param>
        virtual void appMouseButtonCallback(GLFWwindow* pWindow, int button, int action, int mods) = 0;

        /// <summary>
        /// Get how the mouse cursor moved since last frame.
        /// </summary>
        /// <returns>A vec2 containing the change in x and y position since last frame</returns>
        glm::vec2 getCursorDelta() const
        {
            return {mCursorDeltaX, mCursorDeltaY};
        }

        /// <summary>
        /// Check if keyboard inputs are currently captured by the GUI.
        /// </summary>
        /// <returns>True of keyboard events are intended for GUI, false otherwise</returns>
        bool keyboardCapturedByGUI() const
        {
            return mKeyboardCapturedByGUI;
        }

        /// <summary>
        /// Check if mouse inputs are currently captured by the GUI.
        /// </summary>
        /// <returns>True if mouse movements are intended for GUI, false otherwise</returns>
        bool mouseCapturedByGUI() const
        {
            return mMouseCapturedByGUI;
        }

        // GLFW window
        GLFWwindow* mpWindow;

    protected:
        // Screen resolution
        uint32_t mWidth, mHeight;

        // Time since application started in seconds
        float mTime = 0.0f;

    private:
        /// <summary>
        /// Initialize GLFW.
        ///
        /// Create a GLFW context with window for the application.
        /// </summary>
        /// <param name="title">Title to be shown in window frame</param>
        /// <param name="width">Width of the window</param>
        /// <param name="height">Height of the window</param>
        void initGLFW(const std::string& title, uint32_t width, uint32_t height);

        /// <summary>
        /// Initialize ImGUI.
        ///
        /// Create a ImGUI context for the application and initiate the Vulkan backend for ImGUI.
        /// </summary>
        void initImGUI();

        /// <summary>
        /// Request a new frame from the GUI.
        /// </summary>
        /// <returns></returns>
        ImGuiContext* newFrameGUI();

        /// <summary>
        /// Save the next available swapchain image to disk.
        /// </summary>
        /// <param name="pDevice">Currently active device</param>
        /// <param name="pSwapchain">Swapchain to get screenshot from</param>
        void takeScreenshot(ptr<Device> pDevice, ptr<Swapchain> pSwapchain);

        /// <summary>
        /// Toggle between fullscreen and windowed mode.
        /// </summary>
        void toggleFullscreen();

        /// <summary>
        /// Reset to initial framebuffer size.
        /// </summary>
        void resetFramebufferSize();

        /// <summary>
        /// GLFW keyboard callback entry function. This will invoke the app's overriden callback function.
        /// </summary>
        /// <param name="window"></param>
        /// <param name="key"></param>
        /// <param name="scancode"></param>
        /// <param name="action"></param>
        /// <param name="mods"></param>
        static void keyCallbackEntry(GLFWwindow* pWindow, int key, int scancode, int action, int mods);

        /// <summary>
        /// GLFW cursor position callback entry function. This will invoke the app's overridden callback function.
        /// </summary>
        /// <param name="pWindow"></param>
        /// <param name="xPos"></param>
        /// <param name="yPos"></param>
        static void cursorPosCallbackEntry(GLFWwindow* pWindow, double xPos, double yPos);

        /// <summary>
        /// GLFW mouse button callback entry function. This will invoke the app's overridden callback funciton.
        /// </summary>
        /// <param name="pWindow"></param>
        /// <param name="button"></param>
        /// <param name="action"></param>
        /// <param name="mods"></param>
        static void mouseButtonCallbackEntry(GLFWwindow* pWindow, int button, int action, int mods);

        // Time since last frame in seconds
        float mDelta = 0.0f;

        // Smoothed delta for GUI presentation
        float mDeltaSmooth = 0.0f;

        // Give new delta sample 5% weight
        const float kSmoothingFactor = 0.05f;

        // Screen mode
        GLFWmonitor* mpMonitor;
        GLFWvidmode mFullscreenMode;
        bool mFullscreen = false;

        // Mouse movement
        float mCursorX = 0.0f, mCursorY = 0.0f;
        float mCursorDeltaX = 0.0f, mCursorDeltaY = 0.0f;
        float mCursorPrevX = 0.0f, mCursorPrevY = 0.0f;

        // GUI
        bool mCreatedGUI = false;
        bool mShowMainMenu = true;
        bool mShowFrameRate = false;
        bool mShowHelp = false;
        bool mShowAbout = false;

        // Flags for GUI input capture
        bool mKeyboardCapturedByGUI;
        bool mMouseCapturedByGUI;

        // Descriptor pool for ImGUI
        VkDescriptorPool mDescriptorPool;

        // Font for ImGUI
        ImFont* mFont;
    };
} // namespace Mandrill
