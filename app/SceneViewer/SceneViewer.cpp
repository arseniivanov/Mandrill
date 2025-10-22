#include "Mandrill.h"

using namespace Mandrill;

class SceneViewer : public App
{
public:
    enum PipelineType {
        PIPELINE_FILL,
        PIPELINE_LINE,
    };

    struct PushConstants {
        int renderMode;
        int discardOnZeroAlpha;
        alignas(16) glm::vec3 lineColor;
        float lod;
    };

    void loadScene()
    {
        // Create a new scene
        mpScene = std::make_shared<Scene>(mpDevice, mpSwapchain);

        // Load meshes from the scene path
        auto meshIndices = mpScene->addMeshFromFile(mScenePath);


        // Load neural model if path is set
        if (!mNeuralModelPath.empty()) {
            mpScene->loadNeuralModel(mNeuralModelPath);
        } else {
            // Try to autodetect: <obj_filename_stem>.safetensors
            std::filesystem::path autoNeuralPath = mScenePath;
            autoNeuralPath.replace_extension(".safetensors");
            if (std::filesystem::exists(autoNeuralPath)) {
                Log::Info("Auto-detected neural model: {}", autoNeuralPath.string());
                mNeuralModelPath = autoNeuralPath;
                mpScene->loadNeuralModel(mNeuralModelPath);
            } else {
                Log::Warning("Neural model path not set and auto-detection failed for: {}", autoNeuralPath.string());
            }
        }
        mpScene->setNeuralPipelines(mpVqPipeline, mpNonVqPipeline);

        // Add a node to the scene
        std::shared_ptr<Node> pNode = mpScene->addNode();
        pNode->setPipeline(mPipelines[PIPELINE_FILL]);
        pNode->setNeuralPipelines(mpVqPipeline, mpNonVqPipeline);

        // Add all the meshes to the node
        for (auto meshIndex : meshIndices) {
            pNode->addMesh(meshIndex);
        }

        // Indicate which sampler should be used to handle textures
        mpScene->setSampler(mpSampler);

        // Calculate and allocate buffers
        mpScene->compile();

        // Sync to GPU
        mpScene->syncToDevice();
    }

    SceneViewer() : App("SceneViewer", 1920, 1080)
    {
        // Create a Vulkan instance and device
        mpDevice = std::make_shared<Device>(mpWindow);

        // Create a swapchain with 2 frames in flight
        mpSwapchain = std::make_shared<Swapchain>(mpDevice, 2);

        // Create a scene so we can access the layout, the actual scene will be loaded later
        auto pTempScene = std::make_shared<Scene>(mpDevice, mpSwapchain);


        auto pStandardLayout = pTempScene->getLayout(false, false); // for non-neural
        auto pVqLayout = pTempScene->getLayout(true, true);         // for neural VQ
        auto pNonVqLayout = pTempScene->getLayout(true, false);     // for neural non-VQ
                                                                    //
        VkPushConstantRange pushConstantRange = {
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(PushConstants),
        };

        pStandardLayout->addPushConstantRange(pushConstantRange);
        pVqLayout->addPushConstantRange(pushConstantRange);
        pNonVqLayout->addPushConstantRange(pushConstantRange);

        mpScene = nullptr;
        // Create a pass with 1 color attachment, depth attachment and multisampling
        mpPass = std::make_shared<Pass>(mpDevice, mpSwapchain->getExtent(), mpSwapchain->getImageFormat(), 1, true,
                                        mpDevice->getSampleCount());
        std::vector<ShaderDesc> standardShaderDesc;
        standardShaderDesc.emplace_back("SceneViewer/VertexShader.vert", "main", VK_SHADER_STAGE_VERTEX_BIT);
        standardShaderDesc.emplace_back("SceneViewer/FragmentShader.frag", "main", VK_SHADER_STAGE_FRAGMENT_BIT);
        std::shared_ptr<Shader> pStandardShader = std::make_shared<Shader>(mpDevice, standardShaderDesc);

        // VQ Neural Shader
        std::vector<ShaderDesc> vqShaderDesc;
        vqShaderDesc.emplace_back("SceneViewer/VertexShader.vert", "main", VK_SHADER_STAGE_VERTEX_BIT);
        vqShaderDesc.emplace_back("SceneViewer/FragmentShader_VQ.frag", "main",
                                  VK_SHADER_STAGE_FRAGMENT_BIT); // Rename your existing frag shader
        std::shared_ptr<Shader> pVqShader = std::make_shared<Shader>(mpDevice, vqShaderDesc);

        // Raw Packed Neural Shader
        std::vector<ShaderDesc> rawShaderDesc;
        rawShaderDesc.emplace_back("SceneViewer/VertexShader.vert", "main", VK_SHADER_STAGE_VERTEX_BIT);
        rawShaderDesc.emplace_back("SceneViewer/FragmentShader_Raw.frag", "main",
                                   VK_SHADER_STAGE_FRAGMENT_BIT); // The NEW frag shader
        std::shared_ptr<Shader> pRawShader = std::make_shared<Shader>(mpDevice, rawShaderDesc);

        mPipelines.clear(); // Ensure it's empty before we start

        // Create pipeline for standard filled rendering
        mPipelines.emplace_back(std::make_shared<Pipeline>(mpDevice, mpPass, pStandardLayout, pStandardShader));

        // Create a pipeline for standard line rendering
        PipelineDesc linePipelineDesc;
        linePipelineDesc.polygonMode = VK_POLYGON_MODE_LINE;
        mPipelines.emplace_back(
            std::make_shared<Pipeline>(mpDevice, mpPass, pStandardLayout, pStandardShader, linePipelineDesc));

        // Create the two neural pipelines
        PipelineDesc neuralPipelineDesc; // Use default fill settings
        mpVqPipeline = std::make_shared<Pipeline>(mpDevice, mpPass, pVqLayout, pVqShader, neuralPipelineDesc);
        mpNonVqPipeline = std::make_shared<Pipeline>(mpDevice, mpPass, pNonVqLayout, pRawShader, neuralPipelineDesc);

        // Setup camera
        mpCamera = std::make_shared<Camera>(mpDevice, mpWindow, mpSwapchain);
        mpCamera->setPosition(glm::vec3(-135.0f, 430.0f, 100.0f));
        mpCamera->setTarget(glm::vec3(0.0f, 0.0f, 0.0f));
        mpCamera->setDirection(glm::vec3(0.0f, 0.0f, 1.0f));
        mpCamera->setFov(60.0f);

        // Create a sampler that will be used to render materials
        mpSampler = std::make_shared<Sampler>(mpDevice);

        mScenePath = "../res/sponza/sponza.obj";
        loadScene();

        // Initialize GUI
        App::createGUI(mpDevice, mpPass);
    }

    ~SceneViewer()
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
        if (!mpScene) {
            // If no scene is loaded, just clear the screen and draw the GUI
            VkCommandBuffer cmd = mpSwapchain->acquireNextImage();
            mpPass->begin(cmd, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
            App::renderGUI(cmd); // Still draw the GUI so we can load a scene
            mpPass->end(cmd);
            mpSwapchain->present(cmd, mpPass->getOutput());
            return; // Don't do any scene rendering
        }
        // Check if camera matrix and attachments need to be updated
        if (mpSwapchain->recreated()) {
            mpCamera->updateAspectRatio();
            mpPass->update(mpSwapchain->getExtent());
        }

        // Acquire frame from swapchain and prepare rasterizer
        VkCommandBuffer cmd = mpSwapchain->acquireNextImage();
        mpPass->begin(cmd, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

        ptr<Pipeline> pActiveFillPipeline = mPipelines[PIPELINE_FILL];
        if (mRenderMode == 9 && mpScene->hasNeuralModel()) { // 9 is NTC render mode
            pActiveFillPipeline = mpScene->getActiveNeuralPipeline();
        }
        // Push constants for the fill/neural render.
        PushConstants pushConstants = {
            .renderMode = mRenderMode,
            .discardOnZeroAlpha = mDiscardOnZeroAlpha,
            .lod = 0.0,
        };
        // Use the layout from the pipeline we are about to use.
        vkCmdPushConstants(cmd, pActiveFillPipeline->getLayout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pushConstants,
                           &pushConstants);

        // Render scene with the selected pipeline.
        mpScene->render(cmd, mpCamera);

        if (mRenderMode == 9 && mpScene->hasNeuralModel()) {
            mShaderTimings = mpScene->getTimingResults();

            // Only update the average if we got a valid new measurement from the shader
            if (mShaderTimings.total_ns > 0) {
                // A smaller factor means more smoothing. 0.05 is a good starting point.
                const double smoothingFactor = 0.05;

                // The formula for Exponential Moving Average:
                // NewAvg = (NewValue * alpha) + (OldAvg * (1 - alpha))
                // We also convert from nanoseconds to microseconds here by dividing by 1000.0
                mSmoothedGrid0_us =
                    (mShaderTimings.grid0_ns / 1000.0) * smoothingFactor + mSmoothedGrid0_us * (1.0 - smoothingFactor);
                mSmoothedGrid1_us =
                    (mShaderTimings.grid1_ns / 1000.0) * smoothingFactor + mSmoothedGrid1_us * (1.0 - smoothingFactor);
                mSmoothedMlp_us =
                    (mShaderTimings.mlp_ns / 1000.0) * smoothingFactor + mSmoothedMlp_us * (1.0 - smoothingFactor);
                mSmoothedTotal_us =
                    (mShaderTimings.total_ns / 1000.0) * smoothingFactor + mSmoothedTotal_us * (1.0 - smoothingFactor);
            }
        }

        // Render lines
        if (mDrawPolygonLines) {
            // Switch to line rendering
            for (auto& node : mpScene->getNodes()) {
                node.setPipeline(mPipelines[PIPELINE_LINE]);
            }

            PushConstants pushConstants = {
                .renderMode = 10,
                .discardOnZeroAlpha = mDiscardOnZeroAlpha,
                .lineColor = mLineColor,
                .lod = 0.0, // TODO FIX Adjust
            };
            vkCmdPushConstants(cmd, mPipelines[PIPELINE_LINE]->getLayout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof pushConstants, &pushConstants);

            mPipelines[PIPELINE_LINE]->setLineWidth(mLineWidth);

            mpScene->render(cmd, mpCamera);

            // Reset pipeline
            for (auto& node : mpScene->getNodes()) {
                node.setPipeline(mPipelines[PIPELINE_FILL]);
            }
        }

        // Draw GUI
        App::renderGUI(cmd);

        // Submit command buffer to rasterizer and present swapchain frame
        mpPass->end(cmd);
        mpSwapchain->present(cmd, mpPass->getOutput());
    }

    void appGUI(ImGuiContext* pContext)
    {
        ImGui::SetCurrentContext(pContext);

        App::baseGUI(mpDevice, mpSwapchain, mPipelines);

        if (ImGui::Begin("Scene Viewer")) {
            if (ImGui::Button("Load")) {
                mScenePath = OpenFile(mpWindow, "All\0*.*\0Wavefront Object (*.obj)\0*.OBJ\0");
                if (!mScenePath.empty()) {
                    loadScene();
                }
            }

            ImGui::Text("Scene: %s", mScenePath.string().c_str());

            // UI for Neural Model Path
            char neuralPathBuf[1024];
            strncpy(neuralPathBuf, mNeuralModelPath.string().c_str(), sizeof(neuralPathBuf) - 1);
            neuralPathBuf[sizeof(neuralPathBuf) - 1] = 0; // Null terminate

            if (ImGui::InputText("Neural Model (.safetensors)", neuralPathBuf, sizeof(neuralPathBuf),
                                 ImGuiInputTextFlags_EnterReturnsTrue)) {
                mNeuralModelPath = neuralPathBuf;
                if (!mScenePath.empty() && !mNeuralModelPath.empty()) { // Only reload if both paths are set
                    loadScene();                                        // Reload scene with new neural model
                } else if (mNeuralModelPath.empty() && mpScene && mpScene->hasNeuralModel()) {
                    // If path cleared, effectively remove neural model
                    loadScene();
                }
            }
            if (ImGui::Button("Browse Neural Model")) {
                std::filesystem::path tempNeuralPath =
                    OpenFile(mpWindow, "Safetensors (*.safetensors)\0*.safetensors\0All\0*.*\0");
                if (!tempNeuralPath.empty()) {
                    mNeuralModelPath = tempNeuralPath;
                    if (!mScenePath.empty()) { // Only reload if OBJ is also loaded
                        loadScene();
                    }
                }
            }

            const char* renderModes[] = {
                "Diffuse",  "Specular",  "Ambient",
                "Emission", "Shininess", "Index of refraction",
                "Opacity",  "Normal",    "Texture coordinates",
                "NTC",
            };
            ImGui::Combo("Render mode", &mRenderMode, renderModes, IM_ARRAYSIZE(renderModes));
            const char* frontFace[] = {"Counter clockwise", "Clockwise"};
            if (ImGui::Combo("Front face", &mFrontFace, frontFace, IM_ARRAYSIZE(frontFace))) {
                mPipelines[PIPELINE_FILL]->setFrontFace(mFrontFace == 0 ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                                        : VK_FRONT_FACE_CLOCKWISE);
            }
            const char* cullModes[] = {"None", "Front face", "Back face"};
            if (ImGui::Combo("Cull mode", &mCullMode, cullModes, IM_ARRAYSIZE(cullModes))) {
                mPipelines[PIPELINE_FILL]->setCullMode(static_cast<VkCullModeFlagBits>(mCullMode));
            };
            ImGui::Checkbox("Draw polyon lines", &mDrawPolygonLines);
            if (mDrawPolygonLines) {
                ImGui::ColorEdit3("Line color", &mLineColor.x);
                ImGui::SliderFloat("Line width", &mLineWidth, 1.0f, 10.0f);
            }

            bool newSampler = false;
            const char* magFilters[] = {"Linear", "Nearest"};
            if (ImGui::Combo("Mag filter", &mMagFilter, magFilters, IM_ARRAYSIZE(magFilters))) {
                newSampler = true;
            }
            const char* minFilters[] = {"Linear", "Nearest"};
            if (ImGui::Combo("Min filter", &mMinFilter, minFilters, IM_ARRAYSIZE(minFilters))) {
                newSampler = true;
            }
            const char* mipModes[] = {"Linear", "Nearest"};
            if (ImGui::Combo("Mip mode", &mMipMode, mipModes, IM_ARRAYSIZE(mipModes))) {
                newSampler = true;
            }

            if (newSampler) {
                mpSampler = std::make_shared<Sampler>(mpDevice, mMagFilter ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
                                                      mMinFilter ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
                                                      mMipMode ? VK_SAMPLER_MIPMAP_MODE_NEAREST
                                                               : VK_SAMPLER_MIPMAP_MODE_LINEAR);
                mpScene->setSampler(mpSampler);
                mpScene->compile();
                mpScene->syncToDevice();
            }

            ImGui::Checkbox("Discard pixel if diffuse alpha channel is 0", &mDiscardOnZeroAlpha);

            if (ImGui::SliderFloat("Camera move speed", &mCameraMoveSpeed, 0.1f, 100.0f)) {
                mpCamera->setMoveSpeed(mCameraMoveSpeed);
            }
            if (mRenderMode == 9 && mpScene && mpScene->hasNeuralModel()) {
                ImGui::Separator();
                ImGui::Text("NTC Shader Profiling (smoothed, microseconds)"); // Updated title

                // Check against a small floating point threshold instead of just zero
                if (mSmoothedTotal_us > 0.01) {
                    // Display the smoothed, microsecond values with 2 decimal places
                    ImGui::Text("Grid 0 Sampling: %.2f us", mSmoothedGrid0_us);
                    ImGui::Text("Grid 1 Sampling: %.2f us", mSmoothedGrid1_us);
                    ImGui::Text("MLP Evaluation:  %.2f us", mSmoothedMlp_us);
                    ImGui::Text("Total Duration:  %.2f us", mSmoothedTotal_us);
                } else {
                    ImGui::Text("Waiting for data from profiling pixel...");
                }
            }
        }

        ImGui::End();
    }

    void appKeyCallback(GLFWwindow* pWindow, int key, int scancode, int action, int mods)
    {
        App::baseKeyCallback(pWindow, key, scancode, action, mods, mpDevice, mpSwapchain, mPipelines);
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
    std::shared_ptr<Pass> mpPass;
    std::vector<std::shared_ptr<Pipeline>> mPipelines;
    ptr<Pipeline> mpVqPipeline;
    ptr<Pipeline> mpNonVqPipeline;

    std::shared_ptr<Camera> mpCamera;
    float mCameraMoveSpeed = 1.0f;

    std::shared_ptr<Sampler> mpSampler;

    std::shared_ptr<Scene> mpScene;
    std::filesystem::path mScenePath;
    std::filesystem::path mNeuralModelPath;

    int mRenderMode = 0;
    bool mDiscardOnZeroAlpha = false;
    bool mDrawPolygonLines = false;
    glm::vec3 mLineColor = glm::vec3(0.0f, 1.0f, 0.0f);
    float mLineWidth = 1.0f;
    int mFrontFace = 0;
    int mCullMode = 0;
    int mMagFilter = 0;
    int mMinFilter = 0;
    int mMipMode = 0;

    Scene::ShaderTimingResults mShaderTimings;
    double mSmoothedGrid0_us = 0.0;
    double mSmoothedGrid1_us = 0.0;
    double mSmoothedMlp_us = 0.0;
    double mSmoothedTotal_us = 0.0;
};

int main()
{
    SceneViewer app = SceneViewer();
    app.run();
    return 0;
}
