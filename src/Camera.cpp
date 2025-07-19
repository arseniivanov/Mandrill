#include "Camera.h"

#include "Error.h"
#include "Helpers.h"

using namespace Mandrill;

namespace
{
    const float rotationSpeed = 1.0f;
    const float zoomSpeed = 10.0f;
    const float minZoom = 0.1f;
    const float maxZoom = 150.0f;
    const float mouseSpeed = 0.0008f;
} // namespace


Camera::Camera(ptr<Device> pDevice, GLFWwindow* pWindow, ptr<Swapchain> pSwapchain)
    : mpDevice(pDevice), mpWindow(pWindow), mpSwapchain(pSwapchain)
{
    mPosition = glm::vec3(1.0f, 1.0f, 1.0f);
    setTarget(glm::vec3(0.0f, 0.0f, 0.0f));
    mUp = glm::vec3(0.0f, 1.0f, 0.0f);

    updateAspectRatio();

    mFov = 30.0f;
    mNear = 0.01f;
    mFar = 1000.0f;
    mMoveSpeed = 90.0f;

    mpUniforms = std::make_shared<Buffer>(mpDevice, mpSwapchain->getFramesInFlightCount() * sizeof(CameraMatrices),
                                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkDescriptorSetLayoutBinding binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_RAYGEN_BIT_KHR,
    };

    VkDescriptorSetLayoutCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &binding,
    };

    Check::Vk(vkCreateDescriptorSetLayout(mpDevice->getDevice(), &ci, nullptr, &mLayout));

    std::vector<DescriptorDesc> desc;
    desc.emplace_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, mpUniforms);
    desc.back().range = sizeof(CameraMatrices);
    mpDescriptor = std::make_shared<Descriptor>(mpDevice, desc, mLayout);

    // The buffer info allows for using this as push descriptor too
    mBufferInfo.buffer = mpUniforms->getBuffer();
    mBufferInfo.offset = 0;
    mBufferInfo.range = 4 * sizeof(glm::mat4);
}

Camera::~Camera()
{
    Check::Vk(vkDeviceWaitIdle(mpDevice->getDevice()));
    vkDestroyDescriptorSetLayout(mpDevice->getDevice(), mLayout, nullptr);
}

void Camera::updateAspectRatio()
{
    int width, height;
    glfwGetFramebufferSize(mpWindow, &width, &height);
    mAspect = static_cast<float>(width) / static_cast<float>(height);
}

void Camera::update(float delta, glm::vec2 cursorDelta)
{
    // Right vector, used for pitch rotations and strafing
    glm::vec3 right = glm::normalize(glm::cross(mDirection, mUp));

    // Speed control with L-Shift and L-Ctrl
    float speedFactor = 1.0f;
    if (glfwGetKey(mpWindow, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        speedFactor = 2.5f;
    }
    if (glfwGetKey(mpWindow, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
        speedFactor = 0.1f;
    }

    // Move and strafe with WASD
    if (glfwGetKey(mpWindow, GLFW_KEY_W) == GLFW_PRESS) {
        mPosition += speedFactor * mMoveSpeed * delta * mDirection;
    }

    if (glfwGetKey(mpWindow, GLFW_KEY_S) == GLFW_PRESS) {
        mPosition -= speedFactor * mMoveSpeed * delta * mDirection;
    }

    if (glfwGetKey(mpWindow, GLFW_KEY_A) == GLFW_PRESS) {
        mPosition -= speedFactor * mMoveSpeed * delta * right;
    }

    if (glfwGetKey(mpWindow, GLFW_KEY_D) == GLFW_PRESS) {
        mPosition += speedFactor * mMoveSpeed * delta * right;
    }

    // Up and down with Q and E
    if (glfwGetKey(mpWindow, GLFW_KEY_E) == GLFW_PRESS) {
        mPosition += speedFactor * mMoveSpeed * delta * mUp;
    }

    if (glfwGetKey(mpWindow, GLFW_KEY_Q) == GLFW_PRESS) {
        mPosition -= speedFactor * mMoveSpeed * delta * mUp;
    }

    // Direction with arrows
    glm::vec3 newDir(1.0f);
    const float angle = speedFactor * rotationSpeed * delta;
    if (glfwGetKey(mpWindow, GLFW_KEY_UP) == GLFW_PRESS) {
        newDir = glm::rotate(mDirection, angle, right);
    }

    if (glfwGetKey(mpWindow, GLFW_KEY_DOWN) == GLFW_PRESS) {
        newDir = glm::rotate(mDirection, -angle, right);
    }

    if (glfwGetKey(mpWindow, GLFW_KEY_LEFT) == GLFW_PRESS) {
        newDir = glm::rotate(mDirection, angle, mUp);
    }

    if (glfwGetKey(mpWindow, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        newDir = glm::rotate(mDirection, -angle, mUp);
    }

    // Zoom with PERIOD and COMMA
    if (glfwGetKey(mpWindow, GLFW_KEY_PERIOD) == GLFW_PRESS) {
        mFov -= speedFactor * zoomSpeed * delta;
        mFov = std::clamp(mFov, minZoom, maxZoom);
    }

    if (glfwGetKey(mpWindow, GLFW_KEY_COMMA) == GLFW_PRESS) {
        mFov += speedFactor * zoomSpeed * delta;
        mFov = std::clamp(mFov, minZoom, maxZoom);
    }

    // Mouse control
    if ((cursorDelta.x != 0.0f || cursorDelta.y != 0.0f) &&
        (glfwGetMouseButton(mpWindow, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS || mMouseCaptured)) {
        mDirection = glm::rotate(mDirection, -cursorDelta.x * mouseSpeed, mUp);
        newDir = glm::normalize(glm::rotate(mDirection, -cursorDelta.y * mouseSpeed, right));
    }

    // Clamp vertical direction change
    if (fabsf(newDir.y) < 0.99f) {
        mDirection = newDir;
    }

    // Update uniform buffer
    CameraMatrices* matrices = static_cast<CameraMatrices*>(mpUniforms->getHostMap()) + mpSwapchain->getInFlightIndex();
    matrices->view = glm::lookAt(mPosition, mPosition + mDirection, mUp);
    matrices->view_inv = glm::inverse(matrices->view);
    matrices->proj = glm::perspective(glm::radians(mFov), mAspect, mNear, mFar);
    matrices->proj[1][1] *= -1.0f; // GLM and Vulkan are not using the same coordinate system
    matrices->proj_inv = glm::inverse(matrices->proj);
}
