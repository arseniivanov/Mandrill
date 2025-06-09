#pragma once

#include "Common.h"

#include "Buffer.h"
#include "Device.h"
#include "Error.h"
#include "Image.h"
#include "Log.h"

namespace Mandrill
{
    class MANDRILL_API Helpers
    {
    public:
        /// <summary>
        /// Create a new one-time use command buffer. This call should be paired with a corresponding call to cmdEnd().
        /// </summary>
        /// <param name="pDevice">Device to use</param>
        /// <returns>Command buffer</returns>
        inline static VkCommandBuffer cmdBegin(ptr<Device> pDevice)
        {
            VkCommandBufferAllocateInfo ai = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = pDevice->getCommandPool(),
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1,
            };

            VkCommandBuffer cmd;
            Check::Vk(vkAllocateCommandBuffers(pDevice->getDevice(), &ai, &cmd));

            VkCommandBufferBeginInfo bi = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };

            Check::Vk(vkBeginCommandBuffer(cmd, &bi));

            return cmd;
        }

        /// <summary>
        /// End a one-time use command buffer that was initialized with a call to cmdBegin().
        /// </summary>
        /// <param name="pDevice">Device to use</param>
        /// <param name="cmd">Command buffer to end</param>
        inline static void cmdEnd(ptr<Device> pDevice, VkCommandBuffer cmd)
        {
            Check::Vk(vkEndCommandBuffer(cmd));

            VkSubmitInfo si = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1,
                .pCommandBuffers = &cmd,
            };

            Check::Vk(vkQueueSubmit(pDevice->getQueue(), 1, &si, nullptr));
            Check::Vk(vkQueueWaitIdle(pDevice->getQueue()));

            vkFreeCommandBuffers(pDevice->getDevice(), pDevice->getCommandPool(), 1, &cmd);
        }

        /// <summary>
        /// Find a suitable memory type given the memory properties.
        /// </summary>
        /// <param name="pDevice">Device to use</param>
        /// <param name="typeFilter">Type filter from memory requirements</param>
        /// <param name="properties">Requested memory properties</param>
        /// <returns>Memory type</returns>
        inline static uint32_t findMemoryType(ptr<Device> pDevice, uint32_t typeFilter,
                                              VkMemoryPropertyFlags properties)
        {
            for (uint32_t i = 0; i < pDevice->getProperties().memory.memoryTypeCount; i++) {
                if ((typeFilter & (1 << i)) &&
                    (pDevice->getProperties().memory.memoryTypes[i].propertyFlags & properties) == properties) {
                    return i;
                }
            }

            Log::Error("Failed to find suitable memory type");
            return UINT32_MAX;
        }

        /// <summary>
        /// Find supported format for an attachment.
        /// </summary>
        /// <param name="pDevice">Device to use</param>
        /// <param name="candidates">Formats that we want to use</param>
        /// <param name="tiling">Tiling to use</param>
        /// <param name="features">Features to use</param>
        /// <returns>A supported format or VK_FORMAT_UNDEFINED</returns>
        inline static VkFormat findSupportedFormat(ptr<Device> pDevice, std::vector<VkFormat> candidates,
                                                   VkImageTiling tiling, VkFormatFeatureFlags features)
        {
            for (auto& c : candidates) {
                VkFormatProperties properties;
                vkGetPhysicalDeviceFormatProperties(pDevice->getPhysicalDevice(), c, &properties);

                if (tiling == VK_IMAGE_TILING_LINEAR && (properties.linearTilingFeatures & features) == features) {
                    return c;
                } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
                           (properties.optimalTilingFeatures & features) == features) {
                    return c;
                }
            }
            return VK_FORMAT_UNDEFINED;
        }

        /// <summary>
        /// Find a supported depth format for a depth attachment.
        /// </summary>
        /// <param name="pDevice">Device to use</param>
        /// <returns>A supported depth format or VK_FORMAT_UNDEFINED</returns>
        inline static VkFormat findDepthFormat(ptr<Device> pDevice)
        {
            std::vector<VkFormat> candidates;
            candidates.push_back(VK_FORMAT_D32_SFLOAT);
            candidates.push_back(VK_FORMAT_D32_SFLOAT_S8_UINT);
            candidates.push_back(VK_FORMAT_D24_UNORM_S8_UINT);

            return findSupportedFormat(pDevice, candidates, VK_IMAGE_TILING_OPTIMAL,
                                       VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
        }

        /// <summary>
        /// Create an image barrier with layout transition.
        /// </summary>
        /// <param name="cmd">Command buffer to use</param>
        /// <param name="srcStage">Stage the barrier has to wait for</param>
        /// <param name="srcAccess">Type of access the barrier has to wait for</param>
        /// <param name="dstStage">Stage when the barrier has to be finished</param>
        /// <param name="dstAccess">Type of access the barrer must be ready for when finished</param>
        /// <param name="oldLayout">Old image layout</param>
        /// <param name="newLayout">New image layout</param>
        /// <param name="image">Image to transition</param>
        /// <param name="pSubresourceRange">Subresource range to use, or nullptr</param>
        inline static void imageBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                                        VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                                        VkImageLayout oldLayout, VkImageLayout newLayout, VkImage image,
                                        VkImageSubresourceRange* pSubresourceRange = nullptr)
        {
            VkImageSubresourceRange defaultSubresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            };

            VkImageMemoryBarrier2 barrier = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask = srcStage,
                .srcAccessMask = srcAccess,
                .dstStageMask = dstStage,
                .dstAccessMask = dstAccess,
                .oldLayout = oldLayout,
                .newLayout = newLayout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange = pSubresourceRange ? *pSubresourceRange : defaultSubresourceRange,
            };

            VkDependencyInfo dependencyInfo = {
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &barrier,
            };

            vkCmdPipelineBarrier2(cmd, &dependencyInfo);
        }

        /// <summary>
        /// Transition an image layout with some standard synchronizations.
        /// </summary>
        /// <param name="pDevice">Device to use</param>
        /// <param name="image">Image to transition</param>
        /// <param name="format">Image format</param>
        /// <param name="oldLayout">Old image layout</param>
        /// <param name="newLayout">New image layout</param>
        /// <param name="mipLevels">Miplevel to use</param>
        /// <param name="cmd">Command buffer to use, otherwise a new one is created</param>
        inline static void transitionImageLayout(ptr<Device> pDevice, VkImage image, VkFormat format,
                                                 VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels,
                                                 VkCommandBuffer cmd = VK_NULL_HANDLE)
        {
            bool endCmd = false;
            if (!cmd) {
                cmd = cmdBegin(pDevice);
                endCmd = true;
            }

            VkImageMemoryBarrier barrier = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = 0, // Set below
                .dstAccessMask = 0, // Set below
                .oldLayout = oldLayout,
                .newLayout = newLayout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange =
                    {
                        .aspectMask = 0, // Set below
                        .baseMipLevel = 0,
                        .levelCount = mipLevels,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
            };

            VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_NONE_KHR;
            VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_NONE_KHR;

            if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

                // Check if format has stencil component
                if (format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT) {
                    barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
                }
            } else {
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            }

            if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

                srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                       newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

                srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
                       newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

                srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
                srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
                       newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask =
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

                srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR &&
                       newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
                // Before screenshot
                srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL &&
                       newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
                // After screenshot
                srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                dstStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                barrier.dstAccessMask = VK_ACCESS_NONE;
            } else {
                Log::Error("Unsupported layout transition");
            }

            vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

            if (endCmd) {
                cmdEnd(pDevice, cmd);
            }
        }


        inline static void transitionImageLayout(ptr<Device> pDevice, VkImage image, VkFormat format,
                                                 VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels,
                                                 uint32_t arrayLayers) // New parameter
        {
            VkCommandBuffer cmd = cmdBegin(pDevice);

            VkImageMemoryBarrier barrier = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = 0, // Set below
                .dstAccessMask = 0, // Set below
                .oldLayout = oldLayout,
                .newLayout = newLayout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange =
                    {
                        .aspectMask = 0, // Set below
                        .baseMipLevel = 0,
                        .levelCount = mipLevels,
                        .baseArrayLayer = 0,
                        .layerCount = arrayLayers, // Use the new parameter
                    },
            };

            VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_NONE_KHR;
            VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_NONE_KHR;

            if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

                // Check if format has stencil component
                if (format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT) {
                    barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
                }
            } else {
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            }

            if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

                srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                       newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

                srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
                       newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

                srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
                srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
                       newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask =
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

                srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR &&
                       newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
                // Before screenshot
                srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL &&
                       newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
                // After screenshot
                srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                dstStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                barrier.dstAccessMask = VK_ACCESS_NONE;
            } else {
                Log::Error("Unsupported layout transition");
            }

            vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

            cmdEnd(pDevice, cmd);
        }


        /// <summary>
        /// Copy a buffer to an image.
        /// </summary>
        /// <param name="pDevice">Device to use</param>
        /// <param name="buffer">Buffer to use</param>
        /// <param name="image">Image to use</param>
        /// <param name="width">Width of image</param>
        /// <param name="height">Height of image</param>
        /// <param name="depth">Depth of image</param>
        inline static void copyBufferToImage(ptr<Device> pDevice, VkBuffer buffer, VkImage image, uint32_t width,
                                             uint32_t height, uint32_t depth)
        {
            VkCommandBuffer cmd = cmdBegin(pDevice);

            VkBufferImageCopy region = {
                .bufferOffset = 0,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                .imageOffset = {0, 0},
                .imageExtent = {width, height, depth},
            };

            vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            cmdEnd(pDevice, cmd);
        }

        /// <summary>
        /// Copy an image to a buffer.
        /// </summary>
        /// <param name="pDevice">Device to use</param>
        /// <param name="image">Image to use</param>
        /// <param name="buffer">Buffer to use</param>
        /// <param name="width">Width of image</param>
        /// <param name="height">Height of image</param>
        /// <param name="depth">Depth of image</param>
        inline static void copyImageToBuffer(ptr<Device> pDevice, VkImage image, VkBuffer buffer, uint32_t width,
                                             uint32_t height, uint32_t depth)
        {
            VkCommandBuffer cmd = cmdBegin(pDevice);

            VkBufferImageCopy region = {
                .bufferOffset = 0,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                .imageOffset = {0, 0},
                .imageExtent = {width, height, depth},
            };

            vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);

            cmdEnd(pDevice, cmd);
        }

        inline static void copyBufferToImage(ptr<Device> pDevice, VkBuffer buffer, VkImage image, uint32_t width,
                                             uint32_t height, uint32_t depth, uint32_t arrayLayers) // New parameter
        {
            VkCommandBuffer cmd = cmdBegin(pDevice);

            VkBufferImageCopy region = {
                .bufferOffset = 0,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = arrayLayers, // Use the new parameter
                    },
                .imageOffset = {0, 0, 0}, // Use 3D offset
                .imageExtent = {width, height, depth},
            };

            vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            cmdEnd(pDevice, cmd);
        }

        /// <summary>
        /// Aligne a value to a given alignment.
        /// </summary>
        /// <param name="value">Value to align</param>
        /// <param name="alignment">Alignment to use</param>
        /// <returns>Aligned value</returns>
        inline static VkDeviceSize alignTo(VkDeviceSize value, VkDeviceSize alignment)
        {
            return (value + alignment - 1) & ~(alignment - 1);
        }

        /// <summary>
        /// Return a random value from the interval [0.0, 1.0)
        /// </summary>
        /// <param name="reset">If true, the random seed is reset</param>
        /// <returns>A random value</returns>
        inline float random(bool reset = false)
        {
            static std::random_device dev;
            static std::mt19937 rng(dev());
            static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
            if (reset) {
                rng.seed();
            }
            return dis(rng);
        }
    };
} // namespace Mandrill
