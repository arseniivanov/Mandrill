#pragma once

#include "Common.h"

#include "Device.h"
#include "Image.h"
#include "Sampler.h"

namespace Mandrill
{
    class Texture
    {
    public:
        enum class MANDRILL_API Type {
            Texture1D,
            Texture2D,
            Texture3D,
            CubeMap,
            Texture2DArray,
        };

        MANDRILL_NON_COPYABLE(Texture)

        /// <summary>
        /// Create a new texture from a file.
        /// </summary>
        /// <param name="pDevice">Device to use</param>
        /// <param name="type">Type of texture</param>
        /// <param name="format">Format to use</param>
        /// <param name="path">Path to texture file</param>
        /// <param name="mipmaps">Whether to use mipmaps or not</param>
        MANDRILL_API Texture(ptr<Device> pDevice, Type type, VkFormat format, const std::filesystem::path& path,
                             bool mipmaps = false);

        /// <summary>
        /// Create a new texture from a buffer.
        /// </summary>
        /// <param name="pDevice">Device to use</param>
        /// <param name="type">Type of texture</param>
        /// <param name="format">Format to use</param>
        /// <param name="pData">Pointer to texture data</param>
        /// <param name="width">Width of texture</param>
        /// <param name="height">Height of texture</param>
        /// <param name="depth">Depth of texture</param>
        /// <param name="channels">Number of channels in texture</param>
        /// <param name="mipmaps">Whether to use mipmaps or not</param>
        MANDRILL_API Texture(ptr<Device> pDevice, Type type, VkFormat format, const void* pData, uint32_t width,
                             uint32_t height, uint32_t depth, uint32_t channels, bool mipmaps = false);

        /// <summary>
        /// Destructor for texture.
        /// </summary>
        MANDRILL_API ~Texture();

        /// <summary>
        /// Set the sampler for the texture.
        /// </summary>
        /// <param name="pSampler">Sampler to use</param>
        /// <returns></returns>
        MANDRILL_API void setSampler(const ptr<Sampler> pSampler)
        {
            mImageInfo.sampler = pSampler->getSampler();
        }

        /// <summary>
        /// Get the sampler handle currently in use by the texture.
        /// </summary>
        /// <returns>Vulkan sampler handle</returns>
        MANDRILL_API VkSampler getSampler() const
        {
            return mImageInfo.sampler;
        }

        /// <summary>
        /// Get the image of the texture.
        /// </summary>
        /// <returns>Pointer to image</returns>
        MANDRILL_API ptr<Image> getImage() const
        {
            return mpImage;
        }

        /// <summary>
        /// Get the image view handle.
        /// </summary>
        /// <returns>Image view handle</returns>
        MANDRILL_API VkImageView getImageView() const
        {
            return mImageInfo.imageView;
        }

        /// <summary>
        /// Get the write descriptor set. Useful when using push descriptors.
        /// </summary>
        /// <param name="binding">Binding to place the write descriptor in</param>
        /// <returns>Write descriptor set</returns>
        MANDRILL_API VkWriteDescriptorSet getWriteDescriptor(uint32_t binding) const
        {
            VkWriteDescriptorSet descriptor = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstBinding = binding,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &mImageInfo,
            };

            return descriptor;
        }

    private:
        void create(Type type, VkFormat format, const void* pData, uint32_t width, uint32_t height, uint32_t depth,
                    uint32_t bytesPerPixel, bool mipmaps);

        void create_array(VkFormat format, const void* pData, uint32_t width, uint32_t height, uint32_t layerCount,
                          uint32_t bytesPerPixel, bool mipmaps);

        void generateMipmaps();

        ptr<Device> mpDevice;

        ptr<Image> mpImage;
        VkDescriptorImageInfo mImageInfo;
    };
} // namespace Mandrill
