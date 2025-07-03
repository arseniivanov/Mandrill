#include "Buffer.h"

#include "Error.h"
#include "Helpers.h"

using namespace Mandrill;

Buffer::Buffer(ptr<Device> pDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties)
    : mpDevice(pDevice), mSize(size), mUsage(usage), mProperties(properties), mpHostMap(nullptr)
{
    VkBufferCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = mSize,
        .usage = mUsage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    Check::Vk(vkCreateBuffer(mpDevice->getDevice(), &ci, nullptr, &mBuffer));

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(mpDevice->getDevice(), mBuffer, &memReqs);

    VkMemoryAllocateFlagsInfo allocFlagInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };

    VkMemoryAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memReqs.size,
        .memoryTypeIndex = Helpers::findMemoryType(mpDevice, memReqs.memoryTypeBits, mProperties),
    };

    if (mUsage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        allocInfo.pNext = &allocFlagInfo;
    }

    Check::Vk(vkAllocateMemory(mpDevice->getDevice(), &allocInfo, nullptr, &mMemory));

    Check::Vk(vkBindBufferMemory(mpDevice->getDevice(), mBuffer, mMemory, 0));

    // Map memory if it is host coherent
    if (mProperties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
        Check::Vk(vkMapMemory(mpDevice->getDevice(), mMemory, 0, size, 0, &mpHostMap));
    }
}

Buffer::~Buffer()
{
    // vkDeviceWaitIdle(mpDevice->getDevice()); // REMOVE THIS LINE

    if (mpHostMap) {
        vkUnmapMemory(mpDevice->getDevice(), mMemory);
        mpHostMap = nullptr;
    }

    // You might also need to ensure the device is idle before destroying the buffer
    // if it's being used by in-flight commands. A better pattern is to use
    // fences or a deferred destruction queue, but for now, let's see if removing
    // the idle call from this hot path fixes the allocation issue.
    // If you get validation errors about destroying a buffer in use, a single
    // vkDeviceWaitIdle at the end of the entire Scene::compile() is better.
    // Let's assume for now that Helpers::cmdEnd handles synchronization sufficiently.
    vkDestroyBuffer(mpDevice->getDevice(), mBuffer, nullptr);
    vkFreeMemory(mpDevice->getDevice(), mMemory, nullptr);
}
void Buffer::copyFromHost(const void* pData, VkDeviceSize size, VkDeviceSize offset)
{
    // This is the branch for DEVICE_LOCAL buffers that need a staging buffer.
    if (!(mProperties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        // Set up staging buffer
        Buffer staging(mpDevice, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        // Copy to staging buffer. This recursive call will take the `else` branch below.
        staging.copyFromHost(pData, size, 0);

        // Transfer from staging buffer to this (the final DEVICE_LOCAL) buffer
        VkCommandBuffer cmd = Helpers::cmdBegin(mpDevice);

        VkBufferCopy region = {
            .srcOffset = 0, // It's good practice to specify the source offset too.
            .dstOffset = offset,
            .size = size,
        };
        vkCmdCopyBuffer(cmd, staging.getBuffer(), mBuffer, 1, &region);

        Helpers::cmdEnd(mpDevice, cmd);

    }
    // This is the branch for HOST_VISIBLE buffers (like your texture staging buffer).
    else {

        // --- START OF DEBUG CODE ---
        // Check if we're dealing with the large VQ grid staging buffer by checking its unique size.
        if (size == 753664) {
            // Cast the incoming data pointer so we can read from it.
            const uint8_t* sourceData = static_cast<const uint8_t*>(pData);

            printf("--- PRE-MEMCPY CHECK --- First 10 bytes of source data: ");
            for (int i = 0; i < 10; ++i) {
                printf("%s%d", (i > 0 ? ", " : ""), static_cast<int>(sourceData[i]));
            }
            printf("\n");
            fflush(stdout);
        }
        // --- END OF DEBUG CODE ---

        // Perform the direct memory copy.
        char* pOffsettedHostMap = (char*)mpHostMap + offset;
        std::memcpy(pOffsettedHostMap, pData, size);


        // --- OPTIONAL POST-MEMCPY CHECK ---
        // You could add another print here to read back from pOffsettedHostMap
        // to ensure the memcpy itself worked, but it's very unlikely to be the point of failure.
        // The PRE-MEMCPY check is the most important one.
        if (size == 753664) {
            const uint8_t* destData = static_cast<const uint8_t*>(static_cast<void*>(pOffsettedHostMap));
            printf("--- POST-MEMCPY CHECK --- First 10 bytes in mapped buffer: ");
            for (int i = 0; i < 10; ++i) {
                printf("%s%d", (i > 0 ? ", " : ""), static_cast<int>(destData[i]));
            }
            printf("\n");
            fflush(stdout);
        }
        // --- END OPTIONAL CHECK ---
    }
}
