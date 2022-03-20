//
// Created by johnk on 2022/1/26.
//

#include <RHI/Vulkan/Common.h>
#include <RHI/Vulkan/Device.h>
#include <RHI/Vulkan/Buffer.h>
#include <RHI/Vulkan/Gpu.h>

namespace RHI::Vulkan {
    static vk::MemoryPropertyFlags GetVkMemoryType(BufferUsageFlags bufferUsages)
    {
        static std::unordered_map<BufferUsageFlags, vk::MemoryPropertyFlags> rules = {
            { BufferUsageBits::MAP_WRITE | BufferUsageBits::COPY_SRC, vk::MemoryPropertyFlagBits::eHostVisible },
            { BufferUsageBits::MAP_READ | BufferUsageBits::COPY_DST, vk::MemoryPropertyFlagBits::eHostVisible }
            // TODO check other conditions ?
        };
        static vk::MemoryPropertyFlags fallback = vk::MemoryPropertyFlagBits::eDeviceLocal;

        for (const auto& rule : rules) {
            if (bufferUsages & rule.first) {
                return rule.second;
            }
        }
        return fallback;
    }


    static vk::BufferUsageFlags GetVkResourceStates(BufferUsageFlags bufferUsages)
    {
        static std::unordered_map<BufferUsageBits, vk::BufferUsageFlagBits> rules = {
            { BufferUsageBits::COPY_SRC, vk::BufferUsageFlagBits::eTransferSrc },
            { BufferUsageBits::COPY_DST, vk::BufferUsageFlagBits::eTransferDst },
            { BufferUsageBits::INDEX, vk::BufferUsageFlagBits::eIndexBuffer },
            { BufferUsageBits::VERTEX, vk::BufferUsageFlagBits::eVertexBuffer },
            { BufferUsageBits::UNIFORM, vk::BufferUsageFlagBits::eUniformBuffer },
            { BufferUsageBits::STORAGE, vk::BufferUsageFlagBits::eStorageBuffer },
            { BufferUsageBits::INDIRECT, vk::BufferUsageFlagBits::eIndirectBuffer },
        };

        vk::BufferUsageFlags result = {};
        for (const auto& rule : rules) {
            if (bufferUsages & rule.first) {
                result |= rule.second;
            }
        }
        return result;
    }

    VKBuffer::VKBuffer(VKDevice& d, const BufferCreateInfo* createInfo) : Buffer(createInfo), device(d)
    {
        CreateBuffer(createInfo);
        AllocateMemory(createInfo);
    }

    VKBuffer::~VKBuffer()
    {
        FreeMemory();
        DestroyBuffer();
    }

    void* VKBuffer::Map(MapMode mapMode, size_t offset, size_t length)
    {
        void* data;
        if (device.GetVkDevice().mapMemory(vkDeviceMemory, offset, length, {}, &data) != vk::Result::eSuccess) {
            throw VKException("failed to map vulkan device memory");
        }
        return data;
    }

    void VKBuffer::UnMap()
    {
        device.GetVkDevice().unmapMemory(vkDeviceMemory);
    }

    BufferView* VKBuffer::CreateBufferView(const BufferViewCreateInfo* createInfo)
    {
        return nullptr;
    }

    void VKBuffer::Destroy()
    {
        delete this;
    }

    void VKBuffer::CreateBuffer(const BufferCreateInfo* createInfo)
    {
        vk::BufferCreateInfo bufferInfo = {};
        bufferInfo.setSharingMode(vk::SharingMode::eExclusive)
            .setUsage(GetVkResourceStates(createInfo->usages))
            .setSize(createInfo->size);

        if (device.GetVkDevice().createBuffer(&bufferInfo, nullptr, &vkBuffer) != vk::Result::eSuccess) {
            throw VKException("failed to create buffer");
        }
    }

    void VKBuffer::AllocateMemory(const BufferCreateInfo* createInfo)
    {
        vk::MemoryRequirements memoryRequirements = {};
        device.GetVkDevice().getBufferMemoryRequirements(vkBuffer, &memoryRequirements);

        vk::MemoryAllocateInfo memoryInfo = {};
        memoryInfo.setAllocationSize(memoryRequirements.size)
            .setMemoryTypeIndex(device.GetGpu()->FindMemoryType(memoryRequirements.memoryTypeBits,
                                                                GetVkMemoryType(createInfo->usages)));
        if (device.GetVkDevice().allocateMemory(&memoryInfo, nullptr, &vkDeviceMemory) != vk::Result::eSuccess) {
            throw VKException("failed to allocate memory");
        }

        device.GetVkDevice().bindBufferMemory(vkBuffer, vkDeviceMemory, 0);
    }

    void VKBuffer::DestroyBuffer()
    {
        if (vkBuffer) {
            device.GetVkDevice().destroy(vkBuffer);
            vkBuffer = nullptr;
        }
    }

    void VKBuffer::FreeMemory()
    {
        if (vkDeviceMemory) {
            device.GetVkDevice().free(vkDeviceMemory);
            vkDeviceMemory = nullptr;
        }
    }
}
