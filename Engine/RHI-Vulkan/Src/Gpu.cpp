//
// Created by johnk on 12/1/2022.
//

#include <RHI/Vulkan/Gpu.h>
#include <RHI/Vulkan/Common.h>
#include <RHI/Vulkan/Device.h>

namespace RHI::Vulkan {
    VKGpu::VKGpu(vk::PhysicalDevice d) : Gpu(), vkPhysicalDevice(d) {}

    VKGpu::~VKGpu() = default;

    GpuProperty VKGpu::GetProperty()
    {
        vk::PhysicalDeviceProperties vkPhysicalDeviceProperties;
        vkPhysicalDevice.getProperties(&vkPhysicalDeviceProperties);

        GpuProperty property {};
        property.vendorId = vkPhysicalDeviceProperties.vendorID;
        property.deviceId = vkPhysicalDeviceProperties.deviceID;
        property.type = VKEnumCast<vk::PhysicalDeviceType, GpuType>(vkPhysicalDeviceProperties.deviceType);
        return property;
    }

    Device* VKGpu::RequestDevice(const DeviceCreateInfo* createInfo)
    {
        return new VKDevice(*this, createInfo);
    }

    vk::PhysicalDevice VKGpu::GetVkPhysicalDevice()
    {
        return vkPhysicalDevice;
    }

    uint32_t VKGpu::FindMemoryType(uint32_t filter, vk::MemoryPropertyFlags propertyFlags)
    {
        vk::PhysicalDeviceMemoryProperties memProperties;
        vkPhysicalDevice.getMemoryProperties(&memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((filter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & propertyFlags) == propertyFlags) {
                return i;
            }
        }
        throw VKException("failed to found suitable memory type");
    }
}