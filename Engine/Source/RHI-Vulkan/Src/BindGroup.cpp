//
// Created by Zach Lee on 2022/3/20.
//

#include <RHI/Vulkan/BindGroup.h>
#include <RHI/Vulkan/Device.h>
#include <RHI/Vulkan/BindGroupLayout.h>
#include <RHI/Vulkan/Buffer.h>
#include <RHI/Vulkan/BufferView.h>
#include <RHI/Vulkan/Sampler.h>
#include <RHI/Vulkan/TextureView.h>
#include <RHI/Vulkan/Common.h>

namespace RHI::Vulkan {
    VKBindGroup::VKBindGroup(VKDevice& device, const BindGroupCreateInfo& createInfo)
        : BindGroup(createInfo), device(device)
    {
        CreateDescriptorPool(createInfo);
        CreateDescriptorSet(createInfo);
    }

    VKBindGroup::~VKBindGroup() noexcept
    {
        if (descriptorPool) {
            device.GetVkDevice().destroyDescriptorPool(descriptorPool, nullptr);
        }
    }

    void VKBindGroup::Destroy()
    {
        delete this;
    }

    vk::DescriptorSet VKBindGroup::GetVkDescritorSet() const
    {
        return descriptorSet;
    }

    void VKBindGroup::CreateDescriptorPool(const BindGroupCreateInfo& createInfo)
    {
        std::vector<vk::DescriptorPoolSize> poolSizes(createInfo.entryNum);

        for (auto i = 0; i < createInfo.entryNum; i++) {
            const auto& entry = createInfo.entries[i];

            poolSizes[i].setType(VKEnumCast<BindingType, vk::DescriptorType>(entry.binding.type))
                .setDescriptorCount(1);
        }

        vk::DescriptorPoolCreateInfo poolInfo {};
        poolInfo.setPPoolSizes(poolSizes.data())
            .setPoolSizeCount(createInfo.entryNum)
            .setMaxSets(1);

        Assert(device.GetVkDevice().createDescriptorPool(&poolInfo, nullptr, &descriptorPool) == vk::Result::eSuccess);
    }

    void VKBindGroup::CreateDescriptorSet(const BindGroupCreateInfo& createInfo)
    {
        vk::DescriptorSetLayout layout = dynamic_cast<VKBindGroupLayout*>(createInfo.layout)->GetVkDescriptorSetLayout();

        vk::DescriptorSetAllocateInfo allocInfo {};
        allocInfo.setDescriptorSetCount(1)
            .setPSetLayouts(&layout)
            .setDescriptorPool(descriptorPool);

        Assert(device.GetVkDevice().allocateDescriptorSets(&allocInfo, &descriptorSet) == vk::Result::eSuccess);

        std::vector<vk::WriteDescriptorSet> descriptorWrites(createInfo.entryNum);
        std::vector<vk::DescriptorImageInfo> imageInfos;
        std::vector<vk::DescriptorBufferInfo> bufferInfos;
        
        int imageInfosNum = 0, bufferInfosNum = 0;
        for (int i = 0; i < createInfo.entryNum; i++) {
            const auto& entry = createInfo.entries[i];
            if (entry.binding.type == BindingType::UNIFORM_BUFFER) {
                bufferInfosNum++;
            } else if (entry.binding.type == BindingType::SAMPLER || entry.binding.type == BindingType::TEXTURE) {
                imageInfosNum++;
            }
        }
        imageInfos.reserve(imageInfosNum);
        bufferInfos.reserve(bufferInfosNum);
        
        for (int i = 0; i < createInfo.entryNum; i++) {
            const auto& entry = createInfo.entries[i];

            descriptorWrites[i].setDstSet(descriptorSet)
                .setDstBinding(entry.binding.platform.glsl.index)
                .setDescriptorCount(1)
                .setDescriptorType(VKEnumCast<BindingType, vk::DescriptorType>(entry.binding.type));

            if (entry.binding.type == BindingType::UNIFORM_BUFFER) {
                auto* bufferView = dynamic_cast<VKBufferView*>(entry.bufferView);

                bufferInfos.emplace_back();
                bufferInfos.back().setBuffer(bufferView->GetBuffer().GetVkBuffer())
                    .setOffset(bufferView->GetOffset())
                    .setRange(bufferView->GetBufferSize());

                descriptorWrites[i].setPBufferInfo(&bufferInfos.back());
            } else if (entry.binding.type == BindingType::SAMPLER) {
                auto* sampler = dynamic_cast<VKSampler*>(entry.sampler);

                imageInfos.emplace_back();
                imageInfos.back().setSampler(sampler->GetVkSampler());

                descriptorWrites[i].setPImageInfo(&imageInfos.back());
            } else if (entry.binding.type == BindingType::TEXTURE) {
                auto* textureView = dynamic_cast<VKTextureView*>(entry.textureView);

                imageInfos.emplace_back();
                imageInfos.back().setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
                    .setImageView(textureView->GetVkImageView());

                descriptorWrites[i].setPImageInfo(&imageInfos.back());
            } else {
                //TODO
            }
        }
        device.GetVkDevice().updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}
