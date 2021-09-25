//
// Created by John Kindem on 2021/4/26.
//

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include <RHI/Vulkan/VulkanGraphicsPipeline.h>
#include <RHI/Vulkan/VulkanAdapater.h>
#include <RHI/Vulkan/VulkanDriver.h>
#include <RHI/Vulkan/VulkanRenderPass.h>
#include <RHI/Vulkan/VulkanShader.h>

namespace Explosion::RHI {
    VulkanGraphicsPipeline::VulkanGraphicsPipeline(VulkanDriver& driver, Config config)
        : GraphicsPipeline(config), driver(driver), device(*driver.GetDevice())
    {
        CreateDescriptorSetLayout();
        CreatePipelineLayout();
        CreateGraphicsPipeline();
    }

    VulkanGraphicsPipeline::~VulkanGraphicsPipeline()
    {
        DestroyGraphicsPipeline();
        DestroyPipelineLayout();
        DestroyDescriptorSetLayout();
    }

    const VkPipelineLayout& VulkanGraphicsPipeline::GetVkPipelineLayout() const
    {
        return vkPipelineLayout;
    }

    const VkPipeline& VulkanGraphicsPipeline::GetVkPipeline() const
    {
        return vkPipeline;
    }

    const VkDescriptorSetLayout& VulkanGraphicsPipeline::GetVkDescriptorSetLayout() const
    {
        return vkDescriptorSetLayout;
    }

    void VulkanGraphicsPipeline::CreateDescriptorSetLayout()
    {
        std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings(config.descriptorConfig.descriptorAttributes.size());
        for (uint32_t i = 0; i < descriptorSetLayoutBindings.size(); i++) {
            DescriptorAttribute& attribute = config.descriptorConfig.descriptorAttributes[i];
            descriptorSetLayoutBindings[i].binding = attribute.binding;
            descriptorSetLayoutBindings[i].descriptorType = VkConvert<DescriptorType, VkDescriptorType>(attribute.type);
            descriptorSetLayoutBindings[i].descriptorCount = 1;
            descriptorSetLayoutBindings[i].stageFlags = 0;
            descriptorSetLayoutBindings[i].stageFlags = VkGetFlags<ShaderStageBits, VkShaderStageFlagBits>(attribute.shaderStages);
            descriptorSetLayoutBindings[i].pImmutableSamplers = nullptr;
        }

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo {};
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.pNext = nullptr;
        descriptorSetLayoutCreateInfo.flags = 0;
        descriptorSetLayoutCreateInfo.bindingCount = descriptorSetLayoutBindings.size();
        descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings.data();

        if (vkCreateDescriptorSetLayout(device.GetVkDevice(), &descriptorSetLayoutCreateInfo, nullptr, &vkDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create vulkan descriptor set layout");
        }
    }

    void VulkanGraphicsPipeline::DestroyDescriptorSetLayout()
    {
        vkDestroyDescriptorSetLayout(device.GetVkDevice(), vkDescriptorSetLayout, nullptr);
    }

    void VulkanGraphicsPipeline::CreatePipelineLayout()
    {
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo {};
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.pNext = nullptr;
        pipelineLayoutCreateInfo.flags = 0;
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.pSetLayouts = &vkDescriptorSetLayout;
        pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
        pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

        if (vkCreatePipelineLayout(device.GetVkDevice(), &pipelineLayoutCreateInfo, nullptr, &vkPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create vulkan descriptor set layout");
        }
    }

    void VulkanGraphicsPipeline::DestroyPipelineLayout()
    {
        vkDestroyPipelineLayout(device.GetVkDevice(), vkPipelineLayout, nullptr);
    }

    void VulkanGraphicsPipeline::CreateGraphicsPipeline()
    {
        // load shader module and populate shader stage create info
        std::unordered_map<VkShaderStageFlagBits, bool> stageExists;
        std::vector<VkPipelineShaderStageCreateInfo> shaderStageCreateInfos;
        for (auto shader : config.shaderConfig.shaderModules) {
            auto vulkanShader = static_cast<VulkanShader*>(shader);
            auto stage = vulkanShader->GetShaderStage();
            auto iter = stageExists.find(stage);
            if (iter != stageExists.end() && iter->second) {
                throw std::runtime_error("specific shader stage is already added to graphics pipeline");
            }

            VkPipelineShaderStageCreateInfo shaderStageCreateInfo {};
            shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStageCreateInfo.pNext = nullptr;
            shaderStageCreateInfo.flags = 0;
            shaderStageCreateInfo.stage = stage;
            shaderStageCreateInfo.module = vulkanShader->GetShaderModule();
            shaderStageCreateInfo.pName = "main";
            shaderStageCreateInfo.pSpecializationInfo = nullptr;
            shaderStageCreateInfos.emplace_back(shaderStageCreateInfo);
        }

        std::vector<VkVertexInputBindingDescription> vertexInputBindingDescriptions(config.vertexConfig.vertexBindings.size());
        for (uint32_t i = 0; i < vertexInputBindingDescriptions.size(); i++) {
            VertexBinding& vertexBinding = config.vertexConfig.vertexBindings[i];
            vertexInputBindingDescriptions[i].binding = vertexBinding.binding;
            vertexInputBindingDescriptions[i].stride = vertexBinding.stride;
            vertexInputBindingDescriptions[i].inputRate = VkConvert<VertexInputRate, VkVertexInputRate>(vertexBinding.inputRate);
        }
        std::vector<VkVertexInputAttributeDescription> vertexInputAttributeDescriptions(config.vertexConfig.vertexAttributes.size());
        for (uint32_t i = 0; i < vertexInputAttributeDescriptions.size(); i++) {
            VertexAttribute& vertexAttribute = config.vertexConfig.vertexAttributes[i];
            vertexInputAttributeDescriptions[i].binding = vertexAttribute.binding;
            vertexInputAttributeDescriptions[i].location = vertexAttribute.location;
            vertexInputAttributeDescriptions[i].format = VkConvert<Format, VkFormat>(vertexAttribute.format);
            vertexInputAttributeDescriptions[i].offset = vertexAttribute.offset;
        }
        VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo {};
        vertexInputStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputStateCreateInfo.pNext = nullptr;
        vertexInputStateCreateInfo.flags = 0;
        vertexInputStateCreateInfo.vertexBindingDescriptionCount = vertexInputBindingDescriptions.size();
        vertexInputStateCreateInfo.pVertexBindingDescriptions = vertexInputBindingDescriptions.data();
        vertexInputStateCreateInfo.vertexAttributeDescriptionCount = vertexInputAttributeDescriptions.size();
        vertexInputStateCreateInfo.pVertexAttributeDescriptions = vertexInputAttributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo {};
        inputAssemblyStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssemblyStateCreateInfo.pNext = nullptr;
        inputAssemblyStateCreateInfo.flags = 0;
        inputAssemblyStateCreateInfo.topology = VkConvert<PrimitiveTopology, VkPrimitiveTopology>(config.assemblyConfig.topology);
        inputAssemblyStateCreateInfo.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport {};
        Viewport& vpConfig = config.viewportScissorConfig.viewport;
        viewport.x = vpConfig.x;
        viewport.y = vpConfig.y;
        viewport.width = vpConfig.width;
        viewport.height = vpConfig.height;
        viewport.minDepth = vpConfig.minDepth;
        viewport.maxDepth = vpConfig.maxDepth;
        VkRect2D scissor {};
        Scissor& scConfig = config.viewportScissorConfig.scissor;
        scissor.offset.x = scConfig.x;
        scissor.offset.y = scConfig.y;
        scissor.extent.width = scConfig.width;
        scissor.extent.height = scConfig.height;
        VkPipelineViewportStateCreateInfo viewportStateCreateInfo {};
        viewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportStateCreateInfo.pNext = nullptr;
        viewportStateCreateInfo.flags = 0;
        viewportStateCreateInfo.viewportCount = 1;
        viewportStateCreateInfo.pViewports = &viewport;
        viewportStateCreateInfo.scissorCount = 1;
        viewportStateCreateInfo.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo {};
        rasterizationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizationStateCreateInfo.pNext = nullptr;
        rasterizationStateCreateInfo.flags = 0;
        rasterizationStateCreateInfo.depthClampEnable = VkBoolConvert(config.rasterizerConfig.depthClamp);
        rasterizationStateCreateInfo.rasterizerDiscardEnable = VkBoolConvert(config.rasterizerConfig.discard);
        rasterizationStateCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizationStateCreateInfo.cullMode = VkGetFlags<CullModeBits, VkCullModeFlagBits>(config.rasterizerConfig.cullModes);
        rasterizationStateCreateInfo.frontFace = VkConvert<FrontFace, VkFrontFace>(config.rasterizerConfig.frontFace);
        rasterizationStateCreateInfo.depthBiasEnable = VK_FALSE;
        rasterizationStateCreateInfo.depthBiasConstantFactor = 0.0f;
        rasterizationStateCreateInfo.depthBiasClamp = 0.0f;
        rasterizationStateCreateInfo.depthBiasSlopeFactor = 0.0f;
        rasterizationStateCreateInfo.lineWidth = 1.f;

        VkPipelineMultisampleStateCreateInfo multiSampleStateCreateInfo {};
        multiSampleStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multiSampleStateCreateInfo.pNext = nullptr;
        multiSampleStateCreateInfo.flags = 0;
        multiSampleStateCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multiSampleStateCreateInfo.sampleShadingEnable = VK_FALSE;
        multiSampleStateCreateInfo.minSampleShading = 1.0f;
        multiSampleStateCreateInfo.pSampleMask = nullptr;
        multiSampleStateCreateInfo.alphaToCoverageEnable = VK_FALSE;
        multiSampleStateCreateInfo.alphaToOneEnable = VK_FALSE;

        VkPipelineDepthStencilStateCreateInfo depthStencilStateCreateInfo {};
        depthStencilStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencilStateCreateInfo.pNext = nullptr;
        depthStencilStateCreateInfo.flags = 0;
        depthStencilStateCreateInfo.depthTestEnable = VkBoolConvert(config.depthStencilConfig.depthTest);
        depthStencilStateCreateInfo.depthWriteEnable = VkBoolConvert(config.depthStencilConfig.depthWrite);
        depthStencilStateCreateInfo.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencilStateCreateInfo.depthBoundsTestEnable = VK_FALSE;
        depthStencilStateCreateInfo.stencilTestEnable = VkBoolConvert(config.depthStencilConfig.stencilTest);
        depthStencilStateCreateInfo.front = {};
        depthStencilStateCreateInfo.back = {};
        depthStencilStateCreateInfo.minDepthBounds = 0.f;
        depthStencilStateCreateInfo.maxDepthBounds = 1.f;

        VkPipelineColorBlendAttachmentState colorBlendAttachmentState {};
        colorBlendAttachmentState.blendEnable = VkBoolConvert(config.colorBlendConfig.enabled);
        colorBlendAttachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
        VkPipelineColorBlendStateCreateInfo colorBlendStateCreateInfo {};
        colorBlendStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendStateCreateInfo.pNext = nullptr;
        colorBlendStateCreateInfo.flags = 0;
        colorBlendStateCreateInfo.logicOpEnable = VkBoolConvert(config.colorBlendConfig.enabled);
        colorBlendStateCreateInfo.logicOp = VK_LOGIC_OP_COPY;
        colorBlendStateCreateInfo.attachmentCount = 1;
        colorBlendStateCreateInfo.pAttachments = &colorBlendAttachmentState;
        colorBlendStateCreateInfo.blendConstants[0] = 0.f;
        colorBlendStateCreateInfo.blendConstants[1] = 0.f;
        colorBlendStateCreateInfo.blendConstants[2] = 0.f;
        colorBlendStateCreateInfo.blendConstants[3] = 0.f;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo pipelineDynamicStateCreateInfo {};
        pipelineDynamicStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        pipelineDynamicStateCreateInfo.dynamicStateCount = dynamicStates.size();
        pipelineDynamicStateCreateInfo.pDynamicStates = dynamicStates.data();

        VkGraphicsPipelineCreateInfo graphicsPipelineCreateInfo {};
        graphicsPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        graphicsPipelineCreateInfo.pNext = nullptr;
        graphicsPipelineCreateInfo.flags = 0;
        graphicsPipelineCreateInfo.stageCount = shaderStageCreateInfos.size();
        graphicsPipelineCreateInfo.pStages = shaderStageCreateInfos.data();
        graphicsPipelineCreateInfo.pVertexInputState = &vertexInputStateCreateInfo;
        graphicsPipelineCreateInfo.pInputAssemblyState = &inputAssemblyStateCreateInfo;
        graphicsPipelineCreateInfo.pTessellationState = nullptr;
        graphicsPipelineCreateInfo.pViewportState = &viewportStateCreateInfo;
        graphicsPipelineCreateInfo.pRasterizationState = &rasterizationStateCreateInfo;
        graphicsPipelineCreateInfo.pMultisampleState = &multiSampleStateCreateInfo;
        graphicsPipelineCreateInfo.pDepthStencilState = &depthStencilStateCreateInfo;
        graphicsPipelineCreateInfo.pColorBlendState = &colorBlendStateCreateInfo;
        graphicsPipelineCreateInfo.pDynamicState = &pipelineDynamicStateCreateInfo;
        graphicsPipelineCreateInfo.layout = vkPipelineLayout;
        graphicsPipelineCreateInfo.renderPass = dynamic_cast<VulkanRenderPass*>(config.renderPass)->GetVkRenderPass();
        graphicsPipelineCreateInfo.subpass = 0;
        graphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
        graphicsPipelineCreateInfo.basePipelineIndex = -1;

        if (vkCreateGraphicsPipelines(device.GetVkDevice(), VK_NULL_HANDLE, 1, &graphicsPipelineCreateInfo, nullptr, &vkPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create vulkan graphics pipeline");
        }
    }

    void VulkanGraphicsPipeline::DestroyGraphicsPipeline()
    {
        vkDestroyPipeline(device.GetVkDevice(), vkPipeline, nullptr);
    }
}
