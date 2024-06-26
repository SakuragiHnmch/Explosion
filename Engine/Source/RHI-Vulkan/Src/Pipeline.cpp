//
// Created by Zach Lee on 2022/4/2.
//

#include <array>
#include <utility>

#include <RHI/Vulkan/Pipeline.h>
#include <RHI/Vulkan/Device.h>
#include <RHI/Vulkan/ShaderModule.h>
#include <RHI/Vulkan/PipelineLayout.h>
#include <RHI/Vulkan/Common.h>

namespace RHI::Vulkan {
    static VkStencilOpState ConvertStencilOp(const StencilFaceState& stencilOp, const uint32_t readMask, const uint32_t writeMask)
    {
        VkStencilOpState state = {};
        state.compareOp = EnumCast<CompareFunc, VkCompareOp>(stencilOp.compareFunc);
        state.depthFailOp = EnumCast<StencilOp, VkStencilOp>(stencilOp.depthFailOp);
        state.failOp = EnumCast<StencilOp, VkStencilOp>(stencilOp.failOp);
        state.passOp = EnumCast<StencilOp, VkStencilOp>(stencilOp.passOp);
        state.compareMask = readMask;
        state.writeMask = writeMask;
        state.reference = 0;
        return state;
    }

    static VkPipelineDepthStencilStateCreateInfo ConstructDepthStencil(const RasterPipelineCreateInfo& createInfo)
    {
        const auto& dsState = createInfo.depthStencilState;

        VkPipelineDepthStencilStateCreateInfo dsInfo = {};
        dsInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        dsInfo.depthTestEnable = dsState.depthEnabled ? VK_TRUE : VK_FALSE;
        dsInfo.depthWriteEnable = dsState.depthEnabled ? VK_TRUE : VK_FALSE;
        dsInfo.stencilTestEnable = dsState.stencilEnabled ? VK_TRUE : VK_FALSE;
        dsInfo.front = ConvertStencilOp(dsState.stencilFront, dsState.stencilReadMask, dsState.stencilWriteMask);
        dsInfo.back = ConvertStencilOp(dsState.stencilBack, dsState.stencilReadMask, dsState.stencilWriteMask);
        dsInfo.minDepthBounds = -1.f;
        dsInfo.maxDepthBounds = 1.f;
        dsInfo.depthBoundsTestEnable = VK_FALSE;
        dsInfo.depthCompareOp = EnumCast<CompareFunc, VkCompareOp>(dsState.depthCompareFunc);
        return dsInfo;
    }

    static VkPipelineInputAssemblyStateCreateInfo ConstructInputAssembly(const RasterPipelineCreateInfo& createInfo)
    {
        VkPipelineInputAssemblyStateCreateInfo assemblyInfo = {};
        assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        assemblyInfo.topology = EnumCast<PrimitiveTopologyType, VkPrimitiveTopology>(createInfo.primitiveState.topologyType);
        assemblyInfo.primitiveRestartEnable = VK_FALSE;

        return assemblyInfo;
    }

    static VkPipelineRasterizationStateCreateInfo ConstructRasterization(const RasterPipelineCreateInfo& createInfo)
    {
        VkPipelineRasterizationStateCreateInfo rasterState = {};
        rasterState.polygonMode = EnumCast<FillMode, VkPolygonMode>(createInfo.primitiveState.fillMode);
        rasterState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterState.cullMode = EnumCast<CullMode, VkCullModeFlagBits>(createInfo.primitiveState.cullMode);
        rasterState.frontFace = createInfo.primitiveState.frontFace == FrontFace::cw ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterState.depthBiasClamp = createInfo.depthStencilState.depthBiasClamp;
        rasterState.depthBiasSlopeFactor = createInfo.depthStencilState.depthBiasSlopeScale;
        rasterState.depthBiasEnable = createInfo.depthStencilState.depthBias == 0 ? VK_FALSE : VK_TRUE;
        rasterState.depthBiasConstantFactor = static_cast<float>(createInfo.depthStencilState.depthBias);
        rasterState.lineWidth = 1.0;

        // TODO DepthClampEnable requires check depth clamping feature
        rasterState.depthClampEnable = VK_FALSE;
        // rasterState.setDepthClampEnable(createInfo.primitive.depthClip ? VK_FALSE : VK_TRUE);
        // TODO DepthClipEnable requires VK_EXT_depth_clip_enable

        return rasterState;
    }

    static VkPipelineMultisampleStateCreateInfo ConstructMultiSampleState(const RasterPipelineCreateInfo& createInfo)
    {
        // TODO check this
        VkPipelineMultisampleStateCreateInfo multiSampleInfo = {};
        multiSampleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multiSampleInfo.alphaToCoverageEnable = createInfo.multiSampleState.alphaToCoverage ? VK_TRUE : VK_FALSE;
        multiSampleInfo.pSampleMask = &createInfo.multiSampleState.mask;
        multiSampleInfo.rasterizationSamples = static_cast<VkSampleCountFlagBits>(createInfo.multiSampleState.count);
        return multiSampleInfo;
    }

    static VkPipelineViewportStateCreateInfo ConstructViewportInfo(const RasterPipelineCreateInfo&)
    {
        // TODO check this
        VkPipelineViewportStateCreateInfo viewportState = {};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = nullptr;
        viewportState.scissorCount = 1;
        viewportState.pScissors = nullptr;
        return viewportState;
    }

    static VkPipelineColorBlendStateCreateInfo ConstructAttachmentInfo(const RasterPipelineCreateInfo& createInfo, std::vector<VkPipelineColorBlendAttachmentState>& blendStates)
    {
        blendStates.resize(createInfo.fragmentState.colorTargets.size());

        VkPipelineColorBlendStateCreateInfo colorInfo = {};
        colorInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        for (auto i = 0; i < createInfo.fragmentState.colorTargets.size(); ++i) {
            const auto& srcState = createInfo.fragmentState.colorTargets[i];
            blendStates[i].blendEnable = srcState.blendEnabled ? VK_TRUE : VK_FALSE;
            blendStates[i].colorWriteMask = srcState.writeFlags.Value();
            blendStates[i].alphaBlendOp = EnumCast<BlendOp, VkBlendOp>(srcState.colorBlend.op);
            blendStates[i].alphaBlendOp = EnumCast<BlendOp, VkBlendOp>(srcState.alphaBlend.op);
            blendStates[i].srcColorBlendFactor = EnumCast<BlendFactor, VkBlendFactor>(srcState.colorBlend.srcFactor);
            blendStates[i].srcAlphaBlendFactor = EnumCast<BlendFactor, VkBlendFactor>(srcState.alphaBlend.srcFactor);
            blendStates[i].dstColorBlendFactor = EnumCast<BlendFactor, VkBlendFactor>(srcState.colorBlend.dstFactor);
            blendStates[i].dstAlphaBlendFactor = EnumCast<BlendFactor, VkBlendFactor>(srcState.alphaBlend.dstFactor);
        }

        colorInfo.pAttachments = blendStates.data();
        colorInfo.attachmentCount = blendStates.size();
        colorInfo.logicOpEnable = VK_FALSE;
        colorInfo.logicOp = VK_LOGIC_OP_CLEAR;
        colorInfo.blendConstants[0] = 0.0f;
        colorInfo.blendConstants[1] = 0.0f;
        colorInfo.blendConstants[2] = 0.0f;
        colorInfo.blendConstants[3] = 0.0f;
        return colorInfo;
    }

    static VkPipelineVertexInputStateCreateInfo ConstructVertexInput(const RasterPipelineCreateInfo& createInfo,
        std::vector<VkVertexInputAttributeDescription>& attributes,
        std::vector<VkVertexInputBindingDescription>& bindings)
    {
        VkPipelineVertexInputStateCreateInfo vtxInput = {};
        vtxInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        bindings.resize(createInfo.vertexState.bufferLayouts.size());
        for (uint32_t i = 0; i < createInfo.vertexState.bufferLayouts.size(); ++i) {
            const auto& binding = createInfo.vertexState.bufferLayouts[i];
            bindings[i].binding = i;
            bindings[i].inputRate = binding.stepMode == VertexStepMode::perInstance ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX;
            bindings[i].stride = binding.stride;

            for (uint32_t j = 0; j < binding.attributes.size(); ++j) {
                VkVertexInputAttributeDescription desc = {};
                desc.binding = i;
                desc.location = std::get<GlslVertexBinding>(binding.attributes[j].platformBinding).location;
                desc.offset = binding.attributes[j].offset;
                desc.format = EnumCast<VertexFormat, VkFormat>(binding.attributes[j].format);
                attributes.emplace_back(desc);
            }
        }
        vtxInput.vertexAttributeDescriptionCount = attributes.size();
        vtxInput.pVertexAttributeDescriptions = attributes.data();
        vtxInput.vertexBindingDescriptionCount = bindings.size();
        vtxInput.pVertexBindingDescriptions = bindings.data();
        return vtxInput;
    }

    VulkanRasterPipeline::VulkanRasterPipeline(VulkanDevice& inDevice, const RasterPipelineCreateInfo& inCreateInfo)
        : RasterPipeline(inCreateInfo)
        , device(inDevice)
    {
        SavePipelineLayout(inCreateInfo);
        CreateNativeGraphicsPipeline(inCreateInfo);
    }

    VulkanRasterPipeline::~VulkanRasterPipeline()
    {
        if (nativePipeline) {
            vkDestroyPipeline(device.GetNative(), nativePipeline, nullptr);
        }
    }

    VulkanPipelineLayout* VulkanRasterPipeline::GetPipelineLayout() const
    {
        return pipelineLayout;
    }

    void VulkanRasterPipeline::SavePipelineLayout(const RasterPipelineCreateInfo& inCreateInfo)
    {
        auto* layout = static_cast<VulkanPipelineLayout*>(inCreateInfo.layout);
        Assert(layout);
        pipelineLayout = layout;
    }

    void VulkanRasterPipeline::CreateNativeGraphicsPipeline(const RasterPipelineCreateInfo& inCreateInfo)
    {
        std::vector<VkPipelineShaderStageCreateInfo> stages;
        auto setStage = [&stages](ShaderModule* module, const VkShaderStageFlagBits stage) {
            if (module == nullptr) {
                return;
            }
            VkPipelineShaderStageCreateInfo stageInfo = {};
            stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.module = static_cast<const VulkanShaderModule*>(module)->GetNative();
            stageInfo.pName = module->GetEntryPoint().c_str();
            stageInfo.stage = stage;
            stages.emplace_back(stageInfo);
        };
        setStage(inCreateInfo.vertexShader, VK_SHADER_STAGE_VERTEX_BIT);
        setStage(inCreateInfo.pixelShader, VK_SHADER_STAGE_FRAGMENT_BIT);

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY
        };
        VkPipelineDynamicStateCreateInfo dynStateInfo = {};
        dynStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynStateInfo.dynamicStateCount = dynamicStates.size();
        dynStateInfo.pDynamicStates = dynamicStates.data();

        VkPipelineMultisampleStateCreateInfo multiSampleInfo = ConstructMultiSampleState(inCreateInfo);
        VkPipelineDepthStencilStateCreateInfo dsInfo = ConstructDepthStencil(inCreateInfo);
        VkPipelineInputAssemblyStateCreateInfo assemblyInfo = ConstructInputAssembly(inCreateInfo);
        VkPipelineRasterizationStateCreateInfo rasterState = ConstructRasterization(inCreateInfo);
        VkPipelineViewportStateCreateInfo viewportState = ConstructViewportInfo(inCreateInfo);

        std::vector<VkPipelineColorBlendAttachmentState> blendStates;
        VkPipelineColorBlendStateCreateInfo colorInfo = ConstructAttachmentInfo(inCreateInfo, blendStates);

        std::vector<VkVertexInputAttributeDescription> attributes;
        std::vector<VkVertexInputBindingDescription> bindings;
        VkPipelineVertexInputStateCreateInfo vtxInput = ConstructVertexInput(inCreateInfo, attributes, bindings);

        std::vector<VkFormat> pixelFormats(inCreateInfo.fragmentState.colorTargets.size());
        for (size_t i = 0; i < inCreateInfo.fragmentState.colorTargets.size(); i++)
        {
            auto format = inCreateInfo.fragmentState.colorTargets[i].format;
            pixelFormats[i] = EnumCast<PixelFormat, VkFormat>(format);
        }

        VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo;
        pipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        pipelineRenderingCreateInfo.colorAttachmentCount = inCreateInfo.fragmentState.colorTargets.size();
        pipelineRenderingCreateInfo.pColorAttachmentFormats = pixelFormats.data();
        pipelineRenderingCreateInfo.depthAttachmentFormat = inCreateInfo.depthStencilState.depthEnabled ? EnumCast<PixelFormat, VkFormat>(inCreateInfo.depthStencilState.format) : VK_FORMAT_UNDEFINED;
        pipelineRenderingCreateInfo.stencilAttachmentFormat = inCreateInfo.depthStencilState.stencilEnabled ? EnumCast<PixelFormat, VkFormat>(inCreateInfo.depthStencilState.format) : VK_FORMAT_UNDEFINED;
        pipelineRenderingCreateInfo.pNext = nullptr;
        pipelineRenderingCreateInfo.viewMask = 0;

        VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.pStages = stages.data();
        pipelineCreateInfo.stageCount = stages.size();
        pipelineCreateInfo.layout = static_cast<const VulkanPipelineLayout*>(inCreateInfo.layout)->GetNative();
        pipelineCreateInfo.pDynamicState = &dynStateInfo;
        pipelineCreateInfo.pMultisampleState = &multiSampleInfo;
        pipelineCreateInfo.pDepthStencilState = &dsInfo;
        pipelineCreateInfo.pInputAssemblyState = &assemblyInfo;
        pipelineCreateInfo.pRasterizationState = &rasterState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pTessellationState = nullptr;
        pipelineCreateInfo.pColorBlendState = &colorInfo;
        pipelineCreateInfo.pVertexInputState = &vtxInput;
        pipelineCreateInfo.pNext = &pipelineRenderingCreateInfo;

        Assert(vkCreateGraphicsPipelines(device.GetNative(), VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &nativePipeline) == VK_SUCCESS);

#if BUILD_CONFIG_DEBUG
        if (!inCreateInfo.debugName.empty()) {
            device.SetObjectName(VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(nativePipeline), inCreateInfo.debugName.c_str());
        }
#endif
    }

    VkPipeline VulkanRasterPipeline::GetNative() const
    {
        return nativePipeline;
    }

    VulkanComputePipeline::VulkanComputePipeline(VulkanDevice& inDevice, const ComputePipelineCreateInfo& inCreateInfo)
        : ComputePipeline(inCreateInfo)
        , device(inDevice)
        , nativePipeline(VK_NULL_HANDLE)
    {
        SavePipelineLayout(inCreateInfo);
        CreateNativeComputePipeline(inCreateInfo);
    }

    VulkanComputePipeline::~VulkanComputePipeline()
    {
        if (nativePipeline) {
            vkDestroyPipeline(device.GetNative(), nativePipeline, nullptr);
        }
    }

    VulkanPipelineLayout* VulkanComputePipeline::GetPipelineLayout() const
    {
        return pipelineLayout;
    }

    void VulkanComputePipeline::SavePipelineLayout(const ComputePipelineCreateInfo& inCreateInfo)
    {
        auto* layout = static_cast<VulkanPipelineLayout*>(inCreateInfo.layout);
        Assert(layout);
        pipelineLayout = layout;
    }

    void VulkanComputePipeline::CreateNativeComputePipeline(const ComputePipelineCreateInfo& inCreateInfo)
    {
        VkPipelineShaderStageCreateInfo stageInfo = {};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.module = static_cast<const VulkanShaderModule*>(inCreateInfo.computeShader)->GetNative();
        stageInfo.pName = inCreateInfo.computeShader->GetEntryPoint().c_str();
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;

        VkComputePipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = pipelineLayout->GetNative();
        pipelineInfo.stage = stageInfo;

        Assert(vkCreateComputePipelines(device.GetNative(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &nativePipeline) == VK_SUCCESS);
    }

    VkPipeline VulkanComputePipeline::GetNative() const
    {
        return nativePipeline;
    }

}