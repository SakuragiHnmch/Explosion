//
// Created by Zach Lee on 2022/6/4.
//

#include <RHI/Vulkan/CommandEncoder.h>
#include <RHI/Vulkan/Device.h>
#include <RHI/Vulkan/Gpu.h>
#include <RHI/Vulkan/Pipeline.h>
#include <RHI/Vulkan/CommandBuffer.h>
#include <RHI/Vulkan/Buffer.h>
#include <RHI/Vulkan/BufferView.h>
#include <RHI/Vulkan/TextureView.h>
#include <RHI/Vulkan/Texture.h>
#include <RHI/Vulkan/Common.h>
#include <RHI/Vulkan/Instance.h>
#include <RHI/Vulkan/SwapChain.h>
#include <RHI/Vulkan/BindGroup.h>
#include <RHI/Vulkan/PipelineLayout.h>
#include <RHI/Synchronous.h>

namespace RHI::Vulkan {

    VulkanCommandEncoder::VulkanCommandEncoder(VulkanDevice& inDevice, VulkanCommandBuffer& inCmdBuffer)
        : device(inDevice), commandBuffer(inCmdBuffer)
    {
    }
    VulkanCommandEncoder::~VulkanCommandEncoder()
    {
    }

    static std::tuple<VkImageLayout, VkAccessFlags, VkPipelineStageFlags> GetBarrierInfo(TextureState status)
    {
        if (status == TextureState::present) {
            return { VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_MEMORY_READ_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT };
        }
        if (status == TextureState::renderTarget) {
            return { VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        }
        if (status == TextureState::copyDst) {
            return { VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT };
        }
        if (status == TextureState::shaderReadOnly) {
            return { VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
        }
        if (status == TextureState::depthStencilReadonly) {
            return { VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
        }
        if (status == TextureState::depthStencilWrite) {
            return { VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT };
        }
        return {VK_IMAGE_LAYOUT_UNDEFINED, VkAccessFlags {}, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
    }

    void VulkanCommandEncoder::ResourceBarrier(const Barrier& inBarrier)
    {
        if (inBarrier.type == ResourceType::texture) {
            const auto& textureBarrierInfo = inBarrier.texture;
            auto oldLayout = GetBarrierInfo(textureBarrierInfo.before == TextureState::present ? TextureState::undefined : textureBarrierInfo.before);
            auto newLayout = GetBarrierInfo(textureBarrierInfo.after);

            auto* vkTexture = static_cast<VulkanTexture*>(textureBarrierInfo.pointer);
            VkImageMemoryBarrier imageBarrier = {};
            imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imageBarrier.image = vkTexture->GetNative();
            imageBarrier.oldLayout = std::get<0>(oldLayout);
            imageBarrier.srcAccessMask = std::get<1>(oldLayout);
            imageBarrier.newLayout = std::get<0>(newLayout);
            imageBarrier.dstAccessMask = std::get<1>(newLayout);
            imageBarrier.subresourceRange = vkTexture->GetNativeSubResourceFullRange();

            vkCmdPipelineBarrier(commandBuffer.GetNativeCommandBuffer(), std::get<2>(oldLayout), std::get<2>(newLayout), VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 1, &imageBarrier);
        }
    }

    CopyPassCommandEncoder* VulkanCommandEncoder::BeginCopyPass()
    {
        return new VulkanCopyPassCommandEncoder(device, *this, commandBuffer);
    }

    ComputePassCommandEncoder* VulkanCommandEncoder::BeginComputePass()
    {
        return new VulkanComputePassCommandEncoder(device, *this, commandBuffer);
    }

    GraphicsPassCommandEncoder* VulkanCommandEncoder::BeginGraphicsPass(const GraphicsPassBeginInfo& inBeginInfo)
    {
        return new VulkanGraphicsPassCommandEncoder(device, *this, commandBuffer, inBeginInfo);
    }

    void VulkanCommandEncoder::End()
    {
        vkEndCommandBuffer(commandBuffer.GetNativeCommandBuffer());
    }

    void VulkanCommandEncoder::Destroy()
    {
        delete this;
    }

    VulkanCopyPassCommandEncoder::VulkanCopyPassCommandEncoder(VulkanDevice& inDevice, VulkanCommandEncoder& inCmdEncoder, VulkanCommandBuffer& inCmdBuffer)
        : device(inDevice)
        , commandEncoder(inCmdEncoder)
        , commandBuffer(inCmdBuffer)
    {
    }

    VulkanCopyPassCommandEncoder::~VulkanCopyPassCommandEncoder() = default;

    void VulkanCopyPassCommandEncoder::ResourceBarrier(const Barrier& inBarrier)
    {
        commandEncoder.ResourceBarrier(inBarrier);
    }

    void VulkanCopyPassCommandEncoder::CopyBufferToBuffer(Buffer* inSrcBuffer, size_t inSrcOffset, Buffer* inDestBuffer, size_t inDestOffset, size_t inSize)
    {
        auto* srcBuffer = static_cast<VulkanBuffer*>(inSrcBuffer);
        auto* dstBuffer = static_cast<VulkanBuffer*>(inDestBuffer);

        VkBufferCopy copyRegion {};
        copyRegion.srcOffset = inSrcOffset;
        copyRegion.dstOffset = inDestOffset;
        copyRegion.srcOffset = inSize;
        vkCmdCopyBuffer(commandBuffer.GetNativeCommandBuffer(), srcBuffer->GetNative(), dstBuffer->GetNative(), 1, &copyRegion);
    }

    void VulkanCopyPassCommandEncoder::CopyBufferToTexture(Buffer* inSrcBuffer, Texture* inDestTexture, const TextureSubResourceInfo* inSubResourceInfo, const Common::UVec3& inSize)
    {
        auto* buffer = static_cast<VulkanBuffer*>(inSrcBuffer);
        auto* texture = static_cast<VulkanTexture*>(inDestTexture);

        VkBufferImageCopy copyRegion = {};
        copyRegion.imageExtent = {inSize.x, inSize.y, inSize.z };
        copyRegion.imageSubresource = {GetAspectMask(inSubResourceInfo->aspect), inSubResourceInfo->mipLevel, inSubResourceInfo->baseArrayLayer, inSubResourceInfo->arrayLayerNum };

        vkCmdCopyBufferToImage(commandBuffer.GetNativeCommandBuffer(), buffer->GetNative(), texture->GetNative(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
    }

    void VulkanCopyPassCommandEncoder::CopyTextureToBuffer(Texture* inSrcTexture, Buffer* inDestBuffer, const TextureSubResourceInfo* inSubResourceInfo, const Common::UVec3& inSize)
    {
        auto* buffer = static_cast<VulkanBuffer*>(inDestBuffer);
        auto* texture = static_cast<VulkanTexture*>(inSrcTexture);

        VkBufferImageCopy copyRegion = {};
        copyRegion.imageExtent = {inSize.x, inSize.y, inSize.z };
        copyRegion.imageSubresource = {GetAspectMask(inSubResourceInfo->aspect), inSubResourceInfo->mipLevel, inSubResourceInfo->baseArrayLayer, inSubResourceInfo->arrayLayerNum };

        vkCmdCopyImageToBuffer(commandBuffer.GetNativeCommandBuffer(), texture->GetNative(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               buffer->GetNative(), 1, &copyRegion);
    }

    void VulkanCopyPassCommandEncoder::CopyTextureToTexture(Texture* inSrcTexture, const TextureSubResourceInfo* inSrcSubResourceInfo,
                                                            Texture* inDestTexture, const TextureSubResourceInfo* inDestSubResourceInfo, const Common::UVec3& inSize)
    {
        auto* srcTexture = static_cast<VulkanTexture*>(inSrcTexture);
        auto* dstTexture = static_cast<VulkanTexture*>(inDestTexture);

        VkImageCopy copyRegion = {};
        copyRegion.extent = {inSize.x, inSize.y, inSize.z };
        copyRegion.srcSubresource = {GetAspectMask(inSrcSubResourceInfo->aspect), inSrcSubResourceInfo->mipLevel, inSrcSubResourceInfo->baseArrayLayer, inSrcSubResourceInfo->arrayLayerNum };
        copyRegion.dstSubresource = {GetAspectMask(inDestSubResourceInfo->aspect), inDestSubResourceInfo->mipLevel, inDestSubResourceInfo->baseArrayLayer, inDestSubResourceInfo->arrayLayerNum };

        vkCmdCopyImage(commandBuffer.GetNativeCommandBuffer(), srcTexture->GetNative(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       dstTexture->GetNative(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
    }

    void VulkanCopyPassCommandEncoder::EndPass()
    {
    }

    void VulkanCopyPassCommandEncoder::Destroy()
    {
        delete this;
    }

    VulkanComputePassCommandEncoder::VulkanComputePassCommandEncoder(VulkanDevice& inDevice, VulkanCommandEncoder& inCmdEncoder, VulkanCommandBuffer& inCmdBuffer)
        : device(inDevice)
        , commandEncoder(inCmdEncoder)
        , commandBuffer(inCmdBuffer)
    {
    }

    VulkanComputePassCommandEncoder::~VulkanComputePassCommandEncoder() = default;

    void VulkanComputePassCommandEncoder::ResourceBarrier(const Barrier& inBarrier)
    {
        commandEncoder.ResourceBarrier(inBarrier);
    }

    void VulkanComputePassCommandEncoder::SetPipeline(ComputePipeline* inPipeline)
    {
        // TODO
    }

    void VulkanComputePassCommandEncoder::SetBindGroup(uint8_t inLayoutIndex, BindGroup* inBindGroup)
    {
        // TODO
    }

    void VulkanComputePassCommandEncoder::Dispatch(size_t inGroupCountX, size_t inGroupCountY, size_t inGroupCountZ)
    {
        // TODO
    }

    void VulkanComputePassCommandEncoder::EndPass()
    {
        // TODO
    }

    void VulkanComputePassCommandEncoder::Destroy()
    {
        delete this;
    }

    VulkanGraphicsPassCommandEncoder::VulkanGraphicsPassCommandEncoder(VulkanDevice& inDevice, VulkanCommandEncoder& inCmdEncoder, VulkanCommandBuffer& inCmdBuffer, const GraphicsPassBeginInfo& inBeginInfo)
        : device(inDevice)
        , commandEncoder(inCmdEncoder)
        , commandBuffer(inCmdBuffer)
    {
        std::vector<VkRenderingAttachmentInfo> colorAttachmentInfos(inBeginInfo.colorAttachments.size());
        for (size_t i = 0; i < inBeginInfo.colorAttachments.size(); i++)
        {
            auto* colorTextureView = static_cast<VulkanTextureView*>(inBeginInfo.colorAttachments[i].view);
            colorAttachmentInfos[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            colorAttachmentInfos[i].imageView = colorTextureView->GetNative();
            colorAttachmentInfos[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorAttachmentInfos[i].loadOp = VKEnumCast<LoadOp, VkAttachmentLoadOp>(inBeginInfo.colorAttachments[i].loadOp);
            colorAttachmentInfos[i].storeOp = VKEnumCast<StoreOp, VkAttachmentStoreOp>(inBeginInfo.colorAttachments[i].storeOp);
            colorAttachmentInfos[i].clearValue.color = {
                inBeginInfo.colorAttachments[i].clearValue.r,
                inBeginInfo.colorAttachments[i].clearValue.g,
                inBeginInfo.colorAttachments[i].clearValue.b,
                inBeginInfo.colorAttachments[i].clearValue.a
                    };
        }

        auto* textureView = static_cast<VulkanTextureView*>(inBeginInfo.colorAttachments[0].view);
        VkRenderingInfoKHR renderingInfo = {};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.colorAttachmentCount = colorAttachmentInfos.size();
        renderingInfo.pColorAttachments = colorAttachmentInfos.data();
        renderingInfo.layerCount = textureView->GetArrayLayerNum();
        renderingInfo.renderArea = {{0, 0}, {static_cast<uint32_t>(textureView->GetTexture().GetExtent().x), static_cast<uint32_t>(textureView->GetTexture().GetExtent().y)}};
        renderingInfo.viewMask = 0;

        if (inBeginInfo.depthStencilAttachment.has_value())
        {
            auto* depthStencilTextureView = static_cast<VulkanTextureView*>(inBeginInfo.depthStencilAttachment->view);

            VkRenderingAttachmentInfo depthAttachmentInfo = {};
            depthAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depthAttachmentInfo.imageView = depthStencilTextureView->GetNative();
            depthAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAttachmentInfo.loadOp = VKEnumCast<LoadOp, VkAttachmentLoadOp>(inBeginInfo.depthStencilAttachment->depthLoadOp);
            depthAttachmentInfo.storeOp = VKEnumCast<StoreOp, VkAttachmentStoreOp>(inBeginInfo.depthStencilAttachment->depthStoreOp);
            depthAttachmentInfo.clearValue.depthStencil = {inBeginInfo.depthStencilAttachment->depthClearValue, inBeginInfo.depthStencilAttachment->stencilClearValue };

            renderingInfo.pDepthAttachment = &depthAttachmentInfo;

            if (!inBeginInfo.depthStencilAttachment->depthReadOnly) {
                VkRenderingAttachmentInfo stencilAttachmentInfo = {};
                stencilAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                stencilAttachmentInfo.imageView = depthStencilTextureView->GetNative();
                stencilAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                stencilAttachmentInfo.loadOp = VKEnumCast<LoadOp, VkAttachmentLoadOp>(inBeginInfo.depthStencilAttachment->stencilLoadOp);
                stencilAttachmentInfo.storeOp = VKEnumCast<StoreOp, VkAttachmentStoreOp>(inBeginInfo.depthStencilAttachment->stencilStoreOp);
                stencilAttachmentInfo.clearValue.depthStencil = {inBeginInfo.depthStencilAttachment->depthClearValue, inBeginInfo.depthStencilAttachment->stencilClearValue };

                renderingInfo.pStencilAttachment = &stencilAttachmentInfo;
            }
        }

        nativeCmdBuffer = inCmdBuffer.GetNativeCommandBuffer();
        device.GetGpu().GetInstance().pfnVkCmdBeginRenderingKHR(nativeCmdBuffer, &renderingInfo);
    }

    VulkanGraphicsPassCommandEncoder::~VulkanGraphicsPassCommandEncoder() = default;

    void VulkanGraphicsPassCommandEncoder::ResourceBarrier(const Barrier& inBarrier)
    {
        commandEncoder.ResourceBarrier(inBarrier);
    }

    void VulkanGraphicsPassCommandEncoder::SetPipeline(GraphicsPipeline* inPipeline)
    {
        graphicsPipeline = static_cast<VulkanGraphicsPipeline*>(inPipeline);
        Assert(graphicsPipeline);

       vkCmdBindPipeline(nativeCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline->GetNative());
    }

    void VulkanGraphicsPassCommandEncoder::SetBindGroup(uint8_t inLayoutIndex, BindGroup* inBindGroup)
    {
        auto* vBindGroup = static_cast<VulkanBindGroup*>(inBindGroup);
        VkDescriptorSet descriptorSet = vBindGroup->GetNative();
        VkPipelineLayout layout = graphicsPipeline->GetPipelineLayout()->GetNative();

        vkCmdBindDescriptorSets(nativeCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, inLayoutIndex, 1, &descriptorSet, 0, nullptr);
    }

    void VulkanGraphicsPassCommandEncoder::SetIndexBuffer(BufferView *inBufferView)
    {
        auto* mBufferView = static_cast<VulkanBufferView*>(inBufferView);

        VkBuffer indexBuffer = mBufferView->GetBuffer().GetNative();
        auto vkFormat = VKEnumCast<IndexFormat, VkIndexType>(mBufferView->GetIndexFormat());

        vkCmdBindIndexBuffer(nativeCmdBuffer, indexBuffer, 0, vkFormat);
    }

    void VulkanGraphicsPassCommandEncoder::SetVertexBuffer(size_t inSlot, BufferView *inBufferView)
    {
        auto* mBufferView = static_cast<VulkanBufferView*>(inBufferView);

        VkBuffer vertexBuffer = mBufferView->GetBuffer().GetNative();
        VkDeviceSize offset[] = { mBufferView->GetOffset() };
        vkCmdBindVertexBuffers(nativeCmdBuffer, inSlot, 1, &vertexBuffer, offset);
    }

    void VulkanGraphicsPassCommandEncoder::Draw(size_t inVertexCount, size_t inInstanceCount, size_t inFirstVertex, size_t inFirstInstance)
    {
        vkCmdDraw(nativeCmdBuffer, inVertexCount, inInstanceCount, inFirstVertex, inFirstInstance);
    }

    void VulkanGraphicsPassCommandEncoder::DrawIndexed(size_t inIndexCount, size_t inInstanceCount, size_t inFirstIndex, size_t inBaseVertex, size_t inFirstInstance)
    {
        vkCmdDrawIndexed(nativeCmdBuffer, inIndexCount, inInstanceCount, inFirstIndex, inBaseVertex, inFirstInstance);
    }

    void VulkanGraphicsPassCommandEncoder::SetViewport(float inX, float inY, float inWidth, float inHeight, float inMinDepth, float inMaxDepth)
    {
        VkViewport viewport{};
        viewport.x = inX;
        viewport.y = inY;
        viewport.width = inWidth;
        viewport.height = inHeight;
        viewport.minDepth = inMinDepth;
        viewport.maxDepth = inMaxDepth;
        vkCmdSetViewport(nativeCmdBuffer, 0, 1, &viewport);
    }

    void VulkanGraphicsPassCommandEncoder::SetScissor(uint32_t inLeft, uint32_t inTop, uint32_t inRight, uint32_t inBottom)
    {
        VkRect2D rect;
        rect.offset = {static_cast<int32_t>(inLeft), static_cast<int32_t>(inTop) };
        rect.extent = {inRight - inLeft, inBottom - inTop };
        vkCmdSetScissor(nativeCmdBuffer, 0, 1, &rect);
    }

    void VulkanGraphicsPassCommandEncoder::SetPrimitiveTopology(PrimitiveTopology inPrimitiveTopology)
    {
        // check extension
//        cmdHandle.setPrimitiveTopologyEXT(VKEnumCast<PrimitiveTopologyType, vk::PrimitiveTopology>(primitiveTopology)
    }

    void VulkanGraphicsPassCommandEncoder::SetBlendConstant(const float *inConstants)
    {
        vkCmdSetBlendConstants(nativeCmdBuffer, inConstants);
    }

    void VulkanGraphicsPassCommandEncoder::SetStencilReference(uint32_t inReference)
    {
        // TODO stencil face;
        vkCmdSetStencilReference(nativeCmdBuffer, VK_STENCIL_FACE_FRONT_AND_BACK, inReference);
    }

    void VulkanGraphicsPassCommandEncoder::EndPass()
    {
        device.GetGpu().GetInstance().pfnVkCmdEndRenderingKHR(nativeCmdBuffer);
    }

    void VulkanGraphicsPassCommandEncoder::Destroy()
    {
        delete this;
    }
}
