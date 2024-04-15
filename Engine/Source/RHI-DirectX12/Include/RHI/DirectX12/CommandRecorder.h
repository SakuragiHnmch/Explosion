//
// Created by johnk on 2022/3/23.
//

#pragma once

#include <RHI/CommandRecorder.h>

namespace RHI::DirectX12 {
    class DX12Device;
    class DX12CommandBuffer;
    class DX12ComputePipeline;
    class DX12RasterPipeline;
    class DX12PipelineLayout;

    class DX12CommandRecorder : public CommandRecorder {
    public:
        NonCopyable(DX12CommandRecorder)
        explicit DX12CommandRecorder(DX12Device& inDevice, DX12CommandBuffer& inCmdBuffer);
        ~DX12CommandRecorder() override;

        void ResourceBarrier(const Barrier& inBarrier) override;
        Common::UniqueRef<CopyPassCommandRecorder> BeginCopyPass() override;
        Common::UniqueRef<ComputePassCommandRecorder> BeginComputePass() override;
        Common::UniqueRef<RasterPassCommandRecorder> BeginRasterPass(const RasterPassBeginInfo& inBeginInfo) override;
        void End() override;

    private:
        DX12Device& device;
        DX12CommandBuffer& commandBuffer;
    };

    class DX12CopyPassCommandRecorder : public CopyPassCommandRecorder {
    public:
        NonCopyable(DX12CopyPassCommandRecorder)
        explicit DX12CopyPassCommandRecorder(DX12Device& inDevice, DX12CommandRecorder& inCmdRecorder, DX12CommandBuffer& inCmdBuffer);
        ~DX12CopyPassCommandRecorder() override;

        // CommandCommandRecorder
        void ResourceBarrier(const Barrier& inBarrier) override;

        // CopyPassCommandRecorder
        void CopyBufferToBuffer(Buffer* inSrcBuffer, size_t inSrcOffset, Buffer* inDestBuffer, size_t inDestOffset, size_t inSize) override;
        void CopyBufferToTexture(Buffer* inSrcBuffer, Texture* inDestTexture, const TextureSubResourceInfo* inSubResourceInfo, const Common::UVec3& inSize) override;
        void CopyTextureToBuffer(Texture* inSrcTexture, Buffer* inDestBuffer, const TextureSubResourceInfo* inSubResourceInfo, const Common::UVec3& inSize) override;
        void CopyTextureToTexture(Texture* inSrcTexture, const TextureSubResourceInfo* inSrcSubResourceInfo, Texture* inDestTexture, const TextureSubResourceInfo* inDestSubResourceInfo, const Common::UVec3& inSize) override;
        void EndPass() override;

    private:
        DX12Device& device;
        DX12CommandRecorder& commandRecorder;
        DX12CommandBuffer& commandBuffer;
    };

    class DX12ComputePassCommandRecorder : public ComputePassCommandRecorder {
    public:
        NonCopyable(DX12ComputePassCommandRecorder)
        explicit DX12ComputePassCommandRecorder(DX12Device& inDevice, DX12CommandRecorder& inCmdRecorder, DX12CommandBuffer& inCmdBuffer);
        ~DX12ComputePassCommandRecorder() override;

        // CommandCommandRecorder
        void ResourceBarrier(const Barrier& inBarrier) override;

        // ComputePassCommandRecorder
        void SetPipeline(ComputePipeline* inPipeline) override;
        void SetBindGroup(uint8_t inLayoutIndex, BindGroup* inBindGroup) override;
        void Dispatch(size_t inGroupCountX, size_t inGroupCountY, size_t inGroupCountZ) override;
        void EndPass() override;

    private:
        DX12Device& device;
        DX12CommandRecorder& commandRecorder;
        DX12ComputePipeline* computePipeline;
        DX12CommandBuffer& commandBuffer;
    };

    class DX12RasterPassCommandRecorder : public RasterPassCommandRecorder {
    public:
        NonCopyable(DX12RasterPassCommandRecorder)
        explicit DX12RasterPassCommandRecorder(DX12Device& inDevice, DX12CommandRecorder& inCmdRecorder, DX12CommandBuffer& inCmdBuffer, const RasterPassBeginInfo& inBeginInfo);
        ~DX12RasterPassCommandRecorder() override;

        // CommandCommandRecorder
        void ResourceBarrier(const Barrier& inBarrier) override;

        // RasterPassCommandRecorder
        void SetPipeline(RasterPipeline* inPipeline) override;
        void SetBindGroup(uint8_t inLayoutIndex, BindGroup* inBindGroup) override;
        void SetIndexBuffer(BufferView* inBufferView) override;
        void SetVertexBuffer(size_t inSlot, BufferView* inBufferView) override;
        void Draw(size_t inVertexCount, size_t inInstanceCount, size_t inFirstVertex, size_t inFirstInstance) override;
        void DrawIndexed(size_t inIndexCount, size_t inInstanceCount, size_t inFirstIndex, size_t inBaseVertex, size_t inFirstInstance) override;
        void SetViewport(float inX, float inY, float inWidth, float inHeight, float inMinDepth, float inMaxDepth) override;
        void SetScissor(uint32_t inLeft, uint32_t inTop, uint32_t inRight, uint32_t inBottom) override;
        void SetPrimitiveTopology(PrimitiveTopology inPrimitiveTopology) override;
        void SetBlendConstant(const float* inConstants) override;
        void SetStencilReference(uint32_t inReference) override;
        void EndPass() override;

    private:
        DX12Device& device;
        DX12CommandRecorder& commandRecorder;
        DX12RasterPipeline* rasterPipeline;
        DX12CommandBuffer& commandBuffer;
    };
}
