//
// Created by johnk on 2023/3/21.
//

#include <RHI/Dummy/CommandRecorder.h>

namespace RHI::Dummy {
    DummyCopyPassCommandRecorder::DummyCopyPassCommandRecorder(const DummyCommandBuffer& dummyCommandBuffer)
    {
    }

    DummyCopyPassCommandRecorder::~DummyCopyPassCommandRecorder()
    {
    }

    void DummyCopyPassCommandRecorder::ResourceBarrier(const Barrier& barrier)
    {
    }

    void DummyCopyPassCommandRecorder::CopyBufferToBuffer(Buffer* src, size_t srcOffset, Buffer* dst, size_t dstOffset, size_t size)
    {
    }

    void DummyCopyPassCommandRecorder::CopyBufferToTexture(Buffer* src, Texture* dst, const TextureSubResourceInfo* subResourceInfo, const Common::UVec3& size)
    {
    }

    void DummyCopyPassCommandRecorder::CopyTextureToBuffer(Texture* src, Buffer* dst, const TextureSubResourceInfo* subResourceInfo, const Common::UVec3& size)
    {
    }

    void DummyCopyPassCommandRecorder::CopyTextureToTexture(Texture* src, const TextureSubResourceInfo* srcSubResourceInfo, Texture* dst, const TextureSubResourceInfo* dstSubResourceInfo, const Common::UVec3& size)
    {
    }

    void DummyCopyPassCommandRecorder::EndPass()
    {
    }

    DummyComputePassCommandRecorder::DummyComputePassCommandRecorder(const DummyCommandBuffer& dummyCommandBuffer)
    {
    }

    DummyComputePassCommandRecorder::~DummyComputePassCommandRecorder() = default;

    void DummyComputePassCommandRecorder::ResourceBarrier(const Barrier& barrier)
    {
    }

    void DummyComputePassCommandRecorder::SetPipeline(ComputePipeline* pipeline)
    {
    }

    void DummyComputePassCommandRecorder::SetBindGroup(uint8_t layoutIndex, BindGroup* bindGroup)
    {
    }

    void DummyComputePassCommandRecorder::Dispatch(size_t groupCountX, size_t groupCountY, size_t groupCountZ)
    {
    }

    void DummyComputePassCommandRecorder::EndPass()
    {
    }

    DummyRasterPassCommandRecorder::DummyRasterPassCommandRecorder(const DummyCommandBuffer& dummyCommandBuffer)
    {
    }

    DummyRasterPassCommandRecorder::~DummyRasterPassCommandRecorder() = default;

    void DummyRasterPassCommandRecorder::ResourceBarrier(const Barrier& barrier)
    {
    }

    void DummyRasterPassCommandRecorder::SetPipeline(RasterPipeline* pipeline)
    {
    }

    void DummyRasterPassCommandRecorder::SetBindGroup(uint8_t layoutIndex, BindGroup* bindGroup)
    {
    }

    void DummyRasterPassCommandRecorder::SetIndexBuffer(BufferView* bufferView)
    {
    }

    void DummyRasterPassCommandRecorder::SetVertexBuffer(size_t slot, BufferView* bufferView)
    {
    }

    void DummyRasterPassCommandRecorder::Draw(size_t vertexCount, size_t instanceCount, size_t firstVertex, size_t firstInstance)
    {
    }

    void DummyRasterPassCommandRecorder::DrawIndexed(size_t indexCount, size_t instanceCount, size_t firstIndex, size_t baseVertex, size_t firstInstance)
    {
    }

    void DummyRasterPassCommandRecorder::SetViewport(float topLeftX, float topLeftY, float width, float height, float minDepth, float maxDepth)
    {
    }

    void DummyRasterPassCommandRecorder::SetScissor(uint32_t left, uint32_t top, uint32_t right, uint32_t bottom)
    {
    }

    void DummyRasterPassCommandRecorder::SetPrimitiveTopology(PrimitiveTopology primitiveTopology)
    {
    }

    void DummyRasterPassCommandRecorder::SetBlendConstant(const float* constants)
    {
    }

    void DummyRasterPassCommandRecorder::SetStencilReference(uint32_t reference)
    {
    }

    void DummyRasterPassCommandRecorder::EndPass()
    {
    }

    DummyCommandRecorder::DummyCommandRecorder(const DummyCommandBuffer& inDummyCommandBuffer)
        : dummyCommandBuffer(inDummyCommandBuffer)
    {
    }

    DummyCommandRecorder::~DummyCommandRecorder() = default;

    void DummyCommandRecorder::ResourceBarrier(const Barrier& barrier)
    {
    }

    Common::UniqueRef<CopyPassCommandRecorder> DummyCommandRecorder::BeginCopyPass()
    {
        return Common::UniqueRef<CopyPassCommandRecorder>(new DummyCopyPassCommandRecorder(dummyCommandBuffer));
    }

    Common::UniqueRef<ComputePassCommandRecorder> DummyCommandRecorder::BeginComputePass()
    {
        return Common::UniqueRef<ComputePassCommandRecorder>(new DummyComputePassCommandRecorder(dummyCommandBuffer));
    }

    Common::UniqueRef<RasterPassCommandRecorder> DummyCommandRecorder::BeginRasterPass(const RasterPassBeginInfo& beginInfo)
    {
        return Common::UniqueRef<RasterPassCommandRecorder>(new DummyRasterPassCommandRecorder(dummyCommandBuffer));
    }

    void DummyCommandRecorder::End()
    {
    }
}
