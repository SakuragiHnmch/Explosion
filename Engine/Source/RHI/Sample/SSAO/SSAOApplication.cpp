//
// Created by Junkang on 2023/3/8.
//

#include <vector>
#include <array>
#include <random>

#include <Application.h>
#include "GLTFParser.h"

using namespace RHI;

class SSAOApplication final : public Application {
public:
    NonCopyable(SSAOApplication)
    explicit SSAOApplication(const std::string& n) : Application(n) {}
    ~SSAOApplication() override = default;

protected:
    void OnCreate() override
    {
        InitCamera();
        LoadGLTF();
        SelectGPU();
        RequestDeviceAndFetchQueues();
        CreateSwapChain();
        CreateSyncObjects();
        CreateVertexBuffer();
        CreateIndexBuffer();
        CreateQuadBuffer();
        PrepareOffscreen();
        PrepareUniformBuffers();
        CreateShaderModules();
        CreateSampler();
        CreateBindGroupLayoutAndPipelineLayout();
        CreateBindGroup();
        CreatePipeline();
        CreateCommandBuffer();
        GenerateRenderables();
    }

    void OnDrawFrame() override
    {
        uboSceneParams.view = GetCamera().GetViewMatrix();

        auto* pMap = uniformBuffers.sceneParams.buf->Map(MapMode::write, 0, sizeof(UBOSceneParams));
        memcpy(pMap, &uboSceneParams, sizeof(UBOSceneParams));
        uniformBuffers.sceneParams.buf->UnMap();

        inflightFences[nextFrameIndex]->Wait();
        const auto backTextureIndex = swapChain->AcquireBackTexture(backBufferReadySemaphores[nextFrameIndex].Get());
        inflightFences[nextFrameIndex]->Reset();

        const UniqueRef<CommandRecorder> commandRecorder = commandBuffers[nextFrameIndex]->Begin();
        {
            commandRecorder->ResourceBarrier(Barrier::Transition(gBufferPos.texture.Get(), TextureState::shaderReadOnly, TextureState::renderTarget));
            commandRecorder->ResourceBarrier(Barrier::Transition(gBufferNormal.texture.Get(), TextureState::shaderReadOnly, TextureState::renderTarget));
            commandRecorder->ResourceBarrier(Barrier::Transition(gBufferAlbedo.texture.Get(), TextureState::shaderReadOnly, TextureState::renderTarget));
            commandRecorder->ResourceBarrier(Barrier::Transition(gBufferDepth.texture.Get(), TextureState::depthStencilReadonly, TextureState::depthStencilWrite));

            const UniqueRef<RasterPassCommandRecorder> rasterRecorder = commandRecorder->BeginRasterPass(
                RasterPassBeginInfo()
                    .AddColorAttachment(RHI::ColorAttachment(gBufferPos.rtv.Get(), LoadOp::clear, StoreOp::store, LinearColorConsts::black))
                    .AddColorAttachment(RHI::ColorAttachment(gBufferNormal.rtv.Get(), LoadOp::clear, StoreOp::store, LinearColorConsts::black))
                    .AddColorAttachment(RHI::ColorAttachment(gBufferAlbedo.rtv.Get(), LoadOp::clear, StoreOp::store, LinearColorConsts::black))
                    .SetDepthStencilAttachment(DepthStencilAttachment(gBufferDepth.view.Get(), true, LoadOp::clear, StoreOp::store, 0.0f)));
            {
                rasterRecorder->SetPipeline(pipelines.gBuffer.Get());
                rasterRecorder->SetScissor(0, 0, GetWindowWidth(), GetWindowHeight());
                rasterRecorder->SetViewport(0, 0, static_cast<float>(GetWindowWidth()), static_cast<float>(GetWindowHeight()), 0, 1);
                rasterRecorder->SetPrimitiveTopology(PrimitiveTopology::triangleList);
                rasterRecorder->SetBindGroup(0, bindGroups.scene.Get());
                rasterRecorder->SetVertexBuffer(0, vertexBufferView.Get());
                rasterRecorder->SetIndexBuffer(indexBufferView.Get());

                for (const auto& renderable : renderables) {
                    rasterRecorder->SetBindGroup(1, renderable->bindGroup.Get());
                    rasterRecorder->DrawIndexed(renderable->indexCount, 1, renderable->firstIndex, 0, 0);
                }
            }
            rasterRecorder->EndPass();

            commandRecorder->ResourceBarrier(Barrier::Transition(gBufferPos.texture.Get(), TextureState::renderTarget, TextureState::shaderReadOnly));
            commandRecorder->ResourceBarrier(Barrier::Transition(gBufferNormal.texture.Get(), TextureState::renderTarget, TextureState::shaderReadOnly));
            commandRecorder->ResourceBarrier(Barrier::Transition(gBufferAlbedo.texture.Get(), TextureState::renderTarget, TextureState::shaderReadOnly));
            commandRecorder->ResourceBarrier(Barrier::Transition(gBufferDepth.texture.Get(), TextureState::depthStencilWrite, TextureState::depthStencilReadonly));
        }

        {
            // ssao
            commandRecorder->ResourceBarrier(Barrier::Transition(ssaoOutput.texture.Get(), TextureState::shaderReadOnly, TextureState::renderTarget));

            const UniqueRef<RasterPassCommandRecorder> rasterRecorder = commandRecorder->BeginRasterPass(
                RasterPassBeginInfo()
                    .AddColorAttachment(RHI::ColorAttachment(ssaoOutput.rtv.Get(), LoadOp::clear, StoreOp::store, LinearColorConsts::black)));
            {
                rasterRecorder->SetPipeline(pipelines.ssao.Get());
                rasterRecorder->SetScissor(0, 0, GetWindowWidth(), GetWindowHeight());
                rasterRecorder->SetViewport(0, 0, static_cast<float>(GetWindowWidth()), static_cast<float>(GetWindowHeight()), 0, 1);
                rasterRecorder->SetPrimitiveTopology(PrimitiveTopology::triangleList);
                rasterRecorder->SetBindGroup(0, bindGroups.ssao.Get());
                rasterRecorder->SetVertexBuffer(0, quadVertexBufferView.Get());
                rasterRecorder->SetIndexBuffer(quadIndexBufferView.Get());
                rasterRecorder->DrawIndexed(6, 1, 0, 0, 0);
            }
            rasterRecorder->EndPass();
            commandRecorder->ResourceBarrier(Barrier::Transition(ssaoOutput.texture.Get(), TextureState::renderTarget, TextureState::shaderReadOnly));
        }

        {
            // ssaoBlur
            commandRecorder->ResourceBarrier(Barrier::Transition(ssaoBlurOutput.texture.Get(), TextureState::shaderReadOnly, TextureState::renderTarget));

            const UniqueRef<RasterPassCommandRecorder> rasterRecorder = commandRecorder->BeginRasterPass(
                RasterPassBeginInfo()
                    .AddColorAttachment(RHI::ColorAttachment(ssaoBlurOutput.rtv.Get(), LoadOp::clear, StoreOp::store, LinearColorConsts::black)));
            {
                rasterRecorder->SetPipeline(pipelines.ssaoBlur.Get());
                rasterRecorder->SetScissor(0, 0, GetWindowWidth(), GetWindowHeight());
                rasterRecorder->SetViewport(0, 0, static_cast<float>(GetWindowWidth()), static_cast<float>(GetWindowHeight()), 0, 1);
                rasterRecorder->SetPrimitiveTopology(PrimitiveTopology::triangleList);
                rasterRecorder->SetBindGroup(0, bindGroups.ssaoBlur.Get());
                rasterRecorder->SetVertexBuffer(0, quadVertexBufferView.Get());
                rasterRecorder->SetIndexBuffer(quadIndexBufferView.Get());
                rasterRecorder->DrawIndexed(6, 1, 0, 0, 0);
            }
            rasterRecorder->EndPass();
            commandRecorder->ResourceBarrier(Barrier::Transition(ssaoBlurOutput.texture.Get(), TextureState::renderTarget, TextureState::shaderReadOnly));
        }

        {
            // composition
            commandRecorder->ResourceBarrier(Barrier::Transition(swapChainTextures[backTextureIndex], TextureState::present, TextureState::renderTarget));
            const UniqueRef<RasterPassCommandRecorder> rasterRecorder = commandRecorder->BeginRasterPass(
                RasterPassBeginInfo()
                    .AddColorAttachment(RHI::ColorAttachment(swapChainTextureViews[backTextureIndex].Get(), LoadOp::clear, StoreOp::store, LinearColorConsts::black)));
            {
                rasterRecorder->SetPipeline(pipelines.composition.Get());
                rasterRecorder->SetScissor(0, 0, GetWindowWidth(), GetWindowHeight());
                rasterRecorder->SetViewport(0, 0, static_cast<float>(GetWindowWidth()), static_cast<float>(GetWindowHeight()), 0, 1);
                rasterRecorder->SetPrimitiveTopology(PrimitiveTopology::triangleList);
                rasterRecorder->SetBindGroup(0, bindGroups.composition.Get());
                rasterRecorder->SetVertexBuffer(0, quadVertexBufferView.Get());
                rasterRecorder->SetIndexBuffer(quadIndexBufferView.Get());
                rasterRecorder->DrawIndexed(6, 1, 0, 0, 0);
            }
            rasterRecorder->EndPass();
            commandRecorder->ResourceBarrier(Barrier::Transition(swapChainTextures[backTextureIndex], TextureState::renderTarget, TextureState::present));
        }

        commandRecorder->End();

        graphicsQueue->Submit(
            commandBuffers[nextFrameIndex].Get(), QueueSubmitInfo()
                .AddWaitSemaphore(backBufferReadySemaphores[nextFrameIndex].Get())
                .AddSignalSemaphore(renderFinishedSemaphores[nextFrameIndex].Get())
                .SetSignalFence(inflightFences[nextFrameIndex].Get()));

        swapChain->Present(renderFinishedSemaphores[nextFrameIndex].Get());
        nextFrameIndex = (nextFrameIndex + 1) % backBufferCount;
    }

    void OnDestroy() override
    {
        const UniqueRef<Fence> fence = device->CreateFence(false);
        graphicsQueue->Flush(fence.Get());
        fence->Wait();
    }

private:
    static constexpr uint8_t ssaoKernelSize = 64;
    static constexpr uint8_t ssaoNoiseDim = 16;
    static constexpr uint8_t backBufferCount = 2;

    PixelFormat swapChainFormat = PixelFormat::max;
    Gpu* gpu = nullptr;
    UniqueRef<Device> device = nullptr;
    Queue* graphicsQueue = nullptr;
    UniqueRef<Surface> surface = nullptr;
    UniqueRef<SwapChain> swapChain = nullptr;
    UniqueRef<Buffer> vertexBuffer = nullptr;
    UniqueRef<BufferView> vertexBufferView = nullptr;
    UniqueRef<Buffer> indexBuffer = nullptr;
    UniqueRef<BufferView> indexBufferView = nullptr;
    std::array<Texture*, backBufferCount> swapChainTextures {};
    std::array<UniqueRef<TextureView>, backBufferCount> swapChainTextureViews {};

    UniqueRef<Buffer> quadVertexBuffer = nullptr;
    UniqueRef<BufferView> quadVertexBufferView = nullptr;
    UniqueRef<Buffer> quadIndexBuffer = nullptr;
    UniqueRef<BufferView> quadIndexBufferView = nullptr;

    std::array<UniqueRef<CommandBuffer>, backBufferCount> commandBuffers {};
    std::array<UniqueRef<Semaphore>, backBufferCount> backBufferReadySemaphores {};
    std::array<UniqueRef<Semaphore>, backBufferCount> renderFinishedSemaphores {};
    std::array<UniqueRef<Fence>, backBufferCount> inflightFences {};
    uint8_t nextFrameIndex = 0;

    UniqueRef<Sampler> sampler = nullptr;
    UniqueRef<Sampler> noiseSampler = nullptr;

    struct Renderable {
        uint32_t indexCount;
        uint32_t firstIndex;

        UniqueRef<BindGroup> bindGroup;
        UniqueRef<Texture> diffuseColorMap;
        UniqueRef<TextureView> diffuseColorMapView;

        Renderable(Instance& instance, Device& device, BindGroupLayout& bindGroupLayout, Render::ShaderReflectionData& gBufferPsReflectionData, Sampler& sampler, const Mesh& mesh) {
            indexCount = mesh.indexCount;
            firstIndex = mesh.firstIndex;

            // upload diffuseColorMap
            const auto& texData = mesh.materialData->baseColorTexture;

            const BufferCreateInfo bufferCreateInfo = BufferCreateInfo()
                .SetSize(texData->GetSize())
                .SetUsages(BufferUsageBits::uniform | BufferUsageBits::mapWrite | BufferUsageBits::copySrc)
                .SetInitialState(BufferState::staging);

            const UniqueRef<Buffer> pixelBuffer = device.CreateBuffer(bufferCreateInfo);
            if (pixelBuffer != nullptr) {
                auto* mapData = pixelBuffer->Map(MapMode::write, 0, bufferCreateInfo.size);
                memcpy(mapData, texData->buffer.data(), bufferCreateInfo.size);
                pixelBuffer->UnMap();
            }

            diffuseColorMap = device.CreateTexture(
                TextureCreateInfo()
                    .SetFormat(PixelFormat::rgba8Unorm)
                    .SetDimension(TextureDimension::t2D)
                    .SetMipLevels(1)
                    .SetWidth(texData->width)
                    .SetHeight(texData->height)
                    .SetDepthOrArraySize(1)
                    .SetSamples(1)
                    .SetUsages(TextureUsageBits::copyDst | TextureUsageBits::textureBinding)
                    .SetInitialState(TextureState::undefined));

            diffuseColorMapView = diffuseColorMap->CreateTextureView(
                TextureViewCreateInfo()
                    .SetDimension(TextureViewDimension::tv2D)
                    .SetMipLevels(0, 1)
                    .SetArrayLayers(0, 1)
                    .SetAspect(TextureAspect::color)
                    .SetType(TextureViewType::textureBinding));

            const UniqueRef<CommandBuffer> texCommandBuffer = device.CreateCommandBuffer();

            const UniqueRef<CommandRecorder> commandRecorder = texCommandBuffer->Begin();
            {
                const UniqueRef<CopyPassCommandRecorder> copyRecorder = commandRecorder->BeginCopyPass();
                {
                    copyRecorder->ResourceBarrier(Barrier::Transition(diffuseColorMap.Get(), TextureState::undefined, TextureState::copyDst));
                    copyRecorder->CopyBufferToTexture(
                        pixelBuffer.Get(),
                        diffuseColorMap.Get(),
                        BufferTextureCopyInfo(0, TextureSubResourceInfo(), UVec3Consts::zero, UVec3(texData->width, texData->height, 1)));
                    copyRecorder->ResourceBarrier(Barrier::Transition(diffuseColorMap.Get(), TextureState::copyDst, TextureState::shaderReadOnly));
                }
                copyRecorder->EndPass();
            }
            commandRecorder->End();

            const UniqueRef<Fence> fence = device.CreateFence(false);
            QueueSubmitInfo submitInfo {};
            submitInfo.signalFence = fence.Get();
            device.GetQueue(QueueType::graphics, 0)->Submit(texCommandBuffer.Get(), submitInfo);
            fence->Wait();

            // per renderable bindGroup
            bindGroup = device.CreateBindGroup(
                BindGroupCreateInfo(&bindGroupLayout)
                    .AddEntry(BindGroupEntry(gBufferPsReflectionData.QueryResourceBindingChecked("colorTex").second, diffuseColorMapView.Get()))
                    .AddEntry(BindGroupEntry(gBufferPsReflectionData.QueryResourceBindingChecked("colorSampler").second, &sampler)));
        }
    };

    UniqueRef<Model> model = nullptr;
    std::vector<UniqueRef<Renderable>> renderables;

    struct UBuffer {
        UniqueRef<Buffer> buf;
        UniqueRef<BufferView> bufView;
    };

    struct UniformBuffers {
        UBuffer sceneParams;
        UBuffer ssaoKernel;
        UBuffer ssaoParams;

    } uniformBuffers;

    struct UBOSceneParams {
        FMat4x4 projection;
        FMat4x4 model;
        FMat4x4 view;
        float nearPlane = 0.1f;
        float farPlane = 64.0f;
    } uboSceneParams;

    struct UBOSSAOParams {
        FMat4x4 projection;
        int32_t ssao = 1;
        int32_t ssaoOnly = 0;
        int32_t ssaoBlur = 1;
    } ubossaoParams;

    struct Noise {
        UniqueRef<Texture> tex;
        UniqueRef<TextureView> view;
    } noise;

    struct ShaderObjects {
        UniqueRef<ShaderModule> gBufferVs;
        UniqueRef<ShaderModule> gBufferPs;
        UniqueRef<ShaderModule> ssaoVs;
        UniqueRef<ShaderModule> ssaoPs;
        UniqueRef<ShaderModule> ssaoBlurVs;
        UniqueRef<ShaderModule> ssaoBlurPs;
        UniqueRef<ShaderModule> compositionVs;
        UniqueRef<ShaderModule> compositionPs;
        ShaderCompileOutput gBufferVsCompileOutput;
        ShaderCompileOutput gBufferPsCompileOutput;
        ShaderCompileOutput ssaoVsCompileOutput;
        ShaderCompileOutput ssaoPsCompileOutput;
        ShaderCompileOutput ssaoBlurVsCompileOutput;
        ShaderCompileOutput ssaoBlurPsCompileOutput;
        ShaderCompileOutput compositionVsCompileOutput;
        ShaderCompileOutput compositionPsCompileOutput;
    } shaderObjects;

    struct Pipelines {
        UniqueRef<RasterPipeline> gBuffer;
        UniqueRef<RasterPipeline> ssao;
        UniqueRef<RasterPipeline> ssaoBlur;
        UniqueRef<RasterPipeline> composition;
    } pipelines;

    struct PipelineLayouts {
        UniqueRef<PipelineLayout> gBuffer;
        UniqueRef<PipelineLayout> ssao;
        UniqueRef<PipelineLayout> ssaoBlur;
        UniqueRef<PipelineLayout> composition;
    } pipelineLayouts;

    UniqueRef<BindGroupLayout> renderableLayout;

    struct BindGroupLayouts {
        UniqueRef<BindGroupLayout> gBuffer;
        UniqueRef<BindGroupLayout> ssao;
        UniqueRef<BindGroupLayout> ssaoBlur;
        UniqueRef<BindGroupLayout> composition;
    } bindGroupLayouts;

    struct BindGroups {
        UniqueRef<BindGroup> scene;
        UniqueRef<BindGroup> ssao;
        UniqueRef<BindGroup> ssaoBlur;
        UniqueRef<BindGroup> composition;
    } bindGroups;

    struct ColorAttachment {
        UniqueRef<Texture> texture;
        UniqueRef<TextureView> rtv;
        UniqueRef<TextureView> srv;
    };

    ColorAttachment gBufferPos;
    ColorAttachment gBufferNormal;
    ColorAttachment gBufferAlbedo;
    struct {
        UniqueRef<Texture> texture;
        UniqueRef<TextureView> view;
    } gBufferDepth;

    ColorAttachment ssaoOutput;
    ColorAttachment ssaoBlurOutput;

    struct QuadVertex {
        FVec3 pos;
        FVec2 uv;
    };

    void SelectGPU()
    {
        gpu = GetRHIInstance()->GetGpu(0);
    }

    void RequestDeviceAndFetchQueues()
    {
        device = gpu->RequestDevice(
            DeviceCreateInfo()
                .AddQueueRequest(QueueRequestInfo(QueueType::graphics, 1)));
        graphicsQueue = device->GetQueue(QueueType::graphics, 0);
    }

    void CreateSwapChain()
    {
        static std::vector<PixelFormat> swapChainFormatQualifiers = {
            PixelFormat::rgba8Unorm,
            PixelFormat::bgra8Unorm
        };

        surface = device->CreateSurface(SurfaceCreateInfo(GetPlatformWindow()));

        for (const auto format : swapChainFormatQualifiers) {
            if (device->CheckSwapChainFormatSupport(surface.Get(), format)) {
                swapChainFormat = format;
                break;
            }
        }
        Assert(swapChainFormat != PixelFormat::max);

        swapChain = device->CreateSwapChain(
            SwapChainCreateInfo()
                .SetFormat(swapChainFormat)
                .SetPresentMode(PresentMode::immediately)
                .SetTextureNum(backBufferCount)
                .SetWidth(GetWindowWidth())
                .SetHeight(GetWindowHeight())
                .SetSurface(surface.Get())
                .SetPresentQueue(graphicsQueue));

        for (auto i = 0; i < backBufferCount; i++) {
            swapChainTextures[i] = swapChain->GetTexture(i);

            swapChainTextureViews[i] = swapChainTextures[i]->CreateTextureView(
                TextureViewCreateInfo()
                    .SetDimension(TextureViewDimension::tv2D)
                    .SetMipLevels(0, 1)
                    .SetArrayLayers(0, 1)
                    .SetAspect(TextureAspect::color)
                    .SetType(TextureViewType::colorAttachment));
        }
    }

    void CreateVertexBuffer()
    {
        const BufferCreateInfo bufferCreateInfo = BufferCreateInfo()
            .SetSize(model->raw_vertex_buffer.size() * sizeof(Vertex))
            .SetUsages(BufferUsageBits::vertex | BufferUsageBits::mapWrite | BufferUsageBits::copySrc)
            .SetInitialState(BufferState::staging);

        vertexBuffer = device->CreateBuffer(bufferCreateInfo);
        assert(vertexBuffer != nullptr);
        auto* data = vertexBuffer->Map(MapMode::write, 0, bufferCreateInfo.size);
        memcpy(data, model->raw_vertex_buffer.data(), bufferCreateInfo.size);
        vertexBuffer->UnMap();

        const BufferViewCreateInfo bufferViewCreateInfo = BufferViewCreateInfo()
            .SetType(BufferViewType::vertex)
            .SetSize(bufferCreateInfo.size)
            .SetOffset(0)
            .SetExtendVertex(sizeof(Vertex));
        vertexBufferView = vertexBuffer->CreateBufferView(bufferViewCreateInfo);
    }

    void CreateIndexBuffer()
    {
        const BufferCreateInfo bufferCreateInfo = BufferCreateInfo()
            .SetSize(model->raw_index_buffer.size() * sizeof(uint32_t))
            .SetUsages(BufferUsageBits::index | BufferUsageBits::mapWrite | BufferUsageBits::copySrc)
            .SetInitialState(BufferState::staging);

        indexBuffer = device->CreateBuffer(bufferCreateInfo);
        assert(indexBuffer != nullptr);
        auto* data = indexBuffer->Map(MapMode::write, 0, bufferCreateInfo.size);
        memcpy(data, model->raw_index_buffer.data(), bufferCreateInfo.size);
        indexBuffer->UnMap();

        const BufferViewCreateInfo bufferViewCreateInfo = BufferViewCreateInfo()
            .SetType(BufferViewType::index)
            .SetSize(bufferCreateInfo.size)
            .SetOffset(0)
            .SetExtendIndex(IndexFormat::uint32);
        indexBufferView = indexBuffer->CreateBufferView(bufferViewCreateInfo);
    }

    void CreateQuadBuffer()
    {
        // quad vertex buffer
        const std::vector<QuadVertex> vertices {
            {{-1.0f, -1.0f, .0f}, {0.f, 1.0f}},
            {{1.0f, -1.0f, .0f}, {1.0f, 1.0f}},
            {{1.0f, 1.0f, .0f}, {1.0f, .0f}},
            {{-1.0f, 1.0f, .0f}, {0.f, .0f}},
        };

        BufferCreateInfo bufferCreateInfo = BufferCreateInfo()
            .SetSize(vertices.size() * sizeof(QuadVertex))
            .SetUsages(BufferUsageBits::vertex | BufferUsageBits::mapWrite | BufferUsageBits::copySrc)
            .SetInitialState(BufferState::staging);

        quadVertexBuffer = device->CreateBuffer(bufferCreateInfo);
        if (quadVertexBuffer != nullptr) {
            auto* data = quadVertexBuffer->Map(MapMode::write, 0, bufferCreateInfo.size);
            memcpy(data, vertices.data(), bufferCreateInfo.size);
            quadVertexBuffer->UnMap();
        }

        BufferViewCreateInfo bufferViewCreateInfo = BufferViewCreateInfo()
            .SetType(BufferViewType::vertex)
            .SetSize(vertices.size() * sizeof(QuadVertex))
            .SetOffset(0)
            .SetExtendVertex(sizeof(QuadVertex));
        quadVertexBufferView = quadVertexBuffer->CreateBufferView(bufferViewCreateInfo);

        // index buffer
        const std::vector<uint32_t> indices = {0, 1, 2, 0, 2, 3};
        bufferCreateInfo = BufferCreateInfo()
            .SetSize(indices.size() * sizeof(uint32_t))
            .SetUsages(BufferUsageBits::index | BufferUsageBits::mapWrite | BufferUsageBits::copySrc)
            .SetInitialState(BufferState::staging);

        quadIndexBuffer = device->CreateBuffer(bufferCreateInfo);
        if (quadIndexBuffer != nullptr) {
            auto* data = quadIndexBuffer->Map(MapMode::write, 0, bufferCreateInfo.size);
            memcpy(data, indices.data(), bufferCreateInfo.size);
            quadIndexBuffer->UnMap();
        }

        bufferViewCreateInfo = BufferViewCreateInfo()
            .SetType(BufferViewType::index)
            .SetSize(indices.size() * sizeof(uint32_t))
            .SetOffset(0)
            .SetExtendIndex(IndexFormat::uint32);
        quadIndexBufferView = quadIndexBuffer->CreateBufferView(bufferViewCreateInfo);
    }

    void CreateShaderModules()
    {
        CompileShaderAndCreateShaderModule(shaderObjects.gBufferVs, shaderObjects.gBufferVsCompileOutput, "../Test/Sample/RHI/SSAO/Shader/Gbuffer.hlsl", "VSMain", ShaderStageBits::sVertex);
        CompileShaderAndCreateShaderModule(shaderObjects.gBufferPs, shaderObjects.gBufferPsCompileOutput, "../Test/Sample/RHI/SSAO/Shader/Gbuffer.hlsl", "PSMain", ShaderStageBits::sPixel);
        CompileShaderAndCreateShaderModule(shaderObjects.ssaoVs, shaderObjects.ssaoVsCompileOutput, "../Test/Sample/RHI/SSAO/Shader/SSAO.hlsl", "VSMain", ShaderStageBits::sVertex);
        CompileShaderAndCreateShaderModule(shaderObjects.ssaoPs, shaderObjects.ssaoPsCompileOutput, "../Test/Sample/RHI/SSAO/Shader/SSAO.hlsl", "PSMain", ShaderStageBits::sPixel);
        CompileShaderAndCreateShaderModule(shaderObjects.ssaoBlurVs, shaderObjects.ssaoBlurVsCompileOutput, "../Test/Sample/RHI/SSAO/Shader/Blur.hlsl", "VSMain", ShaderStageBits::sVertex);
        CompileShaderAndCreateShaderModule(shaderObjects.ssaoBlurPs, shaderObjects.ssaoBlurPsCompileOutput, "../Test/Sample/RHI/SSAO/Shader/Blur.hlsl", "PSMain", ShaderStageBits::sPixel);
        CompileShaderAndCreateShaderModule(shaderObjects.compositionVs, shaderObjects.compositionVsCompileOutput, "../Test/Sample/RHI/SSAO/Shader/Composition.hlsl", "VSMain", ShaderStageBits::sVertex);
        CompileShaderAndCreateShaderModule(shaderObjects.compositionPs, shaderObjects.compositionPsCompileOutput, "../Test/Sample/RHI/SSAO/Shader/Composition.hlsl", "PSMain", ShaderStageBits::sPixel);
    }

    void CreateSampler()
    {
        sampler = device->CreateSampler(SamplerCreateInfo());
    }

    void CreateCommandBuffer()
    {
        for (auto i = 0; i < backBufferCount; i++) {
            commandBuffers[i] = device->CreateCommandBuffer();
        }
    }

    void CreateSyncObjects()
    {
        for (auto i = 0; i < backBufferCount; i++) {
            backBufferReadySemaphores[i] = device->CreateSemaphore();
            renderFinishedSemaphores[i] = device->CreateSemaphore();
            inflightFences[i] = device->CreateFence(true);
        }
    }

    void CreateBindGroupLayoutAndPipelineLayout()
    {
        const auto& gBufferPsReflectionData = shaderObjects.gBufferPsCompileOutput.reflectionData;
        const auto& ssaoPsReflectionData = shaderObjects.ssaoPsCompileOutput.reflectionData;
        const auto& ssaoBlurPsReflectionData = shaderObjects.ssaoBlurPsCompileOutput.reflectionData;
        const auto& compositionPsReflectionData = shaderObjects.compositionPsCompileOutput.reflectionData;

        //gBuffer
        {
            bindGroupLayouts.gBuffer = device->CreateBindGroupLayout(
                BindGroupLayoutCreateInfo(0)
                    .AddEntry(BindGroupLayoutEntry(gBufferPsReflectionData.QueryResourceBindingChecked("passParams").second, ShaderStageBits::sVertex | ShaderStageBits::sPixel)));
        }

        // renderable layout
        {
            renderableLayout = device->CreateBindGroupLayout(
                BindGroupLayoutCreateInfo(1)
                    .AddEntry(BindGroupLayoutEntry(gBufferPsReflectionData.QueryResourceBindingChecked("colorTex").second, ShaderStageBits::sPixel))
                    .AddEntry(BindGroupLayoutEntry(gBufferPsReflectionData.QueryResourceBindingChecked("colorSampler").second, ShaderStageBits::sPixel)));
        }

        pipelineLayouts.gBuffer = device->CreatePipelineLayout(
            PipelineLayoutCreateInfo()
                .AddBindGroupLayout(bindGroupLayouts.gBuffer.Get())
                .AddBindGroupLayout(renderableLayout.Get()));

        //ssao
        {
            bindGroupLayouts.ssao = device->CreateBindGroupLayout(
                BindGroupLayoutCreateInfo(0)
                    .AddEntry(BindGroupLayoutEntry(ssaoPsReflectionData.QueryResourceBindingChecked("posDepthTex").second, ShaderStageBits::sPixel))
                    .AddEntry(BindGroupLayoutEntry(ssaoPsReflectionData.QueryResourceBindingChecked("normalTex").second, ShaderStageBits::sPixel))
                    .AddEntry(BindGroupLayoutEntry(ssaoPsReflectionData.QueryResourceBindingChecked("ssaoNoiseTex").second, ShaderStageBits::sPixel))
                    .AddEntry(BindGroupLayoutEntry(ssaoPsReflectionData.QueryResourceBindingChecked("texSampler").second, ShaderStageBits::sPixel))
                    .AddEntry(BindGroupLayoutEntry(ssaoPsReflectionData.QueryResourceBindingChecked("ssaoNoiseSampler").second, ShaderStageBits::sPixel))
                    .AddEntry(BindGroupLayoutEntry(ssaoPsReflectionData.QueryResourceBindingChecked("kernalParams").second, ShaderStageBits::sPixel))
                    .AddEntry(BindGroupLayoutEntry(ssaoPsReflectionData.QueryResourceBindingChecked("passParams").second, ShaderStageBits::sPixel)));
        }

        pipelineLayouts.ssao = device->CreatePipelineLayout(
            PipelineLayoutCreateInfo()
                .AddBindGroupLayout(bindGroupLayouts.ssao.Get()));

        // ssaoBlur
        {
            bindGroupLayouts.ssaoBlur = device->CreateBindGroupLayout(
                BindGroupLayoutCreateInfo(0)
                    .AddEntry(BindGroupLayoutEntry(ssaoBlurPsReflectionData.QueryResourceBindingChecked("ssaoTex").second, ShaderStageBits::sPixel))
                    .AddEntry(BindGroupLayoutEntry(ssaoBlurPsReflectionData.QueryResourceBindingChecked("ssaoSampler").second, ShaderStageBits::sPixel)));
        }

        pipelineLayouts.ssaoBlur = device->CreatePipelineLayout(
            PipelineLayoutCreateInfo()
                .AddBindGroupLayout(bindGroupLayouts.ssaoBlur.Get()));

        // composition
        {
            bindGroupLayouts.composition = device->CreateBindGroupLayout(
                BindGroupLayoutCreateInfo(0)
                    .AddEntry(BindGroupLayoutEntry(compositionPsReflectionData.QueryResourceBindingChecked("posTex").second, ShaderStageBits::sPixel))
                    .AddEntry(BindGroupLayoutEntry(compositionPsReflectionData.QueryResourceBindingChecked("normalTex").second, ShaderStageBits::sPixel))
                    .AddEntry(BindGroupLayoutEntry(compositionPsReflectionData.QueryResourceBindingChecked("albedoTex").second, ShaderStageBits::sPixel))
                    .AddEntry(BindGroupLayoutEntry(compositionPsReflectionData.QueryResourceBindingChecked("ssaoTex").second, ShaderStageBits::sPixel))
                    .AddEntry(BindGroupLayoutEntry(compositionPsReflectionData.QueryResourceBindingChecked("ssaoBluredTex").second, ShaderStageBits::sPixel))
                    .AddEntry(BindGroupLayoutEntry(compositionPsReflectionData.QueryResourceBindingChecked("texSampler").second, ShaderStageBits::sPixel))
                    .AddEntry(BindGroupLayoutEntry(compositionPsReflectionData.QueryResourceBindingChecked("passParams").second, ShaderStageBits::sPixel)));
        }

        pipelineLayouts.composition = device->CreatePipelineLayout(
            PipelineLayoutCreateInfo()
                .AddBindGroupLayout(bindGroupLayouts.composition.Get()));
    }

    void CreateBindGroup()
    {
        const auto& gBufferPsReflectionData = shaderObjects.gBufferPsCompileOutput.reflectionData;
        const auto& ssaoPsReflectionData = shaderObjects.ssaoPsCompileOutput.reflectionData;
        const auto& ssaoBlurPsReflectionData = shaderObjects.ssaoBlurPsCompileOutput.reflectionData;
        const auto& compositionPsReflectionData = shaderObjects.compositionPsCompileOutput.reflectionData;

        // GBuffer scene
        {
            bindGroups.scene = device->CreateBindGroup(
                BindGroupCreateInfo(bindGroupLayouts.gBuffer.Get())
                    .AddEntry(BindGroupEntry(gBufferPsReflectionData.QueryResourceBindingChecked("passParams").second, uniformBuffers.sceneParams.bufView.Get())));
        }

        // ssao generation
        {
            bindGroups.ssao = device->CreateBindGroup(
                BindGroupCreateInfo(bindGroupLayouts.ssao.Get())
                    .AddEntry(BindGroupEntry(ssaoPsReflectionData.QueryResourceBindingChecked("posDepthTex").second, gBufferPos.srv.Get()))
                    .AddEntry(BindGroupEntry(ssaoPsReflectionData.QueryResourceBindingChecked("normalTex").second, gBufferNormal.srv.Get()))
                    .AddEntry(BindGroupEntry(ssaoPsReflectionData.QueryResourceBindingChecked("ssaoNoiseTex").second, noise.view.Get()))
                    .AddEntry(BindGroupEntry(ssaoPsReflectionData.QueryResourceBindingChecked("texSampler").second, sampler.Get()))
                    .AddEntry(BindGroupEntry(ssaoPsReflectionData.QueryResourceBindingChecked("ssaoNoiseSampler").second, noiseSampler.Get()))
                    .AddEntry(BindGroupEntry(ssaoPsReflectionData.QueryResourceBindingChecked("kernalParams").second, uniformBuffers.ssaoKernel.bufView.Get()))
                    .AddEntry(BindGroupEntry(ssaoPsReflectionData.QueryResourceBindingChecked("passParams").second, uniformBuffers.ssaoParams.bufView.Get())));
        }

        // ssao blur
        {
            bindGroups.ssaoBlur = device->CreateBindGroup(
                BindGroupCreateInfo(bindGroupLayouts.ssaoBlur.Get())
                    .AddEntry(BindGroupEntry(ssaoBlurPsReflectionData.QueryResourceBindingChecked("ssaoTex").second, ssaoOutput.srv.Get()))
                    .AddEntry(BindGroupEntry(ssaoBlurPsReflectionData.QueryResourceBindingChecked("ssaoSampler").second, sampler.Get())));
        }

        // composition
        {
            bindGroups.composition = device->CreateBindGroup(
                BindGroupCreateInfo(bindGroupLayouts.composition.Get())
                    .AddEntry(BindGroupEntry(compositionPsReflectionData.QueryResourceBindingChecked("posTex").second, gBufferPos.srv.Get()))
                    .AddEntry(BindGroupEntry(compositionPsReflectionData.QueryResourceBindingChecked("normalTex").second, gBufferNormal.srv.Get()))
                    .AddEntry(BindGroupEntry(compositionPsReflectionData.QueryResourceBindingChecked("albedoTex").second, gBufferAlbedo.srv.Get()))
                    .AddEntry(BindGroupEntry(compositionPsReflectionData.QueryResourceBindingChecked("ssaoTex").second, ssaoOutput.srv.Get()))
                    .AddEntry(BindGroupEntry(compositionPsReflectionData.QueryResourceBindingChecked("ssaoBluredTex").second, ssaoBlurOutput.srv.Get()))
                    .AddEntry(BindGroupEntry(compositionPsReflectionData.QueryResourceBindingChecked("texSampler").second, sampler.Get()))
                    .AddEntry(BindGroupEntry(compositionPsReflectionData.QueryResourceBindingChecked("passParams").second, uniformBuffers.ssaoParams.bufView.Get())));
        }
    }

    void PrepareOffscreen()
    {
        CreateAttachments(PixelFormat::rgba32Float, gBufferPos);
        CreateAttachments(PixelFormat::rgba8Unorm, gBufferNormal);
        CreateAttachments(PixelFormat::rgba8Unorm, gBufferAlbedo);
        CreateAttachments(PixelFormat::r8Unorm, ssaoOutput);
        CreateAttachments(PixelFormat::r8Unorm, ssaoBlurOutput);
        CreateDepthAttachment();
    }

    void PrepareUniformBuffers()
    {
        // gltf model axis: y up, x right, z from screen inner to outer
        // to transform gltf coords system to our local coords system: z up, y right, x from screen outer to inner
        const auto aixsTransMat = FMat4x4 {
            0, 0, -1, 0,
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 0, 1
        };

        // scene matries
        uboSceneParams.projection = GetCamera().GetProjectionMatrix();
        uboSceneParams.view = GetCamera().GetViewMatrix();
        uboSceneParams.model = aixsTransMat;

        CreateUniformBuffer(BufferUsageBits::uniform | BufferUsageBits::mapWrite, &uniformBuffers.sceneParams, sizeof(UBOSceneParams), &uboSceneParams);

        // ssao parameters
        ubossaoParams.projection = GetCamera().GetProjectionMatrix();
        CreateUniformBuffer(BufferUsageBits::uniform | BufferUsageBits::mapWrite, &uniformBuffers.ssaoParams, sizeof(UBOSSAOParams), &ubossaoParams);

        // ssao kennel
        std::default_random_engine rndEngine(static_cast<unsigned>(time(nullptr)));
        std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);
        std::vector<FVec4> ssaoKernel(ssaoKernelSize);

        auto lerp = [](float a, float b, float f) ->float {
            return a + f * (b - a);
        };

        for (uint32_t i = 0; i < ssaoKernelSize; ++i)
        {
            FVec3 sample(rndDist(rndEngine) * 2.0 - 1.0, rndDist(rndEngine) * 2.0 - 1.0, rndDist(rndEngine));
            sample.Normalize();
            sample *= rndDist(rndEngine);
            float scale = static_cast<float>(i) / static_cast<float>(ssaoKernelSize);
            scale = lerp(0.1f, 1.0f, scale * scale);
            sample = sample * scale;
            ssaoKernel[i] = FVec4(sample.x, sample.y, sample.z, 0.0f);
        }
        CreateUniformBuffer(BufferUsageBits::uniform | BufferUsageBits::mapWrite, &uniformBuffers.ssaoKernel, ssaoKernel.size() * sizeof(FVec4), ssaoKernel.data());

        // random noise
        std::vector<FVec4> ssaoNoise(ssaoNoiseDim * ssaoNoiseDim);
        for (auto& randomVec : ssaoNoise)
        {
            randomVec = FVec4(rndDist(rndEngine) * 2.0f - 1.0f, rndDist(rndEngine) * 2.0f - 1.0f, 0.0f, 0.0f);
        }

        const BufferCreateInfo bufferCreateInfo = BufferCreateInfo()
            .SetSize(ssaoNoise.size() * sizeof(FVec4))
            .SetUsages(BufferUsageBits::mapWrite | BufferUsageBits::copySrc)
            .SetInitialState(BufferState::staging);

        const UniqueRef<Buffer> pixelBuffer = device->CreateBuffer(bufferCreateInfo);
        if (pixelBuffer != nullptr) {
            auto* data = pixelBuffer->Map(MapMode::write, 0, bufferCreateInfo.size);
            memcpy(data, ssaoNoise.data(), bufferCreateInfo.size);
            pixelBuffer->UnMap();
        }

        noise.tex = device->CreateTexture(
            TextureCreateInfo()
                .SetFormat(PixelFormat::rgba32Float)
                .SetMipLevels(1)
                .SetWidth(ssaoNoiseDim)
                .SetHeight(ssaoNoiseDim)
                .SetDepthOrArraySize(1)
                .SetDimension(TextureDimension::t2D)
                .SetSamples(1)
                .SetUsages(TextureUsageBits::copyDst | TextureUsageBits::textureBinding)
                .SetInitialState(TextureState::undefined));

        noise.view = noise.tex->CreateTextureView(
            TextureViewCreateInfo()
                .SetDimension(TextureViewDimension::tv2D)
                .SetMipLevels(0, 1)
                .SetArrayLayers(0, 1)
                .SetAspect(TextureAspect::color)
                .SetType(TextureViewType::textureBinding));

        noiseSampler = device->CreateSampler(
            SamplerCreateInfo()
                .SetAddressModeU(AddressMode::repeat)
                .SetAddressModeV(AddressMode::repeat));

        const UniqueRef<CommandBuffer> texCommandBuffer = device->CreateCommandBuffer();

        const UniqueRef<CommandRecorder> commandRecorder = texCommandBuffer->Begin();
        {
            const UniqueRef<CopyPassCommandRecorder> copyRecorder = commandRecorder->BeginCopyPass();
            {
                copyRecorder->ResourceBarrier(Barrier::Transition(noise.tex.Get(), TextureState::undefined, TextureState::copyDst));
                copyRecorder->CopyBufferToTexture(
                    pixelBuffer.Get(),
                    noise.tex.Get(),
                    BufferTextureCopyInfo(0, TextureSubResourceInfo(), UVec3Consts::zero, UVec3(ssaoNoiseDim, ssaoNoiseDim, 1)));
                copyRecorder->ResourceBarrier(Barrier::Transition(noise.tex.Get(), TextureState::copyDst, TextureState::shaderReadOnly));
            }
            copyRecorder->EndPass();
        }
        commandRecorder->End();

        const UniqueRef<Fence> fence = device->CreateFence(false);
        QueueSubmitInfo submitInfo {};
        submitInfo.signalFence = fence.Get();
        graphicsQueue->Submit(texCommandBuffer.Get(), submitInfo);
        fence->Wait();
    }

    void CreateDepthAttachment() {
        gBufferDepth.texture = device->CreateTexture(
            TextureCreateInfo()
                .SetFormat(PixelFormat::d32Float)
                .SetMipLevels(1)
                .SetWidth(GetWindowWidth())
                .SetHeight(GetWindowHeight())
                .SetDepthOrArraySize(1)
                .SetDimension(TextureDimension::t2D)
                .SetSamples(1)
                .SetUsages(TextureUsageBits::depthStencilAttachment)
                .SetInitialState(TextureState::depthStencilReadonly));

        gBufferDepth.view = gBufferDepth.texture->CreateTextureView(
            TextureViewCreateInfo()
                .SetDimension(TextureViewDimension::tv2D)
                .SetMipLevels(0, 1)
                .SetArrayLayers(0, 1)
                .SetAspect(TextureAspect::depth)
                .SetType(TextureViewType::depthStencil));
    }

    void CreateAttachments(PixelFormat format, ColorAttachment& attachment)
    {
        attachment.texture = device->CreateTexture(
            TextureCreateInfo()
                .SetFormat(format)
                .SetMipLevels(1)
                .SetWidth(GetWindowWidth())
                .SetHeight(GetWindowHeight())
                .SetDepthOrArraySize(1)
                .SetDimension(TextureDimension::t2D)
                .SetSamples(1)
                .SetUsages(TextureUsageBits::textureBinding | TextureUsageBits::renderAttachment)
                .SetInitialState(TextureState::shaderReadOnly));

        attachment.rtv = attachment.texture->CreateTextureView(
            TextureViewCreateInfo()
                .SetDimension(TextureViewDimension::tv2D)
                .SetMipLevels(0, 1)
                .SetArrayLayers(0, 1)
                .SetAspect(TextureAspect::color)
                .SetType(TextureViewType::colorAttachment));

        attachment.srv = attachment.texture->CreateTextureView(
            TextureViewCreateInfo()
                .SetDimension(TextureViewDimension::tv2D)
                .SetMipLevels(0, 1)
                .SetArrayLayers(0, 1)
                .SetAspect(TextureAspect::color)
                .SetType(TextureViewType::textureBinding));
    }

    void CompileShaderAndCreateShaderModule(UniqueRef<ShaderModule>& outShaderModule, ShaderCompileOutput& outCompileOutput, const std::string& fileName, const std::string& entryPoint, ShaderStageBits shaderStage) const
    {
        const std::vector<std::string> includePath { "../Test/Sample/RHI/SSAO/Shader"};
        outCompileOutput = CompileShader(fileName, entryPoint, shaderStage, includePath);
        outShaderModule = device->CreateShaderModule(ShaderModuleCreateInfo(entryPoint, outCompileOutput.byteCode));
    }

    void CreateUniformBuffer(BufferUsageFlags flags, UBuffer* uBuffer, size_t size, const void* data) const
    {
        const BufferCreateInfo createInfo = BufferCreateInfo()
            .SetSize(size)
            .SetUsages(flags)
            .SetInitialState(BufferState::staging);

        uBuffer->buf = device->CreateBuffer(createInfo);
        if (uBuffer->buf != nullptr && data != nullptr) {
            auto* mapData = uBuffer->buf->Map(MapMode::write, 0, size);
            memcpy(mapData, data, size);
            uBuffer->buf->UnMap();
        }

        const BufferViewCreateInfo viewCreateInfo = BufferViewCreateInfo()
            .SetType(BufferViewType::uniformBinding)
            .SetSize(size)
            .SetOffset(0);
        uBuffer->bufView = uBuffer->buf->CreateBufferView(viewCreateInfo);
    }

    void CreatePipeline()
    {
        const auto& gBufferVsReflectionData = shaderObjects.gBufferVsCompileOutput.reflectionData;
        const auto& ssaoVsReflectionData = shaderObjects.ssaoVsCompileOutput.reflectionData;
        const auto& ssaoBlurVsReflectionData = shaderObjects.ssaoBlurVsCompileOutput.reflectionData;
        const auto& compositionVsReflectionData = shaderObjects.compositionVsCompileOutput.reflectionData;

        // General pipeline infos
        RasterPipelineCreateInfo createInfo = RasterPipelineCreateInfo()
            .SetPrimitiveState(PrimitiveState(PrimitiveTopologyType::triangle, FillMode::solid, IndexFormat::uint32, FrontFace::ccw, CullMode::none, false));

        // Gbuffer
        {
            createInfo
                .SetLayout(pipelineLayouts.gBuffer.Get())
                .SetVertexShader(shaderObjects.gBufferVs.Get())
                .SetPixelShader(shaderObjects.gBufferPs.Get())
                .SetDepthStencilState(DepthStencilState(true, false, PixelFormat::d32Float, CompareFunc::greaterEqual))
                .SetVertexState(
                    VertexState()
                        .AddVertexBufferLayout(
                            VertexBufferLayout(VertexStepMode::perVertex, sizeof(Vertex))
                                .AddAttribute(VertexAttribute(gBufferVsReflectionData.QueryVertexBindingChecked("POSITION"), VertexFormat::float32X4, 0))
                                .AddAttribute(VertexAttribute(gBufferVsReflectionData.QueryVertexBindingChecked("TEXCOORD"), VertexFormat::float32X2, offsetof(Vertex, uv)))
                                .AddAttribute(VertexAttribute(gBufferVsReflectionData.QueryVertexBindingChecked("COLOR"), VertexFormat::float32X4, offsetof(Vertex, color)))
                                .AddAttribute(VertexAttribute(gBufferVsReflectionData.QueryVertexBindingChecked("NORMAL"), VertexFormat::float32X3, offsetof(Vertex, normal)))))
                .SetFragmentState(
                    FragmentState()
                        .AddColorTarget(ColorTargetState(PixelFormat::rgba32Float, ColorWriteBits::all))
                        .AddColorTarget(ColorTargetState(PixelFormat::rgba8Unorm, ColorWriteBits::all))
                        .AddColorTarget(ColorTargetState(PixelFormat::rgba8Unorm, ColorWriteBits::all)));

            pipelines.gBuffer = device->CreateRasterPipeline(createInfo);
        }

        // ssao
        {
            createInfo
                .SetLayout(pipelineLayouts.ssao.Get())
                .SetVertexShader(shaderObjects.ssaoVs.Get())
                .SetPixelShader(shaderObjects.ssaoPs.Get())
                .SetDepthStencilState(DepthStencilState())
                .SetVertexState(
                    VertexState()
                        .AddVertexBufferLayout(
                            VertexBufferLayout(VertexStepMode::perVertex, sizeof(QuadVertex))
                                .AddAttribute(VertexAttribute(ssaoVsReflectionData.QueryVertexBindingChecked("POSITION"), VertexFormat::float32X3, 0))
                                .AddAttribute(VertexAttribute(ssaoVsReflectionData.QueryVertexBindingChecked("TEXCOORD"), VertexFormat::float32X2, offsetof(QuadVertex, uv)))))
                .SetFragmentState(
                    FragmentState()
                        .AddColorTarget(ColorTargetState(PixelFormat::r8Unorm, ColorWriteBits::all)));

            pipelines.ssao = device->CreateRasterPipeline(createInfo);
        }

        // ssaoBlur
        {
            createInfo
                .SetLayout(pipelineLayouts.ssaoBlur.Get())
                .SetVertexShader(shaderObjects.ssaoBlurVs.Get())
                .SetPixelShader(shaderObjects.ssaoBlurPs.Get())
                .SetDepthStencilState(DepthStencilState())
                .SetVertexState(
                    VertexState()
                        .AddVertexBufferLayout(
                            VertexBufferLayout(VertexStepMode::perVertex, sizeof(QuadVertex))
                                .AddAttribute(VertexAttribute(ssaoBlurVsReflectionData.QueryVertexBindingChecked("POSITION"), VertexFormat::float32X3, 0))
                                .AddAttribute(VertexAttribute(ssaoBlurVsReflectionData.QueryVertexBindingChecked("TEXCOORD"), VertexFormat::float32X2, offsetof(QuadVertex, uv)))))
                .SetFragmentState(
                    FragmentState()
                        .AddColorTarget(ColorTargetState(PixelFormat::r8Unorm, ColorWriteBits::all)));

            pipelines.ssaoBlur = device->CreateRasterPipeline(createInfo);
        }

        // composition
        {
            createInfo
                .SetLayout(pipelineLayouts.composition.Get())
                .SetVertexShader(shaderObjects.compositionVs.Get())
                .SetPixelShader(shaderObjects.compositionPs.Get())
                .SetDepthStencilState(DepthStencilState())
                .SetVertexState(
                    VertexState()
                        .AddVertexBufferLayout(
                            VertexBufferLayout(VertexStepMode::perVertex, sizeof(QuadVertex))
                                .AddAttribute(VertexAttribute(compositionVsReflectionData.QueryVertexBindingChecked("POSITION"), VertexFormat::float32X3, 0))
                                .AddAttribute(VertexAttribute(compositionVsReflectionData.QueryVertexBindingChecked("TEXCOORD"), VertexFormat::float32X2, offsetof(QuadVertex, uv)))))
                .SetFragmentState(
                    FragmentState()
                        .AddColorTarget(ColorTargetState(swapChainFormat, ColorWriteBits::all)));

            pipelines.composition = device->CreateRasterPipeline(createInfo);
        }
    }

    void InitCamera()
    {
        auto* camera = new Camera(
            FVec3(.0f, -5.0f, 2.0f),
            FVec3(.0f, .0f, -90.0f),
            Camera::ProjectionParams {
                60.0f,
                static_cast<float>(GetWindowWidth()),
                static_cast<float>(GetWindowHeight()),
                0.1f,
                64.0f
            });

        camera->SetMoveSpeed(0.005f);
        camera->SetRotateSpeed(0.1f);
        SetCamera(camera);
    }

    void LoadGLTF()
    {
        model = MakeUnique<Model>();
        model->LoadFromFile("../Test/Sample/RHI/SSAO/Model/Voyager.gltf");
    }

    void GenerateRenderables()
    {
        for (auto& mesh : model->meshes) {
            renderables.emplace_back(MakeUnique<Renderable>(*GetRHIInstance(), *device, *renderableLayout, shaderObjects.gBufferPsCompileOutput.reflectionData, *sampler, *mesh));
        }
    }
};

int main(int argc, char* argv[])
{
    SSAOApplication application("SSAO");
    if (!application.Initialize(argc, argv)) {
        return -1;
    }
    return application.RunLoop();
}

