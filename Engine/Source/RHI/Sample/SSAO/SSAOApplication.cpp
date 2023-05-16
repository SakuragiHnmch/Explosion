//
// Created by Junkang on 2023/3/8.
//

#include <vector>
#include <array>
#include <memory>
#include <random>

#include <Application.h>
#include "GLTFParser.h"
#include "Camera.h"

using namespace RHI;

class SSAOApplication : public Application {
public:
    NON_COPYABLE(SSAOApplication)
    explicit SSAOApplication(const std::string& n) : Application(n) {}
    ~SSAOApplication() override = default;

    Instance* GetInstance() {
        return instance;
    }

    Device* GetDevice()
    {
        return device.Get();
    }

    BindGroupLayout* GetLayout()
    {
        return renderableLayout.Get();
    }

    Sampler* GetSampler()
    {
        return sampler.Get();
    }

    Queue* GetQueue()
    {
        return graphicsQueue;
    }

protected:
    void OnCreate() override
    {
        InitCamera();
        LoadGLTF();
        CreateInstanceAndSelectGPU();
        RequestDeviceAndFetchQueues();
        CreateSwapChain();
        CreateVertexBuffer();
        CreateIndexBuffer();
        CreateQuadBuffer();
        PrepareOffscreen();
        PrepareUniformBuffers();
        CreateSampler();
        CreateBindGroupLayoutAndPipelineLayout();
        CreateBindGroup();
        CreatePipeline();
        CreateFence();
        CreateCommandBuffer();
        GenerateRenderables();
    }

    void OnDrawFrame() override
    {
        PopulateCommandBuffer();
        SubmitCommandBufferAndPresent();
    }

    void OnDestroy() override
    {
        graphicsQueue->Wait(fence.Get());
        fence->Wait();
    }

private:
    static const uint8_t ssaoKernelSize = 64;
    static const uint8_t ssaoNoiseDim = 16;
    static const uint8_t backBufferCount = 2;

    PixelFormat swapChainFormat = PixelFormat::max;
    Instance* instance = nullptr;
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
    std::array<Common::UniqueRef<TextureView>, backBufferCount> swapChainTextureViews {};

    UniqueRef<Buffer> quadVertexBuffer = nullptr;
    UniqueRef<BufferView> quadVertexBufferView = nullptr;
    UniqueRef<Buffer> quadIndexBuffer = nullptr;
    UniqueRef<BufferView> quadIndexBufferView = nullptr;

    UniqueRef<CommandBuffer> commandBuffer = nullptr;
    UniqueRef<Fence> fence = nullptr;
    UniqueRef<Sampler> sampler = nullptr;
    UniqueRef<Sampler> noiseSampler = nullptr;

    struct Renderable {
        uint32_t indexCount;
        uint32_t firstIndex;

        UniqueRef<BindGroup> bindGroup;
        UniqueRef<Texture> diffuseColorMap;
        UniqueRef<TextureView> diffuseColorMapView;

        Renderable(SSAOApplication* app, UniqueRef<Mesh>& mesh) {
            indexCount =mesh->indexCount;
            firstIndex =mesh->firstIndex;

            // upload diffuseColorMap
            auto& texData =mesh->materialData->baseColorTexture;

            BufferCreateInfo bufferCreateInfo {};
            bufferCreateInfo.size = texData->GetSize();
            bufferCreateInfo.usages = BufferUsageBits::uniform | BufferUsageBits::mapWrite | BufferUsageBits::copySrc;
            auto* pixelBuffer = app->GetDevice()->CreateBuffer(bufferCreateInfo);
            if (pixelBuffer != nullptr) {
                auto* mapData = pixelBuffer->Map(MapMode::write, 0, bufferCreateInfo.size);
                memcpy(mapData, texData->buffer.data(), bufferCreateInfo.size);
                pixelBuffer->UnMap();
            }

            TextureCreateInfo texCreateInfo {};
            texCreateInfo.format = PixelFormat::rgba8Unorm;
            texCreateInfo.dimension = TextureDimension::t2D;
            texCreateInfo.mipLevels = 1;
            texCreateInfo.extent = { texData->width, texData->height, 1};
            texCreateInfo.samples = 1;
            texCreateInfo.usages = TextureUsageBits::copyDst | TextureUsageBits::textureBinding;
            diffuseColorMap = app->GetDevice()->CreateTexture(texCreateInfo);

            TextureViewCreateInfo viewCreateInfo {};
            viewCreateInfo.dimension = TextureViewDimension::tv2D;
            viewCreateInfo.baseArrayLayer = 0;
            viewCreateInfo.arrayLayerNum = 1;
            viewCreateInfo.baseMipLevel = 0;
            viewCreateInfo.mipLevelNum = 1;
            viewCreateInfo.aspect = TextureAspect::color;
            diffuseColorMapView = diffuseColorMap->CreateTextureView(viewCreateInfo);

            auto* texCommandBuffer = app->GetDevice()->CreateCommandBuffer();
            auto* commandEncoder = texCommandBuffer->Begin();
            commandEncoder->ResourceBarrier(Barrier::Transition(diffuseColorMap.Get(), TextureState::undefined, TextureState::copyDst));
            TextureSubResourceInfo subResourceInfo {};
            subResourceInfo.mipLevel = 0;
            subResourceInfo.arrayLayerNum = 1;
            subResourceInfo.baseArrayLayer = 0;
            subResourceInfo.aspect = TextureAspect::color;
            commandEncoder->CopyBufferToTexture(pixelBuffer, diffuseColorMap.Get(), &subResourceInfo, {texData->width, texData->height, 1});
            commandEncoder->ResourceBarrier(Barrier::Transition(diffuseColorMap.Get(), TextureState::copyDst, TextureState::shaderReadOnly));
            commandEncoder->End();

            app->GetQueue()->Submit(texCommandBuffer, nullptr);

            // per renderable bindGroup
            std::vector<BindGroupEntry> entries(2);
            entries[0].binding.type = BindingType::texture;
            entries[0].textureView = diffuseColorMapView.Get();
            entries[1].binding.type = BindingType::sampler;
            entries[1].sampler = app->GetSampler();
            if (app->GetInstance()->GetRHIType() == RHI::RHIType::directX12) {
                entries[0].binding.platform.hlsl = { HlslBindingRangeType::texture, 0 };
                entries[1].binding.platform.hlsl = { HlslBindingRangeType::sampler, 0 };
            } else {
                entries[0].binding.platform.glsl.index = 0;
                entries[1].binding.platform.glsl.index = 1;
            }

            BindGroupCreateInfo createInfo {};
            createInfo.entries = entries.data();
            createInfo.entryNum = static_cast<uint32_t>(entries.size());
            createInfo.layout = app->GetLayout();

            bindGroup = app->GetDevice()->CreateBindGroup(createInfo);
        }
    };

    UniqueRef<Model> model = nullptr;
    std::vector<UniqueRef<Renderable>> renderables;
    Camera camera;

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
        glm::mat4 projection;
        glm::mat4 model;
        glm::mat4 view;
        float nearPlane = 0.1f;
        float farPlane = 64.0f;
    } uboSceneParams;

    struct UBOSSAOParams {
        glm::mat4 projection;
        int32_t ssao = 1;
        int32_t ssaoOnly = 0;
        int32_t ssaoBlur = 1;
    } ubossaoParams;

    struct Noise {
        UniqueRef<Texture> tex;
        UniqueRef<TextureView> view;
    } noise;

    struct ShaderModules {
        UniqueRef<ShaderModule> gBufferVs;
        UniqueRef<ShaderModule> gBufferPs;
        UniqueRef<ShaderModule> ssaoVs;
        UniqueRef<ShaderModule> ssaoPs;
        UniqueRef<ShaderModule> ssaoBlurVs;
        UniqueRef<ShaderModule> ssaoBlurPs;
        UniqueRef<ShaderModule> compositionVs;
        UniqueRef<ShaderModule> compositionPs;

    } shaderModules;

    struct Pipelines {
        UniqueRef<GraphicsPipeline> gBuffer;
        UniqueRef<GraphicsPipeline> ssao;
        UniqueRef<GraphicsPipeline> ssaoBlur;
        UniqueRef<GraphicsPipeline> composition;
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
        UniqueRef<TextureView> view;
    };

    ColorAttachment gBufferPos;
    ColorAttachment gBufferNormal;
    ColorAttachment gBufferAlbedo;
    ColorAttachment gBufferDepth;

    ColorAttachment ssaoOutput;
    ColorAttachment ssaoBlurOutput;

    struct QuadVertex {
        glm::vec3 pos;
        glm::vec2 uv;
    };

    void CreateInstanceAndSelectGPU()
    {
        instance = Instance::GetByType(rhiType);
        gpu = instance->GetGpu(0);
    }

    void RequestDeviceAndFetchQueues()
    {
        std::vector<QueueInfo> queueCreateInfos = { {QueueType::graphics, 1} };
        DeviceCreateInfo createInfo {};
        createInfo.queueCreateInfoNum = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.queueCreateInfos = queueCreateInfos.data();
        device = gpu->RequestDevice(createInfo);
        graphicsQueue = device->GetQueue(QueueType::graphics, 0);
    }

    void CreateSwapChain()
    {
        static std::vector<PixelFormat> swapChainFormatQualifiers = {
            PixelFormat::rgba8Unorm,
            PixelFormat::bgra8Unorm
        };

        SurfaceCreateInfo surfaceCreateInfo {};
        surfaceCreateInfo.window = GetPlatformWindow();
        surface = device->CreateSurface(surfaceCreateInfo);

        for (auto format : swapChainFormatQualifiers) {
            if (device->CheckSwapChainFormatSupport(surface.Get(), format)) {
                swapChainFormat = format;
                break;
            }
        }
        Assert(swapChainFormat != PixelFormat::max);

        SwapChainCreateInfo swapChainCreateInfo {};
        swapChainCreateInfo.format = swapChainFormat;
        swapChainCreateInfo.presentMode = PresentMode::immediately;
        swapChainCreateInfo.textureNum = backBufferCount;
        swapChainCreateInfo.extent = {width, height};
        swapChainCreateInfo.surface = surface.Get();
        swapChainCreateInfo.presentQueue = graphicsQueue;
        swapChain = device->CreateSwapChain(swapChainCreateInfo);

        for (auto i = 0; i < swapChainCreateInfo.textureNum; i++) {
            swapChainTextures[i] = swapChain->GetTexture(i);

            TextureViewCreateInfo viewCreateInfo {};
            viewCreateInfo.dimension = TextureViewDimension::tv2D;
            viewCreateInfo.baseArrayLayer = 0;
            viewCreateInfo.arrayLayerNum = 1;
            viewCreateInfo.baseMipLevel = 0;
            viewCreateInfo.mipLevelNum = 1;
            viewCreateInfo.aspect = TextureAspect::color;
            swapChainTextureViews[i] = swapChainTextures[i]->CreateTextureView(viewCreateInfo);
        }
    }

    void CreateVertexBuffer()
    {
        BufferCreateInfo bufferCreateInfo {};
        bufferCreateInfo.size = model->raw_vertex_buffer.size() * sizeof(Vertex);
        bufferCreateInfo.usages = BufferUsageBits::vertex | BufferUsageBits::mapWrite | BufferUsageBits::copySrc;
        vertexBuffer = device->CreateBuffer(bufferCreateInfo);
        assert(vertexBuffer != nullptr);
        auto* data = vertexBuffer->Map(MapMode::write, 0, bufferCreateInfo.size);
        memcpy(data, model->raw_vertex_buffer.data(), bufferCreateInfo.size);
        vertexBuffer->UnMap();

        BufferViewCreateInfo bufferViewCreateInfo {};
        bufferViewCreateInfo.size = bufferCreateInfo.size;
        bufferViewCreateInfo.offset = 0;
        bufferViewCreateInfo.vertex.stride = sizeof(Vertex);
        vertexBufferView = vertexBuffer->CreateBufferView(bufferViewCreateInfo);
    }

    void CreateIndexBuffer()
    {
        BufferCreateInfo bufferCreateInfo {};
        bufferCreateInfo.size = model->raw_index_buffer.size() * sizeof(uint32_t);
        bufferCreateInfo.usages = BufferUsageBits::index | BufferUsageBits::mapWrite | BufferUsageBits::copySrc;
        indexBuffer = device->CreateBuffer(bufferCreateInfo);
        assert(indexBuffer != nullptr);
        auto* data = indexBuffer->Map(MapMode::write, 0, bufferCreateInfo.size);
        memcpy(data, model->raw_index_buffer.data(), bufferCreateInfo.size);
        indexBuffer->UnMap();


        BufferViewCreateInfo bufferViewCreateInfo {};
        bufferViewCreateInfo.size = bufferCreateInfo.size;
        bufferViewCreateInfo.offset = 0;
        bufferViewCreateInfo.index.format = IndexFormat::uint32;
        indexBufferView = indexBuffer->CreateBufferView(bufferViewCreateInfo);
    }

    void CreateQuadBuffer()
    {
        // vertex buffer
        std::vector<QuadVertex> vertices {
            {{-1.0f, -1.0f, 0.f}, {0.f, 0.f}},
            {{1.0f, -1.0f, 0.f}, {1.f, 0.f}},
            {{1.0f, 1.0f, 0.f}, {1.f, 1.f}},
            {{-1.0f, 1.0f, 0.f}, {0.f, 1.f}},
        };

        BufferCreateInfo bufferCreateInfo {};
        bufferCreateInfo.size = vertices.size() * sizeof(QuadVertex);
        bufferCreateInfo.usages = BufferUsageBits::vertex | BufferUsageBits::mapWrite | BufferUsageBits::copySrc;
        quadVertexBuffer = device->CreateBuffer(bufferCreateInfo);
        if (quadVertexBuffer != nullptr) {
            auto* data = quadVertexBuffer->Map(MapMode::write, 0, bufferCreateInfo.size);
            memcpy(data, vertices.data(), bufferCreateInfo.size);
            quadVertexBuffer->UnMap();
        }

        BufferViewCreateInfo bufferViewCreateInfo {};
        bufferViewCreateInfo.size = vertices.size() * sizeof(Vertex);
        bufferViewCreateInfo.offset = 0;
        bufferViewCreateInfo.vertex.stride = sizeof(Vertex);
        quadVertexBufferView = quadVertexBuffer->CreateBufferView(bufferViewCreateInfo);

        // index buffer
        std::vector<uint32_t> indices = {0, 1, 2, 0, 2, 3};
        bufferCreateInfo.size = indices.size() * sizeof(uint32_t);
        bufferCreateInfo.usages = BufferUsageBits::index | BufferUsageBits::mapWrite | BufferUsageBits::copySrc;
        quadIndexBuffer = device->CreateBuffer(bufferCreateInfo);
        if (quadIndexBuffer != nullptr) {
            auto* data = quadIndexBuffer->Map(MapMode::write, 0, bufferCreateInfo.size);
            memcpy(data, indices.data(), bufferCreateInfo.size);
            quadIndexBuffer->UnMap();
        }

        bufferViewCreateInfo.size = indices.size() * sizeof(uint32_t);
        bufferViewCreateInfo.offset = 0;
        bufferViewCreateInfo.index.format = IndexFormat::uint32;
        quadIndexBufferView = quadIndexBuffer->CreateBufferView(bufferViewCreateInfo);
    }

    void CreateSampler()
    {
        SamplerCreateInfo samplerCreateInfo {};
        sampler = device->CreateSampler(samplerCreateInfo);
    }

    void CreateCommandBuffer()
    {
        commandBuffer = device->CreateCommandBuffer();
    }

    void CreateFence()
    {
        fence = device->CreateFence();
    }

    void CreateBindGroupLayoutAndPipelineLayout()
    {
        std::vector<BindGroupLayoutEntry> entries;
        BindGroupLayoutCreateInfo createInfo {};
        PipelineLayoutCreateInfo pipelineLayoutCreateInfo {};

        //gBuffer
        entries.resize(1);
        entries[0].binding.type = BindingType::uniformBuffer;
        entries[0].shaderVisibility = ShaderStageBits::sVertex | ShaderStageBits::sPixel;
        if (instance->GetRHIType() == RHI::RHIType::directX12) {
            entries[0].binding.platform.hlsl = { HlslBindingRangeType::constantBuffer, 0 };
        } else {
            entries[0].binding.platform.glsl.index = 0;
        }
        createInfo.entries = entries.data();
        createInfo.entryNum = static_cast<uint32_t>(entries.size());
        createInfo.layoutIndex = 0;
        bindGroupLayouts.gBuffer = device->CreateBindGroupLayout(createInfo);

        // renderable layout
        entries.resize(2);
        entries[0].binding.type = BindingType::texture;
        entries[0].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        entries[1].binding.type = BindingType::sampler;
        entries[1].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        if (instance->GetRHIType() == RHI::RHIType::directX12) {
            entries[0].binding.platform.hlsl = { HlslBindingRangeType::texture, 0 };
            entries[1].binding.platform.hlsl = { HlslBindingRangeType::sampler, 0 };
        } else {
            entries[0].binding.platform.glsl.index = 0;
            entries[1].binding.platform.glsl.index = 1;
        }
        createInfo.entries = entries.data();
        createInfo.entryNum = static_cast<uint32_t>(entries.size());
        createInfo.layoutIndex = 1;
        renderableLayout = device->CreateBindGroupLayout(createInfo);

        std::vector<BindGroupLayout*> gBufferLayouts { bindGroupLayouts.gBuffer.Get(), renderableLayout.Get() };
        pipelineLayoutCreateInfo.bindGroupLayoutNum = 2;
        pipelineLayoutCreateInfo.bindGroupLayouts = gBufferLayouts.data();
        pipelineLayouts.gBuffer = device->CreatePipelineLayout(pipelineLayoutCreateInfo);

        //ssao
        entries.resize(7);
        entries[0].binding.type = BindingType::texture;
        entries[0].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        entries[1].binding.type = BindingType::texture;
        entries[1].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        entries[2].binding.type = BindingType::texture;
        entries[2].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        entries[3].binding.type = BindingType::sampler;
        entries[3].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        entries[4].binding.type = BindingType::sampler;
        entries[4].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        entries[5].binding.type = BindingType::uniformBuffer;
        entries[5].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        entries[6].binding.type = BindingType::uniformBuffer;
        entries[6].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        if (instance->GetRHIType() == RHI::RHIType::directX12) {
            entries[0].binding.platform.hlsl = { HlslBindingRangeType::texture, 0 };
            entries[1].binding.platform.hlsl = { HlslBindingRangeType::texture, 1 };
            entries[2].binding.platform.hlsl = { HlslBindingRangeType::texture, 2 };
            entries[3].binding.platform.hlsl = { HlslBindingRangeType::sampler, 0 };
            entries[4].binding.platform.hlsl = { HlslBindingRangeType::sampler, 1 };
            entries[5].binding.platform.hlsl = { HlslBindingRangeType::constantBuffer, 0 };
            entries[6].binding.platform.hlsl = { HlslBindingRangeType::constantBuffer, 1 };
        } else {
            entries[0].binding.platform.glsl.index = 0;
            entries[1].binding.platform.glsl.index = 1;
            entries[2].binding.platform.glsl.index = 2;
            entries[3].binding.platform.glsl.index = 3;
            entries[4].binding.platform.glsl.index = 4;
            entries[5].binding.platform.glsl.index = 5;
            entries[6].binding.platform.glsl.index = 6;
        }
        createInfo.entries = entries.data();
        createInfo.entryNum = static_cast<uint32_t>(entries.size());
        createInfo.layoutIndex = 0;
        bindGroupLayouts.ssao = device->CreateBindGroupLayout(createInfo);

        std::vector<BindGroupLayout*> ssaoLayouts { bindGroupLayouts.ssao.Get() };
        pipelineLayoutCreateInfo.bindGroupLayoutNum = 1;
        pipelineLayoutCreateInfo.bindGroupLayouts = ssaoLayouts.data();
        pipelineLayouts.ssao = device->CreatePipelineLayout(pipelineLayoutCreateInfo);

        // ssaoBlur
        entries.resize(2);
        entries[0].binding.type = BindingType::texture;
        entries[0].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        entries[1].binding.type = BindingType::sampler;
        entries[1].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        if (instance->GetRHIType() == RHI::RHIType::directX12) {
            entries[0].binding.platform.hlsl = { HlslBindingRangeType::texture, 0 };
            entries[1].binding.platform.hlsl = { HlslBindingRangeType::sampler, 0 };
        } else {
            entries[0].binding.platform.glsl.index = 0;
            entries[1].binding.platform.glsl.index = 1;
        }
        createInfo.entries = entries.data();
        createInfo.entryNum = static_cast<uint32_t>(entries.size());
        createInfo.layoutIndex = 0;
        bindGroupLayouts.ssaoBlur = device->CreateBindGroupLayout(createInfo);

        std::vector<BindGroupLayout*> blurLayouts { bindGroupLayouts.ssaoBlur.Get() };
        pipelineLayoutCreateInfo.bindGroupLayouts = blurLayouts.data();
        pipelineLayouts.ssaoBlur = device->CreatePipelineLayout(pipelineLayoutCreateInfo);

        // composition
        entries.resize(7);
        entries[0].binding.type = BindingType::texture;
        entries[0].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        entries[1].binding.type = BindingType::texture;
        entries[1].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        entries[2].binding.type = BindingType::texture;
        entries[2].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        entries[3].binding.type = BindingType::texture;
        entries[3].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        entries[4].binding.type = BindingType::texture;
        entries[4].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        entries[5].binding.type = BindingType::sampler;
        entries[5].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        entries[6].binding.type = BindingType::uniformBuffer;
        entries[6].shaderVisibility = static_cast<ShaderStageFlags>(ShaderStageBits::sPixel);
        if (instance->GetRHIType() == RHI::RHIType::directX12) {
            entries[0].binding.platform.hlsl = { HlslBindingRangeType::texture, 0 };
            entries[1].binding.platform.hlsl = { HlslBindingRangeType::texture, 1 };
            entries[2].binding.platform.hlsl = { HlslBindingRangeType::texture, 2 };
            entries[3].binding.platform.hlsl = { HlslBindingRangeType::texture, 3 };
            entries[4].binding.platform.hlsl = { HlslBindingRangeType::texture, 4 };
            entries[5].binding.platform.hlsl = { HlslBindingRangeType::sampler, 0 };
            entries[6].binding.platform.hlsl = { HlslBindingRangeType::constantBuffer, 0 };
        } else {
            entries[0].binding.platform.glsl.index = 0;
            entries[1].binding.platform.glsl.index = 1;
            entries[2].binding.platform.glsl.index = 2;
            entries[3].binding.platform.glsl.index = 3;
            entries[4].binding.platform.glsl.index = 4;
            entries[5].binding.platform.glsl.index = 5;
            entries[6].binding.platform.glsl.index = 6;
        }
        createInfo.entries = entries.data();
        createInfo.entryNum = static_cast<uint32_t>(entries.size());
        createInfo.layoutIndex = 0;
        bindGroupLayouts.composition = device->CreateBindGroupLayout(createInfo);

        std::vector<BindGroupLayout*> comLayouts { bindGroupLayouts.composition.Get() };
        pipelineLayoutCreateInfo.bindGroupLayouts = comLayouts.data();
        pipelineLayouts.composition = device->CreatePipelineLayout(pipelineLayoutCreateInfo);
    }

    void CreateBindGroup()
    {
        BindGroupCreateInfo createInfo {};
        std::vector<BindGroupEntry> entries(1);

        // GBuffer scene
        entries[0].binding.type = BindingType::uniformBuffer;
        entries[0].bufferView = uniformBuffers.sceneParams.bufView.Get();
        if (instance->GetRHIType() == RHI::RHIType::directX12) {
            entries[0].binding.platform.hlsl = { HlslBindingRangeType::constantBuffer, 0 };
        } else {
            entries[0].binding.platform.glsl.index = 0;
        }
        createInfo.entries = entries.data();
        createInfo.entryNum = static_cast<uint32_t>(entries.size());
        createInfo.layout = bindGroupLayouts.gBuffer.Get();
        bindGroups.scene = device->CreateBindGroup(createInfo);

        // ssao generation
        entries.resize(7);
        entries[0].binding.type = BindingType::texture;
        entries[0].textureView = gBufferPos.view.Get();
        entries[1].binding.type = BindingType::texture;
        entries[1].textureView = gBufferNormal.view.Get();
        entries[2].binding.type = BindingType::texture;
        entries[2].textureView = noise.view.Get();
        entries[3].binding.type = BindingType::sampler;
        entries[3].sampler = sampler.Get();
        entries[4].binding.type = BindingType::sampler;
        entries[4].sampler = noiseSampler.Get();
        entries[5].binding.type = BindingType::uniformBuffer;
        entries[5].bufferView = uniformBuffers.ssaoKernel.bufView.Get();
        entries[6].binding.type = BindingType::uniformBuffer;
        entries[6].bufferView = uniformBuffers.ssaoParams.bufView.Get();
        if (instance->GetRHIType() == RHI::RHIType::directX12) {
            entries[0].binding.platform.hlsl = { HlslBindingRangeType::texture, 0 };
            entries[1].binding.platform.hlsl = { HlslBindingRangeType::texture, 1 };
            entries[2].binding.platform.hlsl = { HlslBindingRangeType::texture, 2 };
            entries[3].binding.platform.hlsl = { HlslBindingRangeType::sampler, 0 };
            entries[4].binding.platform.hlsl = { HlslBindingRangeType::sampler, 1 };
            entries[5].binding.platform.hlsl = { HlslBindingRangeType::constantBuffer, 0 };
            entries[6].binding.platform.hlsl = { HlslBindingRangeType::constantBuffer, 1 };
        } else {
            entries[0].binding.platform.glsl.index = 0;
            entries[1].binding.platform.glsl.index = 1;
            entries[2].binding.platform.glsl.index = 2;
            entries[3].binding.platform.glsl.index = 3;
            entries[4].binding.platform.glsl.index = 4;
            entries[5].binding.platform.glsl.index = 5;
            entries[6].binding.platform.glsl.index = 6;
        }
        createInfo.entries = entries.data();
        createInfo.entryNum = static_cast<uint32_t>(entries.size());
        createInfo.layout = bindGroupLayouts.ssao.Get();
        bindGroups.ssao = device->CreateBindGroup(createInfo);

        // ssao blur
        entries.resize(2);
        entries[0].binding.type = BindingType::texture;
        entries[0].textureView = ssaoOutput.view.Get();
        entries[1].binding.type = BindingType::sampler;
        entries[1].sampler = sampler.Get();
        if (instance->GetRHIType() == RHI::RHIType::directX12) {
            entries[0].binding.platform.hlsl = { HlslBindingRangeType::texture, 0 };
            entries[1].binding.platform.hlsl = { HlslBindingRangeType::sampler, 0 };
        } else {
            entries[0].binding.platform.glsl.index = 0;
            entries[1].binding.platform.glsl.index = 1;
        }
        createInfo.entries = entries.data();
        createInfo.entryNum = static_cast<uint32_t>(entries.size());
        createInfo.layout = bindGroupLayouts.ssaoBlur.Get();
        bindGroups.ssaoBlur = device->CreateBindGroup(createInfo);

        // composition
        entries.resize(7);
        entries[0].binding.type = BindingType::texture;
        entries[0].textureView = gBufferPos.view.Get();
        entries[1].binding.type = BindingType::texture;
        entries[1].textureView = gBufferNormal.view.Get();
        entries[2].binding.type = BindingType::texture;
        entries[2].textureView = gBufferAlbedo.view.Get();
        entries[3].binding.type = BindingType::texture;
        entries[3].textureView = ssaoOutput.view.Get();
        entries[4].binding.type = BindingType::texture;
        entries[4].textureView = ssaoBlurOutput.view.Get();
        entries[5].binding.type = BindingType::sampler;
        entries[5].sampler = sampler.Get();
        entries[6].binding.type = BindingType::uniformBuffer;
        entries[6].bufferView = uniformBuffers.ssaoParams.bufView.Get();
        if (instance->GetRHIType() == RHI::RHIType::directX12) {
            entries[0].binding.platform.hlsl = { HlslBindingRangeType::texture, 0 };
            entries[1].binding.platform.hlsl = { HlslBindingRangeType::texture, 1 };
            entries[2].binding.platform.hlsl = { HlslBindingRangeType::texture, 2 };
            entries[3].binding.platform.hlsl = { HlslBindingRangeType::texture, 3 };
            entries[4].binding.platform.hlsl = { HlslBindingRangeType::texture, 4 };
            entries[5].binding.platform.hlsl = { HlslBindingRangeType::sampler, 0 };
            entries[6].binding.platform.hlsl = { HlslBindingRangeType::constantBuffer, 0 };
        } else {
            entries[0].binding.platform.glsl.index = 0;
            entries[1].binding.platform.glsl.index = 1;
            entries[2].binding.platform.glsl.index = 2;
            entries[3].binding.platform.glsl.index = 3;
            entries[4].binding.platform.glsl.index = 4;
            entries[5].binding.platform.glsl.index = 5;
            entries[6].binding.platform.glsl.index = 6;
        }
        createInfo.entries = entries.data();
        createInfo.entryNum = static_cast<uint32_t>(entries.size());
        createInfo.layout = bindGroupLayouts.composition.Get();
        bindGroups.composition = device->CreateBindGroup(createInfo);
    }

    void PrepareOffscreen()
    {
        CreateAttachments(PixelFormat::rgba32Float, TextureUsageBits::textureBinding | TextureUsageBits::renderAttachment, TextureAspect::color, &gBufferPos, width, height);
        CreateAttachments(PixelFormat::rgba8Unorm, TextureUsageBits::textureBinding | TextureUsageBits::renderAttachment, TextureAspect::color, &gBufferNormal, width, height);
        CreateAttachments(PixelFormat::rgba8Unorm, TextureUsageBits::textureBinding | TextureUsageBits::renderAttachment, TextureAspect::color, &gBufferAlbedo, width, height);
        CreateAttachments(PixelFormat::d32FloatS8Uint, TextureUsageBits::depthStencilAttachment, TextureAspect::depthStencil, &gBufferDepth, width, height);

        CreateAttachments(PixelFormat::r8Unorm, TextureUsageBits::textureBinding | TextureUsageBits::renderAttachment, TextureAspect::color, &ssaoOutput, width, height);

        CreateAttachments(PixelFormat::r8Unorm, TextureUsageBits::textureBinding | TextureUsageBits::renderAttachment, TextureAspect::color, &ssaoBlurOutput, width, height);
    }

    void PrepareUniformBuffers()
    {
        // scene matries
        uboSceneParams.projection = camera.perspective;
        uboSceneParams.view = camera.view;
        uboSceneParams.model = glm::mat4(1.0f);
        CreateUniformBuffer(BufferUsageBits::uniform | BufferUsageBits::mapWrite, &uniformBuffers.sceneParams, sizeof(UBOSceneParams), &uboSceneParams);

        // ssao parameters
        ubossaoParams.projection = camera.perspective;
        CreateUniformBuffer(BufferUsageBits::uniform | BufferUsageBits::mapWrite, &uniformBuffers.ssaoParams, sizeof(UBOSSAOParams), &ubossaoParams);

        // ssao kennel
        std::default_random_engine rndEngine((unsigned)time(nullptr));
        std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);
        std::vector<glm::vec4> ssaoKernel(ssaoKernelSize);

        auto lerp = [](float a, float b, float f) ->float {
            return a + f * (b - a);
        };

        for (uint32_t i = 0; i < ssaoKernelSize; ++i)
        {
            glm::vec3 sample(rndDist(rndEngine) * 2.0 - 1.0, rndDist(rndEngine) * 2.0 - 1.0, rndDist(rndEngine));
            sample = glm::normalize(sample);
            sample *= rndDist(rndEngine);
            float scale = float(i) / float(ssaoKernelSize);
            scale = lerp(0.1f, 1.0f, scale * scale);
            ssaoKernel[i] = glm::vec4(sample * scale, 0.0f);
        }
        CreateUniformBuffer(BufferUsageBits::uniform | BufferUsageBits::mapWrite, &uniformBuffers.ssaoKernel, ssaoKernel.size() * sizeof(glm::vec4), ssaoKernel.data());

        // random noise
        std::vector<glm::vec4> ssaoNoise(ssaoNoiseDim * ssaoNoiseDim);
        for (auto& randomVec : ssaoNoise)
        {
            randomVec = glm::vec4(rndDist(rndEngine) * 2.0f - 1.0f, rndDist(rndEngine) * 2.0f - 1.0f, 0.0f, 0.0f);
        }

        BufferCreateInfo bufferCreateInfo {};
        bufferCreateInfo.size = ssaoNoise.size() * sizeof(glm::vec4);
        // To make this buffer has correct resource state(D3D12_RESOURCE_STATE_GENERIC_READ) in dx, add uniform usage flag
        bufferCreateInfo.usages = BufferUsageBits::uniform | BufferUsageBits::mapWrite | BufferUsageBits::copySrc;
        auto* pixelBuffer = device->CreateBuffer(bufferCreateInfo);
        if (pixelBuffer != nullptr) {
            auto* data = pixelBuffer->Map(MapMode::write, 0, bufferCreateInfo.size);
            memcpy(data, ssaoNoise.data(), bufferCreateInfo.size);
            pixelBuffer->UnMap();
        }

        TextureCreateInfo texCreateInfo {};
        texCreateInfo.format = PixelFormat::rgba32Float;
        texCreateInfo.mipLevels = 1;
        texCreateInfo.extent = {ssaoNoiseDim, ssaoNoiseDim, 1};
        texCreateInfo.dimension = TextureDimension::t2D;
        texCreateInfo.samples = 1;
        texCreateInfo.usages = TextureUsageBits::copyDst | TextureUsageBits::textureBinding;
        noise.tex = device->CreateTexture(texCreateInfo);

        TextureViewCreateInfo viewCreateInfo {};
        viewCreateInfo.dimension = TextureViewDimension::tv2D;
        viewCreateInfo.baseArrayLayer = 0;
        viewCreateInfo.arrayLayerNum = 1;
        viewCreateInfo.baseMipLevel = 0;
        viewCreateInfo.mipLevelNum = 1;
        viewCreateInfo.aspect = TextureAspect::color;
        noise.view = noise.tex->CreateTextureView(viewCreateInfo);

        SamplerCreateInfo samplerCreateInfo {};
        samplerCreateInfo.addressModeU = AddressMode::repeat;
        samplerCreateInfo.addressModeV = AddressMode::repeat;
        noiseSampler = device->CreateSampler(samplerCreateInfo);

        auto* texCommandBuffer = device->CreateCommandBuffer();
        auto* commandEncoder = texCommandBuffer->Begin();
        // Dx need not to transition resource state before copy
        commandEncoder->ResourceBarrier(Barrier::Transition(noise.tex.Get(), TextureState::undefined, TextureState::copyDst));
        TextureSubResourceInfo subResourceInfo {};
        subResourceInfo.mipLevel = 0;
        subResourceInfo.arrayLayerNum = 1;
        subResourceInfo.baseArrayLayer = 0;
        subResourceInfo.aspect = TextureAspect::color;
        commandEncoder->CopyBufferToTexture(pixelBuffer, noise.tex.Get(), &subResourceInfo, {ssaoNoiseDim, ssaoNoiseDim, 1});
        commandEncoder->ResourceBarrier(Barrier::Transition(noise.tex.Get(), TextureState::copyDst, TextureState::shaderReadOnly));
        commandEncoder->End();

        graphicsQueue->Submit(texCommandBuffer, nullptr);

    }

    void CreateAttachments(RHI::PixelFormat format, TextureUsageFlags flags, TextureAspect aspect, ColorAttachment* attachment, uint32_t width, uint32_t height)
    {
        TextureCreateInfo texCreateInfo {};
        texCreateInfo.format = format;
        texCreateInfo.mipLevels = 1;
        texCreateInfo.extent = {width, height, 1};
        texCreateInfo.dimension = TextureDimension::t2D;
        texCreateInfo.samples = 1;
        texCreateInfo.usages = flags;
        attachment->texture = device->CreateTexture(texCreateInfo);

        TextureViewCreateInfo viewCreateInfo {};
        viewCreateInfo.dimension = TextureViewDimension::tv2D;
        viewCreateInfo.baseArrayLayer = 0;
        viewCreateInfo.arrayLayerNum = 1;
        viewCreateInfo.baseMipLevel = 0;
        viewCreateInfo.mipLevelNum = 1;
        viewCreateInfo.aspect = aspect;
        attachment->view = attachment->texture->CreateTextureView(viewCreateInfo);
    }

    ShaderModule* GetShaderModule(const std::string& fileName, const std::string& entryPoint, RHI::ShaderStageBits shaderStage)
    {
        std::vector<uint8_t> byteCode {};
        std::vector<std::string> includePath { "SSAO/Shader"};

        CompileShader(byteCode, fileName, entryPoint, shaderStage, includePath);

        ShaderModuleCreateInfo createInfo {};
        createInfo.size = byteCode.size();
        createInfo.byteCode = byteCode.data();

        return device->CreateShaderModule(createInfo);
    }

    void CreateUniformBuffer(RHI::BufferUsageFlags flags, UBuffer* uBuffer, size_t size, void* data)
    {
        BufferCreateInfo createInfo {};
        createInfo.size = size;
        createInfo.usages = flags;

        uBuffer->buf = device->CreateBuffer(createInfo);
        if (uBuffer->buf != nullptr && data != nullptr) {
            auto* mapData = uBuffer->buf->Map(MapMode::write, 0, size);
            memcpy(mapData, data, size);
            uBuffer->buf->UnMap();
        }

        BufferViewCreateInfo viewCreateInfo {};
        viewCreateInfo.size = size;
        viewCreateInfo.offset = 0;
        uBuffer->bufView = uBuffer->buf->CreateBufferView(viewCreateInfo);
    }

    void CreatePipeline()
    {
        shaderModules.gBufferVs     = GetShaderModule("SSAO/Shader/Gbuffer.hlsl", "VSMain", ShaderStageBits::sVertex);
        shaderModules.gBufferPs     = GetShaderModule("SSAO/Shader/Gbuffer.hlsl", "FSMain", ShaderStageBits::sPixel);
        shaderModules.ssaoVs        = GetShaderModule("SSAO/Shader/Ssao.hlsl", "VSMain", ShaderStageBits::sVertex);
        shaderModules.ssaoPs        = GetShaderModule("SSAO/Shader/Ssao.hlsl", "FSMain", ShaderStageBits::sPixel);
        shaderModules.ssaoBlurVs    = GetShaderModule("SSAO/Shader/Blur.hlsl", "VSMain", ShaderStageBits::sVertex);
        shaderModules.ssaoBlurPs    = GetShaderModule("SSAO/Shader/Blur.hlsl", "FSMain", ShaderStageBits::sPixel);
        shaderModules.compositionVs = GetShaderModule("SSAO/Shader/Composition.hlsl", "VSMain", ShaderStageBits::sVertex);
        shaderModules.compositionPs = GetShaderModule("SSAO/Shader/Composition.hlsl", "FSMain", ShaderStageBits::sPixel);

        // Gbuffer vertex
        std::array<VertexAttribute, 4> vertexAttributes {};
        vertexAttributes[0].format = VertexFormat::float32X3;
        vertexAttributes[0].offset = 0;
        vertexAttributes[0].semanticName = "POSITION";
        vertexAttributes[0].semanticIndex = 0;
        vertexAttributes[1].format = VertexFormat::float32X2;
        vertexAttributes[1].offset = offsetof(Vertex, uv);
        vertexAttributes[1].semanticName = "TEXCOORD";
        vertexAttributes[1].semanticIndex = 0;
        vertexAttributes[2].format = VertexFormat::float32X4;
        vertexAttributes[2].offset = offsetof(Vertex, color);
        vertexAttributes[2].semanticName = "COLOR";
        vertexAttributes[2].semanticIndex = 0;
        vertexAttributes[3].format = VertexFormat::float32X3;
        vertexAttributes[3].offset = offsetof(Vertex, normal);
        vertexAttributes[3].semanticName = "NORMAL";
        vertexAttributes[3].semanticIndex = 0;

        VertexBufferLayout vertexBufferLayout {};
        vertexBufferLayout.stepMode = RHI::VertexStepMode::perVertex;
        vertexBufferLayout.stride = sizeof(Vertex);
        vertexBufferLayout.attributeNum = vertexAttributes.size();
        vertexBufferLayout.attributes = vertexAttributes.data();

        // quad buffer vertex
        std::array<VertexAttribute, 2> quadVertexAttributes {};
        quadVertexAttributes[0].format = VertexFormat::float32X3;
        quadVertexAttributes[0].offset = 0;
        quadVertexAttributes[0].semanticName = "POSITION";
        quadVertexAttributes[0].semanticIndex = 0;
        quadVertexAttributes[1].format = VertexFormat::float32X2;
        quadVertexAttributes[1].offset = offsetof(QuadVertex, uv);
        quadVertexAttributes[1].semanticName = "TEXCOORD";
        quadVertexAttributes[1].semanticIndex = 0;

        VertexBufferLayout quadVertexBufferLayout {};
        quadVertexBufferLayout.stepMode = RHI::VertexStepMode::perVertex;
        quadVertexBufferLayout.stride = sizeof(QuadVertex);
        quadVertexBufferLayout.attributeNum = quadVertexAttributes.size();
        quadVertexBufferLayout.attributes = quadVertexAttributes.data();

        // General pipeline infos
        GraphicsPipelineCreateInfo createInfo {};
        createInfo.vertexState.bufferLayoutNum = 1;
        createInfo.primitiveState.depthClip = false;
        createInfo.primitiveState.frontFace = RHI::FrontFace::ccw;
        createInfo.primitiveState.cullMode = CullMode::none;
        createInfo.primitiveState.topologyType = RHI::PrimitiveTopologyType::triangle;
        createInfo.primitiveState.stripIndexFormat = IndexFormat::uint32;
        createInfo.multiSampleState.count = 1;

        // Gbuffer
        {
            DepthStencilState depthStencilState {};
            depthStencilState.depthEnable = true;
            depthStencilState.depthComparisonFunc = ComparisonFunc::lessEqual;
            depthStencilState.format = PixelFormat::d32FloatS8Uint;
            
            std::array<ColorTargetState, 3> colorTargetStates {};
            colorTargetStates[0].format = PixelFormat::rgba32Float;
            colorTargetStates[0].writeFlags = ColorWriteBits::red | ColorWriteBits::green | ColorWriteBits::blue | ColorWriteBits::alpha;
            colorTargetStates[1].format = PixelFormat::rgba8Unorm;
            colorTargetStates[1].writeFlags = ColorWriteBits::red | ColorWriteBits::green | ColorWriteBits::blue | ColorWriteBits::alpha;
            colorTargetStates[2].format = PixelFormat::rgba8Unorm;
            colorTargetStates[2].writeFlags = ColorWriteBits::red | ColorWriteBits::green | ColorWriteBits::blue | ColorWriteBits::alpha;

            createInfo.depthStencilState = depthStencilState;
            createInfo.vertexState.bufferLayouts = &vertexBufferLayout;
            createInfo.fragmentState.colorTargetNum = colorTargetStates.size();
            createInfo.fragmentState.colorTargets = colorTargetStates.data();
            createInfo.vertexShader = shaderModules.gBufferVs.Get();
            createInfo.pixelShader = shaderModules.gBufferPs.Get();
            createInfo.layout = pipelineLayouts.gBuffer.Get();
            pipelines.gBuffer = device->CreateGraphicsPipeline(createInfo);
        }

        // ssao
        {
            ColorTargetState colorTargetState {};
            colorTargetState.format = PixelFormat::r8Unorm;
            colorTargetState.writeFlags = ColorWriteBits::red | ColorWriteBits::green | ColorWriteBits::blue | ColorWriteBits::alpha;
            
            DepthStencilState depthStencilState {};

            createInfo.depthStencilState = depthStencilState;
            createInfo.vertexState.bufferLayouts = &quadVertexBufferLayout;
            createInfo.fragmentState.colorTargetNum = 1;
            createInfo.fragmentState.colorTargets = &colorTargetState;
            createInfo.vertexShader = shaderModules.ssaoVs.Get();
            createInfo.pixelShader = shaderModules.ssaoPs.Get();
            createInfo.layout = pipelineLayouts.ssao.Get();
            pipelines.ssao = device->CreateGraphicsPipeline(createInfo);
        }

        // ssaoBlur
        {
            ColorTargetState colorTargetState {};
            colorTargetState.format = PixelFormat::r8Unorm;
            colorTargetState.writeFlags = ColorWriteBits::red | ColorWriteBits::green | ColorWriteBits::blue | ColorWriteBits::alpha;

            DepthStencilState depthStencilState {};

            createInfo.depthStencilState = depthStencilState;
            createInfo.vertexState.bufferLayouts = &quadVertexBufferLayout;
            createInfo.fragmentState.colorTargetNum = 1;
            createInfo.fragmentState.colorTargets = &colorTargetState;
            createInfo.vertexShader = shaderModules.ssaoBlurVs.Get();
            createInfo.pixelShader = shaderModules.ssaoBlurPs.Get();
            createInfo.layout = pipelineLayouts.ssaoBlur.Get();
            pipelines.ssaoBlur = device->CreateGraphicsPipeline(createInfo);
        }

        // composition
        {
            ColorTargetState colorTargetState {};
            colorTargetState.format = swapChainFormat;
            colorTargetState.writeFlags = ColorWriteBits::red | ColorWriteBits::green | ColorWriteBits::blue | ColorWriteBits::alpha;
            
            DepthStencilState depthStencilState {};
            
            createInfo.depthStencilState = depthStencilState;
            createInfo.vertexState.bufferLayouts = &quadVertexBufferLayout;
            createInfo.fragmentState.colorTargetNum = 1;
            createInfo.fragmentState.colorTargets = &colorTargetState;
            createInfo.vertexShader = shaderModules.compositionVs.Get();
            createInfo.pixelShader = shaderModules.compositionPs.Get();
            createInfo.layout = pipelineLayouts.composition.Get();
            pipelines.composition = device->CreateGraphicsPipeline(createInfo);
        }
    }

    void PopulateCommandBuffer()
    {
        CommandEncoder* commandEncoder = commandBuffer->Begin();
        {
            commandEncoder->ResourceBarrier(Barrier::Transition(gBufferPos.texture.Get(), TextureState::undefined, TextureState::renderTarget));
            commandEncoder->ResourceBarrier(Barrier::Transition(gBufferNormal.texture.Get(), TextureState::undefined, TextureState::renderTarget));
            commandEncoder->ResourceBarrier(Barrier::Transition(gBufferAlbedo.texture.Get(), TextureState::undefined, TextureState::renderTarget));

            std::array<GraphicsPassColorAttachment, 3> colorAttachments {};
            colorAttachments[0].clearValue = ColorNormalized<4> {0.0f, 0.0f, 0.0f, 1.0f};
            colorAttachments[0].loadOp = LoadOp::clear;
            colorAttachments[0].storeOp = StoreOp::store;
            colorAttachments[0].view = gBufferPos.view.Get();
            colorAttachments[0].resolve = nullptr;
            colorAttachments[1].clearValue = ColorNormalized<4> {0.0f, 0.0f, 0.0f, 1.0f};
            colorAttachments[1].loadOp = LoadOp::clear;
            colorAttachments[1].storeOp = StoreOp::store;
            colorAttachments[1].view = gBufferNormal.view.Get();
            colorAttachments[1].resolve = nullptr;
            colorAttachments[2].clearValue = ColorNormalized<4> {0.0f, 0.0f, 0.0f, 1.0f};
            colorAttachments[2].loadOp = LoadOp::clear;
            colorAttachments[2].storeOp = StoreOp::store;
            colorAttachments[2].view = gBufferAlbedo.view.Get();
            colorAttachments[2].resolve = nullptr;
            
            GraphicsPassDepthStencilAttachment depthAttachment {};
            depthAttachment.view = gBufferDepth.view.Get();
            depthAttachment.depthLoadOp = LoadOp::clear;
            depthAttachment.depthStoreOp = StoreOp::store;
            depthAttachment.depthReadOnly = true;
            depthAttachment.depthClearValue = 1.0;
            depthAttachment.stencilClearValue = 0.0;

            GraphicsPassBeginInfo graphicsPassBeginInfo {};
            graphicsPassBeginInfo.colorAttachmentNum = colorAttachments.size();
            graphicsPassBeginInfo.colorAttachments = colorAttachments.data();
            graphicsPassBeginInfo.depthStencilAttachment = &depthAttachment;

            auto* graphicsEncoder = commandEncoder->BeginGraphicsPass(&graphicsPassBeginInfo);
            {
                graphicsEncoder->SetPipeline(pipelines.gBuffer.Get());
                graphicsEncoder->SetScissor(0, 0, width, height);
                graphicsEncoder->SetViewport(0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1);
                graphicsEncoder->SetPrimitiveTopology(PrimitiveTopology::triangleList);
                graphicsEncoder->SetBindGroup(0, bindGroups.scene.Get());
                graphicsEncoder->SetVertexBuffer(0, vertexBufferView.Get());
                graphicsEncoder->SetIndexBuffer(indexBufferView.Get());

                for (auto& renderable : renderables) {
                    graphicsEncoder->SetBindGroup(1, renderable->bindGroup.Get());
                    graphicsEncoder->DrawIndexed(renderable->indexCount, 1, renderable->firstIndex, 0, 0);
                }
            }
            graphicsEncoder->EndPass();

            commandEncoder->ResourceBarrier(Barrier::Transition(gBufferPos.texture.Get(), TextureState::renderTarget, TextureState::shaderReadOnly));
            commandEncoder->ResourceBarrier(Barrier::Transition(gBufferNormal.texture.Get(), TextureState::renderTarget, TextureState::shaderReadOnly));
            commandEncoder->ResourceBarrier(Barrier::Transition(gBufferAlbedo.texture.Get(), TextureState::renderTarget, TextureState::shaderReadOnly));
        }

        {
            // ssao
            commandEncoder->ResourceBarrier(Barrier::Transition(ssaoOutput.texture.Get(), TextureState::undefined, TextureState::renderTarget));

            std::array<GraphicsPassColorAttachment, 1> colorAttachments {};
            colorAttachments[0].clearValue = ColorNormalized<4> {0.0f, 0.0f, 0.0f, 1.0f};
            colorAttachments[0].loadOp = LoadOp::clear;
            colorAttachments[0].storeOp = StoreOp::store;
            colorAttachments[0].view = ssaoOutput.view.Get();
            colorAttachments[0].resolve = nullptr;

            GraphicsPassBeginInfo graphicsPassBeginInfo {};
            graphicsPassBeginInfo.colorAttachmentNum = colorAttachments.size();
            graphicsPassBeginInfo.colorAttachments = colorAttachments.data();
            graphicsPassBeginInfo.depthStencilAttachment = nullptr;

            auto* graphicsEncoder = commandEncoder->BeginGraphicsPass(&graphicsPassBeginInfo);
            {
                graphicsEncoder->SetPipeline(pipelines.ssao.Get());
                graphicsEncoder->SetScissor(0, 0, width, height);
                graphicsEncoder->SetViewport(0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1);
                graphicsEncoder->SetPrimitiveTopology(PrimitiveTopology::triangleList);
                graphicsEncoder->SetBindGroup(0, bindGroups.ssao.Get());
                graphicsEncoder->SetVertexBuffer(0, quadVertexBufferView.Get());
                graphicsEncoder->SetIndexBuffer(quadIndexBufferView.Get());
                graphicsEncoder->DrawIndexed(6, 1, 0, 0, 0);
            }
            graphicsEncoder->EndPass();
            commandEncoder->ResourceBarrier(Barrier::Transition(ssaoOutput.texture.Get(), TextureState::renderTarget, TextureState::shaderReadOnly));
        }

        {
            // ssaoBlur
            commandEncoder->ResourceBarrier(Barrier::Transition(ssaoBlurOutput.texture.Get(), TextureState::undefined, TextureState::renderTarget));

            std::array<GraphicsPassColorAttachment, 1> colorAttachments {};
            colorAttachments[0].clearValue = ColorNormalized<4> {0.0f, 0.0f, 0.0f, 1.0f};
            colorAttachments[0].loadOp = LoadOp::clear;
            colorAttachments[0].storeOp = StoreOp::store;
            colorAttachments[0].view = ssaoBlurOutput.view.Get();
            colorAttachments[0].resolve = nullptr;

            GraphicsPassBeginInfo graphicsPassBeginInfo {};
            graphicsPassBeginInfo.colorAttachmentNum = colorAttachments.size();
            graphicsPassBeginInfo.colorAttachments = colorAttachments.data();
            graphicsPassBeginInfo.depthStencilAttachment = nullptr;

            auto* graphicsEncoder = commandEncoder->BeginGraphicsPass(&graphicsPassBeginInfo);
            {
                graphicsEncoder->SetPipeline(pipelines.ssaoBlur.Get());
                graphicsEncoder->SetScissor(0, 0, width, height);
                graphicsEncoder->SetViewport(0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1);
                graphicsEncoder->SetPrimitiveTopology(PrimitiveTopology::triangleList);
                graphicsEncoder->SetBindGroup(0, bindGroups.ssaoBlur.Get());
                graphicsEncoder->SetVertexBuffer(0, quadVertexBufferView.Get());
                graphicsEncoder->SetIndexBuffer(quadIndexBufferView.Get());
                graphicsEncoder->DrawIndexed(6, 1, 0, 0, 0);
            }
            graphicsEncoder->EndPass();
            commandEncoder->ResourceBarrier(Barrier::Transition(ssaoBlurOutput.texture.Get(), TextureState::renderTarget, TextureState::shaderReadOnly));
        }

        {
            auto backTextureIndex = swapChain->AcquireBackTexture();

            // composition
            commandEncoder->ResourceBarrier(Barrier::Transition(swapChainTextures[backTextureIndex], TextureState::present, TextureState::renderTarget));

            std::array<GraphicsPassColorAttachment, 1> colorAttachments {};
            colorAttachments[0].clearValue = ColorNormalized<4> {0.0f, 0.0f, 0.0f, 1.0f};
            colorAttachments[0].loadOp = LoadOp::clear;
            colorAttachments[0].storeOp = StoreOp::store;
            colorAttachments[0].view = swapChainTextureViews[backTextureIndex].Get();
            colorAttachments[0].resolve = nullptr;

            GraphicsPassBeginInfo graphicsPassBeginInfo {};
            graphicsPassBeginInfo.colorAttachmentNum = colorAttachments.size();
            graphicsPassBeginInfo.colorAttachments = colorAttachments.data();
            graphicsPassBeginInfo.depthStencilAttachment = nullptr;

            commandEncoder->ResourceBarrier(Barrier::Transition(swapChainTextures[backTextureIndex], TextureState::present, TextureState::renderTarget));
            auto* graphicsEncoder = commandEncoder->BeginGraphicsPass(&graphicsPassBeginInfo);
            {
                graphicsEncoder->SetPipeline(pipelines.composition.Get());
                graphicsEncoder->SetScissor(0, 0, width, height);
                graphicsEncoder->SetViewport(0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1);
                graphicsEncoder->SetPrimitiveTopology(PrimitiveTopology::triangleList);
                graphicsEncoder->SetBindGroup(0, bindGroups.composition.Get());
                graphicsEncoder->SetVertexBuffer(0, quadVertexBufferView.Get());
                graphicsEncoder->SetIndexBuffer(quadIndexBufferView.Get());
                graphicsEncoder->DrawIndexed(6, 1, 0, 0, 0);
            }
            graphicsEncoder->EndPass();
            commandEncoder->ResourceBarrier(Barrier::Transition(swapChainTextures[backTextureIndex], TextureState::renderTarget, TextureState::present));
        }

        commandEncoder->SwapChainSync(swapChain.Get());
        commandEncoder->End();
    }

    void SubmitCommandBufferAndPresent()
    {
        graphicsQueue->Submit(commandBuffer.Get(), fence.Get());
        fence->Wait();
        swapChain->Present();
    }

    void InitCamera()
    {
        camera.type = Camera::CameraType::firstPerson;
        camera.position = { 2.0f, -2.4f, -4.0f };
        camera.setRotation(glm::vec3(10.0f, 30.0f, 0.0f));
        camera.setPerspective(60.0f,
                              static_cast<float>(width) / static_cast<float>(height),
                              uboSceneParams.nearPlane,
                              uboSceneParams.farPlane);
    }

    void LoadGLTF()
    {
        model = std::make_unique<Model>();
        model->LoadFromFile("SSAO/Model/Voyager.gltf");
    }

    void GenerateRenderables()
    {
        for (auto* node : model->linearNodes) {
            for (auto& mesh : node->meshes) {
                renderables.emplace_back(std::make_unique<Renderable>(this, mesh));
            }
        }
    }
};

int main(int argc, char* argv[])
{
    SSAOApplication application("SSAO");
    return application.Run(argc, argv);
}
