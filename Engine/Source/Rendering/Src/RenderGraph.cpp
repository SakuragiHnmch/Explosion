//
// Created by johnk on 2023/11/28.
//

#include <ranges>

#include <Rendering/RenderGraph.h>
#include <Common/Container.h>

namespace Rendering::Internal {
    static void CompilePassForBindGroups(
        std::unordered_set<RGResourceRef>& reads,
        std::unordered_set<RGResourceRef>& writes,
        const std::vector<RGBindGroupRef>& bindGroups)
    {
        auto checkAndMark = [](std::unordered_set<RGResourceRef>& marks, RGResourceViewRef view) -> void {
            RGResourceRef resource = view->GetResourceBase();
            Assert(!marks.contains(resource));
            marks.emplace(resource);
        };
        auto predicateActions = std::unordered_map<RHI::BindingType, std::function<void(const RGBindItemDesc&)>> {
            { RHI::BindingType::uniformBuffer, [&](const RGBindItemDesc& desc) -> void { checkAndMark(reads, desc.bufferView); } },
            { RHI::BindingType::storageBuffer, [&](const RGBindItemDesc& desc) -> void { checkAndMark(writes, desc.bufferView); } },
            { RHI::BindingType::texture, [&](const RGBindItemDesc& desc) -> void { checkAndMark(reads, desc.textureView); } },
            { RHI::BindingType::storageTexture, [&](const RGBindItemDesc& desc) -> void { checkAndMark(writes, desc.textureView); } }
        };

        for (const auto* bindGroup : bindGroups) {
            const auto& bindGroupDesc = bindGroup->GetDesc();
            const auto& bindItems = bindGroupDesc.items;

            for (const auto& itemDesc : bindItems | std::views::values) {
                predicateActions.at(itemDesc.type)(itemDesc);
            }
        }
    }
}

namespace Rendering {
    RGResourceBase::RGResourceBase(RGResType inType)
        : type(inType)
        , forceUsed(false)
    {
    }

    RGResourceBase::~RGResourceBase() = default;

    RGResType RGResourceBase::Type() const
    {
        return type;
    }

    void RGResourceBase::MaskAsUsed()
    {
        forceUsed = true;
    }

    RGResourceViewBase::RGResourceViewBase(Rendering::RGResViewType inType)
        : type(inType)
    {
    }

    RGResourceViewBase::~RGResourceViewBase() = default;

    RGResViewType RGResourceViewBase::Type() const
    {
        return type;
    }

    RGBindGroupDesc RGBindGroupDesc::Create(RHI::BindGroupLayout* inLayout)
    {
        RGBindGroupDesc result;
        result.layout = inLayout;
        return result;
    }

    RGBindGroupDesc& RGBindGroupDesc::Sampler(std::string inName, RHI::Sampler* inSampler)
    {
        Assert(!items.contains(inName));

        RGBindItemDesc item;
        item.type = RHI::BindingType::sampler;
        item.sampler = inSampler;
        items.emplace(std::make_pair(std::move(inName), std::move(item)));
        return *this;
    }

    RGBindGroupDesc& RGBindGroupDesc::UniformBuffer(std::string inName, RGBufferViewRef bufferView)
    {
        Assert(!items.contains(inName));

        RGBindItemDesc item;
        item.type = RHI::BindingType::uniformBuffer;
        item.bufferView = bufferView;
        items.emplace(std::make_pair(std::move(inName), std::move(item)));
        return *this;
    }

    RGBindGroupDesc& RGBindGroupDesc::StorageBuffer(std::string inName, RGBufferViewRef bufferView)
    {
        Assert(!items.contains(inName));

        RGBindItemDesc item;
        item.type = RHI::BindingType::storageBuffer;
        item.bufferView = bufferView;
        items.emplace(std::make_pair(std::move(inName), std::move(item)));
        return *this;
    }

    RGBindGroupDesc& RGBindGroupDesc::Texture(std::string inName, RGTextureViewRef textureView)
    {
        Assert(!items.contains(inName));

        RGBindItemDesc item;
        item.type = RHI::BindingType::texture;
        item.textureView = textureView;
        items.emplace(std::make_pair(std::move(inName), std::move(item)));
        return *this;
    }

    RGBindGroupDesc& RGBindGroupDesc::StorageTexture(std::string inName, RGTextureViewRef textureView)
    {
        Assert(!items.contains(inName));

        RGBindItemDesc item;
        item.type = RHI::BindingType::storageTexture;
        item.textureView = textureView;
        items.emplace(std::make_pair(std::move(inName), std::move(item)));
        return *this;
    }

    RGBindGroup::RGBindGroup(Rendering::RGBindGroupDesc inDesc)
        : desc(std::move(inDesc))
    {
    }

    RGBindGroup::~RGBindGroup() = default;

    const RGBindGroupDesc& RGBindGroup::GetDesc() const
    {
        return desc;
    }

    RHI::BindGroup* RGBindGroup::GetRHI() const
    {
        return rhiHandle;
    }

    RGPass::RGPass(std::string inName, RGPassType inType)
        : name(std::move(inName))
          , type(inType)
          , reads()
          , writes()
    {
    }

    RGPass::~RGPass() = default;

    RGPassType RGPass::Type() const
    {
        return type;
    }

    RGCopyPass::RGCopyPass(std::string inName, RGCopyPassDesc inPassDesc, RGCopyPassExecuteFunc inFunc)
        : RGPass(std::move(inName), RGPassType::copy)
          , passDesc(std::move(inPassDesc))
          , func(std::move(inFunc))
    {
    }

    RGCopyPass::~RGCopyPass() = default;

    void RGCopyPass::Compile()
    {
        for (auto* resource: passDesc.copySrcs) {
            Assert(!reads.contains(resource));
            reads.emplace(resource);
        }
        for (auto* resource: passDesc.copyDsts) {
            Assert(writes.contains(resource));
            writes.emplace(resource);
        }
        Assert(Common::SetUtils::GetIntersection(reads, writes).size() == 0);
    }

    RGComputePass::RGComputePass(std::string inName, std::vector<RGBindGroupRef> inBindGroups
                                 , RGComputePassExecuteFunc inFunc, bool inAsyncCompute)
        : RGPass(std::move(inName), RGPassType::compute)
          , asyncCompute(inAsyncCompute)
          , bindGroups(std::move(inBindGroups))
          , func(std::move(inFunc))
    {
    }

    RGComputePass::~RGComputePass() = default;

    void RGComputePass::Compile()
    {
        Internal::CompilePassForBindGroups(reads, writes, bindGroups);
    }

    RGRasterPass::RGRasterPass(std::string inName, RGRasterPassDesc inPassDesc
                               , std::vector<RGBindGroupRef> inBindGroupds, RGRasterPassExecuteFunc inFunc)
        : RGPass(std::move(inName), RGPassType::raster)
          , passDesc(std::move(inPassDesc))
          , bindGroups(std::move(inBindGroupds))
          , func(std::move(inFunc))
    {
    }

    RGRasterPass::~RGRasterPass() = default;

    void RGRasterPass::Compile()
    {
        auto checkAndMarkWrite = [&](RGResourceViewRef view) -> void
        {
            RGResourceRef resource = view->GetResourceBase();
            Assert(!writes.contains(resource));
            writes.emplace(resource);
        };

        for (auto* colorAttachment: passDesc.colorAttachments) {
            checkAndMarkWrite(colorAttachment);
        }
        if (passDesc.depthStencilAttachment) {
            checkAndMarkWrite(passDesc.depthStencilAttachment);
        }
        Internal::CompilePassForBindGroups(reads, writes, bindGroups);
    }

    RGFencePack::RGFencePack() = default;

    RGFencePack::~RGFencePack() = default;

    RGFencePack::RGFencePack(RHI::Fence* inMainFence, RHI::Fence* inAsyncComputeFence, RHI::Fence* inAsyncCopyFence)
        : mainFence(inMainFence)
        , asyncComputeFence(inAsyncComputeFence)
        , asyncCopyFence(inAsyncCopyFence)
    {
    }

    RGBuilder::RGBuilder(RHI::Device& inDevice)
        : device(inDevice)
    {
    }

    RGBuilder::~RGBuilder() = default;

    void RGBuilder::Execute(const RGFencePack& inFencePack)
    {
        Compile();
        // TODO
    }

    void RGBuilder::Compile()
    {
        CompilePasses();
        DevirtualizeResources();
    }

    void RGBuilder::CompilePasses()
    {
        for (auto& pass : passes) {
            pass->Compile();
        }
    }

    void RGBuilder::DevirtualizeResources()
    {
        // TODO
    }

    void RGBuilder::ExecuteInternal(const Rendering::RGFencePack& inFencePack)
    {
        // TODO
    }
}
