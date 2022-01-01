//
// Created by johnk on 1/1/2022.
//

#include <RHI/DirectX12/CommandQueue.h>
#include <RHI/DirectX12/Enum.h>

namespace RHI::DirectX12 {
    DX12CommandQueue::DX12CommandQueue(DX12LogicalDevice& logicalDevice, const CommandQueueCreateInfo& createInfo)
    {
        CreateCommandQueue(logicalDevice, createInfo);
    }

    DX12CommandQueue::~DX12CommandQueue() = default;

    ComPtr<ID3D12CommandQueue>& DX12CommandQueue::GetDX12CommandQueue()
    {
        return dx12CommandQueue;
    }

    void DX12CommandQueue::CreateCommandQueue(DX12LogicalDevice& logicalDevice, const CommandQueueCreateInfo& createInfo)
    {
        D3D12_COMMAND_QUEUE_DESC desc {};
        desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        desc.Type = EnumCast<CommandQueueType, D3D12_COMMAND_LIST_TYPE>(createInfo.type);

        ThrowIfFailed(
            logicalDevice.GetDX12Device()->CreateCommandQueue(&desc, IID_PPV_ARGS(&dx12CommandQueue)),
            "failed to create dx12 command queue"
        );
    }
}
