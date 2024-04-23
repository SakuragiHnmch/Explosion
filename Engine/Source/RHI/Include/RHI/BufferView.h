//
// Created by johnk on 20/3/2022.
//

#pragma once

#include <cstddef>
#include <variant>

#include <Common/Utility.h>
#include <Common/Hash.h>
#include <RHI/Common.h>

namespace RHI {
    struct VertexBufferViewInfo {
        uint32_t stride;
    };

    struct IndexBufferViewInfo {
        IndexFormat format;
    };

    struct BufferViewCreateInfo {
        BufferViewType type;
        uint32_t size;
        uint32_t offset;
        std::variant<VertexBufferViewInfo, IndexBufferViewInfo> extend;

        BufferViewCreateInfo(
            BufferViewType inType = BufferViewType::max,
            uint32_t inSize = 0,
            uint32_t inOffset = 0,
            const std::variant<VertexBufferViewInfo, IndexBufferViewInfo>& inExtent = {});

        BufferViewCreateInfo& SetType(BufferViewType inType);
        BufferViewCreateInfo& SetOffset(uint32_t inOffset);
        BufferViewCreateInfo& SetSize(uint32_t inSize);
        BufferViewCreateInfo& SetExtendVertex(uint32_t inStride);
        BufferViewCreateInfo& SetExtendIndex(IndexFormat inFormat);

        size_t Hash() const;
    };

    class BufferView {
    public:
        NonCopyable(BufferView)
        virtual ~BufferView();

    protected:
        explicit BufferView(const BufferViewCreateInfo& createInfo);
    };
}
