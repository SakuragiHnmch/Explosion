//
// Created by johnk on 11/1/2022.
//

#pragma once

#include <Common/Debug.h>
#include <RHI/Common.h>
#include <unordered_map>
#include <vulkan/vulkan.h>

#define VK_KHRONOS_VALIDATION_LAYER_NAME "VK_LAYER_KHRONOS_validation"

// enum map definitions
namespace RHI::Vulkan {
    DECLARE_EC_FUNC()
    DECLARE_FC_FUNC()

    ECIMPL_BEGIN(VkPhysicalDeviceType, GpuType)
        ECIMPL_ITEM(VK_PHYSICAL_DEVICE_TYPE_OTHER,          GpuType::software)
        ECIMPL_ITEM(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, GpuType::hardware)
        ECIMPL_ITEM(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,   GpuType::hardware)
        ECIMPL_ITEM(VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,    GpuType::software)
        ECIMPL_ITEM(VK_PHYSICAL_DEVICE_TYPE_CPU,            GpuType::software)
    ECIMPL_END(GpuType)

    ECIMPL_BEGIN(PixelFormat, VkFormat)
        // 8-Bits
        ECIMPL_ITEM(PixelFormat::r8Unorm, VK_FORMAT_R8_UNORM)
        ECIMPL_ITEM(PixelFormat::r8Snorm, VK_FORMAT_R8_SNORM)
        ECIMPL_ITEM(PixelFormat::r8Uint,  VK_FORMAT_R8_UINT)
        ECIMPL_ITEM(PixelFormat::r8Sint,  VK_FORMAT_R8_SINT)
        // 16-Bits
        ECIMPL_ITEM(PixelFormat::r16Uint,  VK_FORMAT_R16_UINT)
        ECIMPL_ITEM(PixelFormat::r16Sint,  VK_FORMAT_R16_SINT)
        ECIMPL_ITEM(PixelFormat::r16Float, VK_FORMAT_R16_SFLOAT)
        ECIMPL_ITEM(PixelFormat::rg8Unorm, VK_FORMAT_R8G8_UNORM)
        ECIMPL_ITEM(PixelFormat::rg8Snorm, VK_FORMAT_R8G8_SNORM)
        ECIMPL_ITEM(PixelFormat::rg8Uint,  VK_FORMAT_R8G8_UINT)
        ECIMPL_ITEM(PixelFormat::rg8Sint,  VK_FORMAT_R8G8_SINT)
        // 32-Bits
        ECIMPL_ITEM(PixelFormat::r32Uint,         VK_FORMAT_R32_UINT)
        ECIMPL_ITEM(PixelFormat::r32Sint,         VK_FORMAT_R32_SINT)
        ECIMPL_ITEM(PixelFormat::r32Float,        VK_FORMAT_R32_SFLOAT)
        ECIMPL_ITEM(PixelFormat::rg16Uint,        VK_FORMAT_R16G16_UINT)
        ECIMPL_ITEM(PixelFormat::rg16Sint,        VK_FORMAT_R16G16_SINT)
        ECIMPL_ITEM(PixelFormat::rg16Float,       VK_FORMAT_R16G16_SFLOAT)
        ECIMPL_ITEM(PixelFormat::rgba8Unorm,      VK_FORMAT_R8G8B8A8_UNORM)
        ECIMPL_ITEM(PixelFormat::rgba8UnormSrgb,  VK_FORMAT_R8G8B8A8_SRGB)
        ECIMPL_ITEM(PixelFormat::rgba8Snorm,      VK_FORMAT_R8G8B8A8_SNORM)
        ECIMPL_ITEM(PixelFormat::rgba8Uint,       VK_FORMAT_R8G8B8A8_UINT)
        ECIMPL_ITEM(PixelFormat::rgba8Sint,       VK_FORMAT_R8G8B8A8_SINT)
        ECIMPL_ITEM(PixelFormat::bgra8Unorm,      VK_FORMAT_B8G8R8A8_UNORM)
        ECIMPL_ITEM(PixelFormat::bgra8UnormSrgb,  VK_FORMAT_B8G8R8A8_SRGB)
        ECIMPL_ITEM(PixelFormat::rgb9E5Float,     VK_FORMAT_E5B9G9R9_UFLOAT_PACK32)
        ECIMPL_ITEM(PixelFormat::rgb10A2Unorm,    VK_FORMAT_A2R10G10B10_UNORM_PACK32)
        ECIMPL_ITEM(PixelFormat::rg11B10Float,    VK_FORMAT_B10G11R11_UFLOAT_PACK32)
        // 64-Bits
        ECIMPL_ITEM(PixelFormat::rg32Uint,        VK_FORMAT_R32G32_UINT)
        ECIMPL_ITEM(PixelFormat::rg32Sint,        VK_FORMAT_R32G32_SINT)
        ECIMPL_ITEM(PixelFormat::rg32Float,       VK_FORMAT_R32G32_SFLOAT)
        ECIMPL_ITEM(PixelFormat::rgba16Uint,      VK_FORMAT_R16G16B16A16_UINT)
        ECIMPL_ITEM(PixelFormat::rgba16Sint,      VK_FORMAT_R16G16B16A16_SINT)
        ECIMPL_ITEM(PixelFormat::rgba16Float,     VK_FORMAT_R16G16B16A16_SFLOAT)
        // 128-Bits
        ECIMPL_ITEM(PixelFormat::rgba32Uint,      VK_FORMAT_R32G32B32A32_UINT)
        ECIMPL_ITEM(PixelFormat::rgba32Sint,      VK_FORMAT_R32G32B32A32_SINT)
        ECIMPL_ITEM(PixelFormat::rgba32Float,     VK_FORMAT_R32G32B32A32_SFLOAT)
        // Depth-Stencil
        ECIMPL_ITEM(PixelFormat::d16Unorm,        VK_FORMAT_D16_UNORM)
        ECIMPL_ITEM(PixelFormat::d24UnormS8Uint,  VK_FORMAT_D24_UNORM_S8_UINT)
        ECIMPL_ITEM(PixelFormat::d32Float,        VK_FORMAT_D32_SFLOAT)
        ECIMPL_ITEM(PixelFormat::d32FloatS8Uint,  VK_FORMAT_D32_SFLOAT_S8_UINT)
        // Undefined
        ECIMPL_ITEM(PixelFormat::max,             VK_FORMAT_UNDEFINED)
    ECIMPL_END(VkFormat)

    ECIMPL_BEGIN(QueueType, VkQueueFlagBits)
        ECIMPL_ITEM(QueueType::graphics, VK_QUEUE_GRAPHICS_BIT)
        ECIMPL_ITEM(QueueType::compute,  VK_QUEUE_COMPUTE_BIT)
        ECIMPL_ITEM(QueueType::transfer, VK_QUEUE_TRANSFER_BIT)
    ECIMPL_END(VkQueueFlagBits)

    ECIMPL_BEGIN(TextureDimension, VkImageType)
        ECIMPL_ITEM(TextureDimension::t1D, VK_IMAGE_TYPE_1D)
        ECIMPL_ITEM(TextureDimension::t2D, VK_IMAGE_TYPE_2D)
        ECIMPL_ITEM(TextureDimension::t3D, VK_IMAGE_TYPE_3D)
    ECIMPL_END(VkImageType)

    ECIMPL_BEGIN(TextureViewDimension, VkImageViewType)
        ECIMPL_ITEM(TextureViewDimension::tv1D,        VK_IMAGE_VIEW_TYPE_1D)
        ECIMPL_ITEM(TextureViewDimension::tv2D,        VK_IMAGE_VIEW_TYPE_2D)
        ECIMPL_ITEM(TextureViewDimension::tv2DArray,   VK_IMAGE_VIEW_TYPE_2D_ARRAY)
        ECIMPL_ITEM(TextureViewDimension::tvCube,      VK_IMAGE_VIEW_TYPE_CUBE)
        ECIMPL_ITEM(TextureViewDimension::tvCubeArray, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)
        ECIMPL_ITEM(TextureViewDimension::tv3D,        VK_IMAGE_VIEW_TYPE_3D)
    ECIMPL_END(VkImageViewType)

    ECIMPL_BEGIN(ShaderStageBits, VkShaderStageFlagBits)
        ECIMPL_ITEM(ShaderStageBits::sVertex,   VK_SHADER_STAGE_VERTEX_BIT)
        ECIMPL_ITEM(ShaderStageBits::sPixel,    VK_SHADER_STAGE_FRAGMENT_BIT)
        ECIMPL_ITEM(ShaderStageBits::sCompute,  VK_SHADER_STAGE_COMPUTE_BIT)
        ECIMPL_ITEM(ShaderStageBits::sGeometry, VK_SHADER_STAGE_GEOMETRY_BIT)
        ECIMPL_ITEM(ShaderStageBits::sHull,     VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)
        ECIMPL_ITEM(ShaderStageBits::sDomain,   VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)
    ECIMPL_END(VkShaderStageFlagBits)

    ECIMPL_BEGIN(PrimitiveTopologyType, VkPrimitiveTopology)
        ECIMPL_ITEM(PrimitiveTopologyType::point,    VK_PRIMITIVE_TOPOLOGY_POINT_LIST)
        ECIMPL_ITEM(PrimitiveTopologyType::line,     VK_PRIMITIVE_TOPOLOGY_LINE_LIST)
        ECIMPL_ITEM(PrimitiveTopologyType::triangle, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
    ECIMPL_END(VkPrimitiveTopology)

    ECIMPL_BEGIN(FillMode, VkPolygonMode)
        ECIMPL_ITEM(FillMode::wireframe,    VK_POLYGON_MODE_LINE)
        ECIMPL_ITEM(FillMode::solid,        VK_POLYGON_MODE_FILL)
    ECIMPL_END(VkPolygonMode)

    ECIMPL_BEGIN(CullMode, VkCullModeFlagBits)
        ECIMPL_ITEM(CullMode::none,  VK_CULL_MODE_NONE)
        ECIMPL_ITEM(CullMode::front, VK_CULL_MODE_FRONT_BIT)
        ECIMPL_ITEM(CullMode::back,  VK_CULL_MODE_BACK_BIT)
    ECIMPL_END(VkCullModeFlagBits)

    ECIMPL_BEGIN(BlendOp, VkBlendOp)
        ECIMPL_ITEM(BlendOp::opAdd,              VK_BLEND_OP_ADD)
        ECIMPL_ITEM(BlendOp::opSubstract,        VK_BLEND_OP_SUBTRACT)
        ECIMPL_ITEM(BlendOp::opReverseSubstract, VK_BLEND_OP_REVERSE_SUBTRACT)
        ECIMPL_ITEM(BlendOp::opMin,              VK_BLEND_OP_MIN)
        ECIMPL_ITEM(BlendOp::opMax,              VK_BLEND_OP_MAX)
    ECIMPL_END(VkBlendOp)

    ECIMPL_BEGIN(BlendFactor, VkBlendFactor)
        ECIMPL_ITEM(BlendFactor::zero,             VK_BLEND_FACTOR_ZERO)
        ECIMPL_ITEM(BlendFactor::one,              VK_BLEND_FACTOR_ONE)
        ECIMPL_ITEM(BlendFactor::src,              VK_BLEND_FACTOR_SRC_COLOR)
        ECIMPL_ITEM(BlendFactor::oneMinusSrc,      VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR)
        ECIMPL_ITEM(BlendFactor::srcAlpha,         VK_BLEND_FACTOR_SRC_ALPHA)
        ECIMPL_ITEM(BlendFactor::oneMinusSrcAlpha, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA)
        ECIMPL_ITEM(BlendFactor::dst,              VK_BLEND_FACTOR_DST_COLOR)
        ECIMPL_ITEM(BlendFactor::oneMinusDst,      VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR)
        ECIMPL_ITEM(BlendFactor::dstAlpha,         VK_BLEND_FACTOR_DST_ALPHA)
        ECIMPL_ITEM(BlendFactor::oneMinusDstAlpha, VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA)
    ECIMPL_END(VkBlendFactor)

    ECIMPL_BEGIN(VertexFormat, VkFormat)
        ECIMPL_ITEM(VertexFormat::uint8X2,   VK_FORMAT_R8G8_UINT)
        ECIMPL_ITEM(VertexFormat::uint8X4,   VK_FORMAT_R8G8B8A8_UINT)
        ECIMPL_ITEM(VertexFormat::sint8X2,   VK_FORMAT_R8G8_SINT)
        ECIMPL_ITEM(VertexFormat::sint8X4,   VK_FORMAT_R8G8B8A8_SINT)
        ECIMPL_ITEM(VertexFormat::unorm8X2,  VK_FORMAT_R8G8_UNORM)
        ECIMPL_ITEM(VertexFormat::unorm8X4,  VK_FORMAT_R8G8B8A8_UNORM)
        ECIMPL_ITEM(VertexFormat::snorm8X2,  VK_FORMAT_R8G8_SNORM)
        ECIMPL_ITEM(VertexFormat::snorm8X4,  VK_FORMAT_R8G8B8A8_SNORM)
        ECIMPL_ITEM(VertexFormat::uint16X2,  VK_FORMAT_R16G16_UINT)
        ECIMPL_ITEM(VertexFormat::uint16X4,  VK_FORMAT_R16G16B16A16_UINT)
        ECIMPL_ITEM(VertexFormat::sint16X2,  VK_FORMAT_R16G16_SINT)
        ECIMPL_ITEM(VertexFormat::sint16X4,  VK_FORMAT_R16G16B16A16_SINT)
        ECIMPL_ITEM(VertexFormat::unorm16X2, VK_FORMAT_R16G16_UNORM)
        ECIMPL_ITEM(VertexFormat::unorm16X4, VK_FORMAT_R16G16B16A16_UNORM)
        ECIMPL_ITEM(VertexFormat::snorm16X2, VK_FORMAT_R16G16_SNORM)
        ECIMPL_ITEM(VertexFormat::snorm16X4, VK_FORMAT_R16G16B16A16_SNORM)
        ECIMPL_ITEM(VertexFormat::float16X2, VK_FORMAT_R16G16_SFLOAT)
        ECIMPL_ITEM(VertexFormat::float16X4, VK_FORMAT_R16G16B16A16_SFLOAT)
        ECIMPL_ITEM(VertexFormat::float32X1, VK_FORMAT_R32_SFLOAT)
        ECIMPL_ITEM(VertexFormat::float32X2, VK_FORMAT_R32G32_SFLOAT)
        ECIMPL_ITEM(VertexFormat::float32X3, VK_FORMAT_R32G32B32_SFLOAT)
        ECIMPL_ITEM(VertexFormat::float32X4, VK_FORMAT_R32G32B32A32_SFLOAT)
        ECIMPL_ITEM(VertexFormat::uint32X1,  VK_FORMAT_R32_UINT)
        ECIMPL_ITEM(VertexFormat::uint32X2,  VK_FORMAT_R32G32_UINT)
        ECIMPL_ITEM(VertexFormat::uint32X3,  VK_FORMAT_R32G32B32_UINT)
        ECIMPL_ITEM(VertexFormat::uint32X4,  VK_FORMAT_R32G32B32A32_UINT)
        ECIMPL_ITEM(VertexFormat::sint32X1,  VK_FORMAT_R32_SINT)
        ECIMPL_ITEM(VertexFormat::sint32X2,  VK_FORMAT_R32G32_SINT)
        ECIMPL_ITEM(VertexFormat::sint32X3,  VK_FORMAT_R32G32B32_SINT)
        ECIMPL_ITEM(VertexFormat::sint32X4,  VK_FORMAT_R32G32B32A32_SINT)
    ECIMPL_END(VkFormat)

    ECIMPL_BEGIN(BindingType, VkDescriptorType)
        ECIMPL_ITEM(BindingType::uniformBuffer,  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
        ECIMPL_ITEM(BindingType::storageBuffer,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
        ECIMPL_ITEM(BindingType::sampler,        VK_DESCRIPTOR_TYPE_SAMPLER)
        ECIMPL_ITEM(BindingType::texture,        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
        ECIMPL_ITEM(BindingType::storageTexture, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
    ECIMPL_END(VkDescriptorType)

    ECIMPL_BEGIN(AddressMode, VkSamplerAddressMode)
        ECIMPL_ITEM(AddressMode::clampToEdge,  VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
        ECIMPL_ITEM(AddressMode::repeat,       VK_SAMPLER_ADDRESS_MODE_REPEAT)
        ECIMPL_ITEM(AddressMode::mirrorRepeat, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT)
    ECIMPL_END(VkSamplerAddressMode)

    ECIMPL_BEGIN(FilterMode, VkFilter)
        ECIMPL_ITEM(FilterMode::nearest, VK_FILTER_NEAREST)
        ECIMPL_ITEM(FilterMode::linear,  VK_FILTER_LINEAR)
    ECIMPL_END(VkFilter)

    ECIMPL_BEGIN(FilterMode, VkSamplerMipmapMode)
        ECIMPL_ITEM(FilterMode::nearest, VK_SAMPLER_MIPMAP_MODE_NEAREST)
        ECIMPL_ITEM(FilterMode::linear,  VK_SAMPLER_MIPMAP_MODE_LINEAR)
    ECIMPL_END(VkSamplerMipmapMode)

    ECIMPL_BEGIN(CompareFunc, VkCompareOp)
        ECIMPL_ITEM(CompareFunc::never, VK_COMPARE_OP_NEVER)
        ECIMPL_ITEM(CompareFunc::less, VK_COMPARE_OP_LESS)
        ECIMPL_ITEM(CompareFunc::equal, VK_COMPARE_OP_EQUAL)
        ECIMPL_ITEM(CompareFunc::lessEqual, VK_COMPARE_OP_LESS_OR_EQUAL)
        ECIMPL_ITEM(CompareFunc::greater, VK_COMPARE_OP_GREATER)
        ECIMPL_ITEM(CompareFunc::notEqual, VK_COMPARE_OP_NOT_EQUAL)
        ECIMPL_ITEM(CompareFunc::greaterEqual, VK_COMPARE_OP_GREATER_OR_EQUAL)
        ECIMPL_ITEM(CompareFunc::always, VK_COMPARE_OP_ALWAYS)
    ECIMPL_END(VkCompareOp)

    ECIMPL_BEGIN(StencilOp, VkStencilOp)
        ECIMPL_ITEM(StencilOp::keep,           VK_STENCIL_OP_KEEP)
        ECIMPL_ITEM(StencilOp::zero,           VK_STENCIL_OP_ZERO)
        ECIMPL_ITEM(StencilOp::replace,        VK_STENCIL_OP_REPLACE)
        ECIMPL_ITEM(StencilOp::invert,         VK_STENCIL_OP_INVERT)
        ECIMPL_ITEM(StencilOp::incrementClamp, VK_STENCIL_OP_INCREMENT_AND_CLAMP)
        ECIMPL_ITEM(StencilOp::decrementClamp, VK_STENCIL_OP_DECREMENT_AND_CLAMP)
        ECIMPL_ITEM(StencilOp::incrementWrap,  VK_STENCIL_OP_INCREMENT_AND_WRAP)
        ECIMPL_ITEM(StencilOp::decrementWrap,  VK_STENCIL_OP_DECREMENT_AND_WRAP)
    ECIMPL_END(VkStencilOp)

    ECIMPL_BEGIN(LoadOp, VkAttachmentLoadOp)
        ECIMPL_ITEM(LoadOp::load,  VK_ATTACHMENT_LOAD_OP_LOAD)
        ECIMPL_ITEM(LoadOp::clear, VK_ATTACHMENT_LOAD_OP_CLEAR)
        ECIMPL_ITEM(LoadOp::max,   VK_ATTACHMENT_LOAD_OP_NONE_EXT)
    ECIMPL_END(VkAttachmentLoadOp)

    ECIMPL_BEGIN(StoreOp, VkAttachmentStoreOp)
        ECIMPL_ITEM(StoreOp::store,   VK_ATTACHMENT_STORE_OP_STORE)
        ECIMPL_ITEM(StoreOp::discard, VK_ATTACHMENT_STORE_OP_DONT_CARE)
        ECIMPL_ITEM(StoreOp::max,     VK_ATTACHMENT_STORE_OP_NONE_EXT)
    ECIMPL_END(VkAttachmentStoreOp)

    ECIMPL_BEGIN(IndexFormat, VkIndexType)
        ECIMPL_ITEM(IndexFormat::uint16, VK_INDEX_TYPE_UINT16)
        ECIMPL_ITEM(IndexFormat::uint32, VK_INDEX_TYPE_UINT32)
        ECIMPL_ITEM(IndexFormat::max,    VK_INDEX_TYPE_NONE_KHR)
    ECIMPL_END(VkIndexType)

    ECIMPL_BEGIN(TextureState, VkImageLayout)
        ECIMPL_ITEM(TextureState::undefined,    VK_IMAGE_LAYOUT_UNDEFINED)
        ECIMPL_ITEM(TextureState::renderTarget, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        ECIMPL_ITEM(TextureState::present,      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
        ECIMPL_ITEM(TextureState::max,          VK_IMAGE_LAYOUT_GENERAL)
    ECIMPL_END(VkImageLayout)

    ECIMPL_BEGIN(PresentMode, VkPresentModeKHR)
        ECIMPL_ITEM(PresentMode::immediately, VK_PRESENT_MODE_IMMEDIATE_KHR)
        ECIMPL_ITEM(PresentMode::vsync,       VK_PRESENT_MODE_FIFO_KHR)
        ECIMPL_ITEM(PresentMode::max,         VK_PRESENT_MODE_IMMEDIATE_KHR) // TODO Set the default present mode to immediate?
    ECIMPL_END(VkPresentModeKHR)

    ECIMPL_BEGIN(TextureAspect, VkImageAspectFlags)
        ECIMPL_ITEM(TextureAspect::color,   VK_IMAGE_ASPECT_COLOR_BIT)
        ECIMPL_ITEM(TextureAspect::depth,   VK_IMAGE_ASPECT_DEPTH_BIT)
        ECIMPL_ITEM(TextureAspect::stencil, VK_IMAGE_ASPECT_STENCIL_BIT)
        ECIMPL_ITEM(TextureAspect::depthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
    ECIMPL_END(VkImageAspectFlags)

    FCIMPL_BEGIN(ShaderStageFlags, VkShaderStageFlags)
        FCIMPL_ITEM(ShaderStageBits::sVertex,   VK_SHADER_STAGE_VERTEX_BIT)
        FCIMPL_ITEM(ShaderStageBits::sPixel,    VK_SHADER_STAGE_FRAGMENT_BIT)
        FCIMPL_ITEM(ShaderStageBits::sCompute,  VK_SHADER_STAGE_COMPUTE_BIT)
        FCIMPL_ITEM(ShaderStageBits::sGeometry, VK_SHADER_STAGE_GEOMETRY_BIT)
        FCIMPL_ITEM(ShaderStageBits::sHull,     VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)
        FCIMPL_ITEM(ShaderStageBits::sDomain,   VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)
    FCIMPL_END(VkShaderStageFlagBits)

    FCIMPL_BEGIN(BufferUsageFlags, VkBufferUsageFlags)
        FCIMPL_ITEM(BufferUsageBits::copySrc,  VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
        FCIMPL_ITEM(BufferUsageBits::copyDst,  VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        FCIMPL_ITEM(BufferUsageBits::index,    VK_BUFFER_USAGE_INDEX_BUFFER_BIT)
        FCIMPL_ITEM(BufferUsageBits::vertex,   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
        FCIMPL_ITEM(BufferUsageBits::uniform,  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
        FCIMPL_ITEM(BufferUsageBits::storage,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        FCIMPL_ITEM(BufferUsageBits::indirect, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT)
    FCIMPL_END(VkBufferUsageFlagBits)

    FCIMPL_BEGIN(TextureUsageFlags, VkImageUsageFlags)
        FCIMPL_ITEM(TextureUsageBits::copySrc,                VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        FCIMPL_ITEM(TextureUsageBits::copyDst,                VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        FCIMPL_ITEM(TextureUsageBits::textureBinding,         VK_IMAGE_USAGE_SAMPLED_BIT)
        FCIMPL_ITEM(TextureUsageBits::storageBinding,         VK_IMAGE_USAGE_STORAGE_BIT)
        FCIMPL_ITEM(TextureUsageBits::renderAttachment,       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        FCIMPL_ITEM(TextureUsageBits::depthStencilAttachment, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
    FCIMPL_END(VkImageUsageFlagBits)
}
