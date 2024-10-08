//
// Created by johnk on 2023/7/7.
//

#pragma once

#include <Common/Math/Transform.h>
#include <Common/Serialization.h>

namespace Common {
    template <typename T>
    struct ViewTransform : Transform<T> {
        static ViewTransform LookAt(const Vec<T, 3>& inPosition, const Vec<T, 3>& inTargetPosition, const Vec<T, 3>& inUpDirection = VecConsts<T, 3>::unitZ);

        ViewTransform();
        ViewTransform(Quaternion<T> inRotation, Vec<T, 3> inTranslation);
        explicit ViewTransform(const Transform<T>& inTransform);
        ViewTransform(const ViewTransform& inOther);
        ViewTransform(ViewTransform&& inOther) noexcept;
        ViewTransform& operator=(const ViewTransform& inOther);

        Mat<T, 4, 4> GetViewMatrix() const;
    };

    using HViewTransform = ViewTransform<HFloat>;
    using FViewTransform = ViewTransform<float>;
    using DViewTransform = ViewTransform<double>;
}

namespace Common { // NOLINT
    template <typename T>
    struct Serializer<ViewTransform<T>> {
        static constexpr bool serializable = true;
        static constexpr uint32_t typeId
            = HashUtils::StrCrc32("Common::ViewTransform")
            + Serializer<T>::typeId;

        static void Serialize(SerializeStream& stream, const ViewTransform<T>& value)
        {
            TypeIdSerializer<ViewTransform<T>>::Serialize(stream);

            Serializer<Transform<T>>::Serialize(stream, value);
        }

        static bool Deserialize(DeserializeStream& stream, ViewTransform<T>& value)
        {
            if (!TypeIdSerializer<ViewTransform<T>>::Deserialize(stream)) {
                return false;
            }

            Serializer<Transform<T>>::Deserialize(stream, value);
            return true;
        }
    };
}

namespace Common {
    template <typename T>
    ViewTransform<T> ViewTransform<T>::LookAt(const Vec<T, 3>& inPosition, const Vec<T, 3>& inTargetPosition, const Vec<T, 3>& inUpDirection)
    {
        return Transform<T>::LookAt(inPosition, inTargetPosition, inUpDirection);
    }

    template <typename T>
    ViewTransform<T>::ViewTransform()
        : Transform<T>()
    {
    }

    template <typename T>
    ViewTransform<T>::ViewTransform(Quaternion<T> inRotation, Vec<T, 3> inTranslation)
        : Transform<T>(std::move(inRotation), std::move(inTranslation))
    {
    }

    template <typename T>
    ViewTransform<T>::ViewTransform(const Transform<T>& inTransform)
        : Transform<T>(inTransform)
    {
    }

    template <typename T>
    ViewTransform<T>::ViewTransform(const ViewTransform& inOther)
        : Transform<T>(inOther)
    {
    }

    template <typename T>
    ViewTransform<T>::ViewTransform(ViewTransform&& inOther) noexcept
        : Transform<T>(std::move(inOther))
    {
    }

    template <typename T>
    ViewTransform<T>& ViewTransform<T>::operator=(const ViewTransform& inOther)
    {
        Transform<T>::operator=(inOther);
        return *this;
    }

    template <typename T>
    Mat<T, 4, 4> ViewTransform<T>::GetViewMatrix() const
    {
        // before apply axis transform Mat:
        //     x+ -> from screen outer to inner
        //     y+ -> from left to right
        //     z+ -> from bottom to top
        // after apply axis transform Mat:
        //     x+ -> from left to right
        //     y+ -> from bottom to top
        //     z+ -> from screen outer to inner
        static Mat<T, 4, 4> axisTransMat = Mat<T, 4, 4>(
            0, 1, 0, 0,
            0, 0, 1, 0,
            1, 0, 0, 0,
            0, 0, 0, 1
        );
        return axisTransMat * this->GetTransformMatrixNoScale().Inverse();
    }
}