//
// Created by Mevol on 2021/8/8.
//

#ifndef EXPLOSION_FILE_H
#define EXPLOSION_FILE_H

#include <FileSystem/Entry.h>

namespace Explosion::FileSystem {
    class File : public Entry<File> {
    public:
        explicit File(const std::string& path);
        ~File() override;
        File(const File& file);
        File& operator=(const File& file);

    public:
        [[nodiscard]] std::string GetName() const;
        [[nodiscard]] std::string GetFullName() const;
        [[nodiscard]] std::string GetExtension() const;


    private:
        friend Entry<File>;

        void MakeImpl();
    };
}

#endif //EXPLOSION_FILE_H