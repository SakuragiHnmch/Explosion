//
// Created by Mevol on 2021/8/8.
//

#include <vector>

#include <FileSystem/Directory.h>
#include <FileSystem/File.h>

namespace Explosion::FileSystem {
    Directory::Directory(const std::string& path) : Entry<Directory>(path) {}

    Directory::~Directory() = default;

    Directory::Directory(const Directory& directory) : Entry<Directory>(directory.path.string()) {}

    Directory& Directory::operator=(const Directory& directory)
    {
        path = directory.path;
        return *this;
    }

    std::vector<File> Directory::ListFile() const
    {
        std::vector<File> result = {};
        if (!IsExists()) {
            return result;
        }
        const fs::directory_entry fsEntry(path);
        if (fsEntry.status().type() != fs::file_type::directory) {
            return result;
        }
        fs::directory_iterator list(fsEntry);
        for (auto& iter : list) {
            if(iter.status().type() != fs::file_type::directory) {
                result.emplace_back(iter.path().string());
            }
        }
        return result;
    }

    std::vector<Directory> Directory::ListDir() const
    {
        std::vector<Directory> result = {};
        if (!IsExists()) {
            return result;
        }
        const fs::directory_entry fsEntry(path);
        if (fsEntry.status().type() != fs::file_type::directory) {
            return result;
        }
        fs::directory_iterator list(fsEntry);
        for (auto& iter : list) {
            if (iter.status().type() == fs::file_type::directory) {
                result.emplace_back(iter.path().string());
            }
        }
        return result;
    }

    void Directory::MakeImpl()
    {
        fs::create_directories(path);
    }
}
