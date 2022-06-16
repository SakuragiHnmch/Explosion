//
// Created by johnk on 2022/5/25.
//

#include <sstream>
#include <filesystem>
#include <unordered_map>

#include <MetaTool/HeaderGenerator.h>
#include <MetaTool/ClangParser.h>
#include <Common/Debug.h>

namespace MetaTool {
    template <typename T>
    std::string GetContextFullName(const std::string& prefix, const T& context)
    {
        return prefix.empty() ? context.name : (prefix + "::" + context.name);
    }

    std::unordered_map<std::string, std::string> ParseMetaDatas(const std::string& name, const std::string& metaData)
    {
        // TODO
        return {
            { "name", name }
        };
    }

    template <typename Context>
    std::string GetMetaDatasCode(Context&& context)
    {
        std::unordered_map<std::string, std::string> metaDatas = ParseMetaDatas(context.name, context.metaData);
        std::stringstream stream;
        for (const auto& iter : metaDatas) {
            stream << ", MData(hash(\"" << iter.first << "\"), " << "\"" << iter.second << "\")";
        }
        return stream.str();
    }
}

namespace MetaTool {
    HeaderGenerator::HeaderGenerator(const HeaderGeneratorInfo& info)
        : info(info)
    {
        std::filesystem::path path(info.outputFilePath);
        std::filesystem::path targetDirPath = path.parent_path();
        if (!std::filesystem::exists(targetDirPath)) {
            std::filesystem::create_directories(targetDirPath);
        }

        file = std::ofstream(info.outputFilePath);
        Assert(file.is_open());
    }

    HeaderGenerator::~HeaderGenerator()
    {
        file.close();
    }

    void HeaderGenerator::Generate(const MetaContext& metaInfo)
    {
        GenerateFileHeader();
        GenerateIncludes();
        GenerateRegistry(metaInfo);
    }

    void HeaderGenerator::GenerateFileHeader()
    {
        file << "/**" << std::endl;
        file << " * Generated by Explosion header generator, do not modify this file anyway." << std::endl;
        file << " */" << std::endl;
        file << std::endl;
        file << "#pragma once" << std::endl;
        file << std::endl;
    }

    void HeaderGenerator::GenerateIncludes()
    {
        file << "#include <string_view>" << std::endl;
        file << "#include <utility>" << std::endl;
        file << std::endl;
        file << "#include <meta/factory.hpp>" << std::endl;
        file << std::endl;
        file << "#include <" << info.sourceFileShortPath << ">" << std::endl;
        file << std::endl;
    }

    void HeaderGenerator::GenerateRegistry(const MetaTool::MetaContext& metaInfo)
    {
        file << "static int _registry = []() -> int {" << std::endl;
        file << "    using MData = std::make_pair<size_t, std::string_view>;" << std::endl;
        file << "    std::hash<std::string_view> hash {};" << std::endl;
        file << std::endl;
        GenerateCodeForNamespace(metaInfo.name, metaInfo);
        file << "    return 0;" << std::endl;
        file << "}();" << std::endl;
        file << std::endl;
    }

    void HeaderGenerator::GenerateCodeForNamespace(const std::string& prefix, const MetaTool::NamespaceContext& namespaceContext)
    {
        std::string fullName = GetContextFullName(prefix, namespaceContext);
        for (const auto& n : namespaceContext.namespaces) {
            GenerateCodeForNamespace(fullName, n);
        }
        for (const auto& c : namespaceContext.classes) {
            GenerateCodeForClasses(fullName, c);
        }
    }

    void HeaderGenerator::GenerateCodeForClasses(const std::string& prefix, const MetaTool::ClassContext& classContext)
    {
        std::string fullName = GetContextFullName(prefix, classContext);
        file << "    meta::reflect<" << fullName << ">(hash(\"" << fullName << "\")" << GetMetaDatasCode(classContext) << ")";
        for (const auto& v : classContext.variables) {
            GenerateCodeForProperty(fullName, v);
        }
        for (const auto& f : classContext.functions) {
            GenerateCodeForFunction(fullName, f);
        }
        file << ";" << std::endl;
        file << std::endl;
    }

    void HeaderGenerator::GenerateCodeForProperty(const std::string& prefix, const VariableContext& variableContext)
    {
        std::string fullName = GetContextFullName(prefix, variableContext);
        file << std::endl;
        file << "        .data<&" << fullName << ">(hash(\"" << variableContext.name << "\")" << GetMetaDatasCode(variableContext) << ")";
    }

    void HeaderGenerator::GenerateCodeForFunction(const std::string& prefix, const FunctionContext& functionContext)
    {
        std::string fullName = GetContextFullName(prefix, functionContext);
        file << std::endl;
        file << "        .func<&" << fullName << ">(hash(\"" << functionContext.name << "\")" << GetMetaDatasCode(functionContext) << ")";
    }
}
