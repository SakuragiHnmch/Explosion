//
// Created by johnk on 2022/11/24.
//

#include <sstream>
#include <filesystem>
#include <utility>

#include <MirrorTool/Generator.h>
#include <Common/Hash.h>
#include <Common/String.h>
#include <Common/IO.h>

namespace MirrorTool {
    static std::string GetFullName(const Node& node)
    {
        std::string outerName = node.outerName;
        const std::string name = node.name;
        return outerName.empty() ? name : fmt::format("{}::{}", outerName, name);
    }

    template <uint8_t TabN>
    static std::string GetMetaDataCode(const Node& node)
    {
        std::stringstream stream;
        for (const auto& [key, value] : node.metaDatas) {
            stream << Common::newline << Common::tab<TabN> << fmt::format(R"(.MetaData("{}", "{}"))", key, value);
        }
        return stream.str();
    }

    static std::string GetBestMatchHeaderPath(const std::string& inputFile, const std::vector<std::string>& headerDirs)
    {
        for (const auto& headerDir : headerDirs) {
            if (headerDir.empty()) {
                return "";
            }
            if (inputFile.starts_with(headerDir)) {
                auto result = Common::StringUtils::Replace(inputFile, headerDir, "");
                return result.starts_with("/") ? result.substr(1) : result;
            }
        }
        return "";
    }

    static std::string GetHeaderNote()
    {
        std::stringstream stream;
        stream << "/* Generated by mirror tool, do not modify this file anyway. */" << Common::newline;
        return stream.str();
    }

    static std::string GetEnumCode(const EnumInfo& enumInfo)
    {
        const auto fullName = GetFullName(enumInfo);

        std::stringstream stream;
        stream << Common::newline;
        stream << Common::tab<1> << fmt::format("Mirror::Registry::Get()") << Common::newline;
        stream << Common::tab<2> << fmt::format(R"(.Enum<{}>("{}"))", fullName, fullName);
        stream << GetMetaDataCode<3>(enumInfo);
        for (const auto& element : enumInfo.elements) {
            const auto elementFullName = GetFullName(element);
            stream << Common::newline;
            stream << Common::tab<3> << fmt::format(R"(.Value<{}>("{}"))", elementFullName, element.name);
            stream << GetMetaDataCode<4>(element);
        }
        stream << ";" << Common::newline;
        return stream.str();
    }

    static std::string GetNamespaceEnumsCode(const NamespaceInfo& ns) // NOLINT
    {
        std::stringstream stream;
        for (const auto& e : ns.enums) {
            stream << GetEnumCode(e);
        }
        for (const auto& cns : ns.namespaces) {
            stream << GetNamespaceEnumsCode(cns);
        }
        return stream.str();
    }

    static std::string GetEnumsCode(const MetaInfo& metaInfo, size_t uniqueId)
    {
        std::stringstream stream;
        stream << Common::newline;
        stream << fmt::format("int _mirrorEnumRegistry_{} = []() -> int", uniqueId) << Common::newline;
        stream << "{";
        stream << GetNamespaceEnumsCode(metaInfo.global);
        for (const auto& ns : metaInfo.namespaces) {
            stream << GetNamespaceEnumsCode(ns);
        }
        stream << Common::newline;
        stream << Common::tab<1> << "return 0;" << Common::newline;
        stream << "}();" << Common::newline;
        return stream.str();
    }

    static std::string GetFieldAccessStr(FieldAccess access)
    {
        static const std::unordered_map<FieldAccess, std::string> map = {
            { FieldAccess::pri, "Mirror::FieldAccess::faPrivate" },
            { FieldAccess::pro, "Mirror::FieldAccess::faProtected" },
            { FieldAccess::pub, "Mirror::FieldAccess::faPublic" },
        };
        return map.at(access);
    }

    static std::string GetClassCode(const ClassInfo& clazz) // NOLINT
    {
        const std::string fullName = GetFullName(clazz);
        auto defaultCtorFieldAccess = FieldAccess::pub;
        for (const auto& constructor : clazz.constructors) {
            if (constructor.parameters.empty()) {
                defaultCtorFieldAccess = constructor.fieldAccess;
            }
        }
        auto detorFieldAccess = clazz.destructor.has_value() ? clazz.destructor->fieldAccess : FieldAccess::pub;

        std::string defaultCtorAndDetorFieldAccessParams = defaultCtorFieldAccess != FieldAccess::pub || detorFieldAccess != FieldAccess::pub
            ? fmt::format(", {}, {}", GetFieldAccessStr(defaultCtorFieldAccess), GetFieldAccessStr(detorFieldAccess))
            : "";

        std::stringstream stream;
        stream << Common::newline;
        stream << fmt::format("int {}::_mirrorRegistry = []() -> int ", fullName) << Common::newline;
        stream << "{" << Common::newline;
        stream << Common::tab<1> << "Mirror::Registry::Get()";
        if (clazz.baseClassName.empty()) {
            stream << Common::newline << Common::tab<2> << fmt::format(R"(.Class<{}, void{}>("{}"))", fullName, defaultCtorAndDetorFieldAccessParams, fullName);
        } else {
            stream << Common::newline << Common::tab<2> << fmt::format(R"(.Class<{}, {}{}>("{}"))", fullName, clazz.baseClassName, defaultCtorAndDetorFieldAccessParams, fullName);
        }
        stream << GetMetaDataCode<3>(clazz);
        for (const auto& constructor : clazz.constructors) {
            const std::string fieldAccessStr = constructor.fieldAccess != FieldAccess::pub ? fmt::format(", {}", GetFieldAccessStr(constructor.fieldAccess)) : "";
            stream << Common::newline << Common::tab<3> << fmt::format(R"(.Constructor<{}{}>("{}"))", constructor.name, fieldAccessStr, constructor.name);
            stream << GetMetaDataCode<4>(constructor);
        }
        for (const auto& staticVariable : clazz.staticVariables) {
            const std::string variableName = GetFullName(staticVariable);
            const std::string fieldAccessStr = staticVariable.fieldAccess != FieldAccess::pub ? fmt::format(", {}", GetFieldAccessStr(staticVariable.fieldAccess)) : "";
            stream << Common::newline << Common::tab<3> << fmt::format(R"(.StaticVariable<&{}{}>("{}"))", variableName, fieldAccessStr, staticVariable.name);
            stream << GetMetaDataCode<4>(staticVariable);
        }
        for (const auto& staticFunction : clazz.staticFunctions) {
            const std::string functionName = GetFullName(staticFunction);
            const std::string fieldAccessStr = staticFunction.fieldAccess != FieldAccess::pub ? fmt::format(", {}", GetFieldAccessStr(staticFunction.fieldAccess)) : "";
            stream << Common::newline << Common::tab<3> << fmt::format(R"(.StaticFunction<&{}{}>("{}"))", functionName, fieldAccessStr, staticFunction.name);
            stream << GetMetaDataCode<4>(staticFunction);
        }
        // TODO overload support
        for (const auto& variable : clazz.variables) {
            const std::string variableName = GetFullName(variable);
            const std::string fieldAccessStr = variable.fieldAccess != FieldAccess::pub ? fmt::format(", {}", GetFieldAccessStr(variable.fieldAccess)) : "";
            stream << Common::newline << Common::tab<3> << fmt::format(R"(.MemberVariable<&{}{}>("{}"))", variableName, fieldAccessStr, variable.name);
            stream << GetMetaDataCode<4>(variable);
        }
        for (const auto& function : clazz.functions) {
            const std::string functionName = GetFullName(function);
            const std::string fieldAccessStr = function.fieldAccess != FieldAccess::pub ? fmt::format(", {}", GetFieldAccessStr(function.fieldAccess)) : "";
            stream << Common::newline << Common::tab<3> << fmt::format(R"(.MemberFunction<&{}{}>("{}"))", functionName, fieldAccessStr, function.name);
            stream << GetMetaDataCode<4>(function);
        }
        // TODO overload support
        stream << ";" << Common::newline;
        stream << Common::tab<1> << "return 0;" << Common::newline;
        stream << "}();" << Common::newline;
        stream << Common::newline;
        stream << fmt::format("const Mirror::Class& {}::GetStaticClass()", fullName) << Common::newline;
        stream << "{" << Common::newline;
        stream << Common::tab<1> << fmt::format("static const Mirror::Class& clazz = Mirror::Class::Get<{}>();", fullName) << Common::newline;
        stream << Common::tab<1> << "return clazz;" << Common::newline;
        stream << "}" << Common::newline;
        stream << fmt::format("const Mirror::Class& {}::GetClass()", fullName) << Common::newline;
        stream << "{" << Common::newline;
        stream << Common::tab<1> << fmt::format("static const Mirror::Class& clazz = Mirror::Class::Get<{}>();", fullName) << Common::newline;
        stream << Common::tab<1> << "return clazz;" << Common::newline;
        stream << "}" << Common::newline;
        stream << Common::newline;

        for (const auto& internalClass : clazz.classes) {
            stream << GetClassCode(internalClass);
        }
        return stream.str();
    }

    static std::string GetNamespaceClassesCode(const NamespaceInfo& ns) // NOLINT
    {
        std::stringstream stream;
        for (const auto& clazz : ns.classes) {
            stream << GetClassCode(clazz);
        }
        for (const auto& cns : ns.namespaces) {
            stream << GetNamespaceClassesCode(cns);
        }
        return stream.str();
    }

    static std::string GetClassesCode(const MetaInfo& metaInfo)
    {
        std::stringstream stream;
        stream << GetNamespaceClassesCode(metaInfo.global);
        for (const auto& ns : metaInfo.namespaces) {
            stream << GetNamespaceClassesCode(ns);
        }
        return stream.str();
    }

    static std::string GetNamespaceGlobalCode(const NamespaceInfo& ns) // NOLINT
    {
        std::stringstream stream;
        for (const auto& var : ns.variables) {
            const auto fullName = GetFullName(var);

            stream << Common::newline;
            stream << Common::tab<1> << "Mirror::Registry::Get()" << Common::newline;
            stream << Common::tab<2> << ".Global()" << Common::newline;
            stream << Common::tab<3> << fmt::format(R"(.Variable<&{}>("{}"))", fullName, fullName);
            stream << GetMetaDataCode<4>(var);
            stream << ";" << Common::newline;
        }
        for (const auto& func : ns.functions) {
            const auto fullName = GetFullName(func);

            stream << Common::newline;
            stream << Common::tab<1> << "Mirror::Registry::Get()" << Common::newline;
            stream << Common::tab<2> << ".Global()" << Common::newline;
            stream << Common::tab<3> << fmt::format(R"(.Function<&{}>("{}"))", fullName, fullName);
            stream << GetMetaDataCode<4>(func);
            stream << ";" << Common::newline;
        }
        // TODO overload support

        for (const auto& cns : ns.namespaces) {
            stream << GetNamespaceGlobalCode(cns);
        }
        return stream.str();
    }

    static std::string GetGlobalCode(const MetaInfo& metaInfo, size_t uniqueId)
    {
        std::stringstream stream;
        stream << Common::newline;
        stream << fmt::format("int _globalRegistry_{} = []() -> int", uniqueId) << Common::newline;
        stream << "{";
        stream << GetNamespaceGlobalCode(metaInfo.global);
        for (const auto& ns : metaInfo.namespaces) {
            stream << GetNamespaceGlobalCode(ns);
        }
        stream << Common::newline;
        stream << Common::tab<1> << "return 0;" << Common::newline;
        stream << "}();" << Common::newline;
        return stream.str();
    }
}

namespace MirrorTool {
    Generator::Generator(std::string inInputFile, std::string inOutputFile, std::vector<std::string> inHeaderDirs, const MetaInfo& inMetaInfo)
        : metaInfo(inMetaInfo)
        , inputFile(std::move(inInputFile))
        , outputFile(std::move(inOutputFile))
        , headerDirs(std::move(inHeaderDirs))
    {
    }

    Generator::~Generator() = default;

    Generator::Result Generator::Generate() const
    {
        if (const std::filesystem::path parentPath = std::filesystem::path(outputFile).parent_path();
            !std::filesystem::exists(parentPath)) {
            std::filesystem::create_directories(parentPath);
        }

        std::ifstream inFile(inputFile);
        if (inFile.fail()) {
            return std::make_pair(false, "failed to open input file");
        }

        std::ofstream outFile(outputFile);
        if (outFile.fail()) {
            return std::make_pair(false, "failed to open output file");
        }

        auto result = GenerateCode(inFile, outFile, Common::HashUtils::CityHash(outputFile.data(), outputFile.size()));
        outFile.close();
        return result;
    }

    Generator::Result Generator::GenerateCode(std::ifstream& inFile, std::ofstream& outFile, size_t uniqueId) const
    {
        std::string bestMatchHeaderPath = GetBestMatchHeaderPath(inputFile, headerDirs);
        if (bestMatchHeaderPath.empty()) {
            return std::make_pair(false, "failed to compute best match header path");
        }

        outFile << GetHeaderNote() << Common::newline;
        outFile << fmt::format("#include <{}>", bestMatchHeaderPath) << Common::newline;
        outFile << "#include <Mirror/Registry.h>" << Common::newline;
        outFile << GetGlobalCode(metaInfo, uniqueId);
        outFile << GetEnumsCode(metaInfo, uniqueId);
        outFile << GetClassesCode(metaInfo);
        return std::make_pair(true, "");
    }
}
