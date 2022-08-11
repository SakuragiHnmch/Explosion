//
// Created by johnk on 2022/8/3.
//

#include <Engine/Engine.h>
#include <Common/Debug.h>
#include <Engine/Application.h>

namespace Engine {
    Engine& Engine::Get()
    {
        static Engine instance;
        return instance;
    }

    Engine::Engine() : application(nullptr) {}

    Engine::~Engine() = default;

    void Engine::Initialize(const EngineInitializer& inInitializer)
    {
        application = inInitializer.application;
        InitPathMapper(inInitializer.execFile, inInitializer.projectFile);
        InitInputManager();
        InitConfigManager();
    }

    void Engine::Tick()
    {
        // TODO
    }

    IApplication* Engine::GetApplication()
    {
        return application;
    }

    Common::PathMapper& Engine::GetPathMapper()
    {
        return *pathMapper;
    }

    InputManager& Engine::GetInputManager()
    {
        return *inputManager;
    }

    ConfigManager& Engine::GetConfigManager()
    {
        return *configManager;
    }

    void Engine::InitPathMapper(const std::string& execFile, const std::string& projectFile)
    {
        std::unordered_map<std::string, std::string> pathMap;
        pathMap["/Engine"] = Common::PathUtils::GetParentPath(execFile);
        pathMap["/Game"] = Common::PathUtils::GetParentPath(projectFile);
        pathMapper = std::make_unique<Common::PathMapper>(Common::PathMapper::From(pathMap));
    }

    void Engine::InitInputManager()
    {
        inputManager = std::make_unique<InputManager>();
    }

    void Engine::InitConfigManager()
    {
        configManager = std::make_unique<ConfigManager>(*pathMapper);
    }
}