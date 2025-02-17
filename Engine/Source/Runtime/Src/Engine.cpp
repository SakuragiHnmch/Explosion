//
// Created by johnk on 2024/8/21.
//

#include <Runtime/Engine.h>
#include <Common/Debug.h>
#include <Common/Time.h>
#include <Core/Module.h>
#include <Core/Paths.h>
#include <Core/Log.h>
#include <Core/Thread.h>
#include <Core/Console.h>
#include <Mirror/Mirror.h>
#include <Runtime/World.h>

namespace Runtime {
    Engine::Engine(const EngineInitParams& inParams)
    {
        Core::ThreadContext::SetTag(Core::ThreadTag::game);
        if (!inParams.projectFile.empty()) {
            Core::Paths::SetCurrentProjectFile(inParams.projectFile);
            LoadProject(inParams.projectFile);
        }

        if (inParams.logToFile) {
            AttachLogFile();
        }
        InitRender(inParams.rhiType);
    }

    Engine::~Engine()
    {
        renderModule->DeInitialize();
        ::Core::ModuleManager::Get().Unload("Render");
    }

    void Engine::MountWorld(World* inWorld)
    {
        worlds.emplace(inWorld);
    }

    void Engine::UnmountWorld(World* inWorld)
    {
        worlds.erase(inWorld);
    }

    Render::RenderModule& Engine::GetRenderModule() const
    {
        return *renderModule;
    }

    void Engine::Tick(float inDeltaTimeSeconds)
    {
        // game thread can run faster than render thread 1 frame as max
        if (last2FrameRenderThreadFence.valid()) {
            last2FrameRenderThreadFence.wait();
        }

        auto& renderThread = renderModule->GetRenderThread();
        renderThread.EmplaceTask([]() -> void {
            Core::Console::Get().PerformRenderThreadSettingsCopy();
        });

        for (auto* world : worlds) {
            if (!world->Playing()) {
                continue;
            }
            world->Tick(inDeltaTimeSeconds);
        }

        last2FrameRenderThreadFence = std::move(lastFrameRenderThreadFence);
        lastFrameRenderThreadFence = renderThread.EmplaceTask([]() -> void {});
    }

    Common::UniquePtr<World> Engine::CreateWorld(const std::string& inName) const // NOLINT
    {
        return new World(inName);
    }

    Project* Engine::GetProject()
    {
        return project.has_value() ? &project.value() : nullptr;
    }

    void Engine::SaveProject() const
    {
        if (!project.has_value()) {
            return;
        }
        Common::JsonSerializeToFile<Project>(Core::Paths::ProjectFile().String(), project.value());
    }

    void Engine::AttachLogFile() // NOLINT
    {
        const auto time = Common::Time(Common::TimePoint::Now());
        const auto logName = Core::Paths::ExecutablePath().FileNameWithoutExtension() + "-" + time.ToString() + ".log";
        const auto logFile = ((Core::Paths::HasSetProjectFile() ? Core::Paths::ProjectLogDir() : Core::Paths::EngineLogDir()) / logName).String();

        Core::Logger::Get().Attach(new Core::FileLogStream(logFile));
        LogInfo(Core, "logger attached to file {}", logFile);
    }

    void Engine::InitRender(const std::string& inRhiTypeStr)
    {
        renderModule = ::Core::ModuleManager::Get().FindOrLoadTyped<Render::RenderModule>("Render");
        Assert(renderModule != nullptr);

        Render::RenderModuleInitParams initParams;
        initParams.rhiType = RHI::GetRHITypeByAbbrString(inRhiTypeStr);
        renderModule->Initialize(initParams);
        LogInfo(Render, "RHI type: {}", inRhiTypeStr);
    }

    void Engine::LoadProject(const std::string& inProjectFile)
    {
        project = Project();
        Common::JsonDeserializeFromFile<Project>(Core::Paths::ProjectFile().String(), project.value());
    }

    Common::UniquePtr<Engine> EngineHolder::engine = nullptr;

    MinEngine::MinEngine(const EngineInitParams& inParams)
        : Engine(inParams)
    {
    }

    MinEngine::~MinEngine() = default;

    bool MinEngine::IsEditor()
    {
        return false;
    }

    void EngineHolder::Load(const std::string& inModuleName, const EngineInitParams& inInitParams)
    {
        Assert(engine == nullptr);
        auto& gameModule = Core::ModuleManager::Get().GetOrLoadTyped<IEngineModule>(inModuleName);
        engine = gameModule.CreateEngine(inInitParams);
    }

    void EngineHolder::Unload()
    {
        Assert(engine != nullptr);
        engine = nullptr;
    }

    Engine& EngineHolder::Get()
    {
        AssertWithReason(engine != nullptr, "no valid engine");
        return *engine;
    }
}
