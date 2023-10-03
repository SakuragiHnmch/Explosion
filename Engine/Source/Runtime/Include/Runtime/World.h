//
// Created by johnk on 2023/9/5.
//

#pragma once

#include <Common/Memory.h>
#include <Runtime/ECS.h>

namespace Runtime {
    class World;

    class RUNTIME_API SystemSchedule {
    public:
        SystemSchedule(World& inWorld, SystemSignature inTarget);
        ~SystemSchedule();

        template <typename S>
        SystemSchedule& ScheduleAfter();

    private:
        SystemSchedule& ScheduleAfterInternal(SystemSignature depend);

        World& world;
        SystemSignature target;
    };

    class RUNTIME_API EventSlot {
    public:
        EventSlot(World& inWorld, SystemEventSignature inTarget);
        ~EventSlot();

        template <typename S>
        EventSlot& Connect();

    private:
        World& world;
        SystemEventSignature target;
    };

    class RUNTIME_API World : public ECSHost {
    public:
        explicit World(std::string inName = "");
        ~World();

        template <typename S>
        SystemSchedule AddSetupSystem()
        {
            SystemSignature signature = CreateSystem(Internal::SignForClass<S>(), new S());
            setupSystems.emplace_back(signature);
            return SystemSchedule(*this, signature);
        }

        template <typename S>
        SystemSchedule AddTickSystem()
        {
            SystemSignature signature = CreateSystem(Internal::SignForClass<S>(), new S());
            tickSystems.emplace_back(signature);
            return SystemSchedule(*this, signature);
        }

        template <typename E>
        EventSlot Event()
        {
            SystemEventSignature signature = Internal::SignForClass<E>();
            if (!systemEventSlots.contains(signature)) {
                systemEventSlots.emplace(std::make_pair(signature, std::vector<SystemEventSignature> {}));
            }
            return EventSlot(*this, signature);
        }

        void Setup();
        void Shutdown();
        void Tick();

    protected:
        void BroadcastSystemEvent(SystemEventSignature eventSignature, const Mirror::Any& eventRef) override;

    private:
        friend class SystemSchedule;
        friend class EventSlot;
        friend class WorldTestHelper;

        SystemSignature CreateSystem(SystemSignature signature, System* systemInstance);
        void ExecuteWorkSystems(const std::vector<SystemSignature>& targets);

        bool setuped;
        std::string name;
        entt::registry registry;
        std::vector<SystemSignature> setupSystems;
        std::vector<SystemSignature> tickSystems;
        std::unordered_map<SystemSignature, Common::UniqueRef<System>> systems;
        std::unordered_map<SystemSignature, std::vector<SystemSignature>> systemDependencies;
        std::unordered_map<SystemEventSignature, std::vector<SystemSignature>> systemEventSlots;
    };

    // used for test only
#if BUILD_TEST
    class RUNTIME_API WorldTestHelper {
    public:
        explicit WorldTestHelper(World& inWorld);
        ~WorldTestHelper();

        SystemCommands HackCreateSystemCommands();

    private:
        World& world;
    };
#endif
}

namespace Runtime {
    template <typename S>
    SystemSchedule& SystemSchedule::ScheduleAfter()
    {
        return ScheduleAfterInternal(Internal::SignForClass<S>());
    }

    template <typename S>
    EventSlot& EventSlot::Connect()
    {
        SystemSignature system = world.CreateSystem(Internal::SignForClass<S>(), new S());
        Assert(world.systemEventSlots.contains(target));
        world.systemEventSlots.at(target).emplace_back(system);
        return *this;
    }
}
