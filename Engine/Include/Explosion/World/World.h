//
// Created by John Kindem on 2021/6/3.
//

#ifndef EXPLOSION_WORLD_H
#define EXPLOSION_WORLD_H

#include <Explosion/ECS/ECS.h>

namespace Explosion {
    class World {
    public:
        World();
        ~World();
        void Tick(float time);

    private:
        void TickSystem(float time);

        ECS::Registry registry;
    };
}

#endif //EXPLOSION_WORLD_H