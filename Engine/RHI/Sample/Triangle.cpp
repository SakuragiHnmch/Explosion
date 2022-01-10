//
// Created by johnk on 9/1/2022.
//

#include "Application.h"
#include <RHI/Instance.h>
using namespace RHI;

class TriangleApplication : public Application {
public:
    NON_COPYABLE(TriangleApplication)
    TriangleApplication(const std::string& n, const uint32_t w, const uint32_t h) : Application(n, w, h) {}
    ~TriangleApplication() override = default;

protected:
    void OnCreate() override
    {
        instance = Instance::CreateInstanceByPlatform();
    }

    void OnDestroy() override {}

    void OnDrawFrame() override {}

private:
    Instance* instance = nullptr;
};

int main(int argc, char* argv[])
{
    TriangleApplication application("RHI-Triangle", 1024, 768);
    return application.Run();
}
