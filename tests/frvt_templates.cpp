#define NIST_EXTERN_FRVT_STRUCTS_VERSION
#define NIST_EXTERN_API_VERSION
#include "frvt11.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>

int main()
{
    auto api = FRVT_11::Interface::getImplementation();
    std::vector<uint8_t> templ(5 + 512 * sizeof(float), 0);
    std::memcpy(templ.data(), "FXT1", 4);
    templ[4] = 1;
    float value = 1.0f;
    std::memcpy(templ.data() + 5, &value, sizeof(value));
    double score = -1;
    assert(api->matchTemplates(templ, templ, score).code == FRVT::ReturnCode::Success && score == 1000);
    auto other = templ;
    value = -1.0f;
    std::memcpy(other.data() + 5, &value, sizeof(value));
    assert(api->matchTemplates(templ, other, score).code == FRVT::ReturnCode::Success && score == 0);
    other[4] = 0;
    assert(api->matchTemplates(templ, other, score).code == FRVT::ReturnCode::VerifTemplateError && score == -1);
    other[4] = 1;
    value = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(other.data() + 5, &value, sizeof(value));
    assert(api->matchTemplates(templ, other, score).code == FRVT::ReturnCode::TemplateFormatError && score == -1);
    other.pop_back();
    assert(api->matchTemplates(templ, other, score).code == FRVT::ReturnCode::TemplateFormatError && score == -1);
}
