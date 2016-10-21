#include "../reader/variant.hh"

int main(int argc, char **argv)
{
    // Store int
    Variant v(10);
    int a = 0;
    a = v.getValue<int>();
    std::cout << "a = " << a << std::endl;

    // Store float
    v.setValue<float>(12.34);
    float d = v.getValue<float>();
    std::cout << "d = " << d << std::endl;

    // Store map<string, string>
    typedef std::map<std::string, std::string> Mapping;
    Mapping m;
    m["one"] = "uno";
    m["two"] = "due";
    m["three"] = "tre";
    v.setValue<Mapping>(m);
    Mapping m2 = v.getValue<Mapping>();
    std::cout << "m2[\"one\"] = " << m2["one"] << std::endl;
    return 0;
}
