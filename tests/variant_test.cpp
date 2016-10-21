#include "../reader/variant.hh"
#include <vector>

int main(int argc, char **argv)
{
    // Store int
    Variant v;
    
    v.setValue<int>(10);
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

    std::vector<int> vv1, vv2;
    vv1.push_back(1);
    vv1.push_back(2);
    vv1.push_back(3);
    
    v.setValue< std::vector<int> >(vv1);    
    vv2 = v.getValue<std::vector<int>>();
    std::cout << "vv2[0] = " << vv2[0] << std::endl;
    std::cout << "vv2[1] = " << vv2[1] << std::endl;
    std::cout << "vv2[2] = " << vv2[2] << std::endl;


    return 0;
}
