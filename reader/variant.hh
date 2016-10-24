
// Variant datatype implementation; based on Stack Overflow post
// http://stackoverflow.com/questions/5319216/implementing-a-variant-class
// See file ./tests/variant_test.cpp for usage examples.

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <memory>

class Variant
{
public:
    Variant() { }

    template<class T>
    Variant(T inValue) :
        mImpl(new VariantImpl<T>(inValue)),
        mClassName(typeid(T).name())
    {
    }

    template<class T>
    T getValue() const
    {
        if (typeid(T).name() != mClassName)
        {
            throw std::logic_error("Non-matching types!");
        }

        return dynamic_cast<VariantImpl<T>*>(mImpl.get())->getValue();
    }

    template<class T>
    void setValue(T inValue)
    {
        mImpl.reset(new VariantImpl<T>(inValue));
        mClassName = typeid(T).name();
    }

private:
    struct AbstractVariantImpl
    {
        virtual ~AbstractVariantImpl() {}
    };

    template<class T>
    struct VariantImpl : public AbstractVariantImpl
    {
        VariantImpl(T inValue) : mValue(inValue) { }

        ~VariantImpl() {}

        T getValue() const { return mValue; }

        T mValue;
    };

    std::unique_ptr<AbstractVariantImpl> mImpl;
    std::string mClassName;
};
