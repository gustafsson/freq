#pragma once

#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <typeinfo>
#include "demangle.h"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define __LOCATION__ (std::string) __FUNCTION__ + " in " + (std::string) __FILE__ + "(" TOSTRING(__LINE__) ")"

#define ThrowInvalidArgument(argument) do {           \
	std::stringstream str;                                                    \
        str << "At " << __LOCATION__ << ", parameter " << vartype(argument)       \
        << " " #argument " = " << argument << ", is not valid. ";             \
	throw std::invalid_argument(str.str());                               \
} while(0)

#define ThrowInvalidArgumentStr(argument, string_expectedValues) do {         \
	std::stringstream str;                                                    \
        str << "At " << __LOCATION__ << ", parameter " << vartype(argument)       \
        << " " #argument " = " << argument << ", is not valid. "              \
	    << string_expectedValues;                                             \
	throw std::invalid_argument(str.str());                               \
} while(0)

#define ThrowLogicErrorMember(argument) do {                                  \
	std::stringstream str;                                                    \
        str << "At " << __LOCATION__ << ", variable " << vartype(argument)        \
        << " " #argument " = " << argument << ", is not valid for "           \
        << typeid(*this).name() << " {* 0x"                                   \
		<< std::oct << this << std::dec << "}. ";                             \
		throw std::logic_error(str.str());                                \
} while(0)

#define ThrowLogicErrorMemberStr(argument, string_expectedValues) do {        \
	std::stringstream str;                                                    \
        str << "At " << __LOCATION__ << ", variable " << vartype(argument)        \
        << " " #argument " = " << argument << ", is not valid for "           \
        << typeid(*this).name() << " {* 0x"                                   \
		<< std::oct << this << std::dec << "}. "                              \
	    << string_expectedValues;                                             \
		throw std::logic_error(str.str());                                \
} while(0)
