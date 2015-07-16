#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <utility>

// A long binary vector, represented with a set of columns indicating where
//  the 1s are located.
// Elements are sorted descendingly for easy removal of the minimum element
//  (i.e. the "leading one").
template <typename T, typename Comp>
class VectorSet
{
public:
	typedef VectorSet<T,Comp> Self; // rust shall soon rule the world!
	typedef typename std::vector<T>::iterator iterator;
	typedef typename std::vector<T>::const_iterator const_iterator;

private:
	std::vector<T> elems;
	Comp comp;

public:
	VectorSet(Comp comp)
	: elems()
	, comp(comp)
	{ }

	template <typename Range>
	VectorSet(const Range & r, Comp comp)
	: elems(r.cbegin(), r.cend())
	, comp(comp)
	{
		std::sort(this->elems.begin(), this->elems.end(), comp);
	}

	// Honestly I would've prefered member functions "xor", "and", etc.
	//  but those are keywords :/
	Self operator^(const Self & other) const { return normal_oper(oper_xor, other); }
	Self operator&(const Self & other) const { return normal_oper(oper_and, other); }
	Self operator|(const Self & other) const { return normal_oper(oper_or, other); }
	Self operator-(const Self & other) const { return normal_oper(oper_minus, other); }

	Self & operator^=(const Self & other) { return assign_oper(oper_xor, other); }
	Self & operator&=(const Self & other) { return assign_oper(oper_and, other); }
	Self & operator|=(const Self & other) { return assign_oper(oper_or, other); }
	Self & operator-=(const Self & other) { return assign_oper(oper_minus, other); }

	std::size_t size() const { return elems.size(); }

	void insert(T value) {
		auto it = std::lower_bound(
			this->elems.begin(), this->elems.end(),
			this->comp);

		if (it != this->elems.end() && *it == value)
			throw std::logic_error("Attempt to insert element already in VectorSet");
		this->elems.insert(it, value);
	}

	bool operator==(const Self & other) const {
		return this->elems == other.elems;
	}

	bool contains(T e) const {
		return std::binary_search(
			this->elems.cbegin(), this->elems.cend(), e,
			this->comp);
	}

	bool empty() const {
		return this->elems.empty();
	}

	iterator begin() { return this->elems.begin(); }
	iterator end() { return this->elems.end(); }
	const_iterator begin() const { return this->elems.cbegin(); }
	const_iterator end() const { return this->elems.cend(); }
	const_iterator cbegin() const { return this->elems.cbegin(); }
	const_iterator cend() const { return this->elems.cend(); }

	// WARNING: Undefined behavior on empty set
	void pop_back() {
		assert(this->elems.size() > 0);
		this->elems.pop_back();
	}

	// WARNING: Undefined behavior on empty set
	T back() const {
		assert(this->elems.size() > 0);
		return this->elems.back();
	}

	// WARNING: Undefined behavior on empty set
	T front() const {
		assert(this->elems.size() > 0);
		return this->elems.front();
	}

	std::string dense_string(T width) const;

private:

	// General wrapper for calling any of <algorithm>'s std::set_XXX methods and writing output
	//  to a new vector
	template <typename F>
	static Self set_operator(const Self & a, const Self & b, F std_set_func, std::size_t maxsize)
	{
		Self result(a.comp);
		result.elems.reserve(maxsize);

		std_set_func(
			a.elems.cbegin(), a.elems.cend(),
			b.elems.cbegin(), b.elems.cend(),
			std::back_inserter(result.elems),
			std::greater<T>()
		);
		return result;
	}

	// pick out the appropriate overload of a std::set_XXX function
	#define SET_ALGO(a) a< \
		decltype(  std::declval<Self>().elems.cbegin()             ), \
		decltype(  std::declval<Self>().elems.cbegin()             ), \
		decltype(  std::back_inserter(std::declval<Self&>().elems) ), \
		Comp >

	// Operators
	static Self oper_xor(const Self & a, const Self & b) {
		return set_operator(a, b, SET_ALGO(std::set_symmetric_difference), a.size() + b.size());
	}

	static Self oper_and(const Self & a, const Self & b) {
		return set_operator(a, b, SET_ALGO(std::set_intersection), std::min(a.size(), b.size()));
	}

	static Self oper_or(const Self & a, const Self & b) {
		return set_operator(a, b, SET_ALGO(std::set_union), a.size() + b.size());
	}

	static Self oper_minus(const Self & a, const Self & b) {
		return set_operator(a, b, SET_ALGO(std::set_difference), a.size());
	}

	// Wraps one of the above into a normal operator
	template <typename F>
	Self normal_oper(F oper, const Self & other) const {
		return oper(*this, other);
	}

	// Wraps one of the above into an assignment operator
	template <typename F>
	Self & assign_oper(F oper, const Self & other) {
		Self tmp = oper(*this, other);
		std::swap(this->elems, tmp.elems);
		return *this;
	}

};

//class oper_xor

template <typename Range>
void print_range(std::ostream & out, const Range & r, std::string left="", std::string mid=" ", std::string right="")
{
	out << left;

	auto it = r.cbegin();
	if (it != r.cend())
		out << *it++;
	while (it != r.cend())
		out << mid << *it++;

	out << right;
}

template <typename Range>
std::string join_range(const Range & r, std::string left="", std::string mid=" ", std::string right="")
{
	std::ostringstream ss;
	print_range(ss, r, left, mid, right);
	return ss.str();
}

template <typename T, typename Comp>
std::string VectorSet<T,Comp>::dense_string(T width) const
{
	if (this->empty())
		return { "[]" };

	std::vector<const char*> item_strs;

//	// kind of silly now that the check is O(N), no?
//	if (width < std::max_element(this.cbegin(), this.cend()))
//		throw std::invalid_argument("specified width is too small to contain the bit set!");

	auto next_one = this->elems.crbegin();
	for (T i = 0; i < width ; i++) {
		if (next_one != this->elems.crend() && i == *next_one) {
			item_strs.push_back("1");
			next_one++;
		} else {
			item_strs.push_back("0");
		}
	}
	return join_range(item_strs, "[", " ", "]");
}

#define MAKE_SPECIALIZED_VEC_SET(name, comp)                    \
	template <typename T>                                       \
	class name                                                  \
	: public VectorSet<T, decltype(comp)>                       \
	{                                                           \
	public:                                                     \
		template <typename... Args>                             \
		name(Args&&... args)                                    \
		: VectorSet<T, decltype(comp)>(args..., (comp))         \
		{ }                                                     \
	}                                                           \

// A forward-sorted VectorSet (providing easy removal of the max element)
MAKE_SPECIALIZED_VEC_SET( MaxVecSet, std::less<T>());

// A reverse-sorted VectorSet (providing easy removal of the min element)
MAKE_SPECIALIZED_VEC_SET( MinVecSet, std::greater<T>());

