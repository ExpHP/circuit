
#include <vector>
#include <algorithm>
#include <iterator>
#include <cstdint>
#include <cassert>
#include <queue>
#include <map>
#include <stdexcept>
#include <sstream>
#include <iostream>

#include "vectorset.hpp"
#include "xorbasis.h"

using namespace std;

//------------------------------------------------------------------------------

bool is_ref(const RowV & rows);
bool is_rref(const RowV & rows);

//------------------------------------------------------------------------------

// Helper methods for working with rows

bool is_zero(const Row & row) { return row.empty(); }

column_t leading_column(const Row & row) { return row.back(); }

bool is_zero(const pair<Row,Aug> & rowaug) { return is_zero(rowaug.first); }

column_t leading_column(const pair<Row,Aug> & rowaug) { return leading_column(rowaug.first); }

template <typename T>
column_t ref_order(const T & a) {
	if (is_zero(a)) {
		return std::numeric_limits<column_t>::max();
	} else {
		return leading_column(a);
	}
}

template <typename T>
bool ref_order_less(const T & a, const T & b) {
	return ref_order(a) < ref_order(b);
}

template <typename T>
bool ref_order_greater(const T & a, const T & b) {
	return ref_order(a) > ref_order(b);
}

//------------------------------------------------------------------------------

// Helper methods for packing/unpacking vectors of pairs
template <typename A, typename B>
vector<pair<A,B>> vec_zip(const vector<A> & a, const vector<B> & b)
{
	if (a.size() != b.size())
		throw logic_error("zipping vecs of unequal length");

	vector<pair<A,B>> ab;
	ab.reserve(a.size());

	auto ita = a.cbegin();
	auto itb = b.cbegin();
	while(ita != a.cend()) {
		ab.emplace_back(*ita++, *itb++);
	}
	return ab;
}

template <typename A, typename B>
pair<vector<A>,vector<B>> vec_unzip(const vector<pair<A,B>> & ab)
{
	vector<A> a; a.reserve(ab.size());
	vector<B> b; b.reserve(ab.size());

	auto it = ab.cbegin();
	while(it != ab.cend()) {
		a.push_back(it->first);
		b.push_back(it->second);
		it++;
	}
	return make_pair(std::move(a), std::move(b));
}

//------------------------------------------------------------------------------

// helper to construct a std::function without specifying the type parameters
template<typename T>
std::function<T> make_function(T *t)
{
	return { t };
}

// helper to construct a std::priority_queue with fewer type parameters
//  (just the contained value type)
template<typename T, typename It, typename F>
std::priority_queue<T, vector<T>, F>
make_priority_queue(It start, It stop, F func)
{
	return { start, stop, func };
}

// Computes rank for rows in REF or RREF form
size_t ref_rank(const RowV & rows) {
	size_t i = rows.size();
	while (i > 0 && is_zero(rows[i-1]))
		i--;
	return i;
}

// Transforms an arbitrary augmented matrix into REF form.
std::pair<RowV, AugV> transform_to_ref(const RowV & rows, const AugV & augs)
{
	// degenerate case: no rows
	if (rows.size() == 0u)
		return { {}, {} };

	// This process mostly consists of just sorting the rows by leading column, although
	//  conflicts must be resolved between rows sharing the same leading column.

	RowAugV rowaugs = vec_zip(rows, augs);

	RowV out_rows;
	AugV out_augs;

	// make a MIN-heap to produce elements in REF order
	auto f_comp = make_function(ref_order_greater<RowAug>);
	auto heap = make_priority_queue<RowAug>(rowaugs.cbegin(), rowaugs.cend(), f_comp);

	// TODO: possible optimization point: try using heap functions directly instead of
	//  through priority_queue.  This would allow moving objects out of the top of the
	//  heap, as well as enabling us to perform a combination pop-push.

	// deal with nonzero rows
	if (!is_zero(heap.top())) {

		// start with one already in the result
		out_rows.push_back(heap.top().first);
		out_augs.push_back(heap.top().second);
		heap.pop();

		while (!heap.empty() && !is_zero(heap.top())) {

			RowAug top = heap.top(); heap.pop();

			// conflict with last inserted row?
			const Row & last = out_rows.back();
			assert(!is_zero(top) && !is_zero(last));
			if (leading_column(top) == leading_column(last)) {

				top.first  ^= out_rows.back();
				top.second ^= out_augs.back();

				assert(ref_order(top) > ref_order(last)); // top's priority has decreased...
				heap.push(std::move(top));                // ...so put it back in line

			// no conflict; insert
			} else {
				out_rows.push_back(std::move(top.first));
				out_augs.push_back(std::move(top.second));
			}
		}
	}

	// deal with empty rows (all of which should have gathered at the end)
	while (!heap.empty()) {
		RowAug top = heap.top(); heap.pop();

		assert(is_zero(top));
		out_rows.push_back(std::move(top.first));
		out_augs.push_back(std::move(top.second));
	}

	assert(out_rows.size() == out_augs.size());
	assert(out_rows.size() == rows.size());
	assert(is_ref(out_rows));
	return { std::move(out_rows), std::move(out_augs) };
}

//------------------------------------------------------------------------------

void ref_transform_to_rref_inplace(RowV & rows, AugV & augs)
{
	assert(is_ref(rows));

	// All remaining conflicts are between a row that "owns" a column and some other row above it.
	// We'll work upwards from the last non-empty row.
	std::map<column_t, size_t> col_owners;

	for (size_t i = ref_rank(rows); i --> 0 ;) {
		assert(!is_zero(rows[i]));

		column_t initial_lead = leading_column(rows[i]);

		// gather rows that "own" ones in this row
		std::vector<size_t> conflicts;
		for (column_t c : rows[i]) {

			auto it = col_owners.find(c);
			if (it != col_owners.end()) // not all columns have an owner!
				conflicts.push_back(it->second);
		}

		// resolve the conflicts
		for (size_t k: conflicts) {
			rows[i] ^= rows[k];
			augs[i] ^= augs[k];
		}

		// leading columns do not change going from REF to RREF
		assert(initial_lead == leading_column(rows[i]));

		col_owners[leading_column(rows[i])] = i;
	}

	assert(rows.size() == augs.size());
	assert(is_rref(rows));
}

// Transforms an REF matrix into RREF form.
pair<RowV, AugV> ref_transform_to_rref(RowV rows, AugV augs) {
	size_t old_size = rows.size();
	ref_transform_to_rref_inplace(rows, augs);

	assert(rows.size() == old_size);
	assert(rows.size() == augs.size());
	assert(is_rref(rows));
	return { std::move(rows), std::move(augs) };
}

// Transforms an arbitrary augmented matrix into RREF form.
pair<RowV, AugV> transform_to_rref(const RowV & in_rows, const AugV & in_augs)
{
	RowV rows; AugV augs;
	std::tie(rows, augs) = transform_to_ref(in_rows, in_augs);

	return ref_transform_to_rref(std::move(rows), std::move(augs));
}

//------------------------------------------------------------------------------

// Adds rows from an REF matrix to the given row until it is either empty, or has
//  a leading 1 different from any other row in the matrix.
// Returns an index to where the row may be inserted to preserve REF.
size_t ref_reduce_row_inplace(const RowV & ref_rows, const AugV & ref_augs, Row & row, Aug & aug)
{
	assert(is_ref(ref_rows));

	auto it = ref_rows.cbegin();
	auto stop = it + ref_rank(ref_rows); // iterator to end of non-empty rows

	while (true) {

		if (is_zero(row))
			break; // no conflict possible

		// find where the row *would* belong
		it = std::lower_bound(it, stop, row, ref_order_less<Row>);

		if (it == stop)
			break; // no conflict (leading one is too large)
		if (ref_order(row) < ref_order(*it))
			break; // no conflict

		assert(ref_order(row) == ref_order(*it)); // conflict...

		size_t i = it - ref_rows.cbegin();
		row ^= ref_rows[i];
		aug ^= ref_augs[i];

		assert(ref_order(row) > ref_order(*it)); // ...resolved!
	}
	return it - ref_rows.cbegin();
}

// Insert an arbitrary row, maintaining REF.  Returns the insertion index.
size_t ref_insert(RowV & ref_rows, AugV & ref_augs, Row row, Aug aug) {
	assert(is_ref(ref_rows));

	size_t i = ref_reduce_row_inplace(ref_rows, ref_augs, row, aug);
	ref_rows.insert(ref_rows.begin() + i, std::move(row));
	ref_augs.insert(ref_augs.begin() + i, std::move(aug));

	assert(is_ref(ref_rows));
	return i;
}

// It is nontrivial to insert a single row without performing a full RREF transform,
//   though not impossible (after all, it used to be the only feature implemented here!)
// But my implementation is... ugly.
// size_t rref_insert(RowV & ref_rows, AugV & ref_augs, Row row, Aug aug);

// Insert a bunch of rows, maintaining RREF
void rref_insert_bunch(RowV & rref_rows, AugV & rref_augs, RowV new_rows, AugV new_augs) {
	assert(is_rref(rref_rows));

	//  Insert rows one at a time maintaining REF.
	//  Note that it is NOT possible to simpy reduce each row against the matrix and
	//    then insert them all at once;  while it would remove all conflicts against
	//    existing rows in the matrix, it would not prevent the new rows from conflicting
	//    __with eachother__.
	for (size_t i=0; i<new_rows.size(); i++)
		ref_insert(rref_rows, rref_augs, std::move(new_rows[i]), std::move(new_augs[i]));

	ref_transform_to_rref_inplace(rref_rows, rref_augs);

	assert(is_rref(rref_rows));
}

bool is_ref(const RowV & rows) {
	size_t nnz = ref_rank(rows);
	for (size_t i=0; i < rows.size(); i++)
		if (is_zero(rows[i]) != (i >= nnz))
			return false;
	for (size_t i=0; i < nnz-1; i++)
		if (leading_column(rows[i]) >= leading_column(rows[i+1]))
			return false;
	return true;
}

bool is_rref(const RowV & rows) {
	if (!is_ref(rows))
		return false;

	vector<column_t> leading_ones;
	for (size_t i = ref_rank(rows); i --> 0 ;) {
		for (auto it=rows[i].cbegin(); it!=rows[i].cend(); ++it) {
			if (std::binary_search(leading_ones.cbegin(), leading_ones.cend(), *it)) {
				return false;
			}
		}
		leading_ones.push_back(leading_column(rows[i]));
	}

	// above loop could falsely succeed if this didn't hold (REF should guarantee it holds)
	assert(std::is_sorted(leading_ones.crbegin(), leading_ones.crend()));

	return true;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
/**        OLD CODE BELOW         **/

/*

	// Remove rows corresponding to the ids in the provided range
	template <typename IdentityTRange>
	void remove_ids(const IdentityTRange & r) {
		// Complexity: I think  r.size() * rows.size().  (there's room for improvement, no doubt!)

		std::vector<identity_t> rvec = { r.cbegin(), r.cend() };

		auto should_remove = [&](const Row & row) {
			return std::find(rvec.begin(), rvec.end(), row.primary_id) != rvec.end();
		};

		auto newend = std::remove_if(this->rows.begin(), this->rows.end(), should_remove);
		this->shorten(newend - rows.begin());

		this->nnz_rows = this->_compute_nnz_rows();
	}

	void remove_empty_rows() {
		this->shorten(this->nnz_rows);
	}

	column_t compute_max_column() const {
		if (this->nnz_rows == 0u)
			throw std::runtime_error("no nonzero rows; max column is degenerate");

		std::vector<column_t> maxes;
		auto it = this->nonzero_cbegin();
		while (it != this->nonzero_cend())
			maxes.push_back((it++)->max_col());

		auto max_it = std::max_element(maxes.cbegin(), maxes.cend());
		assert(max_it != maxes.cend());
		return *max_it;
	}

	std::string dense_string() const {
		column_t width;
		if (this->nnz_rows == 0u)
			width = 0u;
		else
			width = this->compute_max_column();

		std::vector<std::string> item_strs;
		for (auto r : this->rows)
			item_strs.push_back(r.bits.dense_string(width));

		return join_range(item_strs, "[", "\n ", "]");
	}

Row SparseBitRref::reduce_row(Row row) const {

	MinVecSet remaining = row.bits;

	// `remaining` serves somewhat as an iterator over row, taking into account the fact
	//  that the remaining elements of `row` will change as other rows are added into it.
	//  into it.
	while (!remaining.empty()) {
		// FIXME this won't work
		auto it = this->insertion_point(row);  // locate closest leading one

		column_t prev_lead = remaining.back();

		assert((it == this->nonzero_cend()) || !(it->is_zero()));

		if ((it == this->nonzero_cend())    // leading one is larger than any we have
			|| (remaining.back() != it->lead()))  // leading one is between two existing leading ones
		{
			// no conflict; check next element
			remaining.pop_back();
		} else {
			// resolve conflict with leading 1
			row.xor_update(*it);
			remaining ^= it->bits; // update our "iterator" as well
		}

		// While `remaining`s length may not always decrease, the leading element WILL increase
		//  every iteration (and as its value is bounded, the loop will eventually terminate)
		assert(row.is_zero() || (remaining.back() > prev_lead));
	}

	assert(remaining.empty());
	return row;
}

*/

//------------------------------------------------------------------------------

// Unconditionally insert a row, maintaining RREF.
identity_t _XorBasisBuilder::add(const std::vector<column_t> & e)
{
	Row row = { e };
	auto id = assign_identity(row);
	Aug aug = original_aug(id);
	rref_insert_bunch(rows, augs, { row }, { aug });
	return id;
}

// Unconditionally insert many rows, maintaining RREF.
std::vector<identity_t> _XorBasisBuilder::add_many(std::vector<std::vector<column_t>> r)
{
	RowV added_rows;
	AugV added_augs;
	vector<identity_t> ids;
	for (auto e : r) {
		Row row = { e };
		auto id = assign_identity(row);

		added_rows.push_back(std::move(row));
		added_augs.push_back(original_aug(id));
		ids.push_back(id);
	}

	rref_insert_bunch(rows, augs, added_rows, added_augs);
	return std::move(ids);
}

// Insert a single row, maintaining RREF... but only if it is not linearly dependent
//  with rows in the matrix.
std::pair<bool, identity_t> _XorBasisBuilder::add_if_linearly_independent(std::vector<column_t> e)
{

	// copy, add, then revert if an empty row appears.

	// TODO: How (in)-efficient is this?  It begs for that single-row full reduce
	//        algorithm I'm trying so hard to get rid of :/

	AugV augs_copy = augs; // these might change even when a degenerate row is added
						   // (TODO: is this actually true? I mean for the rows that
						   //        weren't empty to begin with...)

#ifndef NDEBUG
	RowV rows_copy = rows;
#endif

	// The "rank" of the matrix is also where the nonempty/empty boundary lies.
	// We can look there to tell which kind of row gets added.

	size_t old_rank = ref_rank(rows);
	identity_t id = add(e); // <-- NOTE: does full RREF transform :(
	bool success = !is_zero(rows[old_rank]);

	if (!success) {
		// aaaaaahhh! added an empty row! revert! revert!
		augs = std::move(augs_copy);
		rows.pop_back();
	}

#ifndef NDEBUG
	assert(success == (rows != rows_copy));
#endif

	return { success, id };
}

void _XorBasisBuilder::remove_zero_rows() {
	size_t rank = ref_rank(rows);
	rows.resize(rank);
	augs.resize(rank);
}

