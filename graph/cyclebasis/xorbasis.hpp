
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

//------------------------------------------------------------------------------

typedef unsigned int column_t;
typedef size_t identity_t;

typedef MinVecSet<column_t>  Row;
typedef MinVecSet<identity_> Aug;
typedef std::pair<Row,Aug>   RowAug;

typedef std::vector<Row>    RowV;
typedef std::vector<Aug>    AugV;
typedef std::vector<RowAug> RowAugV;

bool is_zero(const Row & row) { return row.empty(); }

column_t leading_column(const Row & row) { return row.back(); }

bool is_zero(const pair<Row,Aug> & rowaug) { return is_zero(rowaug.first); }

column_t leading_column(const pair<Row,Aug> & rowaug) { return leading_column(rowaug.first); }

template <typename T>
bool ref_order_less(const T & a, const T & b) {
	return ref_order(a) < ref_order(b);
}

template <typename T>
bool ref_order_greater(const T & a, const T & b) {
	return ref_order(a) > ref_order(b);
}

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
	return result;
}

// Transforms an arbitrary augmented matrix into REF form.
template <typename Range>
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
		out_rows.push_back(heap.top()->first);
		out_augs.push_back(heap.top()->second);
		heap.pop();

		while (!heap.empty() && !is_zero(heap.top())) {

			RowAug top = heap.top(); heap.pop();

			// conflict with last inserted row?
			const Row & last = out_rows.back();
			assert(!is_zero(top) && !is_zero(last));
			if (leading_column(top) == leading_column(last)) {

				top.first  ^= out_rows.back();
				top.second ^= out_augs.back();

				assert(f_comp(top, last)); // top's priority has decreased...
				heap.push(std::move(top)); // ...so put it back in line

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
	return make_pair(std::move(out_rows), std::move(out_augs));
}

// Transforms an arbitrary augmented matrix into RREF form.
template <typename Range>
std::pair<RowV, AugV> transform_to_rref(const RowV & in_rows, const AugV & in_augs)
{
	// start in REF form;  once in REF, row order and leading columns are fixed
	RowV rows; AugV augs;
	std::tie(rows, augs) = transform_to_ref(in_rows, in_augs);

	// All remaining conflicts are between a row that "owns" a column and some other row above it.
	// We'll work upwards from the last non-empty row.
	std::map<column_t, size_t> col_owners;

	for (size_t i = ref_matrix_rank(); i --> 0 ;) {
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
	assert(rows.size() == in_rows.size());
	return result;
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
/**        OLD CODE BELOW         **/

//------------------------------------------------------------------------------

// A sparse bit matrix which maintains reduced row echelon form, modulo 2.
// That is, at any given point in time:
//  * All rows containing nothing but zero are at the end.
//  * The column of each successive (nonzero) row's leading 1 strictly increases.
//  * Any leading 1 is the **only** 1 in its column.
//
// Its construction allows one to determine whether or not a set of bit-vectors is
//  linearly independent modulo 2.
class SparseBitRref {
public:
	typedef typename std::vector<Row>::iterator RowIterator;
	typedef typename std::vector<Row>::const_iterator RowConstIterator;
	typedef typename std::vector<Row>::difference_type RowDifferenceType;

private:

	// members
	RowSpace &            row_space;
	std::vector<Row>      rows;
	std::size_t           nnz_rows;

public:

	SparseBitRref(RowSpace & row_space)
	: row_space(row_space)
	, rows()
	, nnz_rows(0)
	{ }

	// Inserts the rows from range r into the matrix.
	// A full REF transformation will be performed afterwards; it is intended to be used when the
	//  number of rows being added greatly outnumbers the number of rows already in the matrix.
	// (in such a case, however, it should be more performant than inserting the rows individually)
	//
	// The return value is the identity assigned to each row, in the order they appear in the input
	//  range (beginning from r.cbegin())
	template <typename RangeRange>
	std::vector<identity_t> insert_many(const RangeRange & rr) {
		std::vector<identity_t> ids;

		for (const auto & r : rr) {
			Row row = row_space.make_row(r);
			this->insert_direct(this->rows.end(), row);
			ids.push_back(row.primary_id);
		}

		rows = transform_to_rref(rows);
		nnz_rows = _compute_nnz_rows();

		return ids;
	}

	template <typename Range>
	RowIterator insert(const Range & r) {
		Row row = row_space.make_row(r);
		return this->insert_reduced(this->reduce_row(row));
	}

	RowIterator insert(const Row & row) {
		if (!compatible(row_space, row.row_space))
			throw std::logic_error("Attempt to insert row from a different RowSpace!");
		return this->insert_reduced(this->reduce_row(row));
	}

	// adds (via xor) rows from the matrix to the provided row as necessary to eliminate any
	//  conflicts between the row and leading ones in the matrix.
	//
	// If (and only if) reduce_row produces an empty row (i.e. one with all zeros), then the
	//  input row was linearly dependent with rows already in the matrix.
	Row reduce_row(Row row) const;

	RowIterator insertion_point(const Row & row) {
		return std::lower_bound(
			this->nonzero_begin(), this->nonzero_end(), row,
			ref_order_less);
	}

	RowConstIterator insertion_point(const Row & row) const {
		return std::lower_bound(
			this->nonzero_cbegin(), this->nonzero_cend(), row,
			ref_order_less);
	}

	// for iterating through the non-empty rows (i.e. the rows for which the leading column is defined)
	RowIterator nonzero_begin() { return this->rows.begin(); }
	RowIterator nonzero_end()   { return this->nonzero_begin() + RowDifferenceType(this->nnz_rows); }
	RowConstIterator nonzero_cbegin() const { return this->rows.cbegin(); }
	RowConstIterator nonzero_cend()   const { return this->nonzero_cbegin() + RowDifferenceType(this->nnz_rows); }
	// ... and the empty rows
	RowIterator empty_begin() { return this->nonzero_end(); }
	RowIterator empty_end()   { return this->rows.end(); }
	RowConstIterator empty_cbegin() const { return this->nonzero_cend(); }
	RowConstIterator empty_cend()   const { return this->rows.cend(); }

	std::size_t rank() { return this->nnz_rows; }
	std::size_t total_rows() { return this->rows.size(); }
	std::size_t empty_rows() { return this->rows.size() - this->nnz_rows; }

	bool operator==(const SparseBitRref & other) const {
		if (this->nnz_rows != other.nnz_rows)
			return false;
		return this->rows == other.rows;
	}

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

private:

	// Lowest level row insertion function (all other insert methods eventually call this).
	// Inserts a row directly into the underlying container with no further modification.
	// The row is assigned a new id.
	//
	// Returns the id and an iterator to the inserted row (the provided iterator is no longer valid)
	RowIterator insert_direct(RowIterator where, const Row & row) {
		return this->rows.insert(where, row);
	}

	// expects that the row is already reduced (i.e. does not conflict with existing leading
	//  ones in the matrix), and performs further modification to the matrix to ensure existing
	//  rows do not conflict with the new row.
	RowIterator insert_reduced(const Row & row);

	// a limited resize() which only allows the length to be decreased
	// (this limitation is to prevent the creation of rows with unspecified identities)
	void shorten(std::size_t newsize) {
		if (newsize > rows.size())
			throw std::logic_error("shorten() called with longer width");

		if (rows.size() == 0) return; // degenerate case
		Row & dummy = rows.back(); // borrow an existing row because they are tough to construct
		this->rows.resize(newsize, dummy);
	}

};

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


SparseBitRref::RowIterator SparseBitRref::insert_reduced(const Row & row) {
	// special case: row is all zeros
	if (row.is_zero()) {
		// don't increment nnz_rows
		return this->insert_direct(this->rows.end(), row);
	}

	RowIterator insertion_pt = this->insertion_point(row);
	insertion_pt = this->insert_direct(insertion_pt, row);
	this->nnz_rows += 1;

	// Rows above the new row may conflict with the its leading one.
	// Resolve these conflicts.
	assert(!row.is_zero());
	for (auto it = this->nonzero_begin(); it != insertion_pt; ++it) {
		if (it->get(row.lead()))
			(*it).xor_update(row);
		assert(!(it->get(row.lead())));
	}

	return insertion_pt;
}

//------------------------------------------------------------------------------

// The class primarily exported from this module. (with an underscore, so that
//  the bare name may be given to a python wrapper class)
class _XorBasisBuilder
{
private:
	RowSpace row_space;
	SparseBitRref mat;

public:
	_XorBasisBuilder()
	: row_space()
	, mat(row_space)
	{ }

	// owned RowSpace is nontransferrable
	_XorBasisBuilder(const _XorBasisBuilder&) = delete;
	_XorBasisBuilder(_XorBasisBuilder&&) = delete;
	_XorBasisBuilder& operator=(const _XorBasisBuilder&) & = delete;
	_XorBasisBuilder& operator=(_XorBasisBuilder&&) & = delete;
	// TODO: rule of 3 "violation"? (default destructor) or is it safe as written?

	identity_t add(std::vector<column_t> e) {
		auto row = row_space.make_row(e);
		mat.insert(e);
		return row.primary_id;
	}

	std::pair<bool, identity_t> add_if_linearly_independent(std::vector<column_t> e) {
		auto row = row_space.make_row(e);
		auto reduced = this->mat.reduce_row(row);
		if (reduced.is_zero())
			return { false, 0 };

		this->mat.insert(reduced);
		return { true, reduced.primary_id };
	}


	template <typename Range>
	std::vector<identity_t> add_many(Range r) {
		return mat.insert_many(r);
	}

	std::vector<identity_t> get_linearly_dependent_ids() {
		std::vector<identity_t> result;
		for (auto it = this->mat.empty_begin(); it != this->mat.empty_end(); ++it) {
			result.push_back(it->primary_id);
		}
		return result;
	}

	template <typename Range>
	void remove_ids(const Range & r) {
		this->mat.remove_ids(r);
	}

	void remove_linearly_dependent_ids() {
		this->mat.remove_empty_rows();
	}

	const SparseBitRref & matrix() const {
		return this->mat;
	}
};

