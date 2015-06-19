
#pragma once

#include <map>
#include <utility> // pair
#include <vector>

#include "vectorset.hpp"

//------------------------------------------------------------------------------

typedef unsigned int column_t;
typedef size_t identity_t;

typedef MinVecSet<column_t>   Row;
typedef MinVecSet<identity_t> Aug;
typedef std::pair<Row,Aug>    RowAug;

typedef std::vector<Row>    RowV;
typedef std::vector<Aug>    AugV;
typedef std::vector<RowAug> RowAugV;

//------------------------------------------------------------------------------

// anonymous namespace to avoid redefinition by multiple object files that include the header
// (TODO: is this a legitimate reason to use one?)
namespace {

	RowV into_row_v(const std::vector<std::vector<column_t>> & rows) {
		RowV result;
		for (auto & e : rows) {
			result.emplace_back(e);
		}
		return result;
	}

}

//------------------------------------------------------------------------------

// The class primarily exported from this module. (with an underscore, so that
//  the bare name may be given to a python wrapper class)
class _XorBasisBuilder
{
private:
	RowV rows; // Matrix
	AugV augs; // Augmented portion

	identity_t next_identity = 0;
	std::map<identity_t, Row> originals; // Original rows

	// Every row that is used with the matrix gets assigned an unique identity.
	// It also gets a 1 in the corresponding column of its augmented half.
	identity_t assign_identity(Row row) {
		identity_t id = next_identity++;
		originals.insert(std::make_pair(id, std::move(row)));
		return id;
	}

	const Row & original_row(identity_t id) { return originals[id]; }
	Aug         original_aug(identity_t id) { return {std::vector<identity_t>{id}}; }
	RowAug      original_rowaug(identity_t id) { return { original_row(id), original_aug(id) }; }

public:

	_XorBasisBuilder()
	: rows() // a matrix maintained in rref form
	, augs()
	, next_identity(0)
	, originals()
	{ }

	// Unconditionally add a vector to the basis.  Returns its newly assigned identity.
	// (possibly causing a linearly dependent group to emerge)
	identity_t add(Row);

	template <typename ColumnRange>
	identity_t add(const ColumnRange & e) { return add(Row {e}); }

	// Unconditionally add many vectors to the basis.  Returns their newly assigned identities.
	// (possibly causing linearly dependent groups to emerge)
	std::vector<identity_t> add_many(RowV);

	template <typename ColumnRangeRange>
	std::vector<identity_t> add_many(const ColumnRangeRange & r) { return add_many(into_row_v(r)); }

	// Add a vector to the basis, but only if it is not linearly dependent with vectors
	//  already in the basis.  Returns (success, id).
	// Note: the value of id is only meaningful if success is true.
	std::pair<bool, identity_t> add_if_linearly_independent(Row);

	template <typename ColumnRange>
	std::pair<bool, identity_t> add_if_linearly_independent(const ColumnRange & e) {
		return add_if_linearly_independent(Row {e});
	}

	// Removes the specified vectors from the basis.  The resulting state of the basis will be
	//   as though the vectors were never added in the first place.
	void remove_ids(Aug);

	template <typename IdentityRange>
	void remove_ids(const IdentityRange & e) { return remove_ids(Aug {e}); }

	// Returns a list of lists of ids, each of which represent previously-inserted vectors
	//  that sum to zero.
	// The ordering of the returned lists is guaranteed to remain fixed for as long as the
	//  basis is not modified. (this is for use together with remove_from_each_zero_sum())
	std::vector<std::vector<identity_t>> get_zero_sums() const;

	// Removes the specified vectors from the basis, provided that:
	//   * One is provided from every list returned by get_zero_sums(), IN ORDER.
	//     That is, the `i`th id provided must be contained in the `i`th zero sum.
	//   * No id is specified twice.
	// After this method, all vectors remaining in the basis will be linearly independent.
	void remove_from_each_zero_sum(std::vector<identity_t>);

	const RowV & get_rows() const { return rows; }
	const AugV & get_augs() const { return augs; }

	void remove_zero_rows();
};

