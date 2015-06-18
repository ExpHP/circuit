
#pragma once

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
		originals.emplace { id, std::move(row) };
		return id;
	}

	const Row & original_row(identity_t id) { return originals[id]; }
	Aug         original_aug(identity_t id) { return {std::vector<identity_t>{id}}; }
	RowAug      original_rowaug(identity_t id) { return { original_row(id), original_aug(id); }

public:

	_XorBasisBuilder()
	: rows() // a matrix maintained in rref form
	, augs()
	, next_identity(0)
	, originals()
	{ }

	// Unconditionally insert a row, maintaining RREF.
	identity_t add(const std::vector<column_t> & e);

	// Unconditionally insert many rows, maintaining RREF.
	std::vector<identity_t> add_many(std::vector<std::vector<column_t>> r);

	// Insert a single row, maintaining RREF... but only if it is not linearly dependent
	//  with rows in the matrix.
	std::pair<bool, identity_t> add_if_linearly_independent(std::vector<column_t> e);

/*
	// returns lists of ids of rows which xor-sum to zero.
	std::vector<std::vector<identity_t>> get_zero_sums();

	void remove_ids(const std::vector<identity_t> & r);
*/

	void remove_zero_rows();
};

