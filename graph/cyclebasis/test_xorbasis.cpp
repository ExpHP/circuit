#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cassert>

#include "graph/cyclebasis/xorbasis.cpp"

// quick hack to allow the use of an initialization list to construct Rows/Augs
// e.g. Row row = R{ 1, 4, 9, 16 };
#define R std::vector<column_t>
#define A std::vector<identity_t>

using namespace std;

void debug(const RowV & rows, const AugV & augs) {
	for (size_t i=0;i<rows.size(); i++) {
		std::cout << "[ ";
		for (auto col : rows[i])
			std::cout << col << " ";
		std::cout << "]   ";

		std::cout << "[ ";
		for (auto col : augs[i])
			std::cout << col << " ";
		std::cout << "]" << std::endl;
	}
	std::cout << std::endl;
}


Row random_row (size_t n) {
	vector<column_t> row;
	for (size_t j=0; j<n; j++)
		if (rand()%2)
			row.push_back(j);
	return { row };
}

RowV random_matrix (size_t m, size_t n) {
	RowV rows;
	for (size_t i=0; i<m; i++)
		rows.push_back(random_row(n));
	return rows;
}

AugV zero_augmented (size_t m) {
	return AugV(m); // m empty rows
}

AugV identity_augmented (size_t m) {
	AugV augs;
	for (size_t i=0; i<m; i++)
		augs.emplace_back(std::vector<identity_t>{i});
	return augs;
}

void test_rref_fixed() {
	// an example with no zero rows in the result (so that the
	//  augmented matrix is unique as well)
	RowV rows_in = {
		R{ 1, 3, 4}, //  0 1 0 1 1
		R{ 0, 1, 2}, //  1 1 1 0 0
		R{ 3},       //  0 0 0 1 0
		R{ 1, 3}     //  0 1 0 1 0
	};
	RowV rows_good = {
		R{ 0, 2},    //  1 0 1 0 0
		R{ 1},       //  0 1 0 0 0
		R{ 3},       //  0 0 0 1 0
		R{ 4}        //  0 0 0 0 1
	};
	AugV augs_in   = { A{0}, A{1}, A{2}, A{3} };
	AugV augs_good = { A{1,2,3}, A{2,3}, A{2}, A{0,3} };

	RowV rows_out; AugV augs_out;
	tie(rows_out, augs_out) = transform_to_rref(rows_in, augs_in);

	assert(rows_out == rows_good);
	assert(augs_out == augs_good);
}

void test_rref_idempotent() {
	RowV rows1 = random_matrix(50,50);
	AugV augs1 = identity_augmented(50);

	RowV rows2; AugV augs2;
	RowV rows3; AugV augs3;
	tie(rows2, augs2) = transform_to_rref(rows1, augs1);
	tie(rows3, augs3) = transform_to_rref(rows2, augs2);

	assert(rows2 == rows3);
	// (augmented part is allowed to change)
}

// RREF form does not depend on initial row order
void test_rref_insert_order() {
	RowV rows = random_matrix(45,50);
	AugV augs = zero_augmented(50); // don't care 'bout these
	RowV permuted = rows;
	std::random_shuffle(permuted.begin(), permuted.end());

	// empty matrices to insert into
	RowV rref_rows;
	AugV rref_augs;
	RowV rref_permuted;
	AugV rref_permuted_augs;

	rref_insert_bunch(rref_rows, rref_augs, rows, augs);
	rref_insert_bunch(rref_permuted, rref_permuted_augs, permuted, augs);
	assert(rref_rows == rref_permuted);
}

// The result of removing vectors should be the same as if they were never added
//  in the first place.
void test_xorbasis_remove_ids(int n) {
	_XorBasisBuilder xorb;
	RowV rows1 = random_matrix(2*n,4*n);
	RowV rows2 = random_matrix(2*n,4*n);

	xorb.add_many(rows1);
	RowV before = xorb.get_rows();

	// add some, remove them
	vector<identity_t> ids2 = xorb.add_many(rows2);
	xorb.remove_ids(ids2);
	RowV after = xorb.get_rows();

	assert(before == after);
}

int main(int argc, char * argv[]) {
	srand(time(NULL));
	test_rref_fixed();
	test_rref_idempotent();
	test_rref_insert_order();
	std::cout << "lol" << std::endl;
	test_xorbasis_remove_ids(40);
	std::cout << "lol" << std::endl;

	return 0;
}
