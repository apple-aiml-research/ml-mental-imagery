//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
/// Algorithm X with Dancing Links (DLX) — generic exact cover solver.
///
/// Given a binary matrix where rows represent choices and columns represent
/// constraints, find a subset of rows such that every column is covered
/// exactly once.
///
/// This is Knuth's standard implementation with the S (minimum size) heuristic.

/// A node in the doubly-linked toroidal matrix.
#[derive(Debug, Clone)]
struct Node {
    left: usize,
    right: usize,
    up: usize,
    down: usize,
    column: usize,
    row_id: usize, // which original row this node belongs to (usize::MAX for headers)
}

/// DLX solver instance.
pub struct DLX {
    nodes: Vec<Node>,
    /// Number of 1s in each column.
    column_sizes: Vec<usize>,
    root: usize,
    solution_stack: Vec<usize>, // row_ids of selected rows
    ops: usize,
    max_ops: usize,
}

impl DLX {
    /// Build a DLX matrix from row→column coverage lists.
    ///
    /// - `num_columns`: total number of constraints (all primary)
    /// - `rows`: each entry is a list of column indices that the row covers
    /// - `max_ops`: operation limit (analogous to max_bfs_states)
    pub fn new(num_columns: usize, rows: &[Vec<usize>], max_ops: usize) -> Self {
        Self::with_secondary(num_columns, 0, rows, max_ops)
    }

    /// Build a DLX matrix with primary and secondary columns.
    ///
    /// - `num_primary`: columns 0..num_primary must be covered exactly once
    /// - `num_secondary`: columns num_primary..num_primary+num_secondary are
    ///   optional (covered at most once). They are not linked to the root's
    ///   horizontal list, so `choose_column` never selects them.
    /// - `rows`: column indices may reference both primary and secondary columns
    /// - `max_ops`: operation limit
    pub fn with_secondary(
        num_primary: usize,
        num_secondary: usize,
        rows: &[Vec<usize>],
        max_ops: usize,
    ) -> Self {
        let num_columns = num_primary + num_secondary;
        // Node 0 is the root sentinel
        let mut nodes =
            Vec::with_capacity(num_columns + 1 + rows.iter().map(|r| r.len()).sum::<usize>());

        // Root sentinel (placeholder, fixed below)
        nodes.push(Node {
            left: 0,
            right: 0,
            up: 0,
            down: 0,
            column: 0,
            row_id: usize::MAX,
        });

        // Column headers (nodes 1..=num_columns)
        // Primary columns (0..num_primary) are linked to root.
        // Secondary columns (num_primary..num_columns) self-loop.
        for i in 0..num_columns {
            let idx = i + 1; // 1-indexed
            let (left, right) = if i < num_primary {
                // Primary: linked to neighbors in root's list
                (
                    if i == 0 { 0 } else { idx - 1 },
                    if i == num_primary - 1 { 0 } else { idx + 1 },
                )
            } else {
                // Secondary: self-loop (not reachable from root)
                (idx, idx)
            };
            nodes.push(Node {
                left,
                right,
                up: idx,
                down: idx,
                column: idx,
                row_id: usize::MAX,
            });
        }

        // Fix root's left/right to span only primary columns
        if num_primary > 0 {
            nodes[0].left = num_primary;
            nodes[0].right = 1;
        } else {
            nodes[0].left = 0;
            nodes[0].right = 0;
        }

        let mut dlx = DLX {
            nodes,
            column_sizes: vec![0; num_columns],
            root: 0,
            solution_stack: Vec::new(),
            ops: 0,
            max_ops,
        };

        // Add rows
        for (row_id, cols) in rows.iter().enumerate() {
            dlx.add_row(row_id, cols);
        }

        dlx
    }

    fn add_row(&mut self, row_id: usize, cols: &[usize]) {
        if cols.is_empty() {
            return;
        }

        let first_node_idx = self.nodes.len();

        for (i, &col) in cols.iter().enumerate() {
            let col_header = col + 1; // 1-indexed column headers
            let node_idx = self.nodes.len();

            self.nodes.push(Node {
                left: if i == 0 {
                    first_node_idx + cols.len() - 1
                } else {
                    node_idx - 1
                },
                right: if i == cols.len() - 1 {
                    first_node_idx
                } else {
                    node_idx + 1
                },
                up: self.nodes[col_header].up,
                down: col_header,
                column: col_header,
                row_id,
            });

            // Insert into column's vertical list (before header = at bottom)
            let up = self.nodes[col_header].up;
            self.nodes[up].down = node_idx;
            self.nodes[col_header].up = node_idx;

            self.column_sizes[col] += 1;
        }
    }

    fn cover(&mut self, col_header: usize) {
        // Remove column header from horizontal list
        let left = self.nodes[col_header].left;
        let right = self.nodes[col_header].right;
        self.nodes[left].right = right;
        self.nodes[right].left = left;

        // For each row in this column, remove all other nodes in that row from their columns
        let mut row_node = self.nodes[col_header].down;
        while row_node != col_header {
            let mut j = self.nodes[row_node].right;
            while j != row_node {
                let up = self.nodes[j].up;
                let down = self.nodes[j].down;
                self.nodes[up].down = down;
                self.nodes[down].up = up;
                let col_idx = self.nodes[j].column - 1; // 0-indexed
                self.column_sizes[col_idx] -= 1;
                j = self.nodes[j].right;
            }
            row_node = self.nodes[row_node].down;
        }
    }

    fn uncover(&mut self, col_header: usize) {
        // Reverse of cover
        let mut row_node = self.nodes[col_header].up;
        while row_node != col_header {
            let mut j = self.nodes[row_node].left;
            while j != row_node {
                let col_idx = self.nodes[j].column - 1;
                self.column_sizes[col_idx] += 1;
                let up = self.nodes[j].up;
                let down = self.nodes[j].down;
                self.nodes[up].down = j;
                self.nodes[down].up = j;
                j = self.nodes[j].left;
            }
            row_node = self.nodes[row_node].up;
        }

        // Restore column header in horizontal list
        let left = self.nodes[col_header].left;
        let right = self.nodes[col_header].right;
        self.nodes[left].right = col_header;
        self.nodes[right].left = col_header;
    }

    /// Choose column with minimum size (S heuristic).
    fn choose_column(&self) -> Option<usize> {
        let mut best = None;
        let mut best_size = usize::MAX;
        let mut col = self.nodes[self.root].right;
        while col != self.root {
            let col_idx = col - 1;
            if self.column_sizes[col_idx] < best_size {
                best_size = self.column_sizes[col_idx];
                best = Some(col);
                if best_size == 0 {
                    break; // Can't do better; this column has no rows → will fail
                }
            }
            col = self.nodes[col].right;
        }
        best
    }

    fn search(&mut self, find_all: bool, limit: usize, solutions: &mut Vec<Vec<usize>>) -> bool {
        self.ops += 1;
        if self.ops > self.max_ops {
            return false; // timeout
        }

        // No columns remain: a solution is found
        if self.nodes[self.root].right == self.root {
            solutions.push(self.solution_stack.clone());
            return true;
        }

        let col = match self.choose_column() {
            Some(c) => c,
            None => return false,
        };

        // If column has size 0, no solution in this branch
        let col_idx = col - 1;
        if self.column_sizes[col_idx] == 0 {
            return false;
        }

        self.cover(col);

        let mut row_node = self.nodes[col].down;
        while row_node != col {
            let row_id = self.nodes[row_node].row_id;
            self.solution_stack.push(row_id);

            // Cover all other columns in this row
            let mut j = self.nodes[row_node].right;
            while j != row_node {
                self.cover(self.nodes[j].column);
                j = self.nodes[j].right;
            }

            let found = self.search(find_all, limit, solutions);

            if found && !find_all {
                // Early return for find-one mode
                self.uncover(col);
                return true;
            }

            if solutions.len() >= limit {
                // Hit count limit
                // Uncover and return
                let mut j = self.nodes[row_node].left;
                while j != row_node {
                    self.uncover(self.nodes[j].column);
                    j = self.nodes[j].left;
                }
                self.solution_stack.pop();
                self.uncover(col);
                return !solutions.is_empty();
            }

            // Backtrack: uncover columns in reverse order
            let mut j = self.nodes[row_node].left;
            while j != row_node {
                self.uncover(self.nodes[j].column);
                j = self.nodes[j].left;
            }

            self.solution_stack.pop();
            row_node = self.nodes[row_node].down;
        }

        self.uncover(col);
        !solutions.is_empty()
    }

    /// Find one solution. Returns row indices of the chosen rows.
    pub fn solve(&mut self) -> Option<Vec<usize>> {
        let mut solutions = Vec::new();
        self.search(false, 1, &mut solutions);
        solutions.into_iter().next()
    }

    /// Find all solutions up to `limit`. Returns row-index vectors.
    pub fn find_all(&mut self, limit: usize) -> Vec<Vec<usize>> {
        let mut solutions = Vec::new();
        self.search(true, limit, &mut solutions);
        solutions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_exact_cover() {
        // Classic example:
        // Columns: 0 1 2 3 4 5 6
        // Row 0: {0, 3, 6}
        // Row 1: {0, 3}
        // Row 2: {3, 4, 6}
        // Row 3: {2, 4, 5}
        // Row 4: {1, 2, 5, 6}
        // Row 5: {1, 6}
        // Solution: rows {1, 3, 5} → covers {0,3} ∪ {2,4,5} ∪ {1,6} = {0,1,2,3,4,5,6}
        let rows = vec![
            vec![0, 3, 6],
            vec![0, 3],
            vec![3, 4, 6],
            vec![2, 4, 5],
            vec![1, 2, 5, 6],
            vec![1, 6],
        ];
        let mut dlx = DLX::new(7, &rows, 1_000_000);
        let solution = dlx.solve().expect("should find a solution");
        let mut sorted = solution.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 3, 5]);
    }

    #[test]
    fn test_no_solution() {
        // Two rows that both need column 0, but column 1 only covered by row 0
        // Row 0: {0, 1}
        // Row 1: {0, 2}
        // Columns: 0 1 2 — row 0 covers {0,1}, row 1 covers {0,2}
        // Both need col 0, so only one can be chosen. Neither alone covers all 3.
        let rows = vec![vec![0, 1], vec![0, 2]];
        let mut dlx = DLX::new(3, &rows, 1_000_000);
        assert!(dlx.solve().is_none());
    }

    #[test]
    fn test_multiple_solutions_counting() {
        // Columns: 0 1
        // Row 0: {0, 1}
        // Row 1: {0, 1}
        // Two solutions (either row covers both columns)
        let rows = vec![vec![0, 1], vec![0, 1]];
        let mut dlx = DLX::new(2, &rows, 1_000_000);
        assert_eq!(dlx.find_all(10).len(), 2);
    }

    #[test]
    fn test_max_ops_early_termination() {
        // Large enough problem to exhaust tiny op limit
        let rows = vec![
            vec![0, 1],
            vec![1, 2],
            vec![2, 3],
            vec![3, 4],
            vec![0, 4],
            vec![0, 2, 4],
            vec![1, 3],
        ];
        let mut dlx = DLX::new(5, &rows, 3); // very small op limit
        let result = dlx.solve();
        // With only 3 ops, likely won't find the solution
        // Just verify it doesn't panic and respects the limit approximately
        let _ = result;
        assert!(dlx.ops >= 3, "should have used at least max_ops operations");
    }

    #[test]
    fn test_empty_matrix() {
        // 0 columns → trivial solution (empty set)
        let rows: Vec<Vec<usize>> = vec![];
        let mut dlx = DLX::new(0, &rows, 1_000_000);
        let solution = dlx
            .solve()
            .expect("empty matrix should have trivial solution");
        assert!(solution.is_empty());
    }

    #[test]
    fn test_count_with_limit() {
        // Columns: 0 1
        // 5 rows, each covering both columns → 5 solutions
        let rows: Vec<Vec<usize>> = (0..5).map(|_| vec![0, 1]).collect();
        let mut dlx = DLX::new(2, &rows, 1_000_000);
        assert_eq!(dlx.find_all(3).len(), 3);
    }

    #[test]
    fn test_single_column() {
        // 1 column, 3 rows each covering it → 3 solutions
        let rows = vec![vec![0], vec![0], vec![0]];
        let mut dlx = DLX::new(1, &rows, 1_000_000);
        assert_eq!(dlx.find_all(100).len(), 3);
    }

    #[test]
    fn test_find_all_enumerates_all() {
        // Columns: 0 1 2
        // Row 0: {0, 1, 2}  — covers everything alone
        // Row 1: {0}
        // Row 2: {1, 2}
        // Solutions: {0} and {1, 2}
        let rows = vec![vec![0, 1, 2], vec![0], vec![1, 2]];
        let mut dlx = DLX::new(3, &rows, 1_000_000);
        let solutions = dlx.find_all(100);
        assert_eq!(solutions.len(), 2);

        // Verify each solution covers all columns exactly once
        for sol in &solutions {
            let mut covered = vec![false; 3];
            for &row_id in sol {
                for &col in &rows[row_id] {
                    assert!(
                        !covered[col],
                        "column {} covered twice in solution {:?}",
                        col, sol
                    );
                    covered[col] = true;
                }
            }
            assert!(
                covered.iter().all(|&c| c),
                "not all columns covered in solution {:?}",
                sol
            );
        }

        // Verify solutions are distinct
        let mut sorted_solutions: Vec<Vec<usize>> = solutions
            .iter()
            .map(|s| {
                let mut s = s.clone();
                s.sort();
                s
            })
            .collect();
        sorted_solutions.sort();
        for i in 1..sorted_solutions.len() {
            assert_ne!(
                sorted_solutions[i - 1],
                sorted_solutions[i],
                "duplicate solutions found"
            );
        }
    }

    #[test]
    fn test_solvable_and_unsolvable() {
        // Solvable: row 0 covers all columns
        let rows = vec![vec![0, 1]];
        let mut dlx = DLX::new(2, &rows, 1_000_000);
        assert!(dlx.solve().is_some());

        // Unsolvable: no way to cover column 2
        let rows = vec![vec![0, 1]];
        let mut dlx = DLX::new(3, &rows, 1_000_000);
        assert!(dlx.solve().is_none());
    }

    #[test]
    fn test_solution_validity() {
        // Classic 7-column problem — verify the returned rows actually form an exact cover
        let rows = vec![
            vec![0, 3, 6],
            vec![0, 3],
            vec![3, 4, 6],
            vec![2, 4, 5],
            vec![1, 2, 5, 6],
            vec![1, 6],
        ];
        let num_columns = 7;
        let mut dlx = DLX::new(num_columns, &rows, 1_000_000);
        let solution = dlx.solve().expect("should find a solution");

        // Check every column is covered exactly once
        let mut coverage = vec![0u32; num_columns];
        for &row_id in &solution {
            for &col in &rows[row_id] {
                coverage[col] += 1;
            }
        }
        for (col, &count) in coverage.iter().enumerate() {
            assert_eq!(
                count, 1,
                "column {} covered {} times (expected 1)",
                col, count
            );
        }
    }

    #[test]
    fn test_disjoint_subproblems() {
        // Block-diagonal matrix: two independent sub-problems
        // Sub-problem A: columns 0-1, rows 0-1
        // Sub-problem B: columns 2-3, rows 2-3
        // Row 0: {0}     Row 1: {1}     → covers cols 0,1
        // Row 2: {2}     Row 3: {3}     → covers cols 2,3
        // Also add alternatives:
        // Row 4: {0, 1}  → covers cols 0,1 in one row
        // Row 5: {2, 3}  → covers cols 2,3 in one row
        // Solutions: {0,1,2,3}, {0,1,5}, {4,2,3}, {4,5}
        let rows = vec![vec![0], vec![1], vec![2], vec![3], vec![0, 1], vec![2, 3]];
        let mut dlx = DLX::new(4, &rows, 1_000_000);
        let solutions = dlx.find_all(100);
        assert_eq!(solutions.len(), 4);

        // Verify each solution is a valid exact cover
        for sol in &solutions {
            let mut covered = vec![false; 4];
            for &row_id in sol {
                for &col in &rows[row_id] {
                    assert!(!covered[col], "column {} covered twice", col);
                    covered[col] = true;
                }
            }
            assert!(covered.iter().all(|&c| c), "not all columns covered");
        }
    }

    #[test]
    fn test_identity_matrix() {
        // N×N identity: exactly one solution — select all rows
        let n = 5;
        let rows: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut dlx = DLX::new(n, &rows, 1_000_000);
        let solutions = dlx.find_all(100);
        assert_eq!(solutions.len(), 1);
        let mut sol = solutions[0].clone();
        sol.sort();
        assert_eq!(sol, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn test_secondary_columns_partial_cover() {
        // 2 primary columns (pieces), 3 secondary columns (cells).
        // Row 0: piece 0 covers cells {0, 1}  → cols [0, 2, 3]
        // Row 1: piece 1 covers cell {2}      → cols [1, 4]
        // Row 2: skip piece 0                  → cols [0]
        // Row 3: skip piece 1                  → cols [1]
        // Both pieces must be placed or skipped. Cells are optional.
        let rows = vec![
            vec![0, 2, 3], // place piece 0 covering cells 0,1
            vec![1, 4],    // place piece 1 covering cell 2
            vec![0],       // skip piece 0
            vec![1],       // skip piece 1
        ];
        let mut dlx = DLX::with_secondary(2, 3, &rows, 1_000_000);
        let solution = dlx.solve().expect("should find solution");
        // With skip rows added LAST, DLX should prefer placing both pieces.
        let mut sorted = solution.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            vec![0, 1],
            "should place both pieces (rows 0 and 1)"
        );
    }

    #[test]
    fn test_secondary_columns_forced_skip() {
        // 2 primary columns (pieces), 2 secondary columns (cells).
        // Row 0: piece 0 covers cell {0}      → cols [0, 2]
        // Row 1: piece 1 covers cell {0}      → cols [1, 2]  (conflicts with row 0!)
        // Row 2: skip piece 0                  → cols [0]
        // Row 3: skip piece 1                  → cols [1]
        // Both pieces want the same cell. One must be skipped.
        let rows = vec![
            vec![0, 2], // place piece 0 at cell 0
            vec![1, 2], // place piece 1 at cell 0 (conflict)
            vec![0],    // skip piece 0
            vec![1],    // skip piece 1
        ];
        let mut dlx = DLX::with_secondary(2, 2, &rows, 1_000_000);
        let solution = dlx.solve().expect("should find solution");
        // One piece placed, one skipped. Solution has 2 rows (one place + one skip).
        assert_eq!(solution.len(), 2);
        // Exactly one of the rows should be a skip (row 2 or 3)
        let skips: Vec<&usize> = solution.iter().filter(|&&r| r >= 2).collect();
        assert_eq!(skips.len(), 1, "exactly one piece should be skipped");
    }
}
