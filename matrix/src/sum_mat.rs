use crate::mat::{OutOfRangeError, WrappedMatrix};

use num_traits::NumCast;
use utils::safer::safe_lock;

#[derive(Clone)]
pub struct SumMatrix {
    matrix: WrappedMatrix<i64>,
    row_sum: Vec<i64>,
}

impl SumMatrix {
    #[must_use]
    pub fn new(matrix: WrappedMatrix<i64>) -> Self {
        let row_sum = safe_lock(&matrix.mat).iter().map(|row| row.iter().sum()).collect::<Vec<_>>();
        Self { matrix, row_sum }
    }

    #[must_use]
    pub fn get_row_sum(
        &self,
        row: usize,
    ) -> i64 {
        self.row_sum[row]
    }

    /// Set the value at the specified row and column without checking bounds.
    pub fn set_val_unchecked(
        &mut self,
        row: usize,
        col: usize,
        val: i64,
    ) {
        let previous_val = self.matrix.get_unchecked(row, col);
        let delta = val - previous_val;
        self.matrix.set_mut_unchecked(row, col, val);
        self.row_sum[row] += delta;
    }

    /// Set the value at the specified row and column.
    ///
    /// # Errors
    ///
    /// - "Index out of bounds" if the row or column is out of range.
    pub fn set_val(
        &mut self,
        row: usize,
        col: usize,
        val: i64,
    ) -> Result<(), String> {
        if row >= self.matrix.rows() || col >= self.matrix.cols() {
            return Err("Index out of bounds".to_string());
        }
        self.set_val_unchecked(row, col, val);
        Ok(())
    }

    /// Get the value at the specified row and column.
    ///
    /// # Errors
    ///
    /// - "Index out of bounds" if the row or column is out of range.
    pub fn get_val(
        &self,
        row: usize,
        col: usize,
    ) -> Result<i64, OutOfRangeError> {
        self.matrix.get(row, col)
    }

    /// Get the value at the specified row and column without checking bounds.
    #[must_use]
    pub fn get_val_unchecked(
        &self,
        row: usize,
        col: usize,
    ) -> i64 {
        self.matrix.get_unchecked(row, col)
    }

    /// Get the ratio of a value in the matrix to the sum of its row.
    ///
    /// # Panics
    ///
    /// If the numcast fails
    ///
    /// # Errors
    ///
    /// - "Index out of bounds" if the row or column is out of range.
    /// - "Division by zero" if the sum of the row is zero.
    pub fn get_ratio(
        &self,
        row: usize,
        col: usize,
    ) -> Result<f64, String> {
        if row >= self.matrix.rows() || col >= self.matrix.cols() {
            return Err("Index out of bounds".to_string());
        }
        let val = self.matrix.get_unchecked(row, col);
        let row_sum = self.row_sum[row];
        if row_sum == 0 {
            return Err("Division by zero".to_string());
        }
        let val_f64: f64 = NumCast::from(val).unwrap();
        let row_sum_f64: f64 = NumCast::from(row_sum).unwrap();
        Ok(val_f64 / row_sum_f64)
    }
}

// Add unit tests module

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_matrix() {
        let matrix = WrappedMatrix::new(3, 3);
        let mut sum_matrix = SumMatrix::new(matrix);

        sum_matrix.set_val(0, 0, 1).unwrap();
        sum_matrix.set_val(0, 1, 2).unwrap();
        sum_matrix.set_val(0, 2, 3).unwrap();

        assert_eq!(sum_matrix.get_row_sum(0), 6);
        assert_eq!(sum_matrix.get_val(0, 0).unwrap(), 1);
        assert_eq!(sum_matrix.get_val(0, 1).unwrap(), 2);
        assert_eq!(sum_matrix.get_val(0, 2).unwrap(), 3);

        sum_matrix.set_val(0, 1, 5).unwrap();

        assert_eq!(sum_matrix.get_row_sum(0), 9);
        assert_eq!(sum_matrix.get_val(0, 0).unwrap(), 1);
        assert_eq!(sum_matrix.get_val(0, 1).unwrap(), 5);
        assert_eq!(sum_matrix.get_val(0, 2).unwrap(), 3);
    }

    #[test]
    fn test_sum_matrix_unchecked() {
        let matrix = WrappedMatrix::new(3, 3);
        let mut sum_matrix = SumMatrix::new(matrix);

        sum_matrix.set_val_unchecked(0, 0, 1);
        sum_matrix.set_val_unchecked(0, 1, 2);
        sum_matrix.set_val_unchecked(0, 2, 3);

        assert_eq!(sum_matrix.get_row_sum(0), 6);
        assert_eq!(sum_matrix.get_val_unchecked(0, 0), 1);
        assert_eq!(sum_matrix.get_val_unchecked(0, 1), 2);
        assert_eq!(sum_matrix.get_val_unchecked(0, 2), 3);

        sum_matrix.set_val_unchecked(0, 1, 5);

        assert_eq!(sum_matrix.get_row_sum(0), 9);
        assert_eq!(sum_matrix.get_val_unchecked(0, 0), 1);
        assert_eq!(sum_matrix.get_val_unchecked(0, 1), 5);
        assert_eq!(sum_matrix.get_val_unchecked(0, 2), 3);
    }

    #[test]
    fn test_sum_matrix_reports_error() {
        let matrix = WrappedMatrix::new(3, 3);
        let mut sum_matrix = SumMatrix::new(matrix);

        assert!(sum_matrix.set_val(3, 0, 1).is_err());
        assert!(sum_matrix.set_val(0, 3, 1).is_err());
        assert!(sum_matrix.get_val(3, 0).is_err());
        assert!(sum_matrix.get_val(0, 3).is_err());
    }

    #[test]
    fn test_get_ratio() {
        let matrix = WrappedMatrix::new(3, 3);
        let mut sum_matrix = SumMatrix::new(matrix);

        sum_matrix.set_val(0, 0, 1).unwrap();
        sum_matrix.set_val(0, 1, 2).unwrap();
        sum_matrix.set_val(0, 2, 3).unwrap();

        assert_eq!(sum_matrix.get_row_sum(0), 6);
        let epsilon = 1e-6;
        assert!((sum_matrix.get_ratio(0, 0).unwrap() - 1.0 / 6.0).abs() < epsilon);
        assert!((sum_matrix.get_ratio(0, 1).unwrap() - 2.0 / 6.0).abs() < epsilon);
        assert!((sum_matrix.get_ratio(0, 2).unwrap() - 3.0 / 6.0).abs() < epsilon);
    }

    #[test]
    fn test_sum_matrix_get_ratio_reports_division_by_zero() {
        let matrix = WrappedMatrix::new(3, 3);
        let mut sum_matrix = SumMatrix::new(matrix);

        sum_matrix.set_val(0, 0, 0).unwrap();
        sum_matrix.set_val(0, 1, 0).unwrap();
        sum_matrix.set_val(0, 2, 0).unwrap();

        assert_eq!(sum_matrix.get_row_sum(0), 0);
        assert!(sum_matrix.get_ratio(0, 0).is_err());
        assert!(sum_matrix.get_ratio(0, 1).is_err());
        assert!(sum_matrix.get_ratio(0, 2).is_err());
    }
}
