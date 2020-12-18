#ifndef QEXTENDEDGENERICMATRIX_H
#define QEXTENDEDGENERICMATRIX_H

#include <QObject>
#include <QGenericMatrix>

#include <cmath>
#include <iostream>

template <int N, int M, typename T>
class QExtendedGenericMatrix : public QGenericMatrix<N, M, T>
{

public:

    /**
     * @brief QExtendedGenericMatrix<N, M, T> Creates an identity matrix with M rows and N columns
     */
    QExtendedGenericMatrix<N, M, T>();

    /**
     * @brief QExtendedGenericMatrix<N, M, T> Constructs a matrix with M rows and N columns.
     *                                        The matrix entries are given by parameter values
     * @param values Matrix entries provided in row major order
     */
    QExtendedGenericMatrix<N, M, T>(const T *values);

    /**
     * @brief ~QExtendedGenericMatrix<N, M, T> Destructor of this class
     */
    ~QExtendedGenericMatrix<N, M, T>();

    /**
     * @brief determinant Calculates the determinant of this matrix
     * @return Returns the determinant of this matrix
     */
    T determinant() const;

    /**
     * @brief inverted Calculates the inverse of this matrix
     * @param invertible After execution this parameter will be 'true' iff this matrix is invertible (determinant != 0)
     *        or 'false' iff this matrix is not invertible (determinant == 0)
     * @return Returns a new matrix which is the inverse of this matrix
     */
    QExtendedGenericMatrix<N, M, T> inverted(bool *invertible = NULL) const;

    /**
     * @brief row Extracts the i th row of this matrix
     * @param i The index of the row
     * @return Returns the i th row of this matrix
     */
    QExtendedGenericMatrix<N, 1, T> row(const uint &i) const;

    /**
     * @brief col Extracts the j th column of this matrix
     * @param j The index of the column
     * @return Returns the j th column of this matrix
     */
    QExtendedGenericMatrix<1, M, T> col(const uint &j) const;

    /**
     * @brief operator << Writes this matrix to output stream os
     * @param os Output stream
     * @param matrix The matrix whose data is written to the output stream os
     * @return Returns a reference to the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const QExtendedGenericMatrix<N, M, T>& matrix)
    {

        int p = os.precision();
        os.precision(5);
        os.setf(std::ios::fixed);

        for(int i = 0; i < M; ++i)
        {

            for(int j = 0; j < N; ++j)
            {
                os << matrix(i, j) << " ";
            }

            os << std::endl;

        }

        os.precision(p);
        os.unsetf(std::ios::fixed);

        return os;

    }

    /**
     * @brief transposed Determines the transposed of this matrix
     * @return Returns the transposed of this matrix
     */
    QExtendedGenericMatrix<N, M, T> transposed() const;

    /**
     * @brief operator *= Multiplies each entry of this matrix by a factor
     * @param factor The factor this matrix is multiplied by
     * @return Returns a copy of this matrix after multiplication by the factor
     */
    QExtendedGenericMatrix<N, M, T> operator*=(T factor);

    /**
     * @brief operator += Adds matrix other to this matrix
     * @param other
     * @return
     */
    QExtendedGenericMatrix<N, M, T> operator+=(const QGenericMatrix<N, M, T> &other);

    /**
     * @brief operator -= Subtracts matrix other from this matrix
     * @param other
     * @return
     */
    QExtendedGenericMatrix<N, M, T> operator-=(const QGenericMatrix<N, M, T> &other);

    /**
     * @brief operator /= Divides each entry of this matrix by a factor
     * @param divisor The factor this matrix is divided by
     * @return Returns a copy of this matrix after division by the factor
     */
    QExtendedGenericMatrix<N, M, T> operator/=(T divisor);

    /**
     * @brief operator * Multiplies this a matrix with another matrix
     * @param m1 First matrix
     * @param m2 Second matrix
     * @return Returns a new matrix containing the product of m1 and m2
     */
    friend QExtendedGenericMatrix<N, M, T> operator*(const QGenericMatrix<N, M, T> &m1, const QGenericMatrix<M, N, T> &m2)
    {
        auto m = m1 * m2;
        return (*static_cast<QExtendedGenericMatrix<N, M, T>*>(&m));
    }

    /**
     * @brief operator * Multiplies a matrix by a factor
     * @param factor The factor this matrix is multiplied by
     * @param matrix The matrix factor
     * @return Returns a copy of this matrix after multiplication by the factor
     */
    friend QExtendedGenericMatrix<N, M, T> operator*(T factor, const QGenericMatrix<N, M, T> &matrix)
    {
        auto m = factor * matrix;
        return (*static_cast<QExtendedGenericMatrix<N, M, T>*>(&m));
    }

    /**
     * @brief operator * Multiplies a matrix by a factor
     * @param matrix The matrix factor
     * @param factor The factor this matrix is multiplied by
     * @return Returns a copy of this matrix after multiplication by the factor
     */
    friend QExtendedGenericMatrix<N, M, T> operator*(const QGenericMatrix<N, M, T> &matrix, T factor)
    {
        auto m = matrix * factor;
        return (*static_cast<QExtendedGenericMatrix<N, M, T>*>(&m));
    }

    /**
     * @brief operator + Adds matrices m1 and m2
     * @param m1 The first matrix
     * @param m2 The second matrix
     * @return Returns a new matrix after addition of m1 and m2
     */
    friend QExtendedGenericMatrix<N, M, T> operator+(const QGenericMatrix<N, M, T> &m1, const QGenericMatrix<N, M, T> &m2)
    {
        auto m = m1 + m2;
        return (*static_cast<QExtendedGenericMatrix<N, M, T>*>(&m));
    }

    /**
     * @brief operator - Subtracts matrix m2 from matrix m1
     * @param m1 The minuend
     * @param m2 The subtrahend
     * @return Returns a new matrix after subtraction
     */
    friend QExtendedGenericMatrix<N, M, T> operator-(const QGenericMatrix<N, M, T> &m1, const QGenericMatrix<N, M, T> &m2)
    {
        auto m = m1 - m2;
        return (*static_cast<QExtendedGenericMatrix<N, M, T>*>(&m));
    }

    /**
     * @brief operator - Negates argument matrix
     * @param matrix The mateix which is negated
     * @return Returns a new matrix negation
     */
    friend QExtendedGenericMatrix<N, M, T> operator-(const QGenericMatrix<N, M, T> &matrix)
    {
        auto m = -matrix;
        return (*static_cast<QExtendedGenericMatrix<N, M, T>*>(&m));
    }

    /**
     * @brief operator / Divides each entry of this matrix by divisor
     * @param matrix The matrix which is divided by divisor
     * @param divisor The number by which matrix is divided
     * @return Returns a new matrix after division
     */
    friend QExtendedGenericMatrix<N, M, T> operator/(const QGenericMatrix<N, M, T> &matrix, T divisor)
    {
        QGenericMatrix<N, M, T> m = matrix / divisor;
        return (*static_cast<QExtendedGenericMatrix<N, M, T>*>(&m));
    }

protected:

    /**
     * @brief Calculates the LU decomposition of a square matrix A in place
     * @param LUPDecompose A The square matrix whose LU decomposition is calculated
     * @param Tol Tolerance value to detect degenerate matrices
     * @param P Stores pivoting data
     * @return Returns 0 if the LU decomposition was successful or 1 if A is degenerate
     */
    int LUPDecompose(QExtendedGenericMatrix<N, M, T> &A, T Tol, int *P) const;

    /**
     * @brief LUPDeterminant Calculates the determinant of matrix A
     * @param A The matrix whose determinant is calculated
     * @param P Stores pivoting data
     * @return Returns the determinant of A for square matrices. If A is non-square, this function returns 0.
     */
    T LUPDeterminant(QExtendedGenericMatrix<N, M, T> &A, int *P) const;

    /**
     * @brief LUPInvert Calculates the inverse of matrix A
     * @param A The matrix whose inverse is calculated
     * @param P Stores pivoting data
     * @return Returns the inverse matrix of A or the identity matrix if A is not invertible
     */
    void LUPInvert(QExtendedGenericMatrix<N, M, T> &A, int *P, QExtendedGenericMatrix<N, M, T> &IA) const;

};

template <typename T>
using QEMatrix2x2 = QExtendedGenericMatrix<2, 2, T>;

template <typename T>
using QEMatrix2x3 = QExtendedGenericMatrix<2, 3, T>;

template <typename T>
using QEMatrix2x4 = QExtendedGenericMatrix<2, 4, T>;

template <typename T>
using QEMatrix3x2 = QExtendedGenericMatrix<3, 2, T>;

template <typename T>
using QEMatrix3x3 = QExtendedGenericMatrix<3, 3, T>;

template <typename T>
using QEMatrix3x4 = QExtendedGenericMatrix<3, 4, T>;

template <typename T>
using QEMatrix4x2 = QExtendedGenericMatrix<4, 2, T>;

template <typename T>
using QEMatrix4x3 = QExtendedGenericMatrix<4, 3, T>;

template <typename T>
using QEMatrix4x4 = QExtendedGenericMatrix<4, 4, T>;

template <int N, int M, typename T>
QExtendedGenericMatrix<N, M, T>::QExtendedGenericMatrix() : QGenericMatrix<N, M, T>()
{

}

template <int N, int M, typename T>
QExtendedGenericMatrix<N, M, T>::QExtendedGenericMatrix(const T *values) : QGenericMatrix<N, M, T>(values)
{

}

template <int N, int M, typename T>
QExtendedGenericMatrix<N, M, T>::~QExtendedGenericMatrix()
{

}

template <int N, int M, typename T>
T QExtendedGenericMatrix<N, M, T>::determinant() const
{

    if(M != N)
    {

        return 0;

    }
    else
    {

        T *values_copy = new T[N * M];
        this->copyDataTo(values_copy);
        QExtendedGenericMatrix<N, M, T> A(values_copy);
        int *P = new int[N + 1];

        const int retVal = LUPDecompose(A, 1e-3, P);

        T det = 0;
        if(retVal == 1)
        {
            det = LUPDeterminant(A, P);
        }

        delete[] P;

        return det;

    }

}

template <int N, int M, typename T>
QExtendedGenericMatrix<N, M, T> QExtendedGenericMatrix<N, M, T>::inverted(bool *invertible) const
{

    *invertible = (determinant() != 0);

    if(*invertible)
    {

        T *values_copy = new T[N * M];
        this->copyDataTo(values_copy);
        QExtendedGenericMatrix<N, M, T> A(values_copy);

        T *values = new T[N * M];
        QExtendedGenericMatrix<N, M, T> IA(values);

        int *P = new int[N + 1];

        const int retVal = LUPDecompose(A, 1e-3, P);

        if(retVal == 1)
        {
            LUPInvert(A, P, IA);
        }
        else
        {
            IA.setToIdentity();
        }

        delete[] P;

        return IA;

    }
    else
    {

        QExtendedGenericMatrix<N, M, T> inv;
        inv.setToIdentity();
        return inv;

    }

}

template <int N, int M, typename T>
QExtendedGenericMatrix<N, 1, T> QExtendedGenericMatrix<N, M, T>::row(const uint &i) const
{
    QExtendedGenericMatrix<N, 1, T> row;

    for(int j = 0; j < N; ++j)
    {
        row(0, j) = (*this)(i, j);
    }

    return row;

}

template <int N, int M, typename T>
QExtendedGenericMatrix<1, M, T> QExtendedGenericMatrix<N, M, T>::col(const uint &j) const
{

    QExtendedGenericMatrix<1, M, T> col;

    for(int i = 0; i < M; ++i)
    {
        col(i, 0) = (*this)(i, j);
    }

    return col;

}

template <int N, int M, typename T>
QExtendedGenericMatrix<N, M, T> QExtendedGenericMatrix<N, M, T>::transposed() const
{
    auto *m = static_cast<QGenericMatrix<N, M, T>*>(this);
    auto m_T = m->transposed();
    return (*static_cast<QExtendedGenericMatrix<N, M, T>*>(&m_T));
}

template <int N, int M, typename T>
QExtendedGenericMatrix<N, M, T> QExtendedGenericMatrix<N, M, T>::operator*=(T factor)
{
    auto *m_p = static_cast<QGenericMatrix<N, M, T>*>(this);
    (*m_p) *= factor;
    return (*static_cast<QExtendedGenericMatrix<N, M, T>*>(m_p));
}

template <int N, int M, typename T>
QExtendedGenericMatrix<N, M, T> QExtendedGenericMatrix<N, M, T>::operator+=(const QGenericMatrix<N, M, T> &other)
{
    auto *m_p = static_cast<QGenericMatrix<N, M, T>*>(this);
    (*m_p) += other;
    return (*static_cast<QExtendedGenericMatrix<N, M, T>*>(m_p));
}

template <int N, int M, typename T>
QExtendedGenericMatrix<N, M, T> QExtendedGenericMatrix<N, M, T>::operator-=(const QGenericMatrix<N, M, T> &other)
{
    auto *m_p = static_cast<QGenericMatrix<N, M, T>*>(this);
    (*m_p) -= other;
    return (*static_cast<QExtendedGenericMatrix<N, M, T>*>(m_p));
}

template <int N, int M, typename T>
QExtendedGenericMatrix<N, M, T> QExtendedGenericMatrix<N, M, T>::operator/=(T divisor)
{
    auto *m_p = static_cast<QGenericMatrix<N, M, T>*>(this);
    (*m_p) /= divisor;
    return (*static_cast<QExtendedGenericMatrix<N, M, T>*>(m_p));
}

template <int N, int M, typename T>
int QExtendedGenericMatrix<N, M, T>::LUPDecompose(QExtendedGenericMatrix<N, M, T> &A, T Tol, int *P) const
{

    int i, j, k, imax;
    double maxA, absA;

    for (i = 0; i <= N; i++)
        P[i] = i; //Unit permutation matrix, P[N] initialized with N

    for (i = 0; i < N; i++) {
        maxA = 0.0;
        imax = i;

        for (k = i; k < N; k++)
            if ((absA = std::fabs(A(k, i))) > maxA) {
                maxA = absA;
                imax = k;
            }

        if (maxA < Tol) return 0; //failure, matrix is degenerate

        if (imax != i) {
            //pivoting P
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            //pivoting rows of A
            std::array<T, N> A_i;
            for(int l = 0; l < N; ++l)
            {
                A_i.at(l) = A(i, l);
            }
            for(int l = 0; l < N; ++l)
            {
                A(i, l) = A(imax, l);
            }
            for(int l = 0; l < N; ++l)
            {
                A(imax, l) = A_i.at(l);
            }

            //counting pivots starting from N (for determinant)
            P[N]++;
        }

        for (j = i + 1; j < N; j++) {
            A(j, i) /= A(i, i);

            for (k = i + 1; k < N; k++)
                A(j, k) -= A(j, i) * A(i, k);
        }
    }

    return 1;  //decomposition done
}

template <int N, int M, typename T>
T QExtendedGenericMatrix<N, M, T>::LUPDeterminant(QExtendedGenericMatrix<N, M, T> &A, int *P) const
{

    T det = A(0, 0);

    for (int i = 1; i < N; i++)
        det *= A(i, i);

    return (P[N] - N) % 2 == 0 ? det : -det;
}

template <int N, int M, typename T>
void QExtendedGenericMatrix<N, M, T>::LUPInvert(QExtendedGenericMatrix<N, M, T> &A, int *P, QExtendedGenericMatrix<N, M, T> &IA) const
{

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            IA(i, j) = P[i] == j ? 1.0 : 0.0;

            for (int k = 0; k < i; k++)
                IA(i, j) -= A(i, k) * IA(k, j);
        }

        for (int i = N - 1; i >= 0; i--) {
            for (int k = i + 1; k < N; k++)
                IA(i, j) -= A(i, k) * IA(k, j);

            IA(i, j) /= A(i, i);
        }
    }

}

#endif // QEXTENDEDGENERICMATRIX_H
