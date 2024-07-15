#pragma once
#define _CRT_SECURE_NO_WARNINGS
#ifndef _HMATH_H_
#define _HMATH_H_

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include<stdio.h>
#include<math.h>
typedef float logfloat;
typedef void* Ptr;
typedef float* Vector;
typedef float** Matrix;
typedef int* IntVec;
typedef int* ShortVec;
typedef int** IntMat;

#define pi 3.1416

typedef Vector SVector;
typedef Matrix STriMat;
typedef Matrix SMatrix;

typedef struct {	/*���ڴ��渴��������ṹ����Ҫ����Ƶ��Ϳ��ٸ���Ҷ�任*/
	Vector real;	/*ʵ���͸������ֱ��ֱ�洢�����������У���С��ΪN*/
	Vector imag;
	int N;
}XFFT;

/*------------------��������----------------------------------------*/
/*-----------------------------------------------------------------*/
/*�ɹ���������ͨ�õĺ���VectorSize*/
/*�ɹ�Vector��SVector����ͨ�õĺ���������Create��Free������*/
/*-----------------------------------------------------------------*/
/*-----------------------------------------------------------------*/

/*------------------Vector------------------*/
/*------------------float������------------------*/
Vector CreateVector(int n);
int VectorSize(Vector v);	/*���������Ĵ�С*/
void ZeroVector(Vector v);	/*��������*/
void ShowVector(Vector v);
void ShowVectorE(Vector v);	/*�ÿ�ѧ������ʽ��ӡ*/
void FreeVector(Vector v);
float FindMax(Vector v);	/*�ҵ������е����Ԫ��*/
int FindMaxIndex(Vector v);	/*�ҵ����������Ԫ�ص��±�*/
void CopyVector(Vector v1, Vector v2);	/*��v1���Ƶ�v2��v2Ҫ���ȴ�����*/
void CopyVector2(Vector v1, IntVec ind1, Vector v2, IntVec ind2);	/*�����±�ind1��ind2������v1��v2*/
void WriteVectorE(FILE* f, Vector v);	/*�ÿ�ѧ����������д���ļ�*/
void WriteVector(FILE* f, Vector v);
void LoadVector(FILE * f, Vector v);	/*���ļ�����������*/
void LoadVectorE(FILE * f, Vector v);	/*�ÿ�ѧ�������ļ�����������*/

/*------------------IntVec------------------*/
/*------------------��������------------------*/
//VectorSize(IntVec v) ����ȡ��С
IntVec CreateIntVec(int n);	/*����һ����������*/
void FreeIntVec(IntVec v);
void ShowIntVec(IntVec v);
void WriteIntVec(FILE* f, IntVec v);	/*��һ���������浽�ļ�f��*/
void ZeroIntVec(IntVec v);
void CopyIntVec(IntVec v1, IntVec v2);

/*------------------SVector------------------*/
/*------------------float�͹�������------------------*/
//������������ʹ��Vector�ĺ���
SVector CreateSVector(int n);
void FreeSVector(SVector v);


/*------------------������������----------------------------------------*/
/*-----------------------------------------------------------------*/
/*-----------------------------------------------------------------*/
/*-----------------------------------------------------------------*/


/*------------------XFFT------------------*/
/*------------------�洢��������------------------*/
void InitXFFT(XFFT* xfftP, int N);	/*����N����ʼ��XFFT*/
void ShowXFFT(XFFT xf);	/*����Ļ�ϴ�ӡһ�鸴��*/
void ShowXFFTE(XFFT xf);	/*����Ļ�ϴ�ӡ�������ÿ�ѧ����*/
void FreeXFFT(XFFT* xfftP);	
int XFFTSize(XFFT x);	/*������������Ĵ�С*/
void XFFTToVector(XFFT xf, Vector* vp, int power2Flag);	/*�������ṹת��Ϊ������ʽ������ĳ���Ϊ��������ĳ��ȵ�����������洢ʵ�����鲿*/
void VectorToXFFT(XFFT* xfp, Vector v);	/*������Ϊ2N������ת��ΪXFFT��ʽ��Ҫ����v�м���洢ʵ�����鲿*/



/*------------------���󲿷�----------------------------------------*/
/*-----------------------------------------------------------------*/
/*�ɹ�IntMat��Matrix��SMatrixͨ�õĺ���NumRows��NumCols*/
/*�ɹ�Matrix��SMatrixͨ�õĺ���������Create��Free������*/
/*-----------------------------------------------------------------*/
/*-----------------------------------------------------------------*/

/*------------------IntMat------------------*/
/*------------------���ξ���------------------*/
//NumRows(IntMat m),NumCols(IntMat m)����ȡ���������������
IntMat CreateIntMat(int nrows, int ncols);
void FreeIntMat(IntMat m);
void ZeroIntMat(IntMat m);
void WriteIntMat(FILE* f, IntMat m);
void ShowIntMat(IntMat m);


/*------------------Matrix------------------*/
/*------------------float�;���------------------*/
Matrix CreateMatrix(int nrows, int ncols);	/*����������Ҫ������������ֵ������*/
int NumRows(Matrix m);	/*���ؾ���m������*/
int NumCols(Matrix m);	/*���ؾ���m������*/
void ShowMatrix(Matrix m);
void FreeMatrix(Matrix m);
void ZeroMatrix(Matrix m);
void CopyMatrix(Matrix m1, Matrix m2);	
void CopyMatToTri(Matrix m1, STriMat m2);	
void WriteMatrix(FILE* f, Matrix m);
void LoadMatrix(FILE* f, Matrix m);

/*------------------SMatrix------------------*/
/*------------------float�͹�������------------------*/
//������������ʹ��Matrix�ĺ���
SMatrix CreateSMatrix(int nrows, int ncols);
void FreeSMatrix(SMatrix m);

/*------------------STriMat------------------*/
/*------------------float�͹��������Ǿ���------------------*/
Matrix CreateSTriMat(int size);
int STriMatSize(STriMat m);	
void ShowSTriMat(STriMat m);
void FreeSTriMat(STriMat m);
void ZeroSTriMat(STriMat m);
void CopySTriMat(STriMat m1, STriMat m2);
void WriteSTriMat(FILE* f, STriMat m);
void LoadStriMat(FILE*f, STriMat m);

/*------------------Methods for Shared Vec or Mat------------------*/
Ptr GetHook(Ptr m);
void SetHook(Ptr m, Ptr ptr);
void SetUse(Ptr m, int n);


/*------------------�������������ʽ�����------------------*/
/*------------------�㷨����Ϊ�ԳƵķ������ֻ�����������Ǿ���------------------*/
int Choleski(STriMat A, Matrix L);
void MSolve(Matrix L, int i, Vector x, Vector y);
logfloat CovInvert(STriMat c, STriMat invc);	
logfloat CovDet(STriMat c);

/*------------------һЩ��������------------------*/
int mod(int a, int b);	
void reshape(Matrix* mp, Vector v, int r, int c,int dim);	

#ifdef __cplusplus
}
#endif // __cplusplus

#endif
