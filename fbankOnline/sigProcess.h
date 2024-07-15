#pragma once
#define _CRT_SECURE_NO_WARNINGS
#ifndef _SIGPROCESS_H_
#define _SIGPROCESS_H_

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include"hmath.h"


void circshift(Vector v, int shift);
int find(Vector v, float thre, int FrontOrEndFlag);
void pad_signal(Vector* yP, Vector x, int Npad);
void unpad_signal(Vector* yP, Vector x, int res, int target_sz );

/*frame the origin int signal according to frame size and frame shift size*/
/*return a float matrix,each row contains a frame*/
Matrix frameRawSignal(IntVec v, int wlen, int inc,float preEmphasiseCoefft,int enableHamWindow);




/*---------Ԥ����----------------*/
/*zero mean a complete speech waveform nSamples long*/
void ZeroMean(IntVec data);
/*Ԥ����,kһ��ȡ0.9-1�����ȡk=0,�����κ�Ԥ����*/
void PreEmphasise(Vector s, float k);

/*-----------------�����������Լ��Ӵ�����---------------*/
/*����������,���λ��*/
float calBrightness(Vector fftx);
/*�����Ӵ�����,����Ӵ�����ռ�������ı�ֵ*/
void calSubBankE(Vector fftx, Vector subBankEnergy);
/*��������ʣ����ع����������һ֡�Ĳ�����ĸ���*/
float zeroCrossingRate(Vector s, int frameSize);

/*������ϵ��*/
/*�����ֱ�Ϊ��������Ҫ��ֵ�ÿ֡�Ĳ���������֡����ÿ���ƶ��Ĳ��������ϵ����ԭ�źŵľ��룬0��0���Ƿ�򵥲��*/
void Regress(float* data, int vSize, int n, int step, int offset, int delwin, int head, int tail, int simpleDiffs);

void RegressMat(Matrix* m,int delwin, int regressOrder);

void NormaliseLogEnergy(float *data, int n, int step, float silFloor, float escale);

void ZNormalize(float *data, int vSize, int n, int step);

/* GenHamWindow: generate precomputed Hamming window function */
Vector GenHamWindow(int frameSize);
/*Apply Hamming Window to Speech frame s*/
void Ham(Vector s, Vector hamWin, int hamWinSize);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // !_SIGPROCESS_H_
